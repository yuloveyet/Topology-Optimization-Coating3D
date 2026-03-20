import os
import time

import numpy as np
import ufl
from mpi4py import MPI

import dolfinx.io
from dolfinx.fem import (
    Constant,
    Function,
    FunctionSpace,
    VectorFunctionSpace,
    dirichletbc,
    locate_dofs_topological,
)
from dolfinx.mesh import locate_entities_boundary, meshtags as dolfinx_meshtags
from dolfinx import la
from dolfinx.fem import form, assemble_scalar
from dolfinx.fem.petsc import (
    create_matrix,
    create_vector,
    assemble_matrix,
    assemble_vector,
)
from petsc4py import PETSc
from ufl import Measure

from fenitop.parameterize import DensityFilter, Heaviside
from fenitop.optimize import optimality_criteria, mma_optimizer
from fenitop.utility import (
    Communicator,
    Plotter,
    LinearProblem,
    get_date_str,
    save_xdmf,
)


class CG1Filter:
    """A PDE filter mapping from a CG1 space to a CG1 space.
    Used for the secondary smoothing of the base physical density
    to compute smooth gradient norms for the shell/coating."""

    def __init__(self, comm, V, u_out_func, R=1.0, petsc_options={}):
        self.V = V
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        self.u_in = Function(V)
        self.u_out = u_out_func
        self.u_out_wrap = la.create_petsc_vector_wrap(self.u_out.x)
        self.lambda_adj = Function(V)
        self.lambda_wrap = la.create_petsc_vector_wrap(self.lambda_adj.x)

        dx = ufl.Measure("dx", metadata={"quadrature_degree": 2})
        Kf_expr = (R**2 * ufl.dot(ufl.grad(u), ufl.grad(v)) + u * v) * dx
        M_expr = u * v * dx  # Mass matrix for CG1 to CG1

        Kf_form, M_form = form(Kf_expr), form(M_expr)
        Kf_mat, self.M_mat = create_matrix(Kf_form), create_matrix(M_form)

        self.solver = PETSc.KSP().create(comm)
        self.solver.setOperators(Kf_mat)
        prefix = f"cg1filter_{id(self)}"
        self.solver.setOptionsPrefix(prefix)

        opts = PETSc.Options()
        opts.prefixPush(prefix)
        for key, value in petsc_options.items():
            opts[key] = value
        opts.prefixPop()
        self.solver.setFromOptions()
        Kf_mat.setOptionsPrefix(prefix)
        Kf_mat.setFromOptions()

        assemble_matrix(Kf_mat, Kf_form)
        Kf_mat.assemble()
        assemble_matrix(self.M_mat, M_form)
        self.M_mat.assemble()

        self.vec_rhs = self.u_in.vector.copy()

    def forward(self, u_in_func):
        self.M_mat.mult(u_in_func.vector, self.vec_rhs)
        self.solver.solve(self.vec_rhs, self.u_out_wrap)
        self.u_out.x.scatter_forward()
        return self.u_out

    def backward(self, df_dout_vec):
        """Adjoint pass: solves Kf * lambda = df_dout_vec, then returns M * lambda."""
        self.solver.solve(df_dout_vec, self.lambda_wrap)
        self.lambda_adj.x.scatter_forward()
        df_din_vec = self.lambda_adj.vector.copy()
        self.M_mat.mult(self.lambda_adj.vector, df_din_vec)
        return df_din_vec


def _prj(v, eta, beta):
    """Projection function (tanh-based)."""
    denom = ufl.tanh(beta * eta) + ufl.tanh(beta * (1 - eta))
    return (ufl.tanh(beta * eta) + ufl.tanh(beta * (v - eta))) / denom


def form_fem_coating_3d(fem, opt):
    mesh = fem["mesh"]
    dim = mesh.topology.dim

    V = VectorFunctionSpace(mesh, ("CG", 1))
    S0 = FunctionSpace(mesh, ("DG", 0))
    S = FunctionSpace(mesh, ("CG", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    u_field = Function(V)
    lambda_field = Function(V)
    rho_field = Function(S0)
    rho_base = Function(S)
    rho_Nf = Function(S)  # Smoothed base density for shell gradient

    solid_mask = Function(S0)
    solid_mask.vector.array[:] = 0.0

    E0 = fem["young's modulus"]
    nu = fem["poisson's ratio"]
    penal = opt["penalty"]
    penal_shell = opt.get("penal_shell", 1.0)
    lambda_m = opt.get("lambda_m", 0.7)
    lambda_E = opt.get("lambda_E", 0.4)
    lambda_Emin = opt.get("lambda_Emin", opt.get("epsilon", 1e-6))
    q_ext = opt.get("q_ext", 1.0)

    # Shell gradient norm + projection based on rho_Nf
    # In fenitop, filter_radius is directly used as the PDE parameter R.
    # The gradient of a PDE-filtered step function maxes out at 1/(2R).
    # To normalize this gradient to exactly 1.0 at the boundary, alpha must be 2*R.
    rmin_shell = opt.get("filter_radius_shell", opt["filter_radius"])
    alpha = Constant(mesh, float(2.0 * rmin_shell))
    # Add 1e-12 to avoid sqrt(0) singularity which causes NaN in sensitivities!
    grad_norm = alpha * ufl.sqrt(ufl.inner(ufl.grad(rho_Nf), ufl.grad(rho_Nf)) + 1e-12)

    shell_eta = opt.get("shell_eta", 0.5)
    shell_beta = Constant(mesh, float(1.0))
    rho_shell_expr = _prj(grad_norm, shell_eta, shell_beta)
    rho_shell = solid_mask * 1.0 + (1.0 - solid_mask) * rho_shell_expr

    # Material interpolation
    rho_total = lambda_m * rho_base + (1.0 - lambda_m * rho_base) * rho_shell
    rho_base_p = rho_base**penal
    E = E0 * (
        lambda_Emin
        + q_ext * (lambda_E - lambda_Emin) * rho_base_p
        + q_ext * (1.0 - lambda_E * rho_base_p) * rho_shell**penal_shell
    )

    def epsilon(w):
        return ufl.sym(ufl.grad(w))

    _lambda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / (2 * (1 + nu))

    def sigma(w):
        return 2 * mu * epsilon(w) + _lambda * ufl.tr(epsilon(w)) * ufl.Identity(len(w))

    fdim = dim - 1
    disp_facets = locate_entities_boundary(mesh, fdim, fem["disp_bc"])
    bc = dirichletbc(Constant(mesh, np.full(dim, 0.0)), locate_dofs_topological(V, fdim, disp_facets), V)

    tractions, facets, markers = [], [], []
    for marker, (traction, traction_bc) in enumerate(fem["traction_bcs"]):
        tractions.append(Constant(mesh, np.array(traction, dtype=float)))
        current_facets = locate_entities_boundary(mesh, fdim, traction_bc)
        facets.extend(current_facets)
        markers.extend([marker] * len(current_facets))

    facets = np.array(facets, dtype=np.int32)
    markers = np.array(markers, dtype=np.int32)
    if facets.size > 0:
        _, unique_indices = np.unique(facets, return_index=True)
        facets, markers = facets[unique_indices], markers[unique_indices]
        sorted_indices = np.argsort(facets)
        facets = facets[sorted_indices]
        markers = markers[sorted_indices]
        
    facet_tags = dolfinx_meshtags(mesh, fdim, facets, markers)

    metadata = {"quadrature_degree": fem["quadrature_degree"]}
    dx = Measure("dx", metadata=metadata)
    ds = Measure("ds", domain=mesh, metadata=metadata, subdomain_data=facet_tags)

    b = Constant(mesh, np.array(fem["body_force"], dtype=float))
    lhs = ufl.inner(sigma(u), epsilon(v)) * dx
    rhs = ufl.dot(b, v) * dx
    for marker, t in enumerate(tractions):
        rhs += ufl.dot(t, v) * ds(marker)

    linear_problem = LinearProblem(u_field, lambda_field, lhs, rhs, l_vec=None, spring_vec=None, bcs=[bc], petsc_options=fem["petsc_options"])

    # Pass void mask to define physical volume for Vf normalization
    void_mask_for_vol = Function(S0)
    num_elems_local = void_mask_for_vol.vector.array.size
    centers = S0.tabulate_dof_coordinates()[:num_elems_local]
    void_mask_for_vol.vector.array[:] = opt["void_zone"](centers.T).astype(float)
    
    opt["compliance"] = ufl.inner(sigma(u_field), epsilon(u_field)) * dx
    
    # Volume calculation in whole mesh (extended), but normalized by physical domain volume
    opt["volume"] = rho_total * dx
    opt["total_volume"] = (1.0 - void_mask_for_vol) * dx

    return (
        linear_problem,
        u_field,
        rho_field,
        rho_base,
        rho_Nf,
        solid_mask,
        shell_beta,
        rho_shell_expr,
        rho_total,
    )


def project_expression(expr, V, mesh, quadrature_degree):
    out = Function(V)
    w, phi = ufl.TrialFunction(V), ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": quadrature_degree})
    a_form, L_form = form(w * phi * dx), form(expr * phi * dx)
    A, b = create_matrix(a_form), create_vector(L_form)
    assemble_matrix(A, a_form)
    A.assemble()
    assemble_vector(b, L_form)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    solver = PETSc.KSP().create(mesh.comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("gamg")
    solver.setFromOptions()
    x = la.create_petsc_vector_wrap(out.x)
    solver.solve(b, x)
    out.x.scatter_forward()
    solver.destroy(); A.destroy(); b.destroy()
    return out


def save_density_fields_xdmf(mesh, fields, path):
    os.makedirs(path, exist_ok=True)
    out_path = os.path.join(path, "optimized_density_fields.xdmf")
    with dolfinx.io.XDMFFile(mesh.comm, out_path, "w") as xdmf:
        xdmf.write_mesh(mesh)
        for name, f in fields.items():
            f.name = name
            xdmf.write_function(f)


def topopt_coating_3d(fem, opt):
    comm = MPI.COMM_WORLD
    (
        linear_problem, u_field, rho_field, rho_base, rho_Nf,
        solid_mask, shell_beta, rho_shell_expr, rho_total_expr
    ) = form_fem_coating_3d(fem, opt)

    density_filter = DensityFilter(comm, rho_field, rho_base, opt["filter_radius"], fem["petsc_options"])
    heaviside = Heaviside(rho_base)
    # The secondary filter for the shell!
    rmin_shell = opt.get("filter_radius_shell", opt["filter_radius"])
    shell_filter = CG1Filter(comm, rho_base.function_space, rho_Nf, rmin_shell, fem["petsc_options"])

    # Forms for manual sensitivities using UFL automatic differentiation
    C_form = form(opt["compliance"])
    V_form = form(opt["volume"])
    total_vol = comm.allreduce(assemble_scalar(form(opt["total_volume"])), op=MPI.SUM)

    # Partial derivatives w.r.t rho_base (direct) and rho_Nf (via shell)
    dC_drho_base_form = form(-ufl.derivative(opt["compliance"], rho_base))
    dC_drho_Nf_form = form(-ufl.derivative(opt["compliance"], rho_Nf))
    dV_drho_base_form = form(ufl.derivative(opt["volume"], rho_base))
    dV_drho_Nf_form = form(ufl.derivative(opt["volume"], rho_Nf))

    dC_drho_base_vec = create_vector(dC_drho_base_form)
    dC_drho_Nf_vec = create_vector(dC_drho_Nf_form)
    dV_drho_base_vec = create_vector(dV_drho_base_form)
    dV_drho_Nf_vec = create_vector(dV_drho_Nf_form)

    S_comm = Communicator(rho_base.function_space, fem["mesh_serial"])
    if comm.rank == 0:
        plotter = Plotter(fem["mesh_serial"])

    num_elems = rho_field.vector.array.size
    if not opt.get("use_oc", False):
        rho_old1, rho_old2 = np.zeros(num_elems), np.zeros(num_elems)
        low, upp = None, None

    centers = rho_field.function_space.tabulate_dof_coordinates()[:num_elems].T
    solid, void = opt["solid_zone"](centers), opt["void_zone"](centers)
    rho_ini = np.full(num_elems, opt["vol_frac"])
    rho_ini[solid], rho_ini[void] = 0.995, 0.005
    rho_field.vector.array[:] = rho_ini
    rho_min, rho_max = np.zeros(num_elems), np.ones(num_elems)
    rho_min[solid], rho_max[void] = 0.99, 0.01

    solid_mask.vector.array[:] = solid.astype(float)

    date_str = get_date_str(comm)
    filepath = "results/" + date_str + "/"
    if comm.rank == 0 and not os.path.exists(filepath):
        os.makedirs(filepath, exist_ok=True)
    comm.barrier()

    plot_freq = opt.get("plot_freq", 1)
    
    loop, opt_iter, beta, change = 0, 0, 1.0, 2 * opt["opt_tol"]
    while opt_iter < opt["max_iter"] and change > opt["opt_tol"]:
        opt_start_time = time.perf_counter()
        opt_iter += 1
        loop += 1

        # Base filter and projection
        density_filter.forward()
        if opt_iter % opt["beta_interval"] == 0 and beta < opt["beta_max"]:
            beta *= 2.0
            change = opt["opt_tol"] * 2.0
        heaviside.forward(beta, eta=opt.get("base_eta", 0.5))

        # SECONDARY FILTER FOR SHELL (Exactly mimics Matlab)
        shell_beta.value = float(beta)
        shell_filter.forward(rho_base)

        # Solve FEM
        linear_problem.solve_fem()

        # Compute Objective & Volume
        C_value = comm.allreduce(assemble_scalar(C_form), op=MPI.SUM)
        V_value = comm.allreduce(assemble_scalar(V_form), op=MPI.SUM) / total_vol

        # --- Compute Sensitivities ---
        # 1. Compliance
        with dC_drho_base_vec.localForm() as loc: loc.set(0)
        with dC_drho_Nf_vec.localForm() as loc: loc.set(0)
        assemble_vector(dC_drho_base_vec, dC_drho_base_form)
        assemble_vector(dC_drho_Nf_vec, dC_drho_Nf_form)
        dC_drho_base_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dC_drho_Nf_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Scale for log(C) objective BEFORE combining!
        # This makes the optimizer minimize log(C), effectively stabilizing the MMA constraints.
        c_scale = 1.0 / C_value if C_value > 1e-15 else 0.0
        dC_drho_base_vec.scale(c_scale)
        dC_drho_Nf_vec.scale(c_scale)
        
        # Backpropagate shell gradient sensitivity through the secondary filter
        dC_drho_base_from_shell = shell_filter.backward(dC_drho_Nf_vec)
        dC_drho_base_vec.axpy(1.0, dC_drho_base_from_shell) # dJ_base_total = dJ_base + dJ_base_from_shell

        # 2. Volume
        with dV_drho_base_vec.localForm() as loc: loc.set(0)
        with dV_drho_Nf_vec.localForm() as loc: loc.set(0)
        assemble_vector(dV_drho_base_vec, dV_drho_base_form)
        assemble_vector(dV_drho_Nf_vec, dV_drho_Nf_form)
        dV_drho_base_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dV_drho_Nf_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        dV_drho_base_from_shell = shell_filter.backward(dV_drho_Nf_vec)
        dV_drho_base_vec.axpy(1.0, dV_drho_base_from_shell)
        dV_drho_base_vec.scale(1.0 / total_vol)

        # Combine sensitivities (Base Projection -> Density Filter)
        sensitivities = [dC_drho_base_vec, dV_drho_base_vec]
        heaviside.backward(sensitivities)
        [dCdrho, dVdrho] = density_filter.backward(sensitivities)

        g_vec = np.array([V_value - opt["vol_frac"]])
        dJdrho, dgdrho = dCdrho, np.vstack([dVdrho])

        rho_values = rho_field.vector.array.copy()
        if opt.get("use_oc", False):
            rho_new, change = optimality_criteria(rho_values, rho_min, rho_max, g_vec, dJdrho, dgdrho[0], opt["move"])
        else:
            rho_new, change, low, upp = mma_optimizer(1, num_elems, opt_iter, rho_values, rho_min, rho_max, rho_old1, rho_old2, dJdrho, g_vec, dgdrho, low, upp, opt["move"])
            rho_old2 = rho_old1.copy(); rho_old1 = rho_values.copy()

        rho_field.vector.array = rho_new.copy()

        opt_time = time.perf_counter() - opt_start_time
        if comm.rank == 0:
            print(f"It.: {loop} C.: {C_value:.3f} Vf.: {V_value:.3f} beta: {beta} change: {change:.3f} t.: {opt_time: .3f}(s)", flush=True)

        if loop % plot_freq == 0 or change <= opt["opt_tol"] or opt_iter >= opt["max_iter"]:
            rho_tot_plot = project_expression(rho_total_expr, V=rho_base.function_space, mesh=fem["mesh"], quadrature_degree=fem["quadrature_degree"])
            
            # Save total density field as XDMF directly for ParaView (performed on all ranks)
            save_xdmf(fem["mesh"], rho_tot_plot, filepath, filename=f"design_{loop}.xdmf")

            values = S_comm.gather(rho_tot_plot)
            if comm.rank == 0:
                plotter.plot(values, loop, path=filepath, slice_normal='x', slice_origin=(5.0, 15.0, 5.0))

    rho_shell = project_expression(rho_shell_expr, V=rho_base.function_space, mesh=fem["mesh"], quadrature_degree=fem["quadrature_degree"])
    rho_total = project_expression(rho_total_expr, V=rho_base.function_space, mesh=fem["mesh"], quadrature_degree=fem["quadrature_degree"])

    if comm.rank == 0: print(f"Saving density fields to: {filepath}", flush=True)
    save_density_fields_xdmf(fem["mesh"], fields={"density_base": rho_base, "density_shell": rho_shell, "density_total": rho_total}, path=filepath)
    
    # Cleanup explicitly copied PETSc vectors
    dC_drho_base_from_shell.destroy()
    dV_drho_base_from_shell.destroy()
