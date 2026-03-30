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
    functionspace,
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

from fenitop.parameterize import DensityFilter, Heaviside, CG1Filter
from fenitop.optimize import optimality_criteria, mma_optimizer
from fenitop.utility import (
    Communicator,
    Plotter,
    LinearProblem,
    get_date_str,
    save_xdmf,
)


def form_fem_coating_3d(fem, opt):
    mesh = fem["mesh"]
    dim = mesh.topology.dim

    V = functionspace(mesh, ("CG", 1, (dim,)))
    S0 = functionspace(mesh, ("DG", 0))
    S = functionspace(mesh, ("CG", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    u_field = Function(V)
    lambda_field = Function(V)
    rho_field = Function(S0)
    rho_base = Function(S)  # Base density
    rho_nf = Function(S)    # Filtered base density
    rho_shell_func = Function(S) # The projected, physical shell density

    # Padding/void mask for zone-dependent q_ext
    padding_mask = Function(S0)
    num_elems_local = padding_mask.vector.array.size
    centers = S0.tabulate_dof_coordinates()[:num_elems_local]
    padding_mask.vector.array[:] = opt["void_zone"](centers.T).astype(float)

    E0 = fem["young's modulus"]
    nu = fem["poisson's ratio"]
    penal = opt["penalty"]
    penal_shell = opt.get("penal_shell", 1.0)
    lambda_m = opt.get("lambda_m", 0.7)
    lambda_Emin = opt.get("lambda_Emin", opt.get("epsilon", 1e-6))
    q_ext = opt.get("q_ext", 1.0)

    rmin_shell = opt.get("filter_radius_shell", opt["filter_radius"])
    shell_filter = CG1Filter(mesh.comm, S, rho_nf, rmin_shell, fem["petsc_options"])

    # --- Shell field generation (from filtered base field rho_nf) ---
    # 1. Normalized gradient of filtered field. alpha = R/sqrt(3) from Clausen et al.
    alpha_grad = Constant(mesh, float(rmin_shell / np.sqrt(3.0)))
    grad_norm = alpha_grad * ufl.sqrt(ufl.inner(ufl.grad(rho_nf), ufl.grad(rho_nf)) + 1e-12)

    # 2. Heaviside projection of the normalized gradient
    shell_eta = opt.get("shell_eta", 0.5)
    shell_beta = Constant(mesh, float(1.0))
    denominator = ufl.tanh(shell_beta * shell_eta) + ufl.tanh(shell_beta * (1.0 - shell_eta))
    rho_shell_expr = (ufl.tanh(shell_beta * shell_eta) + ufl.tanh(shell_beta * (grad_norm - shell_eta))) / denominator

    # Material interpolation
    # Use the expression directly to allow UFL automatic differentiation
    rho_shell = rho_shell_expr
    rho_total = lambda_m * rho_base + (1.0 - lambda_m * rho_base) * rho_shell
    rho_base_p = rho_base**penal
    q_ext_padding = opt.get("q_ext_padding", 0.2)
    q_ext_zone = ufl.conditional(padding_mask > 0.5, q_ext_padding, q_ext)
    
    # Hashin-Shtrikman upper bound
    lambda_E_hs = lambda_m / (2.0 - lambda_m)
    lambda_E = opt.get("lambda_E", lambda_E_hs)
    
    E = E0 * (
        lambda_Emin
        + q_ext_zone * (lambda_E - lambda_Emin) * rho_base_p
        + q_ext_zone * (1.0 - lambda_E * rho_base_p) * rho_shell**penal_shell
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

    opt["compliance"] = ufl.inner(sigma(u_field), epsilon(u_field)) * dx
    opt["volume"] = rho_total * dx
    opt["total_volume"] = (1.0 - padding_mask) * dx

    return (
        linear_problem,
        u_field,
        rho_field,
        rho_base,
        rho_nf,
        rho_shell_func,
        grad_norm,
        alpha_grad,
        shell_beta,
        rho_shell_expr,
        rho_total,
        shell_filter,
        padding_mask,
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
        linear_problem, u_field, rho_field, rho_base, rho_nf,
        rho_shell_func, grad_norm, alpha_grad, shell_beta_constant,
        rho_shell_expr, rho_total_expr, shell_filter, padding_mask
    ) = form_fem_coating_3d(fem, opt)

    # --- Smoothing and Projection Setup ---
    S = rho_base.function_space
    density_filter = DensityFilter(comm, rho_field, rho_base, opt["filter_radius"], fem["petsc_options"])
    heaviside = Heaviside(rho_base)

    # Forms for objectives and constraints
    C_form = form(opt["compliance"])
    V_form = form(opt["volume"])
    total_vol = comm.allreduce(assemble_scalar(form(opt["total_volume"])), op=MPI.SUM)

    # --- Sensitivity Analysis Forms using UFL ---
    # Partial derivatives w.r.t rho_nf (includes shell path via rho_shell_expr)
    dC_drnf_form = form(-ufl.derivative(opt["compliance"], rho_nf))
    dV_drnf_form = form(ufl.derivative(opt["volume"], rho_nf))
    
    # Partial derivatives w.r.t rho_base (direct path)
    dC_drb_direct_form = form(-ufl.derivative(opt["compliance"], rho_base))
    dV_drb_direct_form = form(ufl.derivative(opt["volume"], rho_base))

    # Pre-allocate sensitivity vectors
    dC_drnf_vec, dV_drnf_vec = create_vector(dC_drnf_form), create_vector(dV_drnf_form)
    dC_drb_direct_vec, dV_drb_direct_vec = create_vector(dC_drb_direct_form), create_vector(dV_drb_direct_form)

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

    date_str = get_date_str(comm)
    filepath = "results/" + date_str + "/"
    if comm.rank == 0 and not os.path.exists(filepath):
        os.makedirs(filepath, exist_ok=True)
    comm.barrier()

    plot_freq = opt.get("plot_freq", 1)
    
    beta_initial = opt.get("beta_initial", 1.0)
    beta_inc = opt.get("beta_inc", 1.2)
    shell_eta = opt.get("shell_eta", 0.5)
    
    loop, opt_iter, mma_iter, beta, change = 0, 0, 0, beta_initial, 1.0
    low, upp = None, None
    while opt_iter < opt["max_iter"] and (change > opt["opt_tol"] or beta < opt["beta_max"]):
        opt_start_time = time.perf_counter()
        opt_iter += 1
        mma_iter += 1
        loop += 1

        # --- Forward Pass ---
        density_filter.forward()
        if opt_iter > 1 and (opt_iter % opt["beta_interval"] == 0 or change <= opt["opt_tol"]) and beta < opt["beta_max"]:
            beta *= beta_inc
            change = 1.0
            mma_iter = 1
            rho_old1[:] = rho_field.vector.array
            rho_old2[:] = rho_old1
            if low is not None:
                low[:] = rho_field.vector.array
                upp[:] = rho_field.vector.array

        heaviside.forward(beta, eta=opt.get("base_eta", 0.5))
        shell_filter.forward(rho_base)
        
        # Apply shell_beta = beta logic
        shell_beta_val = float(beta)
        shell_beta_constant.value = shell_beta_val

        linear_problem.solve_fem()

        C_value = comm.allreduce(assemble_scalar(C_form), op=MPI.SUM)
        V_value = comm.allreduce(assemble_scalar(V_form), op=MPI.SUM) / total_vol

        # --- Analytical Sensitivities using UFL ---
        # Assemble using pre-allocated forms and vectors
        with dC_drnf_vec.localForm() as loc: loc.set(0)
        assemble_vector(dC_drnf_vec, dC_drnf_form); dC_drnf_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        with dV_drnf_vec.localForm() as loc: loc.set(0)
        assemble_vector(dV_drnf_vec, dV_drnf_form); dV_drnf_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        with dC_drb_direct_vec.localForm() as loc: loc.set(0)
        assemble_vector(dC_drb_direct_vec, dC_drb_direct_form); dC_drb_direct_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        with dV_drb_direct_vec.localForm() as loc: loc.set(0)
        assemble_vector(dV_drb_direct_vec, dV_drb_direct_form); dV_drb_direct_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Backpropagate shell path through shell filter
        dC_drb_from_shell = shell_filter.backward(dC_drnf_vec)
        dV_drb_from_shell = shell_filter.backward(dV_drnf_vec)
        
        # Combine direct and shell-path sensitivities
        dC_drb_direct_vec.axpy(1.0, dC_drb_from_shell)
        dV_drb_direct_vec.axpy(1.0, dV_drb_from_shell)
        dV_drb_direct_vec.scale(1.0 / total_vol)

        # CRITICAL: Forward sync before Heaviside backward to ensure ghost nodes are correct
        dC_drb_direct_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        dV_drb_direct_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        # Pass through Heaviside and Density Filter
        sensitivities = [dC_drb_direct_vec, dV_drb_direct_vec]
        heaviside.backward(sensitivities)
        [dCdrho, dVdrho] = density_filter.backward(sensitivities)

        # Only destroy temporary vectors
        dC_drb_from_shell.destroy(); dV_drb_from_shell.destroy()

        g_vec = np.array([V_value - opt["vol_frac"]])
        dJdrho, dgdrho = dCdrho, np.vstack([dVdrho])

        rho_values = rho_field.vector.array.copy()
        if opt.get("use_oc", False):
            rho_new, change = optimality_criteria(rho_values, rho_min, rho_max, g_vec, dJdrho, dgdrho[0], opt["move"])
        else:
            rho_new, change, low, upp = mma_optimizer(1, num_elems, mma_iter, rho_values, rho_min, rho_max, rho_old1, rho_old2, dJdrho, g_vec, dgdrho, low, upp, opt["move"])
            rho_old2 = rho_old1.copy(); rho_old1 = rho_values.copy()

        rho_field.vector.array = rho_new.copy()

        opt_time = time.perf_counter() - opt_start_time
        if comm.rank == 0:
            print(f"It.: {loop} C.: {C_value:.3f} Vf.: {V_value:.3f} beta: {beta} change: {change:.3f} t.: {opt_time: .3f}(s)", flush=True)

        if loop % plot_freq == 0 or change <= opt["opt_tol"] or opt_iter >= opt["max_iter"]:
            rho_tot_plot = project_expression(rho_total_expr, V=rho_base.function_space, mesh=fem["mesh"], quadrature_degree=fem["quadrature_degree"])
            
            save_xdmf(fem["mesh"], rho_tot_plot, filepath, filename=f"design_{loop}.xdmf")

            values = S_comm.gather(rho_tot_plot)
            if comm.rank == 0:
                plotter.plot(values, loop, path=filepath, slice_normal="x", slice_origin=(5.0, 15.0, 5.0), clip_bounds=opt.get("clip_bounds"))

    rho_shell_final = project_expression(rho_shell_expr, V=S, mesh=fem["mesh"], quadrature_degree=fem["quadrature_degree"])
    rho_total_final = project_expression(rho_total_expr, V=S, mesh=fem["mesh"], quadrature_degree=fem["quadrature_degree"])

    if comm.rank == 0: print(f"Saving density fields to: {filepath}", flush=True)
    save_density_fields_xdmf(fem["mesh"], fields={"density_base": rho_base, "density_shell": rho_shell_final, "density_total": rho_total_final}, path=filepath)
