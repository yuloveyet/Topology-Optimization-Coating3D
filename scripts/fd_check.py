import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_box, CellType
from dolfinx.fem import assemble_scalar, form
import ufl

from fenitop.coating_3d import form_fem_coating_3d, CG1Filter
from fenitop.parameterize import DensityFilter, Heaviside
from petsc4py import PETSc

def main():
    comm = MPI.COMM_WORLD
    
    if comm.size > 1:
        if comm.rank == 0:
            print("Please run this FD check script in serial (python3 scripts/fd_check.py) for simplicity.")
        return

    # Very small mesh for fast FEM
    mesh = create_box(
        comm,
        [[0, 0, 0], [4, 4, 4]],
        [4, 4, 4],
        CellType.hexahedron,
    )
    
    fem = {
        "mesh": mesh,
        "mesh_serial": None,
        "young's modulus": 100,
        "poisson's ratio": 0.25,
        "disp_bc": lambda x: np.isclose(x[0], 0.0),
        "traction_bcs": [
            [
                (0, 0, -2.0),
                lambda x: np.isclose(x[0], 4.0),
            ]
        ],
        "body_force": (0, 0, 0),
        "quadrature_degree": 2,
        "petsc_options": {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    }

    opt = {
        "max_iter": 2,
        "opt_tol": 1e-5,
        "vol_frac": 0.3,
        "solid_zone": lambda x: np.full(x.shape[1], False),
        "void_zone": lambda x: np.full(x.shape[1], False),
        "penalty": 3.0,
        "epsilon": 1e-6,
        "filter_radius": 1.2,
        "filter_radius_shell": 1.2,
        "beta_interval": 20,
        "beta_max": 128,
        "use_oc": False,
        "move": 0.02,
        "opt_compliance": True,
        "lambda_m": 0.7,
        "lambda_E": 0.4,
        "penal_shell": 3.0,
        "shell_eta": 0.5,
    }

    # Setup FEM
    (
        linear_problem, u_field, rho_field, rho_base, rho_Nf,
        solid_mask, shell_beta, rho_shell_expr, rho_total_expr
    ) = form_fem_coating_3d(fem, opt)

    density_filter = DensityFilter(comm, rho_field, rho_base, opt["filter_radius"], fem["petsc_options"])
    heaviside = Heaviside(rho_base)
    rmin_shell = opt.get("filter_radius_shell", opt["filter_radius"])
    shell_filter = CG1Filter(comm, rho_base.function_space, rho_Nf, rmin_shell, fem["petsc_options"])

    C_form = form(opt["compliance"])
    V_form = form(opt["volume"])
    total_vol = comm.allreduce(assemble_scalar(form(opt["total_volume"])), op=MPI.SUM)

    dC_drho_base_form = form(-ufl.derivative(opt["compliance"], rho_base))
    dC_drho_Nf_form = form(-ufl.derivative(opt["compliance"], rho_Nf))
    dV_drho_base_form = form(ufl.derivative(opt["volume"], rho_base))
    dV_drho_Nf_form = form(ufl.derivative(opt["volume"], rho_Nf))

    from dolfinx.fem.petsc import create_vector, assemble_vector
    dC_drho_base_vec = create_vector(dC_drho_base_form)
    dC_drho_Nf_vec = create_vector(dC_drho_Nf_form)
    dV_drho_base_vec = create_vector(dV_drho_base_form)
    dV_drho_Nf_vec = create_vector(dV_drho_Nf_form)

    # Randomize rho_field
    np.random.seed(42)
    num_elems = rho_field.vector.array.size
    rho_field.vector.array[:] = np.random.uniform(0.1, 0.9, num_elems)
    rho_field.x.scatter_forward()

    beta = 2.0
    shell_beta.value = float(beta)

    # --- Forward Pass ---
    def forward_pass():
        density_filter.forward()
        heaviside.forward(beta, eta=opt.get("base_eta", 0.5))
        shell_filter.forward(rho_base)
        linear_problem.solve_fem()
        C_val = comm.allreduce(assemble_scalar(C_form), op=MPI.SUM)
        V_val = comm.allreduce(assemble_scalar(V_form), op=MPI.SUM) / total_vol
        return C_val, V_val

    C_value, V_value = forward_pass()

    # --- Analytical Sensitivities ---
    with dC_drho_base_vec.localForm() as loc: loc.set(0)
    with dC_drho_Nf_vec.localForm() as loc: loc.set(0)
    assemble_vector(dC_drho_base_vec, dC_drho_base_form)
    assemble_vector(dC_drho_Nf_vec, dC_drho_Nf_form)
    dC_drho_base_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dC_drho_Nf_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Scale for log(C)
    c_scale = 1.0 / C_value if C_value > 1e-15 else 0.0
    dC_drho_base_vec.scale(c_scale)
    dC_drho_Nf_vec.scale(c_scale)
    
    dC_drho_base_from_shell = shell_filter.backward(dC_drho_Nf_vec)
    dC_drho_base_vec.axpy(1.0, dC_drho_base_from_shell) 

    with dV_drho_base_vec.localForm() as loc: loc.set(0)
    with dV_drho_Nf_vec.localForm() as loc: loc.set(0)
    assemble_vector(dV_drho_base_vec, dV_drho_base_form)
    assemble_vector(dV_drho_Nf_vec, dV_drho_Nf_form)
    dV_drho_base_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dV_drho_Nf_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    
    dV_drho_base_from_shell = shell_filter.backward(dV_drho_Nf_vec)
    dV_drho_base_vec.axpy(1.0, dV_drho_base_from_shell)
    dV_drho_base_vec.scale(1.0 / total_vol)

    sensitivities = [dC_drho_base_vec, dV_drho_base_vec]
    heaviside.backward(sensitivities)
    [dCdrho, dVdrho] = density_filter.backward(sensitivities)

    analytical_dC = dCdrho.copy()
    analytical_dV = dVdrho.copy()

    # Select random elements
    np.random.seed(123)
    if num_elems > 0:
        test_indices = np.random.choice(num_elems, min(5, num_elems), replace=False)
    else:
        test_indices = []

    print("Starting Finite Difference Check...")
    print(f"Base C: {C_value:.6e}, Base V: {V_value:.6e}")
        
    errors_C_dict = {i: [] for i in test_indices}
    errors_V_dict = {i: [] for i in test_indices}
    dh_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

    for i in test_indices:
        a_dC = analytical_dC[i]
        a_dV = analytical_dV[i]
        
        print(f"\n--- Element {i} ---")
        print(f"Analytical d(logC)/drho: {a_dC: .6e}")
        print(f"Analytical dV/drho:      {a_dV: .6e}")
        print(f"{'dh':<10} | {'FD d(logC)':<15} | {'Error %':<10} | {'FD dV':<15} | {'Error %':<10}")
        print("-" * 75)
            
        orig_val = rho_field.vector.array[i]
        
        for dh in dh_values:
            # Plus
            rho_field.vector.array[i] = orig_val + dh
            rho_field.x.scatter_forward()
            C_plus, V_plus = forward_pass()
            
            # Minus
            rho_field.vector.array[i] = orig_val - dh
            rho_field.x.scatter_forward()
            C_minus, V_minus = forward_pass()
            
            # Reset
            rho_field.vector.array[i] = orig_val
            rho_field.x.scatter_forward()
            
            # FD
            fd_dC = (np.log(C_plus) - np.log(C_minus)) / (2 * dh)
            fd_dV = (V_plus - V_minus) / (2 * dh)
            
            err_C = abs(fd_dC - a_dC) / max(abs(a_dC), 1e-15) * 100
            err_V = abs(fd_dV - a_dV) / max(abs(a_dV), 1e-15) * 100
            
            errors_C_dict[i].append(err_C)
            errors_V_dict[i].append(err_V)
            
            print(f"{dh:<10.1e} | {fd_dC:<15.6e} | {err_C:<10.4f} | {fd_dV:<15.6e} | {err_V:<10.4f}")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        for i in test_indices:
            plt.loglog(dh_values, errors_C_dict[i], marker='o', label=f'Elem {i}')
        plt.xlabel('Step size (dh)')
        plt.ylabel('Relative Error (%)')
        plt.title('Sensitivity Error: Objective (log C)')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()

        plt.subplot(1, 2, 2)
        for i in test_indices:
            plt.loglog(dh_values, errors_V_dict[i], marker='o', label=f'Elem {i}')
        plt.xlabel('Step size (dh)')
        plt.ylabel('Relative Error (%)')
        plt.title('Sensitivity Error: Volume (V)')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()

        plt.tight_layout()
        out_plot = 'results/fd_check_results.png'
        import os
        os.makedirs('results', exist_ok=True)
        plt.savefig(out_plot, dpi=300)
        print(f"\n=> Successfully saved convergence plot to {out_plot}")
    except ImportError:
        print("\n=> Matplotlib not installed. Skipping plot generation.")

if __name__ == '__main__':
    main()
