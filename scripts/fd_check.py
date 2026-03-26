import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_box, CellType
from dolfinx.fem import assemble_scalar, form
import ufl
import os

from fenitop.coating_3d import form_fem_coating_3d
from fenitop.parameterize import DensityFilter, Heaviside, CG1Filter
from petsc4py import PETSc

def main():
    comm = MPI.COMM_WORLD

    if comm.size > 1:
        if comm.rank == 0:
            print("Please run this FD check script in serial (python3 scripts/fd_check.py) for simplicity.")
        return

    # Mesh parameters matching coating_beam_3d.py
    mesh_res_phys = [10, 30, 10]  # Same as coating_beam_3d.py
    dx, dy, dz = 10.0 / mesh_res_phys[0], 30.0 / mesh_res_phys[1], 10.0 / mesh_res_phys[2]

    tref = 1.0
    filter_radius_shell = 2.5 * tref
    filter_radius = 2.0 * filter_radius_shell
    d_ext = 1.0 * filter_radius

    n_pad_x = int(np.ceil(d_ext / dx))
    n_pad_y = int(np.ceil(d_ext / dy))
    n_pad_z = int(np.ceil(d_ext / dz))
    actual_d_ext_x = n_pad_x * dx
    actual_d_ext_y = n_pad_y * dy
    actual_d_ext_z = n_pad_z * dz

    mesh_resolution = [
        mesh_res_phys[0] + 2 * n_pad_x,
        mesh_res_phys[1] + 2 * n_pad_y,
        mesh_res_phys[2] + 2 * n_pad_z
    ]

    base_mesh = create_box(
        comm,
        [[-actual_d_ext_x, -actual_d_ext_y, -actual_d_ext_z], 
         [10.0 + actual_d_ext_x, 30.0 + actual_d_ext_y, 10.0 + actual_d_ext_z]],
        mesh_resolution,
        CellType.hexahedron,
    )

    from dolfinx.mesh import create_submesh
    import dolfinx
    
    # Compute midpoints of all cells in the base mesh
    c_coords = dolfinx.mesh.compute_midpoints(base_mesh, 3, np.arange(base_mesh.topology.index_map(3).size_local, dtype=np.int32))
    
    # Remove top padded elements at the load region: y > 30, x in [4.5, 5.5], z in [4.5, 5.5]
    remove_top = (c_coords[:, 1] > 30.0 - 1e-6) & (c_coords[:, 0] >= 4.5 - 1e-6) & (c_coords[:, 0] <= 5.5 + 1e-6) & (c_coords[:, 2] >= 4.5 - 1e-6) & (c_coords[:, 2] <= 5.5 + 1e-6)
    
    # Remove bottom padded elements at the support regions: y < 0, x in [0, 1.5] and [8.5, 10], z in [0, 10]
    remove_bottom = (c_coords[:, 1] < 1e-6) & (c_coords[:, 0] >= -1e-6) & (c_coords[:, 0] <= 10.0 + 1e-6) & (c_coords[:, 2] >= -1e-6) & (c_coords[:, 2] <= 10.0 + 1e-6) & ((c_coords[:, 0] <= 1.5 + 1e-6) | (c_coords[:, 0] >= 8.5 - 1e-6))
    
    # Keep cells that are not in the removed regions
    cells_to_keep = np.where(~(remove_top | remove_bottom))[0].astype(np.int32)
    mesh, _, _, _ = create_submesh(base_mesh, 3, cells_to_keep)
    mesh.topology.create_connectivity(2, 3)

    fem = {
        "mesh": mesh,
        "mesh_serial": None,
        "young's modulus": 100,
        "poisson's ratio": 0.25,
        "disp_bc": lambda x: np.isclose(x[1], 0.0)
        & (x[0] >= -1e-6) & (x[0] <= 10.0 + 1e-6) & (x[2] >= -1e-6) & (x[2] <= 10.0 + 1e-6)
        & (np.less(x[0], 1.5 + dx/2 + 1e-6) | np.greater(x[0], 8.5 - dx/2 - 1e-6)),
        "traction_bcs": [
            [
                (0.0, 0.0, -2.0),
                lambda x: np.isclose(x[1], 30.0)
                & (
                    np.greater(x[0], 4.5 - dx/2 - 1e-6)
                    & np.less(x[0], 5.5 + dx/2 + 1e-6)
                    & np.greater(x[2], 4.5 - dz/2 - 1e-6)
                    & np.less(x[2], 5.5 + dz/2 + 1e-6)
                ),
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
        "vol_frac": 0.20,
        "solid_zone": lambda x: np.full(x.shape[1], False),
        "void_zone": lambda x: (
            (x[0] < -1e-6) | (x[0] > 10.0 + 1e-6) | 
            (x[1] < -1e-6) | (x[1] > 30.0 + 1e-6) | 
            (x[2] < -1e-6) | (x[2] > 10.0 + 1e-6)
        ),
        "penalty": 3.0,
        "epsilon": 1e-6,
        "filter_radius": filter_radius,
        "filter_radius_shell": filter_radius_shell,
        "beta_interval": 20,
        "beta_max": 64,
        "use_oc": False,
        "move": 0.05,
        "opt_compliance": True,
        "lambda_m": 0.7,
        "lambda_E": 0.7 / (3.0 - 2.0 * 0.7),
        "penal_shell": 1.0,
        "shell_eta": 0.5,
        "q_ext_padding": 0.2,
    }

    # Setup FEM
    (
        linear_problem, u_field, rho_field, rho_base, rho_nf,
        shell_beta, rho_shell_expr, rho_total_expr,
        shell_filter
    ) = form_fem_coating_3d(fem, opt)

    density_filter = DensityFilter(comm, rho_field, rho_base, opt["filter_radius"], fem["petsc_options"])
    heaviside = Heaviside(rho_base)

    C_form = form(opt["compliance"])
    V_form = form(opt["volume"])
    total_vol = comm.allreduce(assemble_scalar(form(opt["total_volume"])), op=MPI.SUM)

    dC_drho_base_form = form(-ufl.derivative(opt["compliance"], rho_base))
    dV_drho_base_form = form(ufl.derivative(opt["volume"], rho_base))
    dC_drho_nf_form = form(-ufl.derivative(opt["compliance"], rho_nf))
    dV_drho_nf_form = form(ufl.derivative(opt["volume"], rho_nf))

    from dolfinx.fem.petsc import create_vector, assemble_vector
    dC_drho_base_vec = create_vector(dC_drho_base_form)
    dV_drho_base_vec = create_vector(dV_drho_base_form)
    dC_drho_nf_vec = create_vector(dC_drho_nf_form)
    dV_drho_nf_vec = create_vector(dV_drho_nf_form)

    # Initialize with uniform density 0.8
    rho_field.vector.array[:] = 0.8
    rho_field.x.scatter_forward()

    beta = 2.0
    shell_beta.value = float(beta / 2.0)

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
    assemble_vector(dC_drho_base_vec, dC_drho_base_form)
    dC_drho_base_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    with dC_drho_nf_vec.localForm() as loc: loc.set(0)
    assemble_vector(dC_drho_nf_vec, dC_drho_nf_form)
    dC_drho_nf_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Removed log(C) scaling
    
    with dV_drho_base_vec.localForm() as loc: loc.set(0)
    assemble_vector(dV_drho_base_vec, dV_drho_base_form)
    dV_drho_base_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dV_drho_base_vec.scale(1.0 / total_vol)

    with dV_drho_nf_vec.localForm() as loc: loc.set(0)
    assemble_vector(dV_drho_nf_vec, dV_drho_nf_form)
    dV_drho_nf_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dV_drho_nf_vec.scale(1.0 / total_vol)

    dC_drho_base_from_shell = shell_filter.backward(dC_drho_nf_vec)
    dV_drho_base_from_shell = shell_filter.backward(dV_drho_nf_vec)

    dC_drho_base_vec.axpy(1.0, dC_drho_base_from_shell)
    dV_drho_base_vec.axpy(1.0, dV_drho_base_from_shell)

    sensitivities = [dC_drho_base_vec, dV_drho_base_vec]
    heaviside.backward(sensitivities)
    [dCdrho, dVdrho] = density_filter.backward(sensitivities)
    
    dC_drho_base_from_shell.destroy()
    dV_drho_base_from_shell.destroy()

    analytical_dC = dCdrho.copy()
    analytical_dV = dVdrho.copy()

    # Selection of elements specifically at the structural boundary and in padding regions
    import time
    np.random.seed(int(time.time())) 
    
    num_elems = rho_field.vector.array.size
    centers = rho_field.function_space.tabulate_dof_coordinates()[:num_elems]
    
    # Identify padding elements (void_zone)
    is_padding = opt["void_zone"](centers.T)
    padding_indices = np.where(is_padding)[0]
    design_indices = np.where(~is_padding)[0]
    
    # Calculate distance to center axis (x=5, z=5)
    dist_to_axis = np.sqrt((centers[:, 0] - 5.0)**2 + (centers[:, 2] - 5.0)**2)

    # Point 1-3: Inside "cylinder" (deep in design domain, r < 3.0)
    inner_indices = np.where((~is_padding) & (dist_to_axis < 3.0))[0]
    point_1_3 = np.random.choice(inner_indices, 3, replace=False)
    
    # Point 4-6: On "cylinder wall" (physical domain boundary, r approx 5.0 or x,z boundaries)
    dist_to_boundary_x = np.minimum(np.abs(centers[:, 0] - 0.0), np.abs(10.0 - centers[:, 0]))
    dist_to_boundary_z = np.minimum(np.abs(centers[:, 2] - 0.0), np.abs(10.0 - centers[:, 2]))
    min_dist_to_boundary = np.minimum(dist_to_boundary_x, dist_to_boundary_z)
    
    # Use <= with a small tolerance to ensure boundary elements (at distance 0.5*dx) are picked
    wall_indices = np.where((~is_padding) & (min_dist_to_boundary <= 0.5 * dx + 1e-6))[0]
    if len(wall_indices) < 3:
        wall_indices = design_indices # Fallback to any design element if boundary is tricky
    point_4_6 = np.random.choice(wall_indices, 3, replace=False)

    test_indices = np.concatenate([point_1_3, point_4_6])
    test_labels = ["1: Inner", "2: Inner", "3: Inner", "4: Boundary", "5: Boundary", "6: Boundary"]

    print("Starting Finite Difference Check...")
    print(f"Base C: {C_value:.6e}, Base V: {V_value:.6e}")
        
    errors_C_dict = {i: [] for i in test_indices}
    errors_V_dict = {i: [] for i in test_indices}
    dh_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

    analytical_dC_vals = {i: analytical_dC[i] for i in test_indices}
    numerical_dC_vals = {i: [] for i in test_indices}
    analytical_dV_vals = {i: analytical_dV[i] for i in test_indices}
    numerical_dV_vals = {i: [] for i in test_indices}

    for idx, i in enumerate(test_indices):
        label = test_labels[idx]
        a_dC = analytical_dC[i]
        a_dV = analytical_dV[i]
        
        print(f"\n--- Element {i} ({label}) ---")
        print(f"Analytical dC/drho: {a_dC: .6e}")
        print(f"Analytical dV/drho:      {a_dV: .6e}")
        print(f"{'dh':<10} | {'FD dC':<15} | {'Error %':<10} | {'FD dV':<15} | {'Error %':<10}")
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
            fd_dC = (C_plus - C_minus) / (2 * dh)
            fd_dV = (V_plus - V_minus) / (2 * dh)
            
            # Signed Relative Difference (%)
            rel_diff_C = (fd_dC - a_dC) / max(abs(a_dC), 1e-15) * 100
            rel_diff_V = (fd_dV - a_dV) / max(abs(a_dV), 1e-15) * 100
            
            errors_C_dict[i].append(abs(rel_diff_C))
            errors_V_dict[i].append(abs(rel_diff_V))
            
            numerical_dC_vals[i].append(fd_dC)
            numerical_dV_vals[i].append(fd_dV)
            
            print(f"{dh:<10.1e} | {fd_dC:<15.6e} | {rel_diff_C:<10.4f} | {fd_dV:<15.6e} | {rel_diff_V:<10.4f}")

    try:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Absolute Error convergence for dC/drho (log-log)
        for idx, i in enumerate(test_indices):
            axs[0,0].loglog(dh_values, errors_C_dict[i], marker='o', label=test_labels[idx])
        axs[0,0].set_title('Sensitivity Error Convergence: dC/drho')
        axs[0,0].set_xlabel('Step size (dh)')
        axs[0,0].set_ylabel('Relative Error (%)')
        axs[0,0].legend()
        axs[0,0].grid(True, which="both", ls="--", alpha=0.5)
        axs[0,0].invert_xaxis()

        # Plot 2: Absolute Error convergence for dV/drho (log-log)
        for idx, i in enumerate(test_indices):
            axs[0,1].loglog(dh_values, errors_V_dict[i], marker='o', label=test_labels[idx])
        axs[0,1].set_title('Sensitivity Error Convergence: dV/drho')
        axs[0,1].set_xlabel('Step size (dh)')
        axs[0,1].set_ylabel('Relative Error (%)')
        axs[0,1].legend()
        axs[0,1].grid(True, which="both", ls="--", alpha=0.5)
        axs[0,1].invert_xaxis()

        # Plot 3: Analytical vs Numerical values for dC sensitivity (semilogx to show signs)
        for idx, i in enumerate(test_indices):
            axs[1,0].semilogx(dh_values, numerical_dC_vals[i], marker='x', label=f'FD {test_labels[idx]}')
            axs[1,0].axhline(y=analytical_dC_vals[i], color='gray', linestyle='--', alpha=0.5)
        axs[1,0].set_title('FD dC/drho vs dh (signed values)')
        axs[1,0].set_xlabel('Step size (dh)')
        axs[1,0].set_ylabel('dC/drho')
        axs[1,0].legend()
        axs[1,0].grid(True, which="both", ls="--", alpha=0.5)
        axs[1,0].invert_xaxis()

        # Plot 4: Analytical vs Numerical values for V sensitivity (semilogx to show signs)
        for idx, i in enumerate(test_indices):
            axs[1,1].semilogx(dh_values, numerical_dV_vals[i], marker='x', label=f'FD {test_labels[idx]}')
            axs[1,1].axhline(y=analytical_dV_vals[i], color='gray', linestyle='--', alpha=0.5)
        axs[1,1].set_title('FD dV/drho vs dh (signed values)')
        axs[1,1].set_xlabel('Step size (dh)')
        axs[1,1].set_ylabel('dV/drho')
        axs[1,1].legend()
        axs[1,1].grid(True, which="both", ls="--", alpha=0.5)
        axs[1,1].invert_xaxis()

        plt.tight_layout(); os.makedirs('results', exist_ok=True)
        plt.savefig('results/fd_check_distribution.png', dpi=300)
    except ImportError: pass

if __name__ == '__main__':
    main()
