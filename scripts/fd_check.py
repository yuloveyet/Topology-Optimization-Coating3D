import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_box, CellType
from dolfinx.fem import assemble_scalar, form, Function
import ufl
import os

from fenitop.coating_3d import form_fem_coating_3d, project_expression
from fenitop.parameterize import DensityFilter, Heaviside, CG1Filter
from dolfinx.fem.petsc import (
    create_vector,
    assemble_vector,
    create_matrix,
    assemble_matrix,
)
from petsc4py import PETSc


def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # Mesh parameters matching coating_beam_3d.py
    # 几何基准尺寸与比例 (长:宽:高)
    length = 50.0
    aspect_ratio = (1.0, 1.0, 3.0)  # Length:Width:Height

    width = length * (aspect_ratio[1] / aspect_ratio[0])
    height = length * (aspect_ratio[2] / aspect_ratio[0])

    # 网格分辨率基准 (nx)，其余方向将根据 aspect_ratio 自动缩放
    # 可自由修改此参数以整体调整网格精细度
    base_mesh_res = 30 
    mesh_res_phys = [
        int(base_mesh_res * aspect_ratio[0]),  # nx
        int(base_mesh_res * aspect_ratio[2]),  # ny (Height direction)
        int(base_mesh_res * aspect_ratio[1]),  # nz (Width direction)
    ]
    dx_m, dy_m, dz_m = (
        length / mesh_res_phys[0],
        height / mesh_res_phys[1],
        width / mesh_res_phys[2],
    )

    filter_radius = 0.2 * length
    filter_radius_shell = filter_radius / 2.0
    tref = filter_radius_shell / 5.0
    d_ext = 1.0 * filter_radius

    n_pad_x = int(np.ceil(d_ext / dx_m))
    n_pad_y = int(np.ceil(d_ext / dy_m))
    n_pad_z = int(np.ceil(d_ext / dz_m))
    actual_d_ext_x = n_pad_x * dx_m
    actual_d_ext_y = n_pad_y * dy_m
    actual_d_ext_z = n_pad_z * dz_m

    mesh_resolution = [
        mesh_res_phys[0] + 2 * n_pad_x,
        mesh_res_phys[1] + 2 * n_pad_y,
        mesh_res_phys[2] + 2 * n_pad_z,
    ]

    base_mesh = create_box(
        comm,
        [
            [-actual_d_ext_x, -actual_d_ext_y, -actual_d_ext_z],
            [length + actual_d_ext_x, height + actual_d_ext_y, width + actual_d_ext_z],
        ],
        mesh_resolution,
        CellType.hexahedron,
    )

    from dolfinx.mesh import create_submesh
    import dolfinx

    # Compute midpoints of all cells in the base mesh
    c_coords = dolfinx.mesh.compute_midpoints(
        base_mesh,
        3,
        np.arange(base_mesh.topology.index_map(3).size_local, dtype=np.int32),
    )

    # Remove top padded elements at the load region
    remove_top = (
        (c_coords[:, 1] > height - 1e-6)
        & (c_coords[:, 0] >= length / 2 - 0.5 - actual_d_ext_x - 1e-6)
        & (c_coords[:, 0] <= length / 2 + 0.5 + actual_d_ext_x + 1e-6)
        & (c_coords[:, 2] >= width / 2 - 0.5 - actual_d_ext_z - 1e-6)
        & (c_coords[:, 2] <= width / 2 + 0.5 + actual_d_ext_z + 1e-6)
    )

    # Remove bottom padded elements at the support regions
    remove_bottom = (
        (c_coords[:, 1] < 1e-6)
        & ((c_coords[:, 0] <= 1.5 + actual_d_ext_x + 1e-6) | (c_coords[:, 0] >= length - 1.5 - actual_d_ext_x - 1e-6))
    )

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
        & (x[0] >= -1e-6)
        & (x[0] <= length + 1e-6)
        & (x[2] >= -1e-6)
        & (x[2] <= width + 1e-6)
        & (
            np.less(x[0], 1.5 + dx_m / 2 + 1e-6)
            | np.greater(x[0], length - 1.5 - dx_m / 2 - 1e-6)
        ),
        "traction_bcs": [
            [
                (0.0, 0.0, -2.0),
                lambda x: np.isclose(x[1], height)
                & (
                    np.greater(x[0], length / 2 - 0.5 - dx_m / 2 - 1e-6)
                    & np.less(x[0], length / 2 + 0.5 + dx_m / 2 + 1e-6)
                    & np.greater(x[2], width / 2 - 0.5 - dz_m / 2 - 1e-6)
                    & np.less(x[2], width / 2 + 0.5 + dz_m / 2 + 1e-6)
                ),
            ]
        ],
        "body_force": (0, 0, 0),
        "quadrature_degree": 4,
        "petsc_options": {
            "ksp_type": "cg",
            "pc_type": "gamg",
            "ksp_rtol": 1e-13,
            "ksp_atol": 1e-15,
        },
    }

    opt = {
        "max_iter": 2,
        "opt_tol": 1e-5,
        "vol_frac": 0.20,
        "solid_zone": lambda x: np.full(x.shape[1], False),
        "void_zone": lambda x: (
            (x[0] < -1e-6)
            | (x[0] > length + 1e-6)
            | (x[1] < -1e-6)
            | (x[1] > height + 1e-6)
            | (x[2] < -1e-6)
            | (x[2] > width + 1e-6)
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
        "lambda_E": 0.7 / (2.0 - 0.7),
        "penal_shell": 1.0,
        "shell_eta": 0.5,
        "q_ext_padding": 0.2,
    }

    # Setup FEM
    (
        linear_problem,
        u_field,
        rho_field,
        rho_base,
        rho_nf,
        rho_shell_func,
        grad_norm,
        alpha_grad,
        shell_beta_constant,
        rho_shell_expr,
        rho_total_expr,
        shell_filter,
        padding_mask,
    ) = form_fem_coating_3d(fem, opt)
    density_filter = DensityFilter(
        comm, rho_field, rho_base, opt["filter_radius"], fem["petsc_options"]
    )
    heaviside = Heaviside(rho_base)
    S = rho_base.function_space

    # Forms for objectives and constraints
    C_form = form(opt["compliance"])
    V_form = form(opt["volume"])
    total_vol = comm.allreduce(assemble_scalar(form(opt["total_volume"])), op=MPI.SUM)

    # Initialize rho_field as a cylinder along the Y-axis (height direction)
    num_elems = rho_field.vector.array.size
    centers = rho_field.function_space.tabulate_dof_coordinates()[:num_elems].T

    # Calculate distance to central Y-axis (x=length/2, z=width/2)
    dist_to_axis_ini = np.sqrt(
        (centers[0] - length / 2.0) ** 2 + (centers[2] - width / 2.0) ** 2
    )
    r_cyl = min(length, width) * 0.3

    # Set cylinder: 0.9 inside, 0.1 outside, within design domain
    rho_ini = np.where(dist_to_axis_ini < r_cyl, 0.9, 0.1)

    # Apply void/padding zone (overwrite to 0.005)
    void = opt["void_zone"](centers)
    rho_ini[void] = 0.005

    rho_field.vector.array[:] = rho_ini
    rho_field.x.scatter_forward()

    # Define beta and shell_eta here before analytical sensitivities
    beta = 2.0
    shell_eta = opt.get("shell_eta", 0.5)

    # --- Forward Pass ---
    def forward_pass(current_beta):
        density_filter.forward()
        heaviside.forward(current_beta, eta=opt.get("base_eta", 0.5))
        shell_filter.forward(rho_base)

        # Apply shell_beta = beta logic
        shell_beta_val = float(current_beta)
        shell_beta_constant.value = shell_beta_val

        linear_problem.solve_fem()
        C_val = comm.allreduce(assemble_scalar(C_form), op=MPI.SUM)
        V_val = comm.allreduce(assemble_scalar(V_form), op=MPI.SUM) / total_vol
        return C_val, V_val

    C_value, V_value = forward_pass(beta)

    # --- Simplified Analytical Sensitivities using UFL (Matching topopt_coating_3d) ---
    dC_drnf_form = form(-ufl.derivative(opt["compliance"], rho_nf))
    dV_drnf_form = form(ufl.derivative(opt["volume"], rho_nf))
    dC_drb_direct_form = form(-ufl.derivative(opt["compliance"], rho_base))
    dV_drb_direct_form = form(ufl.derivative(opt["volume"], rho_base))

    dC_drnf_vec, dV_drnf_vec = create_vector(dC_drnf_form), create_vector(dV_drnf_form)
    dC_drb_direct_vec, dV_drb_direct_vec = create_vector(
        dC_drb_direct_form
    ), create_vector(dV_drb_direct_form)

    # Assemble
    with dC_drnf_vec.localForm() as loc:
        loc.set(0)
    assemble_vector(dC_drnf_vec, dC_drnf_form)
    dC_drnf_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    with dV_drnf_vec.localForm() as loc:
        loc.set(0)
    assemble_vector(dV_drnf_vec, dV_drnf_form)
    dV_drnf_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    with dC_drb_direct_vec.localForm() as loc:
        loc.set(0)
    assemble_vector(dC_drb_direct_vec, dC_drb_direct_form)
    dC_drb_direct_vec.ghostUpdate(
        addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
    )
    with dV_drb_direct_vec.localForm() as loc:
        loc.set(0)
    assemble_vector(dV_drb_direct_vec, dV_drb_direct_form)
    dV_drb_direct_vec.ghostUpdate(
        addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
    )

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

    analytical_dC_vec = dCdrho.copy()
    analytical_dV_vec = dVdrho.copy()

    # --- Cleanup ---
    dC_drnf_vec.destroy()
    dV_drnf_vec.destroy()
    dC_drb_direct_vec.destroy()
    dV_drb_direct_vec.destroy()
    dC_drb_from_shell.destroy()
    dV_drb_from_shell.destroy()

    # Selection of elements specifically at the structural boundary and in padding regions
    import time

    np.random.seed(int(time.time()) + rank)

    num_elems_local = rho_field.function_space.dofmap.index_map.size_local
    centers = rho_field.function_space.tabulate_dof_coordinates()[:num_elems_local]

    # Identify padding elements (void_zone)
    is_padding = opt["void_zone"](centers.T)
    design_indices = np.where(~is_padding)[0]

    # Calculate distance to center axis
    dist_to_axis = np.sqrt(
        (centers[:, 0] - length / 2.0) ** 2 + (centers[:, 2] - width / 2.0) ** 2
    )

    # Local Candidates: Pick up to 5 of each type to ensure Rank 0 has a diverse pool
    # 1. Inner: Deep inside solid OR deep inside void
    inner_indices = np.where((~is_padding) & ((dist_to_axis < r_cyl * 0.5) | (dist_to_axis > r_cyl * 1.5)))[0]
    local_candidates = []
    if len(inner_indices) > 0:
        num_pick = min(len(inner_indices), 5)
        picked_indices = np.random.choice(inner_indices, num_pick, replace=False)
        for idx in picked_indices:
            local_candidates.append((rank, int(idx), "Inner"))

    # 2. Structural Boundary (Surface): Exactly on the cylinder interface where gradient is max
    surface_indices = np.where((~is_padding) & (np.abs(dist_to_axis - r_cyl) < 1.0 * dx_m))[0]
    if len(surface_indices) > 0:
        num_pick = min(len(surface_indices), 5)
        picked_indices = np.random.choice(surface_indices, num_pick, replace=False)
        for idx in picked_indices:
            local_candidates.append((rank, int(idx), "Boundary"))

    # Gather all candidates to Rank 0
    all_candidates = comm.gather(local_candidates, root=0)
    test_points = []
    if rank == 0:
        flattened = [item for sublist in all_candidates for item in sublist]

        # Separate by type
        inner_pool = [p for p in flattened if p[2] == "Inner"]
        bound_pool = [p for p in flattened if p[2] == "Boundary"]

        # Pick a few of each
        import random

        selected_inner = random.sample(inner_pool, min(len(inner_pool), 3))
        selected_bound = random.sample(bound_pool, min(len(bound_pool), 3))
        test_points = selected_inner + selected_bound

        print(
            f"Selected {len(selected_inner)} Inner and {len(selected_bound)} Surface Boundary points for FD check."
        )

    test_points = comm.bcast(test_points, root=0)

    if rank == 0:
        print(f"Starting Finite Difference Check on {len(test_points)} points...")
        print(f"Base C: {C_value:.6e}, Base V: {V_value:.6e}")

    dh_values = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    all_results = []

    for point in test_points:
        p_rank, p_idx, label = point

        # All ranks need analytical values for comparison (broadcasted from owner)
        a_dC = 0.0
        a_dV = 0.0
        if rank == p_rank:
            a_dC = analytical_dC_vec[p_idx]
            a_dV = analytical_dV_vec[p_idx]
        a_dC = comm.bcast(a_dC, root=p_rank)
        a_dV = comm.bcast(a_dV, root=p_rank)

        if rank == 0:
            print(f"\n--- Point {label} (Rank {p_rank}, Local Idx {p_idx}) ---")
            print(f"Analytical dC/drho: {a_dC: .10e}")
            print(f"Analytical dV/drho: {a_dV: .10e}")
            print(
                f"{'dh':<10} | {'FD dC':<15} | {'RelErr C%':<10} | {'FD dV':<15} | {'RelErr V%':<10}"
            )
            print("-" * 80)

        point_errors_C = []
        point_errors_V = []
        point_fd_C = []
        point_fd_V = []

        for dh in dh_values:
            # Perturb on owner rank
            orig_val = 0.0
            if rank == p_rank:
                orig_val = rho_field.x.array[p_idx]
                rho_field.x.array[p_idx] = orig_val + dh
            rho_field.x.scatter_forward()
            C_plus, V_plus = forward_pass(beta)

            if rank == p_rank:
                rho_field.x.array[p_idx] = orig_val - dh
            rho_field.x.scatter_forward()
            C_minus, V_minus = forward_pass(beta)

            # Reset
            if rank == p_rank:
                rho_field.x.array[p_idx] = orig_val
            rho_field.x.scatter_forward()

            fd_dC = (C_plus - C_minus) / (2 * dh)
            fd_dV = (V_plus - V_minus) / (2 * dh)

            rel_diff_C = (fd_dC - a_dC) / max(abs(a_dC), 1e-15) * 100
            rel_diff_V = (fd_dV - a_dV) / max(abs(a_dV), 1e-15) * 100

            if rank == 0:
                print(
                    f"{dh:<10.1e} | {fd_dC:<15.6e} | {rel_diff_C:<10.4f} | {fd_dV:<15.6e} | {rel_diff_V:<10.4f}"
                )

            point_errors_C.append(abs(rel_diff_C))
            point_errors_V.append(abs(rel_diff_V))
            point_fd_C.append(fd_dC)
            point_fd_V.append(fd_dV)

        all_results.append(
            {
                "label": label,
                "p_idx": p_idx,
                "a_dC": a_dC,
                "a_dV": a_dV,
                "errors_C": point_errors_C,
                "errors_V": point_errors_V,
                "fd_C": point_fd_C,
                "fd_V": point_fd_V,
            }
        )

    if rank == 0:
        try:
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(2, 2, figsize=(15, 12))

            inner_results = [r for r in all_results if r["label"] == "Inner"]
            bound_results = [r for r in all_results if r["label"] == "Boundary"]

            # Row 0: Inner Points
            for res in inner_results:
                l_c = f"Idx {res['p_idx']} (C={res['a_dC']:.1e})"
                axs[0, 0].loglog(dh_values, res["errors_C"], marker="o", label=l_c)
                l_v = f"Idx {res['p_idx']} (V={res['a_dV']:.1e})"
                axs[0, 1].loglog(dh_values, res["errors_V"], marker="s", label=l_v)

            # Row 1: Boundary Points
            for res in bound_results:
                l_c = f"Idx {res['p_idx']} (C={res['a_dC']:.1e})"
                axs[1, 0].loglog(dh_values, res["errors_C"], marker="o", label=l_c)
                l_v = f"Idx {res['p_idx']} (V={res['a_dV']:.1e})"
                axs[1, 1].loglog(dh_values, res["errors_V"], marker="s", label=l_v)

            # Labels and Titles
            axs[0, 0].set_title("Inner Points: Rel Error dC/drho")
            axs[0, 0].set_ylabel("Rel Error %")
            axs[0, 1].set_title("Inner Points: Rel Error dV/drho")
            axs[0, 1].set_ylabel("Rel Error %")
            axs[1, 0].set_title("Boundary Points: Rel Error dC/drho")
            axs[1, 0].set_ylabel("Rel Error %")
            axs[1, 1].set_title("Boundary Points: Rel Error dV/drho")
            axs[1, 1].set_ylabel("Rel Error %")

            for ax in axs.flat:
                ax.set_xlabel("Step size (dh)")
                ax.legend(fontsize="x-small", loc="best")
                ax.grid(True, which="both", ls="--", alpha=0.5)
                ax.invert_xaxis()

            plt.tight_layout()
            os.makedirs("results", exist_ok=True)
            plt.savefig("results/fd_check_parallel.png", dpi=300)
            print("\nError plots saved to 'results/fd_check_parallel.png'")
        except (ImportError, PermissionError):
            pass


if __name__ == "__main__":
    main()
