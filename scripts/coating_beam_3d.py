import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_box, CellType

from fenitop.coating_3d import topopt_coating_3d

# 几何基准尺寸与比例 (长:宽:高)
length = 30.0
aspect_ratio = (1.0, 1.0, 3.0)  # Length:Width:Height

width = length * (aspect_ratio[1] / aspect_ratio[0])
height = length * (aspect_ratio[2] / aspect_ratio[0])

# 网格分辨率基准 (nx)，其余方向将根据 aspect_ratio 自动缩放
# 可自由修改此参数以整体调整网格精细度 (若遇到 OOM 问题，请调低此值，如 30 或 40)
base_mesh_res = 60 
mesh_res_phys = [
    int(base_mesh_res * aspect_ratio[0]),  # nx (Length direction)
    int(base_mesh_res * aspect_ratio[2]),  # ny (Height direction)
    int(base_mesh_res * aspect_ratio[1]),  # nz (Width direction)
]
# Calculate element size (h_x, h_y, h_z) from physical domain
h_x, h_y, h_z = length / mesh_res_phys[0], height / mesh_res_phys[1], width / mesh_res_phys[2]

# 2. Define shell thickness and filter radii

filter_radius = 10  # Base filter radius (R1)
filter_radius_shell = filter_radius / 2.0  # Shell filter radius (R2 = R1 / 2.0)
tref = filter_radius_shell / 5.0  # Reference shell thickness (tref = R2 / 5.0)
# 3. d_ext set to exactly 1.0 * filter_radius as per the paper and Matlab implementation
d_ext = 1.0 * filter_radius

# 4. Calculate padding elements to ensure perfect boundary alignment
n_pad_x = int(np.ceil(d_ext / h_x))
n_pad_y = int(np.ceil(d_ext / h_y))
n_pad_z = int(np.ceil(d_ext / h_z))
# Re-calculate exact d_ext to be a multiple of element size
actual_d_ext_x = n_pad_x * h_x
actual_d_ext_y = n_pad_y * h_y
actual_d_ext_z = n_pad_z * h_z

mesh_resolution = [
    mesh_res_phys[0] + 2 * n_pad_x,
    mesh_res_phys[1] + 2 * n_pad_y,
    mesh_res_phys[2] + 2 * n_pad_z,
]


def create_domain_mesh(comm):
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

    c_coords = dolfinx.mesh.compute_midpoints(
        base_mesh,
        3,
        np.arange(base_mesh.topology.index_map(3).size_local, dtype=np.int32),
    )

    remove_top = (
        (c_coords[:, 1] > height - 1e-6)
        & (c_coords[:, 0] >= length / 2 - 0.5 - actual_d_ext_x - 1e-6)
        & (c_coords[:, 0] <= length / 2 + 0.5 + actual_d_ext_x + 1e-6)
        & (c_coords[:, 2] >= width / 2 - 0.5 - actual_d_ext_z - 1e-6)
        & (c_coords[:, 2] <= width / 2 + 0.5 + actual_d_ext_z + 1e-6)
    )
    remove_bottom = (
        (c_coords[:, 1] < 1e-6)
        & ((c_coords[:, 0] <= 1.5 + actual_d_ext_x + 1e-6) | (c_coords[:, 0] >= length - 1.5 - actual_d_ext_x - 1e-6))
    )

    cells_to_keep = np.where(~(remove_top | remove_bottom))[0].astype(np.int32)
    mesh, _, _, _ = create_submesh(base_mesh, 3, cells_to_keep)
    mesh.topology.create_connectivity(2, 3)
    return mesh


mesh = create_domain_mesh(MPI.COMM_WORLD)
if MPI.COMM_WORLD.rank == 0:
    mesh_serial = create_domain_mesh(MPI.COMM_SELF)
else:
    mesh_serial = None

fem = {
    "mesh": mesh,
    "mesh_serial": mesh_serial,
    "young's modulus": 100,
    "poisson's ratio": 0.25,
    "disp_bc": lambda x: np.isclose(x[1], 0.0)
    & (x[0] >= -1e-6)
    & (x[0] <= length + 1e-6)
    & (x[2] >= -1e-6)
    & (x[2] <= width + 1e-6)
    & (np.less(x[0], 1.5 + h_x / 2 + 1e-6) | np.greater(x[0], length - 1.5 - h_x / 2 - 1e-6)),
    "traction_bcs": [
        [
            (0.0, 0.0, -2.0),
            lambda x: np.isclose(x[1], height)
            & (
                np.greater(x[0], length / 2 - 0.5 - h_x / 2 - 1e-6)
                & np.less(x[0], length / 2 + 0.5 + h_x / 2 + 1e-6)
                & np.greater(x[2], width / 2 - 0.5 - h_z / 2 - 1e-6)
                & np.less(x[2], width / 2 + 0.5 + h_z / 2 + 1e-6)
            ),
        ]
    ],
    "body_force": (0, 0, 0),
    "quadrature_degree": 3,
    "petsc_options": {
        "ksp_type": "cg",
        "pc_type": "gamg",
        "ksp_rtol": 1e-10,
        "ksp_atol": 1e-12,
        "ksp_max_it": 1000,
    },
}

opt = {
    "max_iter": 200,
    "opt_tol": 1e-3,
    "vol_frac": 0.20,
    "solid_zone": lambda x: np.full(x.shape[1], False),
    # Add a tiny tolerance (1e-6) to prevent exact boundary nodes from being marked as void
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
    "filter_radius_shell": filter_radius_shell,  # Crucial: shell filter must be smaller than base filter
    "beta_initial": 1.0,
    "beta_inc": 2,
    "beta_interval": 20,
    "beta_max": 64*2,
    "use_oc": False,
    "move": 0.02,
    "opt_compliance": True,
    "plot_freq": 10,
    # Parameters aligned with Clausen et al. (2017) paper
    "lambda_m": 0.5,
    "q_ext_padding": 0.2,  # Soft modulus for padding region
    "penal_shell": 1.0,  # Coating penalty pg=1 (Section 2.3 in paper)
    "shell_eta": 0.5,
    "slice_normal": "x",
    "slice_origin": (length / 2.0, height / 2.0, width / 2.0),
    "clip_bounds": [
        0.0,
        length,
        0.0,
        height,
        0.0,
        width,
    ],  # Bounds of the physical domain [xmin, xmax, ymin, ymax, zmin, zmax]
}


if __name__ == "__main__":
    topopt_coating_3d(fem, opt)

# Execute in parallel:
# mpirun -n 8 python3 scripts/coating_beam_3d.py
