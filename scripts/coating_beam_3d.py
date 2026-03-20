import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_box, CellType

from fenitop.coating_3d import topopt_coating_3d


filter_radius = 5
# 1. d_ext set to exactly 2.0 * filter_radius as per the paper's safety standard
d_ext = 2.0 * filter_radius 

# 2. Define custom physical mesh resolution (original intended density)
mesh_res_phys = [50, 150, 50] 
# Calculate element size (dx, dy, dz) from physical domain [10, 30, 10]
dx, dy, dz = 10.0 / mesh_res_phys[0], 30.0 / mesh_res_phys[1], 10.0 / mesh_res_phys[2]

# 3. Calculate padding elements to ensure perfect boundary alignment
n_pad_x = int(np.ceil(d_ext / dx))
n_pad_z = int(np.ceil(d_ext / dz))
# Re-calculate exact d_ext to be a multiple of element size
actual_d_ext_x = n_pad_x * dx
actual_d_ext_z = n_pad_z * dz

mesh_resolution = [
    mesh_res_phys[0] + 2 * n_pad_x,
    mesh_res_phys[1],
    mesh_res_phys[2] + 2 * n_pad_z
]

mesh = create_box(
    MPI.COMM_WORLD,
    [[-actual_d_ext_x, 0, -actual_d_ext_z], [10 + actual_d_ext_x, 30, 10 + actual_d_ext_z]],
    mesh_resolution,
    CellType.hexahedron,
)
if MPI.COMM_WORLD.rank == 0:
    mesh_serial = create_box(
        MPI.COMM_SELF,
        [[-actual_d_ext_x, 0, -actual_d_ext_z], [10 + actual_d_ext_x, 30, 10 + actual_d_ext_z]],
        mesh_resolution,
        CellType.hexahedron,
    )
else:
    mesh_serial = None

fem = {
    "mesh": mesh,
    "mesh_serial": mesh_serial,
    "young's modulus": 100,
    "poisson's ratio": 0.25,
    "disp_bc": lambda x: np.isclose(x[1], 0)
    & (x[0] >= 0.0) & (x[0] <= 10.0) & (x[2] >= 0.0) & (x[2] <= 10.0)
    & (np.less(x[0], 1.5) | np.greater(x[0], 8.5)),
    "traction_bcs": [
        [
            (0, 0, -2.0),
            lambda x: np.isclose(x[1], 30)
            & (
                np.greater(x[0], 4.5)
                & np.less(x[0], 5.5)
                & np.greater(x[2], 4.5)
                & np.less(x[2], 5.5)
            ),
        ]
    ],
    "body_force": (0, 0, 0),
    "quadrature_degree": 2,
    "petsc_options": {
        "ksp_type": "cg",
        "pc_type": "gamg",
    },
}

opt = {
    "max_iter": 200,
    "opt_tol": 1e-5,
    "vol_frac": 0.20,
    "solid_zone": lambda x: np.full(x.shape[1], False),
    # Add a tiny tolerance (1e-6) to prevent exact boundary nodes from being marked as void
    "void_zone": lambda x: (x[0] < -1e-6) | (x[0] > 10.0 + 1e-6) | (x[2] < -1e-6) | (x[2] > 10.0 + 1e-6),
    "penalty": 3.0,
    "epsilon": 1e-6,
    "filter_radius": filter_radius,
    "filter_radius_shell": 2.5, # Crucial: shell filter must be smaller than base filter
    "beta_interval": 20,
    "beta_max": 64,
    "use_oc": False,
    "move": 0.05,
    "opt_compliance": True,
    "plot_freq": 10,
    # Parameters aligned with Clausen et al. (2017) paper
    "lambda_m": 0.7,
    "lambda_E": 0.7 / (2.0 - 0.7), # 3D Hashin-Shtrikman upper bound (Eq. 1 in paper)
    "penal_shell": 1.0,           # Coating penalty pg=1 (Section 2.3 in paper)
    "shell_eta": 0.5,
}


if __name__ == "__main__":
    topopt_coating_3d(fem, opt)

# Execute in parallel:
# mpirun -n 8 python3 scripts/coating_beam_3d.py
