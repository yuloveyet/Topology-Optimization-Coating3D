import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_box, CellType

from fenitop.topopt import topopt

# SI unit: m-N-Kg-s-Pa-J-Kg/m3, mm-N-tonne-s-MPa-mJ-tonne/mm3
Lx, mesh_x, beam_ratio = 30, 50, 3
Lz, Ly = Lx, np.int32(Lx * beam_ratio)
mesh_z, mesh_y = mesh_x, np.int32(mesh_x * beam_ratio)
t_ref = 0.04 * Lx
block_size = 2 * t_ref
mesh_size = Lx / mesh_x
""" z
    |
    o---x
     \
      y
"""

mesh = create_box(
    MPI.COMM_WORLD,
    [[0, -Ly, 0], [Lx, Ly, Lz]],
    [mesh_x, mesh_y, mesh_z],
    CellType.hexahedron,
)
if MPI.COMM_WORLD.rank == 0:
    mesh_serial = create_box(
        MPI.COMM_SELF,
        [[0, -Ly, 0], [Lx, Ly, Lz]],
        [mesh_x, mesh_y, mesh_z],
        CellType.hexahedron,
    )
else:
    mesh_serial = None

width = Lx / 20
fem = {  # FEA parameters
    "mesh": mesh,
    "mesh_serial": mesh_serial,
    "young's modulus": 69e3,
    "poisson's ratio": 0.33,
    "disp_bc": lambda x: np.isclose(x[2], 0)
    & (np.less(x[1], -Ly + width) | np.greater(x[1], Ly - width)),
    "traction_bcs": [
        [
            (0, 0, -10.0),
            lambda x: np.isclose(x[2], Lz)
            & (np.greater(x[1], -width) & np.less(x[1], width))
            & (np.greater(x[0], Lx / 2 - width) & np.less(x[0], Lx / 2 + width)),
        ]
    ],
    "body_force": (0, 0, 0),
    "quadrature_degree": 2,
    "petsc_options": {
        "ksp_type": "cg",
        "pc_type": "gamg",
    },
}

opt = {  # Topology optimization parameters
    "max_iter": 200,
    "opt_tol": 1e-5,
    "vol_frac": 0.2,
    "solid_zone": lambda x: np.full(x.shape[1], False),
    "void_zone": lambda x: np.full(x.shape[1], False),
    "penalty": 3.0,
    "epsilon": 1e-6,
    "filter_radius": mesh_size * 2 * 3 / np.log(2),
    "beta_interval": 20,
    "beta_max": 128,
    "use_oc": False,
    "move": 0.02,
    "opt_compliance": True,
}

if __name__ == "__main__":
    topopt(fem, opt)

# Execute the code in parallel:
# mpirun -n 8 python3 scripts/beam_3d.py
