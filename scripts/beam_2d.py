import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_rectangle, CellType

from fenitop.topopt import topopt

L, H = 1.5, 1.0
mesh_L, mesh_H = 150, 100
t_ref = 0.04 * L
block_size = 2 * t_ref

mesh = create_rectangle(
    MPI.COMM_WORLD, [[0, 0], [L, H]], [mesh_L, mesh_H], CellType.quadrilateral
)
if MPI.COMM_WORLD.rank == 0:
    mesh_serial = create_rectangle(
        MPI.COMM_SELF, [[0, 0], [L, H]], [mesh_L, mesh_H], CellType.quadrilateral
    )
else:
    mesh_serial = None

fem = {  # FEM parameters
    "mesh": mesh,
    "mesh_serial": mesh_serial,
    "young's modulus": 100,
    "poisson's ratio": 0.25,
    "disp_bc": lambda x: np.isclose(x[0], 0),
    "traction_bcs": [
        [
            (0, -100.0),
            lambda x: (
                np.isclose(x[0], L)
                & np.greater(x[1], H / 2 - H / 10)
                & np.less(x[1], H/2+H/10)
            ),
        ]
    ],
    "body_force": (0, 0),
    "quadrature_degree": 2,
    "petsc_options": {
        "ksp_type": "cg",
        "pc_type": "gamg",
    },
}

opt = {  # Topology optimization parameters
    "max_iter": 300,
    "opt_tol": 1e-5,
    "vol_frac": 0.4,
    "solid_zone": lambda x: np.full(x.shape[1], False),
    "void_zone": lambda x: np.full(x.shape[1], False),
    "penalty": 3.0,
    "epsilon": 1e-6,
    "filter_radius": L/10,
    "beta_interval": 20,
    "beta_max": 128,
    "use_oc": False,
    "move": 0.1,
    "opt_compliance": True,
}

if __name__ == "__main__":
    topopt(fem, opt)

# Execute the code in parallel:
# mpirun -n 8 python3 scripts/beam_2d.py
