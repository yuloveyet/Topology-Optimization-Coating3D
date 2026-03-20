# Topology Optimization of 3D Shell Structures with Porous Infill

This repository contains a high-performance, parallelized 3D topology optimization framework implemented in **FEniCSx**. It is specifically designed to create shell structures with porous/void interiors, extending the 2D coating approach to 3D as described in the seminal paper:

> **Clausen, A., Andreassen, E., & Sigmund, O. (2017).**
> *Topology optimization of 3D shell structures with porous infill.*
> Acta Mechanica Sinica, 33(4), 778-791.

## Key Features

- **3D Coating Interpolation:** A specialized material interpolation model that strictly enforces a uniform solid shell (coating) over a porous or void base structure, using spatial gradients of the smoothed density field.
- **Two-Step PDE Filtering:** Employs a robust Helmholtz PDE-based filtering scheme:
  1. Base structure smoothing ($R_1$) and Heaviside projection.
  2. Shell gradient extraction ($R_2$) and secondary projection.
- **Domain Extension Technique:** Automatically pads the design domain with void regions and manipulates filter boundary conditions. This eliminates boundary truncation effects, ensuring uniform coating thickness even at the edges of the physical domain.
- **Hashin-Shtrikman Bounds:** The stiffness of the porous infill is modeled using the 3D Hashin-Shtrikman upper bounds, strictly aligning with physical limits.
- **High-Performance MPI Parallelization:** Built on `dolfinx` and `PETSc` (using `GAMG` preconditioners), allowing for the optimization of multi-million degree-of-freedom 3D structures across distributed memory clusters.
- **Real-time 3D Visualization:** Automatically generates longitudinal cross-section slices (PNG) and exports raw density fields (`.xdmf`) for `ParaView` at specified intervals without blocking the MPI execution.

## Installation

The project relies on the modern FEniCSx stack. An `environment.yml` is provided for easy setup using Conda/Mamba:

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate fenitop
```

**Core Dependencies:**
- `python >= 3.10`
- `fenics-dolfinx`
- `mpi4py`
- `petsc4py`
- `pyvista` (for fast PNG slicing)

## Usage

### 1. Running an Optimization
To run the 3D MBB/Cantilever beam coating example, execute the script using `mpirun`. Adjust the number of processes (`-n`) based on your hardware:

```bash
mpirun -n 8 python3 scripts/coating_beam_3d.py
```

*Note: The script includes automated Domain Extension padding. If you encounter Out-Of-Memory (OOM / Signal 9) errors, reduce the `mesh_res_phys` or the `filter_radius` in the script to lower the degrees of freedom.*

### 2. Verifying Sensitivities
A finite difference testing script is included to verify the exactness of the analytical gradients (derived via UFL automatic differentiation and adjoint backward passes through the PDE filters).

```bash
# Run in serial for accurate finite difference checks
python3 scripts/fd_check.py
```
This will output the relative error between analytical and numerical sensitivities and generate a Log-Log convergence plot.

## Results & Output
All results are saved in a time-stamped directory under `results/` (synchronized to UTC+8).
- `design_*.png`: Longitudinal cross-section slices showing the dense outer shell and porous/void interior.
- `design_*.xdmf`: Raw 3D density fields (`rho_total`) ready for rendering and thresholding in ParaView.

## Acknowledgements
This implementation is heavily inspired by the theoretical derivations of the TopOpt research group at DTU. The codebase translates their methodology into the modern FEniCSx (DOLFINx) paradigm.
