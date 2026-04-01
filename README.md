# Topology Optimization of 3D Shell Structures with Porous Infill: A Vibe Coding Case Study

This repository showcases a high-performance, parallelized implementation of the 3D shell-infill (coating) topology optimization methodology. 

## What is Vibe Coding?

**Vibe Coding** is a modern software engineering paradigm where the developer focuses on conveying high-level architectural "vibes," mathematical intent, and physical constraints to an AI agent (in this case, **Gemini CLI**), which then handles the heavy lifting of implementation, debugging, and cross-platform synchronization. 

This project serves as a **Vibe Coding Case Study**, demonstrating how complex scientific computing tasks—specifically 3D finite element analysis and gradient-based optimization—can be rapidly realized from theoretical papers (e.g., Clausen et al., 2017) into a production-ready FEniCSx environment with AI as a primary collaborative partner.

## Project Overview

The implementation is built upon the foundational **[`FEniTop`](https://github.com/missionlab/fenitop)** framework, extending its capabilities to realize complex, gradient-based shell definitions in a 3D FEniCSx environment.

## Key Features

-   **3D Coating Interpolation:** A specialized material interpolation model that strictly enforces a uniform solid shell over a porous or void base structure, driven by spatial gradients of the smoothed density field.
-   **Two-Step PDE Filtering:** Employs a robust, Helmholtz PDE-based filtering scheme:
    1.  Base structure smoothing ($R_1$) and Heaviside projection.
    2.  Shell gradient extraction ($R_2$) and secondary projection.
-   **Domain Extension Technique:** Automatically pads the design domain with void regions and manipulates filter boundary conditions. This effectively eliminates boundary truncation artifacts, ensuring uniform coating thickness even at the extreme edges of the physical domain.
-   **Hashin-Shtrikman Bounds:** The stiffness of the internal porous infill is modeled using the 3D Hashin-Shtrikman upper bounds, strictly aligning the optimization process with physical limits.
-   **High-Performance MPI Parallelization:** Built entirely on FEniCSx (`dolfinx`) and `PETSc` (utilizing `GAMG` preconditioners), allowing for the scalable optimization of multi-million degree-of-freedom 3D structures across distributed memory clusters.
-   **Real-time 3D Visualization:** Automatically generates longitudinal cross-section slices (PNG) using `pyvista` and exports raw density fields (`.xdmf`) for `ParaView` at user-specified intervals, all without blocking the main MPI execution loop.

## Project Architecture

The repository is structured as follows:

-   **`fenitop/`**: The core framework library containing implementations for PDE filters, material interpolation models, sensitivity analysis, optimization loops, and parallel visualization utilities.
-   **`scripts/`**: Ready-to-use execution and validation scripts, such as the main 3D coating beam runner (`coating_beam_3d.py`) and the parallel finite difference verifier (`fd_check.py`).
-   **`results/`**: The default output directory storing iterative optimization results, including PNG cross-section slices and 3D XDMF density fields.

## Installation

This project relies on the modern FEniCSx stack. An `environment.yml` is provided for a streamlined setup using Conda/Mamba:

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate fenitop
```

**Core Dependencies:**
-   `python >= 3.10`
-   `fenics-dolfinx >= 0.8.0` (DOLFINx)
-   `mpi4py >= 3.1.0`
-   `petsc4py >= 3.18.0`
-   `numpy >= 1.21.0`
-   `scipy >= 1.7.0`
-   `pyvista >= 0.34.0`
-   `numba >= 0.55.0`
-   `scikit-image >= 0.19.0`

## Usage

### 1. Running an Optimization
To execute the 3D MBB beam coating optimization example, run the script using `mpirun`. Adjust the number of processes (`-n`) according to your hardware capabilities:

```bash
mpirun -n 8 python3 scripts/coating_beam_3d.py
```

> **Note:** The script incorporates automated Domain Extension padding. If you encounter Out-Of-Memory (OOM / Signal 9) errors, reduce the `base_mesh_res` or the `filter_radius` in the script to lower the total degrees of freedom.

### 2. Verifying Sensitivities
A finite difference testing script is provided to rigorously verify the exactness of the analytical gradients. These gradients are derived via UFL automatic differentiation and adjoint backward passes through the PDE filters.

```bash
# Run in parallel for high-speed finite difference checks
mpirun -n 4 python3 scripts/fd_check.py
```
This script evaluates representative inner and boundary points, outputs the relative error between analytical and numerical sensitivities, and saves a Log-Log convergence plot to the `results/` directory.

## Optimization Design Results

Here is a cross-sectional example of the optimization result for a 3D structural beam, highlighting the dense outer shell and the optimized porous interior:

![3D Beam Optimization Result](images/result_20260331.png)

### Results & Output Format
All results are systematically saved in a time-stamped directory under `results/` (synchronized to UTC+8).
-   `design_*.png`: High-resolution longitudinal cross-section slices rendered with `pyvista` (using the 'turbo' colormap).
-   `design_*.xdmf`: Raw 3D density fields (`rho_total`) ready for advanced rendering and thresholding in ParaView.

## Acknowledgements

### About the FEniTop Framework
The foundation of this codebase is **[`FEniTop`](https://github.com/missionlab/fenitop)**, a simple, powerful, and parallel-computing-enabled FEniCSx implementation for 2D and 3D topology optimization, originally presented by **Jia et al. (2024)**. It serves as an excellent launchpad for exploring advanced techniques.

### About This Project
This repository extends the `FEniTop` framework by integrating the specialized 3D shell-infill (coating) methodology from the seminal paper by **Clausen et al. (2017)**. The core contribution here is the successful implementation of Clausen's complex, gradient-based shell definition and filtering within a modern, distributed finite element paradigm.

### Role of Gemini CLI
The development, refactoring, and debugging of this complex implementation were significantly accelerated with the assistance of **Gemini CLI**, an AI-powered software engineering agent, which acted as a collaborative partner throughout the project's lifecycle.

## Citation

If you utilize this code or the `FEniTop` framework in your research, please cite the foundational framework:

> **Jia, Y., Wang, C., & Zhang, X. S. (2024).**
> *FEniTop: a simple FEniCSx implementation for 2D and 3D topology optimization supporting parallel computing.*
> Structural and Multidisciplinary Optimization, 67(6), 84.
> [DOI: 10.1007/s00158-024-03780-6](https://doi.org/10.1007/s00158-024-03780-6)

Additionally, please cite the theoretical foundation for the 3D shell-infill methodology implemented in this repository:

> **Clausen, A., Andreassen, E., & Sigmund, O. (2017).**
> *Topology optimization of 3D shell structures with porous infill.*
> Acta Mechanica Sinica, 33(4), 778-791.
> [DOI: 10.1007/s10409-017-0637-x](https://doi.org/10.1007/s10409-017-0637-x)

## License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for the full text.
