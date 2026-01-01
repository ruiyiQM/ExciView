# ExciView

**BSE/Excited State Analysis Tools for FHI-aims**

**Implemented by:** Ruiyi Zhou (ETH Zurich) ruiyi.zhou@phys.chem.ethz.ch

ExciView is a Python-based toolkit designed to analyze excited state wavefunctions (excitons) calculated using the Bethe-Salpeter Equation (BSE) formalism in **FHI-aims**. It bridges the gap between reciprocal space (k-space) eigenvector weights, real-space atomic contributions, and 3D volumetric densities.

## 🚀 Key Features

*   **Efficient Binary I/O:** Reads specific eigenvectors directly from large ELSI binary files without loading the full matrix into memory.
*   **Reciprocal Space Analysis:** Calculates the momentum distribution ($k$-weights) of the exciton in the Brillouin zone.
*   **Real-Space Projection (PBC):** Performs Mulliken decomposition analysis for particle (electron) and hole densities in systems with Periodic Boundary Conditions.
*   **Orbital Decomposition:** Decomposes contributions by angular momentum channel ($s, p, d, f$) per atom.
*   **Volumetric Visualization:** Generates average electron and hole density `.cube` files by performing weighted summations of eigenstate densities.
*   **Smart Thresholding:** Optimizes computational cost by identifying dominant $k$-points and bands, generating optimized input snippets for FHI-aims.

## 📋 Workflow

The tool operates via an interactive menu with two main analysis pipelines:

### Pipeline A: Atomic Population Analysis (Mulliken)
1.  **Phase 1 (Screening):** Scans the binary BSE file to identify dominant $k$-points ($W_k > \epsilon$) and generates a `mulliken_snippet.in` for FHI-aims.
2.  **Phase 2 (Analysis):** Maps the resulting FHI-aims `band_mulliken` outputs back to the BSE weights to compute the final spatial distribution (Atom/Orbital breakdown).

### Pipeline B: Volumetric Analysis (Cube Files)
3.  **Phase 3 (Cube Generation):** Identifies specific (k-point, band) pairs that contribute significantly to the hole/electron and generates a `cube_snippet.in` for FHI-aims.
4.  **Phase 4 (Density Summation):** Reads the requested `.cube` files (validated against `control.in`) and sums them according to their BSE weights to produce `avg_hole.cube` and `avg_elec.cube`.

## 🛠 Roadmap / To-Do

- [ ] **Conditional Density:** Calculate electron density distribution for a fixed hole position ($\rho_e(\mathbf{r} | \mathbf{r}_h)$).
- [ ] **Molecular Analysis:** Support for finite systems (Molecules) covering LR-TDDFT, BSE, and full-frequency GW analysis.

## 📦 Requirements
*   Python 3.x
*   NumPy
*   SciPy
*   **ASE (Atomic Simulation Environment)** (Required for Phase 4)

## ✉️ Contact
For questions or issues, please contact **Ruiyi Zhou** (ruiyi.zhou@phys.chem.ethz.ch) at ETH Zurich.
