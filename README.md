# ExciView

**BSE/Excited State Analysis Tools for FHI-aims**

**Implemented by:** Ruiyi Zhou (ETH Zurich) ruiyi.zhou@phys.chem.ethz.ch

ExciView is a Python-based toolkit designed to analyze excited state wavefunctions (excitons) calculated using the Bethe-Salpeter Equation (BSE) formalism in **FHI-aims**. It bridges the gap between reciprocal space (k-space) eigenvector weights and real-space atomic contributions.

## 🚀 Key Features

*   **Efficient Binary I/O:** Reads specific eigenvectors directly from large ELSI binary files without loading the full matrix into memory.
*   **Reciprocal Space Analysis:** Calculates and plots the momentum distribution ($k$-weights) of the exciton in the Brillouin zone.
*   **Real-Space Projection (PBC):** Performs Mulliken decomposition analysis for particle (electron) and hole densities in systems with Periodic Boundary Conditions.
*   **Orbital Decomposition:** Decomposes contributions by angular momentum channel ($s, p, d, f$) per atom.
*   **Smart Thresholding:** Optimizes computational cost by identifying dominant $k$-points and skipping negligible contributions.

## 📋 Workflow

The tool operates in a two-phase pipeline to ensure efficiency:

1.  **Phase 1 (Screening):** Scans the binary BSE file, calculates $k$-point weights ($W_k$), and generates an optimized `mulliken_snippet.in` for FHI-aims.
2.  **Phase 2 (Analysis):** Maps the resulting FHI-aims Mulliken outputs back to the BSE weights to compute the final spatial distribution of the Hole and Electron.

## 🛠 Roadmap / To-Do

- [ ] **Volumetric Plotting:** Generate `.cube` files for average hole/electron density visualization.
- [ ] **Conditional Density:** Calculate electron density distribution for a fixed hole position ($\rho_e(\mathbf{r} | \mathbf{r}_h)$).
- [ ] **Molecular Analysis:** Support for finite systems (Molecules) covering LR-TDDFT, BSE, and full-frequency GW analysis.

## 📦 Requirements
*   Python 3.x
*   NumPy
*   SciPy
*   Matplotlib

## ✉️ Contact
For questions or issues, please contact **Ruiyi Zhou** (ruiyi.zhou@phys.chem.ethz.ch) at ETH Zurich.
