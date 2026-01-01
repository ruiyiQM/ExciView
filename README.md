# ExciView

**BSE/Excited State Analysis Tools for FHI-aims**

**Implemented by:** Ruiyi Zhou (ETH Zurich) ruiyi.zhou@phys.chem.ethz.ch

ExciView is a Python-based toolkit designed to analyze excited state wavefunctions (excitons) calculated using the Bethe-Salpeter Equation (BSE) formalism in **FHI-aims**. It bridges the gap between reciprocal space (k-space) eigenvector weights, real-space atomic contributions, and complex 3D volumetric wavefunctions.

## 🚀 Key Features

*   **Efficient Binary I/O:** Reads specific eigenvectors directly from large ELSI binary files without loading the full matrix into memory.
*   **Reciprocal Space Analysis:** Analyzes momentum distribution ($k$-weights) and identifies dominant orbital transitions (e.g., HOMO/LUMO character) across the Brillouin zone.
*   **Real-Space Projection (Mulliken):** Performs decomposition analysis for particle (electron) and hole densities in systems with Periodic Boundary Conditions, including per-atom orbital angular momentum ($s, p, d, f$) breakdown.
*   **Average Volumetric Density:** Generates `.cube` files for the average hole and electron densities by performing weighted incoherent sums of band densities.
*   **Conditional Wavefunction Analysis:** Calculates the correlated electron density distribution for a specific *fixed* hole position ($\rho_e(\mathbf{r} | \mathbf{r}_h)$) by performing coherent wavefunction summation.
*   **Smart Thresholding:** Optimizes computational cost by identifying dominant $k$-points/bands and generating optimized input snippets for FHI-aims.

## 📋 Workflow

The tool operates via an interactive menu with four main analysis pipelines:

### 1. Atomic Population Pipeline (Mulliken)
*   **Phase 1 (Screening):** Scans the binary BSE file to identify dominant $k$-points ($W_k > \epsilon$) and generates a `mulliken_snippet.in` for FHI-aims.
*   **Phase 2 (Analysis):** Maps the resulting FHI-aims `band_mulliken` outputs back to the BSE weights to compute the final spatial distribution and orbital character.

### 2. Average Density Pipeline (Volumetric)
*   **Phase 3 (Cube Generation):** Identifies specific (k-point, band) pairs that contribute significantly to the hole/electron and generates a `cube_snippet.in` requesting `eigenstate_density`.
*   **Phase 4 (Summation):** Reads the requested `.cube` files (validated against `control.in`) and sums them according to their BSE weights to produce `avg_hole.cube` and `avg_elec.cube`.

### 3. Brillouin Zone & Band Analysis
*   **Phase 5 (Statistics):** Performs a pure statistical analysis of the BSE eigenvector to report the dominant $k$-points and the dominant band transitions (e.g., HOMO-1 $\to$ LUMO) responsible for the excitation.

### 4. Conditional Density Pipeline (To be added)
*   **Phase 6 (Input Gen):** Generates input requests for raw `eigenstate` wavefunctions (preserving phase information) for relevant bands.
*   **Phase 7 (Coherent Sum):** Asks for a fixed hole coordinate $\mathbf{r}_h$, extracts hole amplitudes, computes mixing coefficients, and constructs the conditional electron density $\rho_{cond}(\mathbf{r})$.

## 🛠 Roadmap / To-Do

- [x] **Volumetric Plotting:** Generate `.cube` files for average hole/electron density visualization.
- [ ] **Conditional Density:** Calculate electron density distribution for a fixed hole position.
- [ ] **Molecular Analysis:** Support for finite systems (Molecules) covering LR-TDDFT, BSE, and full-frequency GW analysis.

## 📦 Requirements
*   Python 3.x
*   NumPy
*   SciPy
*   **ASE (Atomic Simulation Environment)** (Required for Cube file operations)

## ✉️ Contact
For questions or issues, please contact **Ruiyi Zhou** (ruiyi.zhou@phys.chem.ethz.ch) at ETH Zurich.
