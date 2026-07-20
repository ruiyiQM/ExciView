"""Conditional electron-density analysis for a fixed hole position."""

import numpy as np
import glob
from exciview.io.cube_tools import read_complex_pair, get_grid_val, write_cube, ASE_AVAILABLE

def generate_cond_inputs(exciton, threshold=0.01, filename="cube_cond_snippet.in"):
    """Generate requests for real and imaginary wavefunction cube files."""
    w_h = exciton.get_hole_weights()
    w_e = exciton.get_electron_weights()
    
    with open(filename, "w") as f:
        f.write(f"# Conditional Snippet\n")
        # The complete request-generation logic is still a placeholder in this release.
    print(f"Written to {filename}")

def analyze_conditional(exciton, r_fix, pat_real, pat_imag, thresh=0.01):
    """Build a conditional electron density by coherently summing wavefunctions."""
    if not ASE_AVAILABLE: return
    
    # 1. Sample each relevant hole wavefunction at the fixed hole coordinate.
    print("Extracting hole amplitudes...")
    hole_vals = {}
    w_h = exciton.get_hole_weights()
    
    for k in range(exciton.nk):
        for v in range(exciton.nv):
            if w_h[k, v] >= thresh:
                b_abs = exciton.v_start + v
                data, atoms = read_complex_pair(pat_real, pat_imag, b_abs, k+1)
                if data is not None:
                    hole_vals[(k, v)] = get_grid_val(data, atoms, r_fix)

    # 2. Use the sampled hole amplitudes to construct the electron wavefunction.
    print("Constructing electron wavefunction...")
    total_psi = None; ref_atoms = None
    w_e = exciton.get_electron_weights()
    
    for k in range(exciton.nk):
        for c in range(exciton.nc):
            # Calculate the hole-conditioned mixing coefficient for this conduction band.
            coeff = 0j
            for v in range(exciton.nv):
                if (k, v) in hole_vals:
                    # The BSE amplitude is multiplied by the complex-conjugate hole value.
                    val = exciton.coefficients[k, v, c] * np.conjugate(hole_vals[(k, v)])
                    coeff += val
            
            if abs(coeff) < 1e-10: continue
            if w_e[k, c] < thresh: continue
            
            b_abs = exciton.c_start + c
            data, atoms = read_complex_pair(pat_real, pat_imag, b_abs, k+1)
            
            if data is not None:
                if total_psi is None: total_psi = np.zeros_like(data); ref_atoms = atoms
                total_psi += coeff * data

    # 3. Convert the coherent wavefunction into a density and write a cube file.
    if total_psi is not None:
        rho = np.abs(total_psi)**2
        out = f"cond_density_{exciton.id}.cube"
        with open(out, 'w') as f: write_cube(f, ref_atoms, data=rho)
        print(f"Done: {out}")
