import numpy as np
from exciview.io.aims_parser import parse_mulliken_file, parse_snippet_for_mapping

def generate_mulliken_inputs(exciton, threshold=0.01, filename="mulliken_snippet.in"):
    """Phase 1: Generate FHI-aims input."""
    k_weights = np.sum(exciton.weights, axis=(1, 2))
    norm = np.sum(k_weights)
    if norm > 0: k_weights /= norm
    
    count = 0
    with open(filename, "w") as f:
        f.write(f"# ExciView Mulliken Snippet | Thresh {threshold*100}%\n")
        for i, w in enumerate(k_weights):
            if w >= threshold:
                count += 1
                kx, ky, kz = exciton.get_k_coords(i)
                label = f"K_{i}"
                f.write(f"output band_mulliken {kx:.6f} {ky:.6f} {kz:.6f} "
                        f"{kx:.6f} {ky:.6f} {kz:.6f} 2 {label} {label}\n")
    print(f"Generated {count} requests in {filename}")

def analyze_mulliken_output(exciton, pattern, offset, snippet_file, v_start, c_start):
    """Phase 2: Parse outputs and compute populations."""
    k_indices = parse_snippet_for_mapping(snippet_file)
    if not k_indices: return
    
    hole_pop = None; elec_pop = None
    hole_orbs = {}; elec_orbs = {}; max_orb = 0
    
    print("Processing...")
    for i, k_idx in enumerate(k_indices):
        fname = pattern.format(offset + i)
        m_data = parse_mulliken_file(fname)
        if not m_data: continue
        
        if hole_pop is None:
            atoms = sorted(list(m_data[list(m_data.keys())[0]].keys()))
            hole_pop = np.zeros(max(atoms)+1); elec_pop = np.zeros(max(atoms)+1)

        # Helper to accumulate
        def acc(pop, orbs_dict, band_idx, weight):
            nonlocal max_orb
            if band_idx in m_data:
                for atom in atoms:
                    d = m_data[band_idx][atom]
                    pop[atom] += weight * d['total']
                    if atom not in orbs_dict: orbs_dict[atom] = np.zeros(6)
                    n = len(d['orbitals'])
                    if n > max_orb: max_orb = n
                    orbs_dict[atom][:n] += weight * d['orbitals']

        # Hole Loop
        for v in range(exciton.nv):
            w = np.sum(exciton.weights[k_idx, v, :])
            acc(hole_pop, hole_orbs, v_start + v, w)
            
        # Elec Loop
        for c in range(exciton.nc):
            w = np.sum(exciton.weights[k_idx, :, c])
            acc(elec_pop, elec_orbs, c_start + c, w)

    # Normalize
    h_sum = np.sum(hole_pop); e_sum = np.sum(elec_pop)
    if h_sum > 0: 
        hole_pop /= h_sum
        for a in hole_orbs: hole_orbs[a] /= h_sum
    if e_sum > 0:
        elec_pop /= e_sum
        for a in elec_orbs: elec_orbs[a] /= e_sum
        
    # Write output
    out_name = f"exciton_analysis_state_{exciton.id}.dat"
    with open(out_name, 'w') as f:
        f.write("# ExciView Analysis\n# SECTION 1: ATOMIC BREAKDOWN\n")
        labels = ['s', 'p', 'd', 'f', 'g']
        atoms = sorted(hole_orbs.keys())
        for a in atoms:
            f.write(f"Atom: {a:<4} H: {hole_pop[a]:.4f} E: {elec_pop[a]:.4f}\n")
            for l in range(max_orb):
                f.write(f"  {labels[l]}: H {hole_orbs[a][l]:.4f} E {elec_orbs[a][l]:.4f}\n")
    print(f"Results written to {out_name}")