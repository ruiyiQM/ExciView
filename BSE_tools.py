import struct
import numpy as np
import scipy.sparse as sp
import sys
import os
import re
import glob

# Try to import ASE (Required for Phase 4 & 6)
try:
    from ase.io.cube import read_cube, write_cube
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False

# =============================================================================
#                                CORE FUNCTIONS
# =============================================================================

def read_elsi_state(filename, state_index):
    """Reads a specific eigenvector from the ELSI binary file."""
    fmt_i8 = "l"; fmt_i4 = "i"; fmt_d  = "d"
    size_i8 = 8; size_i4 = 4; size_d  = 8
    
    with open(filename, "rb") as f:
        header_data = f.read(128)
        header = struct.unpack(fmt_i8 * 16, header_data)
        is_complex = (header[2] != 0); n_basis = header[3]; nnz_total = header[5]
        
        if state_index < 0 or state_index >= n_basis:
            raise ValueError(f"State index {state_index} is out of bounds (Max: {n_basis-1})")

        offset_col_ptr = 128
        offset_row_idx = offset_col_ptr + (n_basis * size_i8)
        offset_values  = offset_row_idx + (nnz_total * size_i4)
        
        f.seek(offset_col_ptr + (state_index * size_i8))
        ptr_start = struct.unpack(fmt_i8, f.read(size_i8))[0]
        ptr_end = nnz_total + 1 if state_index == n_basis - 1 else struct.unpack(fmt_i8, f.read(size_i8))[0]
        n_elements = ptr_end - ptr_start
        
        if n_elements == 0:
            return sp.csc_matrix((n_basis, 1), dtype=complex if is_complex else float)

        data_offset = ptr_start - 1
        f.seek(offset_row_idx + (data_offset * size_i4))
        row_indices = np.array(struct.unpack(fmt_i4 * n_elements, f.read(n_elements * size_i4))) - 1
        
        offset_v = offset_values + (data_offset * (16 if is_complex else 8))
        f.seek(offset_v)
        if not is_complex:
            values = np.array(struct.unpack(fmt_d * n_elements, f.read(n_elements * 8)))
        else:
            raw = struct.unpack(fmt_d * (n_elements * 2), f.read(n_elements * 16))
            values = np.array(raw[0::2]) + 1j * np.array(raw[1::2])

    col_ptr = np.array([0, n_elements])
    return sp.csc_matrix((values, row_indices, col_ptr), shape=(n_basis, 1))

def parse_aims_mulliken_file(filename):
    """Parses a single FHI-aims band_mulliken output file."""
    mulliken_data = {} 
    current_band = None
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.split()
            if not parts: continue
            if parts[0] == "State" and len(parts) == 2 and parts[1].isdigit():
                current_band = int(parts[1])
                mulliken_data[current_band] = {}
                continue
            if current_band is not None and parts[0].isdigit() and int(parts[0]) == current_band:
                try:
                    atom_id = int(parts[3])
                    total_val = float(parts[4])
                    # Handle variable orbital lengths (s, p, d...)
                    orb_vals = np.array([float(x) for x in parts[5:]])
                    mulliken_data[current_band][atom_id] = {'total': total_val, 'orbitals': orb_vals}
                except (ValueError, IndexError): continue
    except FileNotFoundError: return None
    return mulliken_data

def get_bse_weights(filename, state_idx, nk, nv, nc):
    """Returns probability weights tensor (Nk, Nv, Nc)."""
    sparse_vec = read_elsi_state(filename, state_idx)
    dense_vec = sparse_vec.toarray().flatten()
    expected_size = nk * nv * nc
    if dense_vec.shape[0] != expected_size:
        raise ValueError(f"Dimension mismatch! File: {dense_vec.shape[0]}, Expected: {expected_size}")
    coeffs = dense_vec.reshape((nk, nv, nc))
    return np.abs(coeffs)**2

def parse_snippet_for_mapping(snippet_filename):
    """Reads mapping from input snippet."""
    k_indices = []
    try:
        with open(snippet_filename, 'r') as f:
            for line in f:
                if line.strip().startswith("output band_mulliken"):
                    match = re.search(r"K_(\d+)", line)
                    if match:
                        k_indices.append(int(match.group(1)))
    except FileNotFoundError:
        print(f"[Error] Mapping file not found: {snippet_filename}")
        return None
    return k_indices

def parse_control_for_cubes(control_file="cube_snippet.in"):
    """Parses control.in to find which cubes were actually requested."""
    valid_cubes = set()
    current_band = None
    try:
        with open(control_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            parts = line.split()
            if not parts: continue
            if parts[0] == "output" and "cube" in parts and ("eigenstate_density" in parts or "eigenstate" in parts):
                try: current_band = int(parts[-1])
                except ValueError: current_band = None
            elif "cube" in parts and "kpoint" in parts:
                try:
                    k_idx = int(parts[-1])
                    if current_band is not None: valid_cubes.add((k_idx, current_band))
                except ValueError: pass
    except FileNotFoundError:
        print(f"[Error] Could not read {control_file}")
        return None
    return valid_cubes

def get_k_coords(k_index, nkx, nky, nkz):
    if nkx * nky * nkz == 0: return (0.0, 0.0, 0.0)
    iz = k_index % nkz; iy = (k_index // nkz) % nky; ix = k_index // (nkz * nky)
    return (ix/nkx if nkx>1 else 0, iy/nky if nky>1 else 0, iz/nkz if nkz>1 else 0)

def get_grid_val_at_coord(data, atoms, coord):
    """Finds value in grid closest to Cartesian coord."""
    cell = atoms.get_cell()
    try: inv_cell = np.linalg.inv(cell)
    except: 
        inv_cell = np.zeros((3,3))
        for i in range(3): inv_cell[i,i] = 1.0/cell[i,i]
    frac = np.dot(coord, inv_cell) % 1.0
    shape = np.array(data.shape)
    indices = np.floor(frac * shape).astype(int)
    return data[indices[0], indices[1], indices[2]]

# =============================================================================
#                                WORKFLOWS
# =============================================================================

def workflow_generate_inputs_mulliken():
    print("\n--- PHASE 1: Generate Mulliken Inputs ---")
    fname = input("Binary filename: ").strip()
    if not os.path.exists(fname): return print("File not found.")
    
    state_idx = int(input("State Index: "))
    nv = int(input("Nv: ")); nc = int(input("Nc: ")); nk = int(input("Nk: "))
    nkx = int(input("Kx: ")); nky = int(input("Ky: ")); nkz = int(input("Kz: "))
    
    try: weights = get_bse_weights(fname, state_idx, nk, nv, nc)
    except Exception as e: return print(f"Error: {e}")

    k_weights = np.sum(weights, axis=(1, 2))
    norm = np.sum(k_weights)
    if norm > 0: k_weights /= norm

    thresh = float(input("Threshold %: ")) / 100.0
    out_file = "mulliken_snippet.in"
    count = 0
    with open(out_file, "w") as f:
        f.write(f"# Mulliken Snippet | State {state_idx} | Thresh {thresh*100}%\n")
        for i, w in enumerate(k_weights):
            if w >= thresh:
                count += 1
                kx, ky, kz = get_k_coords(i, nkx, nky, nkz)
                label = f"K_{i}"
                line = (f"output band_mulliken {kx:.6f} {ky:.6f} {kz:.6f} "
                        f"{kx:.6f} {ky:.6f} {kz:.6f} 2 {label} {label}")
                f.write(line + "\n")
    print(f"Success! Wrote {count} lines to {out_file}")

def workflow_analyze_spatial():
    print("\n--- PHASE 2: Spatial Analysis (Mulliken) ---")
    fname = input("Binary filename: ").strip()
    if not os.path.exists(fname): return print("File not found.")
    state_idx = int(input("State Index: "))
    nv = int(input("Nv: ")); nc = int(input("Nc: ")); nk = int(input("Nk: "))
    
    snippet_file = input("Snippet file (default: mulliken_snippet.in): ").strip()
    if not snippet_file: snippet_file = "mulliken_snippet.in"
    k_indices = parse_snippet_for_mapping(snippet_file)
    if not k_indices: return

    v_start = int(input("First Valence Band ID: "))
    c_start = int(input("First Conduction Band ID: "))
    pat = input("File Pattern (e.g. bandmlk{}.out): ")
    offset = int(input("Start Offset (e.g. 1001): "))

    try: weights = get_bse_weights(fname, state_idx, nk, nv, nc)
    except Exception as e: return print(f"Error: {e}")

    hole_pop = None; elec_pop = None; atom_ids = []
    hole_atom_orbs = {}; elec_atom_orbs = {}
    MAX_ORBS = 6; max_orb = 0; files_proc = 0

    print("Processing...")
    for i, k_idx in enumerate(k_indices):
        m_file = pat.format(offset + i)
        m_data = parse_aims_mulliken_file(m_file)
        if not m_data: continue
        files_proc += 1
        
        if hole_pop is None:
            first = list(m_data.keys())[0]
            atom_ids = sorted(list(m_data[first].keys()))
            hole_pop = np.zeros(max(atom_ids)+1); elec_pop = np.zeros(max(atom_ids)+1)
        
        for v in range(nv):
            if (v_start + v) in m_data:
                w = np.sum(weights[k_idx, v, :])
                for atom in atom_ids:
                    d = m_data[v_start + v][atom]
                    hole_pop[atom] += w * d['total']
                    if atom not in hole_atom_orbs: hole_atom_orbs[atom] = np.zeros(MAX_ORBS)
                    orbs = d['orbitals']; n = len(orbs)
                    if n > max_orb: max_orb = n
                    if n <= MAX_ORBS: hole_atom_orbs[atom][:n] += w * orbs

        for c in range(nc):
            if (c_start + c) in m_data:
                w = np.sum(weights[k_idx, :, c])
                for atom in atom_ids:
                    d = m_data[c_start + c][atom]
                    elec_pop[atom] += w * d['total']
                    if atom not in elec_atom_orbs: elec_atom_orbs[atom] = np.zeros(MAX_ORBS)
                    orbs = d['orbitals']; n = len(orbs)
                    if n <= MAX_ORBS: elec_atom_orbs[atom][:n] += w * orbs

    sum_h = np.sum(hole_pop); sum_e = np.sum(elec_pop)
    if sum_h > 1e-9: 
        hole_pop /= sum_h
        for a in hole_atom_orbs: hole_atom_orbs[a] /= sum_h
    if sum_e > 1e-9: 
        elec_pop /= sum_e
        for a in elec_atom_orbs: elec_atom_orbs[a] /= sum_e

    out_name = f"exciton_analysis_state_{state_idx}.dat"
    valid_atoms = [idx for idx in range(len(hole_pop)) if idx in atom_ids]
    orb_labels = ['s', 'p', 'd', 'f', 'g', 'h']
    
    with open(out_name, "w") as f:
        f.write(f"# Exciton Analysis State {state_idx}\n# Files: {files_proc}\n\n")
        f.write("# SECTION 1: DETAILED ATOMIC ORBITAL BREAKDOWN\n")
        f.write("#" + "-"*60 + "\n")
        for atom in valid_atoms:
            f.write(f"Atom: {atom:<4d} | Hole Pop: {hole_pop[atom]:.6f} | Elec Pop: {elec_pop[atom]:.6f}\n")
            h_orbs = hole_atom_orbs.get(atom, np.zeros(MAX_ORBS))
            e_orbs = elec_atom_orbs.get(atom, np.zeros(MAX_ORBS))
            for i in range(max_orb):
                lab = f"l={i} ({orb_labels[i]})"
                f.write(f"    {lab:<10} H: {h_orbs[i]:.6f}   E: {e_orbs[i]:.6f}\n")
            f.write("\n")
    print(f"Done! Saved to {out_name}")

def workflow_generate_cube_inputs():
    print("\n--- PHASE 3: Generate Cube Inputs ---")
    fname = input("Binary filename: ").strip()
    if not os.path.exists(fname): return print("File not found.")
    
    state_idx = int(input("State Index: "))
    nv = int(input("Nv: ")); nc = int(input("Nc: ")); nk = int(input("Nk: "))
    v_start = int(input("Absolute ID First Valence Band: "))
    c_start = int(input("Absolute ID First Conduction Band: "))

    try: weights = get_bse_weights(fname, state_idx, nk, nv, nc)
    except Exception as e: return print(f"Error: {e}")

    w_hole = np.sum(weights, axis=2) 
    w_elec = np.sum(weights, axis=1) 

    thresh = float(input("Threshold % (e.g. 1.0): ")) / 100.0
    
    out_file = "cube_snippet.in"
    count = 0
    
    with open(out_file, "w") as f:
        f.write(f"# Cube Snippet | State {state_idx} | Thresh {thresh*100}%\n")
        f.write("\n# --- HOLE CUBES ---\n")
        for k in range(nk):
            for v in range(nv):
                if w_hole[k, v] >= thresh:
                    count += 1
                    band_abs = v_start + v
                    f.write(f"output cube eigenstate_density {band_abs}\n")
                    f.write(f"   cube kpoint {k+1}\n")

        f.write("\n# --- ELECTRON CUBES ---\n")
        for k in range(nk):
            for c in range(nc):
                if w_elec[k, c] >= thresh:
                    count += 1
                    band_abs = c_start + c
                    f.write(f"output cube eigenstate_density {band_abs}\n")
                    f.write(f"   cube kpoint {k+1}\n")
                    
    print(f"Success! Generated {count} cube requests in {out_file}")

def workflow_analyze_cubes():
    print("\n--- PHASE 4: Average Density (Cube Summation) ---")
    if not ASE_AVAILABLE: return print("[Error] ASE not installed.")

    fname = input("Binary filename: ").strip()
    if not os.path.exists(fname): return print("File not found.")
    
    state_idx = int(input("State Index: "))
    nv = int(input("Nv: ")); nc = int(input("Nc: ")); nk = int(input("Nk: "))
    v_start = int(input("Absolute ID First Valence Band: "))
    c_start = int(input("Absolute ID First Conduction Band: "))
    
    print("\n[Validation]")
    ctrl_file = input("Control file (default: cube_snippet.in): ").strip()
    if not ctrl_file: ctrl_file = "cube_snippet.in"
    active_cubes = parse_control_for_cubes(ctrl_file)
    if not active_cubes:
        if input("  -> No valid requests found. Continue anyway? (y/n): ").lower() != 'y': return
        use_validation = False
    else:
        print(f"  -> Found {len(active_cubes)} active cube targets.")
        use_validation = True

    print("\n[Filename Pattern]")
    default_pat = "cube_*_eigenstate_density_{:05d}_spin_1_k_point_{:04d}.cube"
    pat = input(f"Pattern (default: {default_pat}): ").strip()
    if not pat: pat = default_pat

    print("\nReading Vectors...")
    try: weights = get_bse_weights(fname, state_idx, nk, nv, nc)
    except Exception as e: return print(f"Error: {e}")

    w_hole = np.sum(weights, axis=2); w_elec = np.sum(weights, axis=1)
    processed_hole_weight = 0.0; processed_elec_weight = 0.0

    print("Summing Hole Density...")
    avg_hole_data = None; ref_atoms = None; count_h = 0
    for k in range(nk):
        for v in range(nv):
            band_abs = v_start + v
            if use_validation and (k+1, band_abs) not in active_cubes: continue
            
            search_pat = pat.format(band_abs, k+1)
            files = glob.glob(search_pat)
            if not files: continue
            
            with open(files[0], 'r') as f:
                content = read_cube(f, read_data=True)
            data = content['data'] if isinstance(content, dict) else content[0]
            atoms = content['atoms'] if isinstance(content, dict) else content[1]
            
            if avg_hole_data is None: avg_hole_data = np.zeros_like(data); ref_atoms = atoms
            avg_hole_data += w_hole[k, v] * data
            processed_hole_weight += w_hole[k, v]
            count_h += 1

    if avg_hole_data is not None and processed_hole_weight > 0:
        avg_hole_data /= processed_hole_weight
        out_h = f"avg_hole_state_{state_idx}.cube"
        with open(out_h, 'w') as f: write_cube(f, ref_atoms, data=avg_hole_data)
        print(f"  -> Wrote {out_h}")

    print("Summing Electron Density...")
    avg_elec_data = None; count_e = 0
    for k in range(nk):
        for c in range(nc):
            band_abs = c_start + c
            if use_validation and (k+1, band_abs) not in active_cubes: continue
            
            search_pat = pat.format(band_abs, k+1)
            files = glob.glob(search_pat)
            if not files: continue
            
            with open(files[0], 'r') as f:
                content = read_cube(f, read_data=True)
            data = content['data'] if isinstance(content, dict) else content[0]
            
            if avg_elec_data is None: avg_elec_data = np.zeros_like(data)
            avg_elec_data += w_elec[k, c] * data
            processed_elec_weight += w_elec[k, c]
            count_e += 1

    if avg_elec_data is not None and processed_elec_weight > 0:
        avg_elec_data /= processed_elec_weight
        out_e = f"avg_elec_state_{state_idx}.cube"
        with open(out_e, 'w') as f: write_cube(f, ref_atoms, data=avg_elec_data)
        print(f"  -> Wrote {out_e}")
    print("Done!")

def workflow_analyze_bz():
    print("\n--- PHASE 5: BZ & Band Analysis (Text Only) ---")
    fname = input("Binary filename: ").strip()
    if not os.path.exists(fname): return print("File not found.")
    
    state_idx = int(input("State Index: "))
    nv = int(input("Nv: ")); nc = int(input("Nc: ")); nk = int(input("Nk: "))
    nkx = int(input("Kx: ")); nky = int(input("Ky: ")); nkz = int(input("Kz: "))
    
    try: weights = get_bse_weights(fname, state_idx, nk, nv, nc)
    except Exception as e: return print(f"Error: {e}")

    # --- PART 1: K-POINT ANALYSIS ---
    k_weights = np.sum(weights, axis=(1, 2))
    total_norm = np.sum(k_weights)
    if total_norm > 0: 
        k_weights_norm = k_weights / total_norm
        weights_norm = weights / total_norm
    else:
        k_weights_norm = k_weights
        weights_norm = weights
    
    dom_k_idx = np.argmax(k_weights_norm)
    kx, ky, kz = get_k_coords(dom_k_idx, nkx, nky, nkz)
    
    print("\n" + "="*40 + "\n       A. RECIPROCAL SPACE (K-POINTS)\n" + "="*40)
    print(f"Total Norm Captured: {total_norm:.6f}")
    print(f"Dominant K-point:    Index {dom_k_idx} ({k_weights_norm[dom_k_idx]:.2%})")
    print(f"                     Coord ({kx:.3f}, {ky:.3f}, {kz:.3f})")
    
    # --- PART 2: BAND ANALYSIS ---
    print("\n" + "="*40 + "\n       B. BAND TRANSITIONS (ORBITALS)\n" + "="*40)
    trans_matrix = np.sum(weights_norm, axis=0) # Sum over K
    
    h_band_contrib = np.sum(trans_matrix, axis=1) # Sum over conduction
    e_band_contrib = np.sum(trans_matrix, axis=0) # Sum over valence
    
    print(f"\n[Dominant Hole Bands] (Total Nv={nv})")
    print(f"{'Band ID':<10} {'Label':<12} {'Contribution':<15}")
    print("-" * 40)
    h_sort = np.argsort(h_band_contrib)[::-1]
    for i in h_sort[:5]: 
        val = h_band_contrib[i]
        if val < 0.001: continue
        diff = (nv - 1) - i
        label = "HOMO" if diff == 0 else f"HOMO-{diff}"
        print(f"{i:<10d} {label:<12} {val:.2%}")

    print(f"\n[Dominant Electron Bands] (Total Nc={nc})")
    print(f"{'Band ID':<10} {'Label':<12} {'Contribution':<15}")
    print("-" * 40)
    e_sort = np.argsort(e_band_contrib)[::-1]
    for i in e_sort[:5]: 
        val = e_band_contrib[i]
        if val < 0.001: continue
        label = "LUMO" if i == 0 else f"LUMO+{i}"
        print(f"{i:<10d} {label:<12} {val:.2%}")

    print(f"\n[Dominant Transitions (v -> c)]")
    print(f"{'Transition':<25} {'Weight':<10}")
    print("-" * 40)
    flat_indices = np.argsort(trans_matrix.flatten())[::-1]
    for idx in flat_indices[:5]:
        val = trans_matrix.flatten()[idx]
        if val < 0.001: continue
        v = idx // nc; c = idx % nc
        h_label = "HOMO" if v == nv-1 else f"HOMO-{nv-1-v}"
        e_label = "LUMO" if c == 0 else f"LUMO+{c}"
        print(f"{h_label} -> {e_label:<12} {val:.2%}")

def workflow_generate_cond_inputs():
    print("\n--- PHASE 5: Generate Conditional Wavefunction Inputs ---")
    fname = input("Binary filename: ").strip()
    if not os.path.exists(fname): return print("File not found.")
    
    state_idx = int(input("State Index: "))
    nv = int(input("Nv: ")); nc = int(input("Nc: ")); nk = int(input("Nk: "))
    v_start = int(input("Absolute ID First Valence Band: "))
    c_start = int(input("Absolute ID First Conduction Band: "))

    try: weights = get_bse_weights(fname, state_idx, nk, nv, nc)
    except Exception as e: return print(f"Error: {e}")

    w_hole = np.sum(weights, axis=2) 
    w_elec = np.sum(weights, axis=1)
    thresh = float(input("Threshold % (e.g. 1.0): ")) / 100.0
    
    out_file = "cube_cond_snippet.in"
    count = 0
    with open(out_file, "w") as f:
        f.write(f"# Conditional Snippet | State {state_idx} | Thresh {thresh*100}%\n")
        f.write("\n# --- HOLE WAVEFUNCTIONS ---\n")
        for k in range(nk):
            for v in range(nv):
                if w_hole[k, v] >= thresh:
                    count += 1
                    band_abs = v_start + v
                    f.write(f"output cube eigenstate {band_abs}\n")
                    f.write(f"   cube kpoint {k+1}\n")
        f.write("\n# --- ELECTRON WAVEFUNCTIONS ---\n")
        for k in range(nk):
            for c in range(nc):
                if w_elec[k, c] >= thresh:
                    count += 1
                    band_abs = c_start + c
                    f.write(f"output cube eigenstate {band_abs}\n")
                    f.write(f"   cube kpoint {k+1}\n")
    print(f"Success! Generated {count} wavefunction requests in {out_file}")

def workflow_analyze_cond():
    print("\n--- PHASE 6: Conditional Density Analysis ---")
    if not ASE_AVAILABLE: return print("[Error] ASE not installed.")
    fname = input("Binary filename: ").strip()
    if not os.path.exists(fname): return print("File not found.")
    
    state_idx = int(input("State Index: "))
    nv = int(input("Nv: ")); nc = int(input("Nc: ")); nk = int(input("Nk: "))
    v_start = int(input("Abs Valence ID: "))
    c_start = int(input("Abs Conduction ID: "))

    print("\n[Fixed Hole Position]")
    hx = float(input("  x (Ang): ")); hy = float(input("  y (Ang): ")); hz = float(input("  z (Ang): "))
    r_fix = np.array([hx, hy, hz])

    print("\n[Pattern]")
    default_pat = "cube_*_eigenstate_{:05d}_spin_1_k_point_{:04d}.cube"
    pat = input(f"Pattern (default: {default_pat}): ").strip() or default_pat

    print("\nReading Wavefunction Coefficients...")
    sparse_vec = read_elsi_state(fname, state_idx)
    dense_vec = sparse_vec.toarray().flatten()
    coeffs = dense_vec.reshape((nk, nv, nc)) 

    thresh = float(input("Threshold % used in Phase 5: ")) / 100.0
    w_check = np.sum(np.abs(coeffs)**2, axis=2)

    print("Step A: Extracting hole values...")
    hole_vals = {}
    for k in range(nk):
        for v in range(nv):
            if w_check[k, v] >= thresh:
                band_abs = v_start + v
                search_pat = pat.format(band_abs, k+1)
                files = glob.glob(search_pat)
                if not files: continue
                with open(files[0], 'r') as f: content = read_cube(f, read_data=True)
                data = content['data'] if isinstance(content, dict) else content[0]
                atoms = content['atoms'] if isinstance(content, dict) else content[1]
                val = get_grid_val_at_coord(data, atoms, r_fix)
                hole_vals[(k, v)] = val

    if not hole_vals: return print("Error: No valid hole wavefunctions found.")

    print("Step B: Constructing Conditional Electron Wavefunction...")
    total_psi = None; ref_atoms = None
    we_check = np.sum(np.abs(coeffs)**2, axis=1)

    for k in range(nk):
        for c in range(nc):
            mixing_coeff = 0.0 + 0.0j
            for v in range(nv):
                if (k, v) in hole_vals:
                    mixing_coeff += coeffs[k, v, c] * np.conjugate(hole_vals[(k, v)])
            
            if abs(mixing_coeff) < 1e-10: continue
            if we_check[k, c] < thresh: continue 
            
            band_abs = c_start + c
            search_pat = pat.format(band_abs, k+1)
            files = glob.glob(search_pat)
            if not files: continue
            
            with open(files[0], 'r') as f: content = read_cube(f, read_data=True)
            data = content['data'] if isinstance(content, dict) else content[0]
            atoms = content['atoms'] if isinstance(content, dict) else content[1]

            if total_psi is None:
                total_psi = np.zeros_like(data, dtype=complex)
                ref_atoms = atoms
            total_psi += mixing_coeff * data

    if total_psi is None: return print("Error: No electron contributions.")
    print("Step C: Squaring wavefunction...")
    cond_density = np.abs(total_psi)**2
    out_name = f"cond_density_state_{state_idx}.cube"
    with open(out_name, 'w') as f: write_cube(f, ref_atoms, data=cond_density)
    print(f"Done! Written to {out_name}")

# =============================================================================
#                                MAIN MENU
# =============================================================================

def main():
    print("==================================================")
    print("          BSE ANALYSIS TOOLKIT (ExciView)         ")
    print("==================================================")
    print("1. Phase 1: Generate Mulliken Inputs")
    print("2. Phase 2: Analyze Mulliken Output")
    print("3. Phase 3: Generate Cube Inputs (Avg Density)")
    print("4. Phase 4: Analyze Cube Files (Avg Density)")
    print("5. Phase 5: BZ & Band Analysis (Text Only)")
    print("6. Phase 6: Generate Conditional Inputs")
    print("7. Phase 7: Analyze Conditional Density")
    print("0. Exit")
    
    choice = input("\nSelect Option: ").strip()
    
    if choice == "1": workflow_generate_inputs_mulliken()
    elif choice == "2": workflow_analyze_spatial()
    elif choice == "3": workflow_generate_cube_inputs()
    elif choice == "4": workflow_analyze_cubes()
    elif choice == "5": workflow_analyze_bz()
    #lif choice == "6": workflow_generate_cond_inputs()
    #elif choice == "7": workflow_analyze_cond()
    elif choice == "0": sys.exit(0)
    else: print("Invalid option.")

if __name__ == "__main__":
    main()