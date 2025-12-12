import struct
import numpy as np
import scipy.sparse as sp
import sys
import os
import re

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
                    # parts[3] is Atom ID, parts[4] is Total Mulliken Charge
                    mulliken_data[current_band][int(parts[3])] = float(parts[4])
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
    """Reads mulliken_snippet.in to determine which k-index corresponds to which file."""
    k_indices = []
    try:
        with open(snippet_filename, 'r') as f:
            for line in f:
                if line.strip().startswith("output band_mulliken"):
                    # Regex to find labels like K_12
                    match = re.search(r"K_(\d+)", line)
                    if match:
                        k_indices.append(int(match.group(1)))
    except FileNotFoundError:
        print(f"[Error] Could not find mapping file: {snippet_filename}")
        return None
    return k_indices

# =============================================================================
#                                WORKFLOWS
# =============================================================================

def workflow_generate_inputs():
    print("\n--- PHASE 1: Generate Inputs ---")
    fname = input("Binary filename: ").strip()
    if not os.path.exists(fname): return print("File not found.")
    
    state_idx = int(input("State Index: "))
    nv = int(input("Valence Bands (Nv): "))
    nc = int(input("Conduction Bands (Nc): "))
    nk = int(input("Total K-points (Nk): "))
    nkx = int(input("Grid Kx (for coordinate output): "))
    nky = int(input("Grid Ky: "))
    nkz = int(input("Grid Kz: "))
    
    try:
        weights_tensor = get_bse_weights(fname, state_idx, nk, nv, nc)
    except Exception as e: return print(f"Error: {e}")

    k_weights = np.sum(weights_tensor, axis=(1, 2))
    norm = np.sum(k_weights)
    if norm > 0: k_weights /= norm

    thresh_percent = float(input("Enter threshold % (e.g., 5.0): "))
    thresh_val = thresh_percent / 100.0

    out_file = "mulliken_snippet.in"
    count = 0
    with open(out_file, "w") as f:
        f.write(f"# Auto-generated for State {state_idx}, Threshold {thresh_percent}%\n")
        for i, w in enumerate(k_weights):
            if w >= thresh_val:
                count += 1
                iz = i % nkz; iy = (i // nkz) % nky; ix = i // (nkz * nky)
                kx_f = ix/nkx if nkx>1 else 0; ky_f = iy/nky if nky>1 else 0; kz_f = iz/nkz if nkz>1 else 0
                
                label = f"K_{i}"
                line = (f"output band_mulliken {kx_f:.6f} {ky_f:.6f} {kz_f:.6f} "
                        f"{kx_f:.6f} {ky_f:.6f} {kz_f:.6f} 2 {label} {label}")
                f.write(line + "\n")
                
    print(f"Success! Wrote {count} lines to '{out_file}'.")
    print("Use this file for your FHI-aims calculation.")


def workflow_analyze_spatial():
    print("\n--- PHASE 2: Spatial Analysis ---")
    
    fname = input("Binary filename: ").strip()
    if not os.path.exists(fname): return print("File not found.")

    state_idx = int(input("State Index: "))
    nv = int(input("Valence Bands (Nv): "))
    nc = int(input("Conduction Bands (Nc): "))
    nk = int(input("Total K-points (Nk): "))
    
    print("\n[Mapping]")
    snippet_file = input("Enter snippet filename (default: mulliken_snippet.in): ").strip()
    if not snippet_file: snippet_file = "mulliken_snippet.in"
    
    calculated_k_indices = parse_snippet_for_mapping(snippet_file)
    if not calculated_k_indices: return
    print(f"-> Found {len(calculated_k_indices)} k-points in mapping.")
    
    print("\n[Band IDs]")
    v_start = int(input("Absolute ID of First Valence Band: "))
    c_start = int(input("Absolute ID of First Conduction Band: "))
    
    print("\n[Output Files]")
    pat = input("Pattern (e.g., 'bandmlk{}.out'): ")
    offset = int(input("Starting index offset (e.g. 1001): "))

    print("\nReading Vectors...")
    try:
        weights_tensor = get_bse_weights(fname, state_idx, nk, nv, nc)
    except Exception as e: return print(f"Error: {e}")

    hole_pop = None
    elec_pop = None
    atom_ids = []
    files_processed = 0

    print("Processing...")
    
    for i, k_actual in enumerate(calculated_k_indices):
        file_id = offset + i
        mull_file = pat.format(file_id)
        
        m_data = parse_aims_mulliken_file(mull_file)
        if m_data is None:
            print(f"  [Warning] Missing file '{mull_file}'")
            continue
            
        files_processed += 1
        
        if hole_pop is None:
            first_band = list(m_data.keys())[0]
            atom_ids = sorted(list(m_data[first_band].keys()))
            max_atom = max(atom_ids)
            hole_pop = np.zeros(max_atom + 1)
            elec_pop = np.zeros(max_atom + 1)

        # Accumulate Hole
        for v in range(nv):
            band_abs = v_start + v
            if band_abs in m_data:
                w = np.sum(weights_tensor[k_actual, v, :]) 
                for atom in atom_ids:
                    hole_pop[atom] += w * m_data[band_abs][atom]

        # Accumulate Electron
        for c in range(nc):
            band_abs = c_start + c
            if band_abs in m_data:
                w = np.sum(weights_tensor[k_actual, :, c])
                for atom in atom_ids:
                    elec_pop[atom] += w * m_data[band_abs][atom]

    if hole_pop is None: return print("No valid data processed.")

    # --- NORMALIZATION STEP ---
    # Since we skipped k-points, the raw sums will be < 1.0. 
    # We rescale them to exactly 1.0.
    
    sum_h = np.sum(hole_pop)
    sum_e = np.sum(elec_pop)
    
    print(f"\n[Normalization]")
    print(f"  Raw Hole Sum: {sum_h:.6f} (Reflects % coverage of calculated k-points)")
    print(f"  Raw Elec Sum: {sum_e:.6f}")
    
    if sum_h > 1e-9:
        hole_pop /= sum_h
        print("  -> Renormalized Hole distribution to 1.0")
        
    if sum_e > 1e-9:
        elec_pop /= sum_e
        print("  -> Renormalized Electron distribution to 1.0")

    # --- WRITE OUTPUT ---
    out_name = f"exciton_pop_state_{state_idx}.dat"
    valid_atoms = [idx for idx in range(len(hole_pop)) if idx in atom_ids]
    
    with open(out_name, "w") as f:
        f.write(f"# Exciton Spatial Analysis | State {state_idx}\n")
        f.write(f"# Mapped {files_processed} files using {snippet_file}\n")
        f.write(f"# Normalization Applied: Yes\n")
        f.write(f"# {'Atom':<6} {'Hole':<12} {'Electron':<12} {'Diff(E-H)':<12}\n")
        f.write("#" + "-"*50 + "\n")
        for atom in valid_atoms:
            h = hole_pop[atom]; e = elec_pop[atom]
            f.write(f"  {atom:<6d} {h:<12.6f} {e:<12.6f} {e-h:<12.6f}\n")
            
    print(f"\nDone! Results written to {out_name}")

# =============================================================================
#                                MAIN MENU
# =============================================================================

def main():
    print("==================================================")
    print("          BSE & MULLIKEN ANALYSIS TOOL            ")
    print("==================================================")
    print("1. Phase 1: Generate Input Files (with threshold)")
    print("2. Phase 2: Analyze Output (using snippet map)")
    print("0. Exit")
    
    choice = input("\nSelect Option: ").strip()
    
    if choice == "1": workflow_generate_inputs()
    elif choice == "2": workflow_analyze_spatial()
    elif choice == "0": sys.exit(0)
    else: print("Invalid option.")

if __name__ == "__main__":
    main()