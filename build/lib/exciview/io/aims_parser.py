import numpy as np
import re

def parse_mulliken_file(filename):
    """Parses band_mulliken output. Returns dict[band][atom] = {total, orbitals}"""
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
                    orb_vals = np.array([float(x) for x in parts[5:]])
                    mulliken_data[current_band][atom_id] = {'total': total_val, 'orbitals': orb_vals}
                except: continue
    except FileNotFoundError: return None
    return mulliken_data

def parse_control_for_cubes(control_file="cube_snippet.in"):
    """Returns set of (k_idx, band_idx) found in control file."""
    valid_cubes = set()
    current_band = None
    try:
        with open(control_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                # Detect output cube
                if parts[0] == "output" and "cube" in parts:
                    try: current_band = int(parts[-1])
                    except: current_band = None
                # Detect kpoint modifier
                elif "cube" in parts and "kpoint" in parts:
                    try:
                        k_idx = int(parts[-1])
                        if current_band is not None: valid_cubes.add((k_idx, current_band))
                    except: pass
    except FileNotFoundError: return None
    return valid_cubes

def parse_snippet_for_mapping(filename):
    """Extracts K-indices from generated mulliken_snippet.in"""
    k_indices = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.strip().startswith("output band_mulliken"):
                    match = re.search(r"K_(\d+)", line)
                    if match: k_indices.append(int(match.group(1)))
    except: return None
    return k_indices