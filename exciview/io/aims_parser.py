"""Parsers for FHI-aims Mulliken output and generated input snippets."""

import numpy as np
import re

def parse_mulliken_file(filename):
    """Parse band_mulliken output into ``band -> atom -> values`` mappings."""
    mulliken_data = {} 
    current_band = None
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        # The output is line-oriented, so identify state headers before atom rows.
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
    """Return the (k-point, band) cube requests found in an input snippet."""
    valid_cubes = set()
    current_band = None
    try:
        with open(control_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                # Remember the band associated with the next k-point modifier.
                if parts[0] == "output" and "cube" in parts:
                    try: current_band = int(parts[-1])
                    except: current_band = None
                # Pair the remembered band with the requested k-point index.
                elif "cube" in parts and "kpoint" in parts:
                    try:
                        k_idx = int(parts[-1])
                        if current_band is not None: valid_cubes.add((k_idx, current_band))
                    except: pass
    except FileNotFoundError: return None
    return valid_cubes

def parse_snippet_for_mapping(filename):
    """Extract k-point labels from the generated Mulliken snippet."""
    k_indices = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.strip().startswith("output band_mulliken"):
                    match = re.search(r"K_(\d+)", line)
                    if match: k_indices.append(int(match.group(1)))
    except: return None
    return k_indices
