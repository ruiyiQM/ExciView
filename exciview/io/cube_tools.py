"""Helpers for reading, combining, sampling, and writing Gaussian cube data."""

import numpy as np
import glob
try:
    from ase.io.cube import read_cube as ase_read
    from ase.io.cube import write_cube
    ASE_AVAILABLE = True
except ImportError:
    # Keep imports usable when ASE is absent; callers receive a focused error later.
    ASE_AVAILABLE = False

def safe_read_cube(filename):
    """Read a cube file while supporting ASE's dictionary and tuple return formats."""
    if not ASE_AVAILABLE: raise ImportError("ASE not installed")
    
    with open(filename, 'r') as f:
        content = ase_read(f, read_data=True)
        
    if isinstance(content, dict):
        return content['data'], content['atoms']
    else:
        return content[0], content[1]

def read_complex_pair(pat_real, pat_imag, band, k):
    """Read real/imaginary cube files and combine them into one complex array."""
    search_real = pat_real.format(band, k)
    files_real = glob.glob(search_real)
    if not files_real: return None, None
    
    data_r, atoms = safe_read_cube(files_real[0])
    
    search_imag = pat_imag.format(band, k)
    files_imag = glob.glob(search_imag)
    
    # Gamma-point calculations may provide only a real wavefunction.
    if files_imag:
        data_i, _ = safe_read_cube(files_imag[0])
        data_c = data_r + 1j * data_i
    else:
        data_c = data_r.astype(complex)
        
    return data_c, atoms

def get_grid_val(data, atoms, coord):
    """Sample a volumetric array at a Cartesian coordinate using periodic wrapping."""
    cell = atoms.get_cell()
    try: inv = np.linalg.inv(cell)
    except: 
        inv = np.zeros((3,3))
        for i in range(3): inv[i,i] = 1.0/cell[i,i]
    
    # Convert Cartesian coordinates to fractional coordinates and wrap into the cell.
    frac = np.dot(coord, inv) % 1.0
    shape = np.array(data.shape)
    indices = np.floor(frac * shape).astype(int)
    return data[indices[0], indices[1], indices[2]]
