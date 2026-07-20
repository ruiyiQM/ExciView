"""Data structures for loading and manipulating exciton eigenvectors."""

import numpy as np
from .io.elsi import read_elsi_state

class Exciton:
    """Store one BSE exciton state and its band/k-point contributions."""

    def __init__(self, state_idx, nk, nv, nc, nk_grid=(1,1,1)):
        # Store the dimensions needed to reshape the flattened BSE vector.
        self.id = state_idx
        self.nk = nk
        self.nv = nv
        self.nc = nc
        
        # Grid dimensions (kx, ky, kz)
        self.nk_grid = nk_grid
        
        # Core data: coefficients are complex amplitudes and weights are |A|^2.
        self.coefficients = None # shape (Nk, Nv, Nc) [Complex]
        self.weights = None      # shape (Nk, Nv, Nc) [Real, |A|^2]
        
        # Absolute band offsets map local valence/conduction indices to FHI-aims IDs.
        self.v_start = 0 # Absolute band ID offset
        self.c_start = 0

    def load_from_aims(self, filename):
        """Load one sparse ELSI state and reshape it into (k, valence, conduction)."""
        print(f"Loading State {self.id} from {filename}...")
        # The reader seeks directly to the requested state to avoid loading the file.
        sparse_vec = read_elsi_state(filename, self.id)
        dense_vec = sparse_vec.toarray().flatten()
        
        # Validate the supplied physical dimensions before reshaping the vector.
        expected = self.nk * self.nv * self.nc
        if dense_vec.shape[0] != expected:
            raise ValueError(f"Dimension mismatch: Expected {expected}, got {dense_vec.shape[0]}")
            
        self.coefficients = dense_vec.reshape((self.nk, self.nv, self.nc))
        self.weights = np.abs(self.coefficients)**2
        print("Data loaded successfully.")

    def get_hole_weights(self):
        """Return weights summed over conduction bands: W(k, v)."""
        return np.sum(self.weights, axis=2)

    def get_electron_weights(self):
        """Return weights summed over valence bands: W(k, c)."""
        return np.sum(self.weights, axis=1)

    def get_k_coords(self, k_linear_idx):
        """Convert a flattened k-point index to fractional grid coordinates."""
        nkx, nky, nkz = self.nk_grid
        if nkx * nky * nkz == 0: return (0.0, 0.0, 0.0)
        
        # The stored order is z-fastest, followed by y, then x.
        iz = k_linear_idx % nkz
        iy = (k_linear_idx // nkz) % nky
        ix = k_linear_idx // (nkz * nky)
        
        return (ix/nkx if nkx>1 else 0, 
                iy/nky if nky>1 else 0, 
                iz/nkz if nkz>1 else 0)
