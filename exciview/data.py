import numpy as np
from .io.elsi import read_elsi_state

class Exciton:
    def __init__(self, state_idx, nk, nv, nc, nk_grid=(1,1,1)):
        self.id = state_idx
        self.nk = nk
        self.nv = nv
        self.nc = nc
        
        # Grid dimensions (kx, ky, kz)
        self.nk_grid = nk_grid
        
        # Core Data
        self.coefficients = None # shape (Nk, Nv, Nc) [Complex]
        self.weights = None      # shape (Nk, Nv, Nc) [Real, |A|^2]
        
        # Auxiliary
        self.v_start = 0 # Absolute band ID offset
        self.c_start = 0

    def load_from_aims(self, filename):
        """Loads data using the ELSI reader."""
        print(f"Loading State {self.id} from {filename}...")
        sparse_vec = read_elsi_state(filename, self.id)
        dense_vec = sparse_vec.toarray().flatten()
        
        expected = self.nk * self.nv * self.nc
        if dense_vec.shape[0] != expected:
            raise ValueError(f"Dimension mismatch: Expected {expected}, got {dense_vec.shape[0]}")
            
        self.coefficients = dense_vec.reshape((self.nk, self.nv, self.nc))
        self.weights = np.abs(self.coefficients)**2
        print("Data loaded successfully.")

    def get_hole_weights(self):
        """Returns weights summed over electrons: W(k, v)"""
        return np.sum(self.weights, axis=2)

    def get_electron_weights(self):
        """Returns weights summed over holes: W(k, c)"""
        return np.sum(self.weights, axis=1)

    def get_k_coords(self, k_linear_idx):
        """Returns fractional coordinates (kx, ky, kz)"""
        nkx, nky, nkz = self.nk_grid
        if nkx * nky * nkz == 0: return (0.0, 0.0, 0.0)
        
        iz = k_linear_idx % nkz
        iy = (k_linear_idx // nkz) % nky
        ix = k_linear_idx // (nkz * nky)
        
        return (ix/nkx if nkx>1 else 0, 
                iy/nky if nky>1 else 0, 
                iz/nkz if nkz>1 else 0)
