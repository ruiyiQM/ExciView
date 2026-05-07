import struct
import numpy as np
import scipy.sparse as sp

def read_elsi_state(filename, state_index):
    """
    Reads a specific column (state) from an ELSI binary format file.
    Uses file seeking for memory efficiency.
    """
    fmt_i8 = "l"; fmt_i4 = "i"; fmt_d  = "d"
    size_i8 = 8; size_i4 = 4; size_d  = 8
    
    with open(filename, "rb") as f:
        header = struct.unpack(fmt_i8 * 16, f.read(128))
        is_complex = (header[2] != 0)
        n_basis = header[3]
        nnz_total = header[5]
        
        # Validation
        if state_index < 0 or state_index >= n_basis:
            raise ValueError(f"State index {state_index} out of bounds.")

        offset_col_ptr = 128
        offset_row_idx = offset_col_ptr + (n_basis * size_i8)
        offset_values  = offset_row_idx + (nnz_total * size_i4)
        
        # Get Pointers
        f.seek(offset_col_ptr + (state_index * size_i8))
        ptr_start = struct.unpack(fmt_i8, f.read(size_i8))[0]
        
        if state_index == n_basis - 1:
            ptr_end = nnz_total + 1
        else:
            ptr_end = struct.unpack(fmt_i8, f.read(size_i8))[0]
            
        n_elements = ptr_end - ptr_start
        if n_elements == 0:
            return sp.csc_matrix((n_basis, 1), dtype=complex if is_complex else float)

        data_offset = ptr_start - 1
        
        # Read Rows
        f.seek(offset_row_idx + (data_offset * size_i4))
        row_indices = np.array(struct.unpack(fmt_i4 * n_elements, f.read(n_elements * size_i4))) - 1
        
        # Read Values
        val_offset_bytes = data_offset * (16 if is_complex else 8)
        f.seek(offset_values + val_offset_bytes)
        
        if not is_complex:
            values = np.array(struct.unpack(fmt_d * n_elements, f.read(n_elements * 8)))
        else:
            raw = struct.unpack(fmt_d * (n_elements * 2), f.read(n_elements * 16))
            values = np.array(raw[0::2]) + 1j * np.array(raw[1::2])

    col_ptr = np.array([0, n_elements])
    return sp.csc_matrix((values, row_indices, col_ptr), shape=(n_basis, 1))
