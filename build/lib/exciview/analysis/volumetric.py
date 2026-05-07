import numpy as np
import glob
from exciview.io.cube_tools import safe_read_cube, write_cube, ASE_AVAILABLE
from exciview.io.aims_parser import parse_control_for_cubes

def generate_cube_inputs(exciton, threshold=0.01, filename="cube_snippet.in"):
    """Phase 3: Generate cube requests."""
    w_h = exciton.get_hole_weights()
    w_e = exciton.get_electron_weights()
    
    with open(filename, "w") as f:
        f.write(f"# Cube Snippet | Thresh {threshold*100}%\n")
        f.write("\n# Hole\n")
        for k in range(exciton.nk):
            for v in range(exciton.nv):
                if w_h[k, v] >= threshold:
                    f.write(f"output cube eigenstate_density {exciton.v_start + v}\n")
                    f.write(f"   cube kpoint {k+1}\n")
        
        f.write("\n# Electron\n")
        for k in range(exciton.nk):
            for c in range(exciton.nc):
                if w_e[k, c] >= threshold:
                    f.write(f"output cube eigenstate_density {exciton.c_start + c}\n")
                    f.write(f"   cube kpoint {k+1}\n")
    print(f"Requests written to {filename}")

def sum_average_density(exciton, pattern, control_file="cube_snippet.in"):
    """Phase 4: Sum cubes."""
    if not ASE_AVAILABLE: return print("ASE required.")
    
    valid_set = parse_control_for_cubes(control_file)
    w_h = exciton.get_hole_weights()
    w_e = exciton.get_electron_weights()
    
    def process_sum(weights, start_band, out_name):
        avg_data = None; ref_atoms = None; total_w = 0
        for k in range(exciton.nk):
            for b in range(weights.shape[1]):
                b_abs = start_band + b
                if valid_set and (k+1, b_abs) not in valid_set: continue
                
                # File search
                pat = pattern.format(b_abs, k+1)
                files = glob.glob(pat)
                if not files: continue
                
                data, atoms = safe_read_cube(files[0])
                if avg_data is None: avg_data = np.zeros_like(data); ref_atoms = atoms
                
                w = weights[k, b]
                avg_data += w * data
                total_w += w
        
        if avg_data is not None:
            if total_w > 0: avg_data /= total_w
            with open(out_name, 'w') as f: write_cube(f, ref_atoms, data=avg_data)
            print(f"Saved {out_name}")

    print("Summing Hole...")
    process_sum(w_h, exciton.v_start, f"avg_hole_{exciton.id}.cube")
    print("Summing Electron...")
    process_sum(w_e, exciton.c_start, f"avg_elec_{exciton.id}.cube")