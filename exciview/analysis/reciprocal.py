import numpy as np

def analyze_bz_and_bands(exciton):
    """Analyzes Reciprocal Space and Band contributions."""
    print("\n" + "="*40 + "\n       RECIPROCAL & BAND ANALYSIS\n" + "="*40)
    
    # 1. K-Points
    k_weights = np.sum(exciton.weights, axis=(1, 2))
    norm = np.sum(k_weights)
    if norm > 0: k_weights /= norm
    
    dom_k = np.argmax(k_weights)
    kx, ky, kz = exciton.get_k_coords(dom_k)
    
    print(f"Total Norm: {norm:.6f}")
    print(f"Dominant K: Index {dom_k} ({k_weights[dom_k]:.2%}) -> ({kx:.3f}, {ky:.3f}, {kz:.3f})")
    
    # 2. Bands
    weights_norm = exciton.weights / norm if norm > 0 else exciton.weights
    trans_matrix = np.sum(weights_norm, axis=0) # Sum over K
    
    h_contrib = np.sum(trans_matrix, axis=1) # Sum over C
    e_contrib = np.sum(trans_matrix, axis=0) # Sum over V
    
    print(f"\n[Hole Bands] (Nv={exciton.nv})")
    for i in np.argsort(h_contrib)[::-1][:5]:
        if h_contrib[i] < 0.001: continue
        diff = (exciton.nv - 1) - i
        label = "HOMO" if diff == 0 else f"HOMO-{diff}"
        print(f"  {i:<4} {label:<10} {h_contrib[i]:.2%}")

    print(f"\n[Electron Bands] (Nc={exciton.nc})")
    for i in np.argsort(e_contrib)[::-1][:5]:
        if e_contrib[i] < 0.001: continue
        label = "LUMO" if i == 0 else f"LUMO+{i}"
        print(f"  {i:<4} {label:<10} {e_contrib[i]:.2%}")