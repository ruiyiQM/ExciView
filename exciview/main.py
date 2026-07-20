"""Interactive command-line entry point for ExciView."""

import sys
import os
import subprocess
import platform
import importlib.resources
import numpy as np
from exciview.io.aims_parser import parse_mulliken_file
from exciview.data import Exciton
from exciview.analysis import reciprocal, mulliken, volumetric, conditional



def open_manual():
    """Locate and open the packaged user manual with the platform default viewer."""
    # Locate the PDF file within the installed package.
    try:
        # For Python 3.9+
        path_ref = importlib.resources.files("exciview.docs") / "manual.pdf"
        pdf_path = str(path_ref)
    except:
        # Fallback or if not installed yet (running from source)
        pdf_path = os.path.join(os.path.dirname(__file__), "docs", "manual.pdf")

    if not os.path.exists(pdf_path):
        print(f"[Error] Manual not found at: {pdf_path}")
        return

    print(f"Opening manual: {pdf_path}")
    
    # Use the native file-opening command for the current operating system.
    system = platform.system()
    if system == 'Darwin':       # macOS
        subprocess.call(('open', pdf_path))
    elif system == 'Windows':    # Windows
        os.startfile(pdf_path)
    else:                        # Linux
        subprocess.call(('xdg-open', pdf_path))

def main():
    """Collect calculation metadata, load an exciton, and run the selected analysis."""
    print("==================================================")
    print("          ExciView: BSE Analysis Toolkit          ")
    print("==================================================")
    
    # Read the ELSI filename and dimensions required to interpret its state vector.
    fname = input("Binary filename: ").strip() or "BSE_eigenvectors.dat"
    if not os.path.exists(fname): return print("File not found.")
    
    s_idx = int(input("State Index: "))
    nv = int(input("Nv: ")); nc = int(input("Nc: ")); nk = int(input("Nk: "))
    kx = int(input("Kx: ")); ky = int(input("Ky: ")); kz = int(input("Kz: "))
    
    # Load the selected state once; all menu actions reuse this in-memory object.
    exciton = Exciton(s_idx, nk, nv, nc, (kx, ky, kz))
    exciton.load_from_aims(fname)
    
    while True:
        print("\n--- Menu ---")
        print("1. [Mulliken] Generate Inputs")
        print("2. [Mulliken] Analyze Output")
        print("3. [Volumetric] Generate Inputs (Avg)")
        print("4. [Volumetric] Sum Cubes (Avg)")
        print("5. [Reciprocal] Band/BZ Analysis")
        print("6. [Conditional] Generate Inputs")
        print("7. [Conditional] Analyze Density")
        print("8. [Help] Open User Manual (PDF)")
        print("0. Exit")
        
        opt = input("Select: ").strip()
        
        if opt == "0": break
        elif opt == "1":
            t = float(input("Thresh %: "))/100
            mulliken.generate_mulliken_inputs(exciton, t)
        elif opt == "2":
            exciton.v_start = int(input("Abs Valence Start: "))
            exciton.c_start = int(input("Abs Conduction Start: "))
            pat = input("Pattern: ")
            off = int(input("Offset: "))
            mulliken.analyze_mulliken_output(exciton, pat, off, "mulliken_snippet.in", exciton.v_start, exciton.c_start)
        elif opt == "3":
            exciton.v_start = int(input("Abs Valence Start: "))
            exciton.c_start = int(input("Abs Conduction Start: "))
            t = float(input("Thresh %: "))/100
            volumetric.generate_cube_inputs(exciton, t)
        elif opt == "4":
            exciton.v_start = int(input("Abs Valence Start: "))
            exciton.c_start = int(input("Abs Conduction Start: "))
            pat = input("Pattern: ")
            volumetric.sum_average_density(exciton, pat)
        elif opt == "5":
            reciprocal.analyze_bz_and_bands(exciton)
        elif opt == "6":
            t = float(input("Thresh %: "))/100
            conditional.generate_cond_inputs(exciton, t)
        elif opt == "7":
            # The current CLI uses the origin as a default fixed-hole coordinate.
            conditional.analyze_conditional(exciton, np.array([0,0,0]), "pat_r", "pat_i")
        elif opt == "8":
            open_manual()


if __name__ == "__main__":
    main()
