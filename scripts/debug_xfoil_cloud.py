"""
Debug script to manually trigger XFOIL on Streamlit Cloud and print output.
Added to bypass the Streamlit UI and get raw execution logs.
"""
import os
import sys
import subprocess
import shutil

def run_debug():
    print("=== XFOIL DEBUG TOOL ===")
    
    # Check XFOIL
    xfoil_path = shutil.which("xfoil")
    if not xfoil_path:
        print("ERROR: xfoil not in PATH!")
        return
        
    print(f"Found xfoil at: {xfoil_path}")
    
    cmd = []
    if shutil.which("xvfb-run"):
        print("Found xvfb-run, wrapping command.")
        cmd = ["xvfb-run", "-a", xfoil_path]
    else:
        print("xvfb-run not found, running xfoil directly.")
        cmd = [xfoil_path]
        
    print(f"Command: {cmd}")
    
    # 1. Create a dummy airfoil .dat file
    dat_path = os.path.join(os.getcwd(), "debug_airfoil.dat")
    with open(dat_path, "w") as f:
        f.write("DEBUG_AIRFOIL\n")
        f.write("  1.000000  0.001000\n")
        f.write("  0.500000  0.050000\n")
        f.write("  0.000000  0.000000\n")
        f.write("  0.500000 -0.050000\n")
        f.write("  1.000000 -0.001000\n")
        
    print(f"Created dummy airfoil at: {dat_path}")
    
    # 2. Build full execution script
    script = f"LOAD {dat_path}\n"
    script += "PANE\n"
    script += "OPER\n"
    script += "VISC 500000\n"
    script += "ITER 100\n"
    script += "PACC\n"
    script += "debug_polar.txt\n\n"
    script += "ALFA 5.0\n"
    script += "PACC\n"
    script += "QUIT\n"
    
    print("Running script:")
    print(script)
    
    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(input=script, timeout=15)
        
        print("\n--- STDOUT ---")
        print(stdout)
        print("--- STDERR ---")
        print(stderr)
        print(f"Exit Code: {process.returncode}")
        
    except Exception as e:
        print(f"\nEXCEPTION CAUGHT: {e}")
    finally:
        try:
            os.remove(dat_path)
            os.remove(os.path.join(os.getcwd(), "debug_polar.txt"))
        except:
            pass

if __name__ == "__main__":
    run_debug()
