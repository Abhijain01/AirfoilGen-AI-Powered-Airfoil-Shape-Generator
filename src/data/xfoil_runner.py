
"""
XFOIL automation — run aerodynamic analysis on airfoils.

This version uses the `xfoil.exe` command directly via subprocess.
No Python wrappers required.
"""

import os
import subprocess
import numpy as np
import pandas as pd
import warnings
import shutil
import time
from tqdm import tqdm

# Constants
# Constants
xfoil_path = shutil.which("xfoil")
if xfoil_path is None:
    # Try generic path if not in PATH (fallback for Windows)
    possible_paths = [
        r"C:\Users\abhis\XFOIL6.99\xfoil.exe",
        "xfoil.exe"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            xfoil_path = p
            break

XFOIL_CMD = xfoil_path if xfoil_path else "xfoil"
TEMP_DIR = "xfoil_temp"

if xfoil_path:
    print(f"XFOIL executable found at: {xfoil_path}")
else:
    warnings.warn("XFOIL executable not found in PATH. Analysis may fail.")

def _write_airfoil_file(x, y, filename):
    """Write airfoil coordinates to XFOIL format"""
    with open(filename, 'w') as f:
        f.write(f"{os.path.basename(filename)}\n")
        # XFOIL format: output coords from TE (upper) -> LE -> TE (lower)
        # Note: input x, y are usually separated. 
        # But here checking how they come from cst_to_coords:
        # x_upper (0->1), y_upper, x_lower (0->1), y_lower
        # XFOIL expects top surface TE->LE then bottom surface LE->TE
        
        # Check input format from caller. 
        # Typically called with (x_upper, y_upper, x_lower, y_lower)
        # x_upper is 0 (LE) to 1 (TE).
        pass

def analyze_airfoil(x_upper, y_upper, x_lower, y_lower,
                    reynolds_numbers, alpha_range,
                    n_crit=9, max_iter=100, timeout=10):
    """
    Run XFOIL analysis on a single airfoil at multiple conditions.
    Optimized to run all Re numbers in one XFOIL session.
    """
    # Setup temp directory
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Unique ID for this run to avoid collisions
    run_id = f"{time.time()}_{np.random.randint(0, 1000)}"
    airfoil_file = os.path.join(TEMP_DIR, f"airfoil_{run_id}.dat")
    
    # 1. Write Airfoil File
    with open(airfoil_file, 'w') as f:
        f.write("Generated Airfoil\n")
        # Upper: TE to LE
        for i in range(len(x_upper)-1, -1, -1):
            f.write(f"  {x_upper[i]:.6f}  {y_upper[i]:.6f}\n")
        # Lower: LE to TE (skip LE if duplicate)
        for i in range(1, len(x_lower)):
            f.write(f"  {x_lower[i]:.6f}  {y_lower[i]:.6f}\n")

    # 2. Construct Input Script for ALL Re numbers
    input_script = []
    
    # Disable graphics
    input_script.append("PLOP")
    input_script.append("G")
    input_script.append("")
    
    # Load airfoil
    input_script.append(f"LOAD {airfoil_file}")
    input_script.append("") # Confirm name
    
    # Clean up geometry using XFOIL's built-in commands
    # This specifically prevents SIGFPE floating-point crashes on Linux
    input_script.append("MDES")    # Enter design menu
    input_script.append("FILT")    # Smooth geometry
    input_script.append("EXEC")    # Execute changes
    input_script.append("")        # Exit MDES
    
    # Panel operations
    input_script.append("PANE")
    
    # Operational mode
    input_script.append("OPER")
    input_script.append(f"ITER {max_iter}")
    
    # Track polar files
    polar_files = {}

    for Re in reynolds_numbers:
        polar_file = os.path.join(TEMP_DIR, f"polar_{run_id}_Re{int(Re)}.txt")
        polar_files[Re] = polar_file
        
        # Remove if exists
        if os.path.exists(polar_file):
            os.remove(polar_file)
            
        input_script.append(f"Visc {Re}")
        
        # Start accumulation
        input_script.append("PACC")
        input_script.append(f"{polar_file}")
        input_script.append("") # No dump file
        
        # Alpha sequence
        alpha_min = alpha_range[0]
        alpha_max = alpha_range[-1]
        step = alpha_range[1] - alpha_range[0] if len(alpha_range) > 1 else 1.0
        
        input_script.append(f"ASEQ {alpha_min} {alpha_max} {step}")
        
        # Stop accumulation
        input_script.append("PACC")
    
    # Quit
    input_script.append("QUIT") # Quit OPER
    input_script.append("QUIT") # Quit XFOIL
    
    # 3. Run XFOIL (Single Process)
    try:
        startupinfo = None
        creationflags = 0
        cmd = [XFOIL_CMD]
        
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            creationflags = subprocess.CREATE_NO_WINDOW
        else:
            # On Linux (like Streamlit Cloud), XFOIL often hangs without an X display.
            # Use xvfb-run to provide a virtual framebuffer if available.
            if shutil.which("xvfb-run"):
                cmd = ["xvfb-run", "-a", XFOIL_CMD]

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.getcwd(),
            startupinfo=startupinfo,
            creationflags=creationflags
        )
        
        # Send input
        # Increase timeout based on number of Re runs
        total_timeout = timeout * len(alpha_range) * len(reynolds_numbers) * 0.5
        
        stdout_data, stderr_data = process.communicate(
            input="\n".join(input_script),
            timeout=max(total_timeout, 10)
        )
        
        if process.returncode != 0:
            print(f"\n[XFOIL ERROR] Exit code {process.returncode}")
            print(f"STDOUT: {stdout_data}")
            print(f"STDERR: {stderr_data}")
        
    except subprocess.TimeoutExpired:
        process.kill()
        pass
    except Exception as e:
        warnings.warn(f"XFOIL execution error: {e}")
        pass
    
    # 4. Parse Outputs
    results = []
    n_converged = 0
    n_total = len(reynolds_numbers) * len(alpha_range)
    
    for Re, polar_file in polar_files.items():
        if os.path.exists(polar_file):
            try:
                with open(polar_file, 'r') as f:
                    lines = f.readlines()
                
                data_start = -1
                for i, line in enumerate(lines):
                    if "-------" in line:
                        data_start = i + 1
                        break
                
                if data_start > 0 and data_start < len(lines):
                    for line in lines[data_start:]:
                        parts = line.split()
                        if len(parts) >= 7:
                            results.append({
                                'alpha': float(parts[0]),
                                'Re': float(Re),
                                'Cl': float(parts[1]),
                                'Cd': float(parts[2]),
                                'Cm': float(parts[4]),
                                'converged': True
                            })
                            n_converged += 1
            except Exception:
                pass
            
            # Cleanup polar
            try:
                os.remove(polar_file)
            except:
                pass
                
    # Cleanup airfoil
    try:
        os.remove(airfoil_file)
    except:
        pass
        
    return results, n_converged, n_total


def batch_analyze(airfoils, reynolds_numbers, alpha_range,
                  n_crit=9, max_iter=100):
    """
    Run XFOIL analysis on MANY airfoils.
    """
    all_results = []
    total_converged = 0
    total_attempted = 0
    failed_airfoils = []
    
    # Ensure temp dir exists
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    print(f"\n[XFOIL] Analyzing {len(airfoils)} airfoils using XFOIL executable")
    
    for airfoil in tqdm(airfoils, desc="Analyzing airfoils"):
        name = airfoil['name']
        
        try:
            results, n_conv, n_total = analyze_airfoil(
                airfoil['x'], airfoil['y_upper'],
                airfoil['x'], airfoil['y_lower'],
                reynolds_numbers, alpha_range,
                n_crit=n_crit, max_iter=max_iter
            )
            
            # Add airfoil name
            for r in results:
                r['airfoil_name'] = name
            
            all_results.extend(results)
            total_converged += n_conv
            total_attempted += n_total
            
            if n_conv == 0:
                failed_airfoils.append(name)
                
        except Exception as e:
            failed_airfoils.append(name)
            warnings.warn(f"Failed on {name}: {e}")
    
    # Cleanup temp dir
    try:
        shutil.rmtree(TEMP_DIR)
    except:
        pass
        
    convergence_rate = total_converged / max(total_attempted, 1) * 100
    
    summary = {
        'total_airfoils': len(airfoils),
        'total_attempted': total_attempted,
        'total_converged': total_converged,
        'convergence_rate': convergence_rate,
        'failed_airfoils': failed_airfoils,
        'n_failed': len(failed_airfoils),
    }
    
    print(f"\n[XFOIL] Results Summary:")
    print(f"  Converged: {total_converged}/{total_attempted} "
          f"({convergence_rate:.1f}%)")
    
    return all_results, summary