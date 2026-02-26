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
    
    # Simple script to just boot XFOIL and quit
    script = "QUIT\n"
    
    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(input=script, timeout=10)
        
        print("\n--- STDOUT ---")
        print(stdout)
        print("--- STDERR ---")
        print(stderr)
        print(f"Exit Code: {process.returncode}")
        
    except Exception as e:
        print(f"\nEXCEPTION CAUGHT: {e}")

if __name__ == "__main__":
    run_debug()
