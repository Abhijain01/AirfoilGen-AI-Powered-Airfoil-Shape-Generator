import os
import sys

# Ensure XFOIL is in PATH for this session before import
xfoil_path = r"C:\Users\abhis\XFOIL6.99"
if xfoil_path not in os.environ['PATH']:
    os.environ['PATH'] += ";" + xfoil_path

try:
    from xfoil_wrapper import XFoil
    print("SUCCESS: 'xfoil-wrapper' imported successfully (found XFoil class).")
except ImportError as e:
    print(f"FAILURE: Could not import 'xfoil-wrapper'. Error: {e}")
    sys.exit(1)

# Basic check if executable is found (wrapper usually checks on init or run)
# We can just try to run a dummy command or check if shutil.which finds it
import shutil
if shutil.which("xfoil"):
    print(f"SUCCESS: 'xfoil' executable found at: {shutil.which('xfoil')}")
else:
    print("FAILURE: 'xfoil' executable NOT found in PATH.")
    sys.exit(1)
