"""
Export airfoil coordinates to various file formats.

Supported formats:
  - .dat (XFOIL/Selig format)
  - .csv (general purpose)
  - .json (web applications)

These files can be imported into:
  - XFOIL for analysis
  - AutoCAD / SolidWorks / Fusion 360 for 3D modeling
  - Any CAD software that accepts point data
"""

import os
import json
import numpy as np
from datetime import datetime


def export_dat(x_upper, y_upper, x_lower, y_lower, filepath, name="airfoil"):
    """
    Export airfoil in Selig/XFOIL .dat format.
    
    Format:
      Line 1: airfoil name
      Line 2+: x  y  (upper surface from TE to LE)
      Then: x  y  (lower surface from LE to TE)
    
    Parameters
    ----------
    x_upper : array-like
        Upper surface x-coordinates (LE to TE order)
    y_upper : array-like
        Upper surface y-coordinates
    x_lower : array-like
        Lower surface x-coordinates (LE to TE order)
    y_lower : array-like
        Lower surface y-coordinates
    filepath : str
        Output file path (e.g., "output/my_airfoil.dat")
    name : str
        Airfoil name (written in header)
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write(f"{name}\n")
        
        # Upper surface: from TE to LE (reverse order)
        for i in range(len(x_upper) - 1, -1, -1):
            f.write(f"  {x_upper[i]:.6f}  {y_upper[i]:.6f}\n")
        
        # Lower surface: from LE to TE (skip LE point to avoid duplicate)
        for i in range(1, len(x_lower)):
            f.write(f"  {x_lower[i]:.6f}  {y_lower[i]:.6f}\n")
    
    print(f"[EXPORT] Saved .dat file: {filepath}")


def export_csv(x_upper, y_upper, x_lower, y_lower, filepath, 
               name="airfoil", metadata=None):
    """
    Export airfoil coordinates to CSV format.
    
    Parameters
    ----------
    x_upper, y_upper : array-like
        Upper surface coordinates
    x_lower, y_lower : array-like
        Lower surface coordinates
    filepath : str
        Output file path
    name : str
        Airfoil name
    metadata : dict, optional
        Additional info (Cl, Cd, Re, etc.) to include in header
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'w') as f:
        # Header
        f.write(f"# Airfoil: {name}\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Generator: AirfoilGen v1.0\n")
        
        if metadata:
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
        
        f.write(f"# Upper surface: {len(x_upper)} points\n")
        f.write(f"# Lower surface: {len(x_lower)} points\n")
        f.write("#\n")
        
        # Upper surface
        f.write("# UPPER SURFACE\n")
        f.write("surface,x,y\n")
        for i in range(len(x_upper)):
            f.write(f"upper,{x_upper[i]:.6f},{y_upper[i]:.6f}\n")
        
        # Lower surface
        f.write("# LOWER SURFACE\n")
        for i in range(len(x_lower)):
            f.write(f"lower,{x_lower[i]:.6f},{y_lower[i]:.6f}\n")
    
    print(f"[EXPORT] Saved .csv file: {filepath}")


def export_json(x_upper, y_upper, x_lower, y_lower, filepath,
                name="airfoil", metadata=None):
    """
    Export airfoil coordinates to JSON format.
    
    Parameters
    ----------
    x_upper, y_upper : array-like
        Upper surface coordinates
    x_lower, y_lower : array-like
        Lower surface coordinates
    filepath : str
        Output file path
    name : str
        Airfoil name
    metadata : dict, optional
        Additional info to include
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    data = {
        "name": name,
        "generated": datetime.now().isoformat(),
        "generator": "AirfoilGen v1.0",
        "n_points_per_surface": len(x_upper),
        "upper_surface": {
            "x": [round(float(v), 6) for v in x_upper],
            "y": [round(float(v), 6) for v in y_upper],
        },
        "lower_surface": {
            "x": [round(float(v), 6) for v in x_lower],
            "y": [round(float(v), 6) for v in y_lower],
        },
    }
    
    if metadata:
        data["metadata"] = metadata
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"[EXPORT] Saved .json file: {filepath}")


def export_all_formats(x_upper, y_upper, x_lower, y_lower,
                       output_dir, name="airfoil", metadata=None):
    """
    Export airfoil in ALL supported formats.
    
    Parameters
    ----------
    x_upper, y_upper, x_lower, y_lower : array-like
        Surface coordinates
    output_dir : str
        Output directory
    name : str
        Airfoil name (used for filenames)
    metadata : dict, optional
        Additional info
    """
    safe_name = name.replace(" ", "_").replace("/", "_").lower()
    
    export_dat(x_upper, y_upper, x_lower, y_lower,
               os.path.join(output_dir, f"{safe_name}.dat"), name)
    
    export_csv(x_upper, y_upper, x_lower, y_lower,
               os.path.join(output_dir, f"{safe_name}.csv"), name, metadata)
    
    export_json(x_upper, y_upper, x_lower, y_lower,
                os.path.join(output_dir, f"{safe_name}.json"), name, metadata)