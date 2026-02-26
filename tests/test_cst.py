"""
Tests for CST parameterization.

Run with: python -m pytest tests/test_cst.py -v
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometry.cst import (
    cst_shape,
    cst_to_coordinates,
    coordinates_to_cst,
    compute_airfoil_properties,
    generate_random_cst,
    generate_x_cosine,
    validate_airfoil
)
from src.geometry.naca import naca4digit, naca4digit_from_string


def test_cosine_spacing():
    """Test that cosine spacing is correct"""
    x = generate_x_cosine(100)
    assert len(x) == 100
    assert abs(x[0] - 0.0) < 1e-10
    assert abs(x[-1] - 1.0) < 1e-10
    # More points near LE and TE
    assert x[1] < 0.001  # First point very close to LE
    print("  ✓ Cosine spacing test passed")


def test_cst_shape():
    """Test basic CST shape function"""
    x = generate_x_cosine(100)
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    y = cst_shape(weights, x)
    
    assert len(y) == 100
    assert abs(y[0]) < 1e-6       # Zero at LE
    assert abs(y[-1]) < 1e-6      # Zero at TE
    assert np.max(y) > 0           # Positive somewhere
    assert not np.any(np.isnan(y)) # No NaN
    print("  ✓ CST shape test passed")


def test_cst_to_coordinates():
    """Test coordinate generation from CST parameters"""
    cst_upper = np.array([0.17, 0.20, 0.16, 0.13, 0.15, 0.12, 0.10, 0.08])
    cst_lower = np.array([0.15, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02])
    
    x_all, y_all, x_u, y_u, x_l, y_l = cst_to_coordinates(
        cst_upper, cst_lower, n_points=100
    )
    
    assert len(x_all) == 200
    assert len(y_all) == 200
    assert len(x_u) == 100
    assert len(y_u) == 100
    assert len(x_l) == 100
    assert len(y_l) == 100
    
    # Upper surface should be above lower surface
    assert np.all(y_u >= y_l)
    
    # Thickness should be positive
    assert np.all((y_u - y_l) >= -1e-6)
    
    print("  ✓ CST to coordinates test passed")


def test_roundtrip_cst():
    """Test that coordinates → CST → coordinates gives same shape"""
    # Generate a NACA airfoil
    x, y_upper, y_lower = naca4digit(0.02, 0.4, 0.12, n_points=100)
    
    # Fit CST
    cst_upper, cst_lower, err_u, err_l = coordinates_to_cst(
        x, y_upper, x, y_lower, n_weights=8
    )
    
    # Reconstruct
    _, _, _, y_u_recon, _, y_l_recon = cst_to_coordinates(
        cst_upper, cst_lower, n_points=100
    )
    
    # Compare
    rms_upper = np.sqrt(np.mean((y_upper - y_u_recon) ** 2))
    rms_lower = np.sqrt(np.mean((y_lower - y_l_recon) ** 2))
    
    assert rms_upper < 0.005, f"Upper surface RMS too high: {rms_upper}"
    assert rms_lower < 0.005, f"Lower surface RMS too high: {rms_lower}"
    
    print(f"  ✓ Roundtrip test passed (RMS: upper={rms_upper:.6f}, lower={rms_lower:.6f})")


def test_naca_generation():
    """Test NACA airfoil generation"""
    x, yu, yl, name = naca4digit_from_string("2412", 100)
    
    assert name == "NACA 2412"
    assert len(x) == 100
    assert np.all(yu >= yl)
    assert np.max(yu - yl) > 0.10  # 12% thickness
    assert np.max(yu - yl) < 0.14
    
    print("  ✓ NACA generation test passed")


def test_random_cst():
    """Test random CST airfoil generation"""
    cst_params, valid = generate_random_cst(n_airfoils=50, seed=42)
    
    assert len(cst_params) > 0, "No valid airfoils generated"
    assert cst_params.shape[1] == 16
    assert np.all(valid)
    
    print(f"  ✓ Random CST test passed ({len(cst_params)} airfoils)")


def test_airfoil_properties():
    """Test property computation"""
    x, yu, yl = naca4digit(0.02, 0.4, 0.12, 100)
    props = compute_airfoil_properties(x, yu, yl)
    
    assert abs(props['max_thickness'] - 0.12) < 0.02
    assert 0.2 < props['max_thickness_loc'] < 0.4
    assert props['max_camber'] > 0
    
    print(f"  ✓ Properties test passed (t/c={props['max_thickness']:.4f})")


def test_validate_airfoil():
    """Test airfoil validation"""
    # Valid airfoil
    x, yu, yl = naca4digit(0.02, 0.4, 0.12, 100)
    valid, issues = validate_airfoil(x, yu, yl)
    assert valid, f"Valid airfoil flagged as invalid: {issues}"
    
    # Invalid airfoil (upper below lower)
    valid2, issues2 = validate_airfoil(x, yl, yu)  # swapped!
    assert not valid2, "Invalid airfoil not detected"
    
    print("  ✓ Validation test passed")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  RUNNING CST TESTS")
    print("=" * 60 + "\n")
    
    test_cosine_spacing()
    test_cst_shape()
    test_cst_to_coordinates()
    test_roundtrip_cst()
    test_naca_generation()
    test_random_cst()
    test_airfoil_properties()
    test_validate_airfoil()
    
    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)