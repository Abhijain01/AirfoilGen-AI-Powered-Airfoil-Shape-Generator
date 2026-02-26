"""
CST (Class Shape Transformation) Parameterization

This is the CORE of our project.
Every airfoil shape is represented as 16 numbers (CST weights).
The model will predict these 16 numbers to generate new airfoils.

CST guarantees:
  - Smooth airfoil shape (always)
  - Closed shape (always)
  - No self-intersection (almost always)
  - Valid leading edge (always)
  - Any 16 numbers = valid airfoil

Reference: Kulfan, B. "Universal Parametric Geometry Representation Method"
           AIAA Journal, 2008
"""

import numpy as np
from math import comb
from scipy.optimize import least_squares


def cst_shape(weights, x, n1=0.5, n2=1.0):
    """
    Compute CST surface coordinates.
    
    The CST method represents an airfoil surface as:
        y(x) = C(x) * S(x) + x * y_te
    
    Where:
        C(x) = x^n1 * (1-x)^n2      (class function — defines general shape)
        S(x) = sum of weighted Bernstein polynomials (shape function — defines details)
    
    Parameters
    ----------
    weights : array-like
        CST weights (typically 8 values per surface)
    x : array-like
        x-coordinates, must be in range [0, 1]
    n1 : float
        Class function exponent 1 (0.5 for airfoil)
    n2 : float
        Class function exponent 2 (1.0 for airfoil)
    
    Returns
    -------
    y : numpy array
        y-coordinates of the surface
    """
    weights = np.asarray(weights, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    
    # Class function: defines general airfoil-like shape
    # C(x) = x^0.5 * (1-x)^1.0
    # This gives: zero at LE and TE, rounded LE, pointed TE
    C = x ** n1 * (1.0 - x) ** n2
    
    # Shape function: Bernstein polynomial basis
    n = len(weights) - 1  # polynomial order
    S = np.zeros_like(x)
    
    for i, w in enumerate(weights):
        # Bernstein basis polynomial
        K = comb(n, i)
        B = K * (x ** i) * ((1.0 - x) ** (n - i))
        S += w * B
    
    # Surface coordinate
    y = C * S
    
    return y


def generate_x_cosine(n_points=100):
    """
    Generate x-coordinates with cosine spacing.
    
    Cosine spacing puts more points near leading edge and trailing edge
    where the curvature is highest. This is standard practice.
    
    Parameters
    ----------
    n_points : int
        Number of points (default 100 per surface)
    
    Returns
    -------
    x : numpy array
        x-coordinates from 0 to 1 with cosine spacing
    """
    theta = np.linspace(0, np.pi, n_points)
    x = 0.5 * (1.0 - np.cos(theta))
    return x


def cst_to_coordinates(cst_upper, cst_lower, n_points=100):
    """
    Convert CST parameters to airfoil (x,y) coordinates.
    
    THIS IS THE KEY FUNCTION:
    16 numbers → 200 (x,y) coordinates
    
    Parameters
    ----------
    cst_upper : array-like, shape (8,)
        CST weights for upper surface
    cst_lower : array-like, shape (8,)
        CST weights for lower surface
    n_points : int
        Points per surface (default 100, total 200)
    
    Returns
    -------
    x_all : numpy array, shape (2 * n_points,)
        x-coordinates (upper surface LE→TE, then lower surface LE→TE)
    y_all : numpy array, shape (2 * n_points,)
        y-coordinates
    x_upper : numpy array, shape (n_points,)
        Upper surface x
    y_upper : numpy array, shape (n_points,)
        Upper surface y
    x_lower : numpy array, shape (n_points,)
        Lower surface x
    y_lower : numpy array, shape (n_points,)
        Lower surface y
    """
    cst_upper = np.asarray(cst_upper, dtype=np.float64)
    cst_lower = np.asarray(cst_lower, dtype=np.float64)
    
    # Generate x-coordinates with cosine spacing
    x = generate_x_cosine(n_points)
    
    # Compute upper and lower surfaces
    y_upper = cst_shape(cst_upper, x)
    y_lower = -cst_shape(cst_lower, x)  # negative because lower surface
    
    # Introduce extremely small blunt trailing edge (TE)
    # The Linux version of XFOIL crashes with SIGFPE if TE thickness is exactly 0.0
    TE_THICKNESS = 0.001
    y_upper += x * (TE_THICKNESS / 2.0)
    y_lower -= x * (TE_THICKNESS / 2.0)
    
    # Combine: upper (LE→TE) then lower (LE→TE)
    x_all = np.concatenate([x, x])
    y_all = np.concatenate([y_upper, y_lower])
    
    return x_all, y_all, x, y_upper, x, y_lower


def coordinates_to_cst(x_upper, y_upper, x_lower, y_lower, n_weights=8):
    """
    Fit CST parameters to existing airfoil coordinates.
    
    THIS IS THE INVERSE:
    (x,y) coordinates → 16 CST numbers
    
    Uses least-squares optimization to find best CST weights.
    
    Parameters
    ----------
    x_upper : array-like
        Upper surface x-coordinates (0 to 1)
    y_upper : array-like
        Upper surface y-coordinates
    x_lower : array-like
        Lower surface x-coordinates (0 to 1)
    y_lower : array-like
        Lower surface y-coordinates (should be negative or near zero)
    n_weights : int
        Number of CST weights per surface (default 8)
    
    Returns
    -------
    cst_upper : numpy array, shape (n_weights,)
        CST weights for upper surface
    cst_lower : numpy array, shape (n_weights,)
        CST weights for lower surface
    fit_error_upper : float
        RMS fitting error for upper surface
    fit_error_lower : float
        RMS fitting error for lower surface
    """
    x_upper = np.asarray(x_upper, dtype=np.float64)
    y_upper = np.asarray(y_upper, dtype=np.float64)
    x_lower = np.asarray(x_lower, dtype=np.float64)
    y_lower = np.asarray(y_lower, dtype=np.float64)
    
    def fit_surface(x, y_target, is_lower=False):
        """Fit CST weights to one surface"""
        
        if is_lower:
            y_target = -y_target  # CST uses positive values
        
        def residuals(weights):
            y_cst = cst_shape(weights, x)
            return y_cst - y_target
        
        # Initial guess
        w0 = np.ones(n_weights) * 0.15
        
        # Bounds: weights typically in range [-0.5, 0.5]
        lower_bounds = -0.5 * np.ones(n_weights)
        upper_bounds = 0.5 * np.ones(n_weights)
        
        # Fit using least squares
        result = least_squares(
            residuals, w0,
            bounds=(lower_bounds, upper_bounds),
            method='trf',
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=10000
        )
        
        # Compute fit error
        y_fitted = cst_shape(result.x, x)
        if is_lower:
            rms_error = np.sqrt(np.mean((-y_fitted - y_lower) ** 2))
        else:
            rms_error = np.sqrt(np.mean((y_fitted - y_target) ** 2))
        
        return result.x, rms_error
    
    # Fit upper surface
    cst_upper, error_upper = fit_surface(x_upper, y_upper, is_lower=False)
    
    # Fit lower surface
    cst_lower, error_lower = fit_surface(x_lower, y_lower, is_lower=True)
    
    return cst_upper, cst_lower, error_upper, error_lower


def compute_airfoil_properties(x, y_upper, y_lower):
    """
    Compute geometric properties of an airfoil.
    
    Parameters
    ----------
    x : array-like
        x-coordinates (same for upper and lower)
    y_upper : array-like
        Upper surface y-coordinates
    y_lower : array-like
        Lower surface y-coordinates
    
    Returns
    -------
    properties : dict
        Dictionary with geometric properties
    """
    thickness = y_upper - y_lower
    camber = (y_upper + y_lower) / 2.0
    
    # Find max thickness and location
    idx_max_t = np.argmax(thickness)
    max_thickness = thickness[idx_max_t]
    max_thickness_loc = x[idx_max_t]
    
    # Find max camber and location
    idx_max_c = np.argmax(np.abs(camber))
    max_camber = camber[idx_max_c]
    max_camber_loc = x[idx_max_c]
    
    # Leading edge radius (approximation)
    if len(x) > 1 and x[1] > 0:
        le_radius = (thickness[1] ** 2) / (4.0 * x[1])
    else:
        le_radius = 0.0
    
    # Trailing edge thickness
    te_thickness = thickness[-1]
    
    # Trailing edge angle
    if len(x) > 2:
        dx = x[-1] - x[-2]
        dy_upper = y_upper[-1] - y_upper[-2]
        dy_lower = y_lower[-1] - y_lower[-2]
        te_angle = np.degrees(np.arctan2(dy_upper, dx) - np.arctan2(dy_lower, dx))
    else:
        te_angle = 0.0
    
    # Areas
    upper_area = np.trapz(y_upper, x)
    lower_area = np.trapz(np.abs(y_lower), x)
    
    properties = {
        'max_thickness': float(max_thickness),
        'max_thickness_loc': float(max_thickness_loc),
        'max_camber': float(max_camber),
        'max_camber_loc': float(max_camber_loc),
        'le_radius': float(le_radius),
        'te_thickness': float(te_thickness),
        'te_angle': float(te_angle),
        'upper_area': float(upper_area),
        'lower_area': float(lower_area),
    }
    
    return properties


def generate_random_cst(n_airfoils=100, n_weights=8, seed=42):
    """
    Generate random valid airfoil shapes using CST parameters.
    
    This creates training data beyond what UIUC database provides.
    
    Parameters
    ----------
    n_airfoils : int
        Number of random airfoils to generate
    n_weights : int
        Number of CST weights per surface
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    cst_params : numpy array, shape (n_airfoils, 2 * n_weights)
        CST parameters [upper_weights, lower_weights] for each airfoil
    valid_mask : numpy array, shape (n_airfoils,), dtype=bool
        Which airfoils passed validity checks
    """
    rng = np.random.RandomState(seed)
    
    cst_params = np.zeros((n_airfoils, 2 * n_weights))
    valid_mask = np.zeros(n_airfoils, dtype=bool)
    
    x = generate_x_cosine(100)
    
    generated = 0
    attempts = 0
    max_attempts = n_airfoils * 10
    
    while generated < n_airfoils and attempts < max_attempts:
        attempts += 1
        
        # Random CST weights
        # Upper surface: mostly positive (creates upper surface above x-axis)
        upper_weights = rng.uniform(0.05, 0.35, n_weights)
        # Add some variation
        upper_weights += rng.normal(0, 0.05, n_weights)
        upper_weights = np.clip(upper_weights, 0.01, 0.50)
        
        # Lower surface: mostly positive (CST convention, shape is flipped)
        lower_weights = rng.uniform(0.02, 0.25, n_weights)
        lower_weights += rng.normal(0, 0.05, n_weights)
        lower_weights = np.clip(lower_weights, 0.00, 0.40)
        
        # Compute shape
        y_upper = cst_shape(upper_weights, x)
        y_lower = -cst_shape(lower_weights, x)
        
        # Validity checks
        thickness = y_upper - y_lower
        
        # Check 1: Minimum thickness in central region (exclude LE/TE)
        if np.min(thickness[20:-20]) < 0.005:
            continue
        
        # Check 2: Maximum thickness between 4% and 30%
        max_t = np.max(thickness)
        if max_t < 0.04 or max_t > 0.30:
            continue
        
        # Check 3: Upper surface above lower surface everywhere
        if np.any(y_upper < y_lower):
            continue
        
        # Check 4: No extreme camber
        camber = (y_upper + y_lower) / 2.0
        if np.max(np.abs(camber)) > 0.10:
            continue
        
        # Check 5: Leading edge not too blunt or sharp
        if y_upper[1] < 0.001 or y_upper[1] > 0.06:
            continue
        
        # Valid airfoil
        cst_params[generated, :n_weights] = upper_weights
        cst_params[generated, n_weights:] = lower_weights
        valid_mask[generated] = True
        generated += 1
    
    print(f"[CST] Generated {generated}/{n_airfoils} valid random airfoils "
          f"({attempts} attempts)")
    
    return cst_params[:generated], valid_mask[:generated]


def validate_airfoil(x, y_upper, y_lower):
    """
    Check if an airfoil shape is physically valid.
    
    Parameters
    ----------
    x : array-like
        x-coordinates
    y_upper : array-like
        Upper surface y-coordinates
    y_lower : array-like
        Lower surface y-coordinates
    
    Returns
    -------
    is_valid : bool
        Whether the airfoil is valid
    issues : list of str
        List of issues found (empty if valid)
    """
    issues = []
    
    thickness = y_upper - y_lower
    
    # Check 1: Upper above lower
    if np.any(y_upper < y_lower):
        issues.append("Upper surface below lower surface")
    
    # Check 2: Positive thickness
    if np.any(thickness < -0.001):
        issues.append(f"Negative thickness: min = {np.min(thickness):.4f}")
    
    # Check 3: Reasonable thickness range
    max_t = np.max(thickness)
    if max_t < 0.02:
        issues.append(f"Too thin: max thickness = {max_t:.4f}")
    if max_t > 0.40:
        issues.append(f"Too thick: max thickness = {max_t:.4f}")
    
    # Check 4: x-range
    if x[0] < -0.01 or x[-1] > 1.01:
        issues.append(f"x-range invalid: [{x[0]:.4f}, {x[-1]:.4f}]")
    
    # Check 5: Smoothness (no sudden jumps)
    dy_upper = np.diff(y_upper)
    dy_lower = np.diff(y_lower)
    if np.max(np.abs(dy_upper)) > 0.1:
        issues.append("Upper surface not smooth")
    if np.max(np.abs(dy_lower)) > 0.1:
        issues.append("Lower surface not smooth")
    
    is_valid = len(issues) == 0
    return is_valid, issues