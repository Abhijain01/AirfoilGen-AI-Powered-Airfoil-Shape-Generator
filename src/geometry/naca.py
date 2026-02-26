"""
NACA 4-digit airfoil generator.

Generates standard NACA airfoils (e.g., NACA 0012, NACA 2412, NACA 4415).
These provide well-known, tested shapes for our training data.

NACA MPXX naming:
  M = maximum camber (% of chord)
  P = position of max camber (tenths of chord)
  XX = maximum thickness (% of chord)

Example: NACA 2412
  M = 2% max camber
  P = 40% chord position
  XX = 12% max thickness
"""

import numpy as np
from src.geometry.cst import generate_x_cosine


def naca4digit(m, p, t, n_points=100):
    """
    Generate NACA 4-digit airfoil coordinates.
    
    Parameters
    ----------
    m : float
        Maximum camber as fraction of chord (e.g., 0.02 for 2%)
    p : float
        Position of maximum camber as fraction of chord (e.g., 0.4 for 40%)
    t : float
        Maximum thickness as fraction of chord (e.g., 0.12 for 12%)
    n_points : int
        Number of points per surface
    
    Returns
    -------
    x : numpy array, shape (n_points,)
        x-coordinates (same for upper and lower)
    y_upper : numpy array, shape (n_points,)
        Upper surface y-coordinates
    y_lower : numpy array, shape (n_points,)
        Lower surface y-coordinates
    """
    # Cosine spacing for x-coordinates
    x = generate_x_cosine(n_points)
    
    # Thickness distribution (NACA formula)
    yt = 5.0 * t * (
        0.2969 * np.sqrt(x + 1e-10)
        - 0.1260 * x
        - 0.3516 * x ** 2
        + 0.2843 * x ** 3
        - 0.1015 * x ** 4
    )
    
    # Camber line
    if m == 0 or p == 0:
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
    else:
        yc = np.where(
            x < p,
            (m / p ** 2) * (2.0 * p * x - x ** 2),
            (m / (1.0 - p) ** 2) * ((1.0 - 2.0 * p) + 2.0 * p * x - x ** 2)
        )
        
        dyc_dx = np.where(
            x < p,
            (2.0 * m / p ** 2) * (p - x),
            (2.0 * m / (1.0 - p) ** 2) * (p - x)
        )
    
    # Angle of camber line
    theta = np.arctan(dyc_dx)
    
    # Upper and lower surfaces
    y_upper = yc + yt * np.cos(theta)
    y_lower = yc - yt * np.cos(theta)
    
    return x, y_upper, y_lower


def naca4digit_from_string(naca_string, n_points=100):
    """
    Generate NACA airfoil from string name.
    
    Parameters
    ----------
    naca_string : str
        NACA designation like "0012", "2412", "4415"
    n_points : int
        Points per surface
    
    Returns
    -------
    x, y_upper, y_lower : numpy arrays
    name : str
        Full NACA name
    """
    naca_string = naca_string.strip()
    
    if len(naca_string) != 4:
        raise ValueError(f"Expected 4-digit NACA string, got '{naca_string}'")
    
    m = int(naca_string[0]) / 100.0   # max camber
    p = int(naca_string[1]) / 10.0    # camber position
    t = int(naca_string[2:4]) / 100.0  # thickness
    
    # Handle symmetric airfoils (p=0 when m=0)
    if m == 0:
        p = 0.5  # doesn't matter, no camber
    
    x, y_upper, y_lower = naca4digit(m, p, t, n_points)
    name = f"NACA {naca_string}"
    
    return x, y_upper, y_lower, name


def generate_naca_family(n_points=100):
    """
    Generate a systematic family of NACA 4-digit airfoils.
    
    This creates ~500 different airfoils covering a wide range of:
    - Thickness: 6% to 24%
    - Camber: 0% to 6%
    - Camber position: 20% to 70%
    
    Returns
    -------
    airfoils : list of dict
        Each dict has: name, x, y_upper, y_lower, m, p, t
    """
    airfoils = []
    
    # Symmetric airfoils (no camber)
    for t_pct in range(6, 25, 2):  # 6% to 24%
        t = t_pct / 100.0
        naca_str = f"00{t_pct:02d}"
        x, yu, yl, name = naca4digit_from_string(naca_str, n_points)
        airfoils.append({
            'name': name,
            'naca': naca_str,
            'x': x, 'y_upper': yu, 'y_lower': yl,
            'm': 0.0, 'p': 0.0, 't': t
        })
    
    # Cambered airfoils
    for m_pct in range(1, 7):          # 1% to 6% camber
        for p_pct in range(2, 8):      # 20% to 70% camber position
            for t_pct in range(6, 25, 3):  # 6% to 24% thickness
                m = m_pct / 100.0
                p = p_pct / 10.0
                t = t_pct / 100.0
                
                naca_str = f"{m_pct}{p_pct}{t_pct:02d}"
                
                try:
                    x, yu, yl, name = naca4digit_from_string(naca_str, n_points)
                    
                    # Basic validity check
                    if np.all(yu >= yl) and np.max(yu - yl) > 0.03:
                        airfoils.append({
                            'name': name,
                            'naca': naca_str,
                            'x': x, 'y_upper': yu, 'y_lower': yl,
                            'm': m, 'p': p, 't': t
                        })
                except Exception:
                    continue
    
    print(f"[NACA] Generated {len(airfoils)} NACA 4-digit airfoils")
    return airfoils