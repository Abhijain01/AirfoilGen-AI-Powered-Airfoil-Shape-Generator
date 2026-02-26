
import os
import sys
import torch
import numpy as np

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.inference import AirfoilGenerator

def debug():
    print("Initializing Generator...")
    try:
        gen = AirfoilGenerator(checkpoint_dir='checkpoints', device='cpu')
    except Exception as e:
        print(f"Failed to load generator: {e}")
        return
    
    print("\nRunning generation with user parameters (approximate):")
    # User values based on interaction
    Cl = 1.0
    Re = 500000
    alpha = 5.0
    
    # Constraints
    min_t = 0.06
    max_t = 0.20
    max_Cd = 0.035
    
    print(f"Target: Cl={Cl}, Re={Re}, alpha={alpha}")
    print(f"Constraints: Thickness={min_t}-{max_t}, Max Cd={max_Cd}")
    
    designs = gen.generate(
        Cl=Cl, Re=Re, alpha=alpha,
        min_thickness=min_t, max_thickness=max_t,
        max_Cd=max_Cd,
        n_designs=5, n_candidates=200
    )
    
    if not designs:
        print("\nFAILURE: No designs returned.")
    else:
        print(f"\nSUCCESS: {len(designs)} designs returned.")
        for i, d in enumerate(designs):
            print(f"Design {i+1}: Cl={d.predicted_cl:.3f}, Cd={d.predicted_cd:.4f}, t/c={d.thickness:.3f}")
            if hasattr(d, 'failed_constraints') and d.failed_constraints:
                print(f"  [WARNING] Failed constraints: {d.failed_constraints}")
            else:
                 print("  [OK] Meets all constraints")

if __name__ == "__main__":
    debug()
