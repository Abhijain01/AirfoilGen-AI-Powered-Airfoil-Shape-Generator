"""
╔══════════════════════════════════════════════════════════╗
║  SCRIPT 1/3: CREATE CALIBRATION DATASET                  ║
║                                                          ║
║  Generates diverse airfoil shapes from CVAE              ║
║  + camber variants, labels them with XFOIL.              ║
║                                                          ║
║  Run: python scripts/create_calibration_data.py          ║
║  Time: ~20 minutes                                       ║
║  Output: data/calibration_data.pkl                       ║
╚══════════════════════════════════════════════════════════╝
"""

import sys
import os
import pickle
import numpy as np
import torch
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.generator import CVAE
from src.geometry.cst import cst_to_coordinates, validate_airfoil, compute_airfoil_properties


def generate_base_shapes(cvae, scaler, device, n_shapes=300):
    """Generate diverse base shapes from CVAE using multiple conditions + temps."""
    print(f"\n[1/4] Generating {n_shapes} base shapes from CVAE...")

    all_cst = []

    # Diverse conditions
    cl_values = np.linspace(-0.2, 1.8, 10)
    thickness_values = [0.08, 0.10, 0.12, 0.15, 0.18]
    temps = [0.3, 0.5, 0.8, 1.0, 1.3, 1.8, 2.5]

    per_combo = max(1, n_shapes // (len(cl_values) * len(temps)))

    for cl in cl_values:
        for temp in temps:
            cd_est = 0.005 + 0.04 * 0.12 + 0.0001 * (5.0 ** 2)
            cond_raw = np.array([cl, cd_est, np.log10(500000), 5.0, 0.12],
                                dtype=np.float32)

            if scaler is not None and 'cond_mean' in scaler:
                cond_norm = (cond_raw - scaler['cond_mean']) / scaler['cond_std']
            else:
                cond_norm = cond_raw

            cond_t = torch.from_numpy(cond_norm).unsqueeze(0).to(device)

            with torch.no_grad():
                cond_exp = cond_t.expand(per_combo, -1)
                z = torch.randn(per_combo, cvae.latent_dim, device=device) * temp
                cst_norm = cvae.decode(z, cond_exp)

                if scaler is not None and 'cst_mean' in scaler:
                    cst_mean = torch.tensor(scaler['cst_mean'], device=device,
                                            dtype=torch.float32)
                    cst_std = torch.tensor(scaler['cst_std'], device=device,
                                           dtype=torch.float32)
                    cst = cst_norm * cst_std + cst_mean
                else:
                    cst = cst_norm

                all_cst.append(cst.cpu().numpy())

    all_cst = np.vstack(all_cst)

    # Validate
    valid = []
    for cst in all_cst:
        try:
            upper, lower = cst[:8], cst[8:]
            _, _, x_u, y_u, x_l, y_l = cst_to_coordinates(upper, lower, 100)
            ok, _ = validate_airfoil(x_u, y_u, y_l)
            if ok:
                valid.append(cst)
        except Exception:
            continue

    valid = np.array(valid)
    print(f"  Valid base shapes: {len(valid)}/{len(all_cst)}")
    return valid


def create_camber_variants(base_shapes, n_target=600):
    """Create camber-scaled variants from base shapes."""
    print(f"\n[2/4] Creating camber variants...")

    scales = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4,
              1.6, 1.8, 2.0, 2.3, 2.6, 3.0]

    # Pick subset of diverse bases
    n_bases = min(50, len(base_shapes))
    # Sort by thickness for diversity
    thicknesses = []
    for cst in base_shapes:
        try:
            _, _, x_u, y_u, x_l, y_l = cst_to_coordinates(cst[:8], cst[8:], 100)
            props = compute_airfoil_properties(x_u, y_u, y_l)
            thicknesses.append(props['max_thickness'])
        except Exception:
            thicknesses.append(0.12)

    sorted_idx = np.argsort(thicknesses)
    step = max(1, len(sorted_idx) // n_bases)
    base_idx = sorted_idx[::step][:n_bases]
    selected_bases = base_shapes[base_idx]

    variants = []
    for cst in selected_bases:
        upper, lower = cst[:8], cst[8:]
        thick_part = (upper + lower) / 2.0
        camber_part = (upper - lower) / 2.0

        # If camber too small, add default
        if np.mean(np.abs(camber_part)) < 0.005:
            camber_part = np.array([0.01, 0.02, 0.03, 0.04,
                                    0.04, 0.03, 0.02, 0.01])

        for scale in scales:
            new_upper = np.clip(thick_part + scale * camber_part, 0.01, 0.50)
            new_lower = np.clip(thick_part - scale * camber_part, 0.005, 0.40)
            new_cst = np.concatenate([new_upper, new_lower])

            try:
                _, _, x_u, y_u, x_l, y_l = cst_to_coordinates(
                    new_upper, new_lower, 100)
                ok, _ = validate_airfoil(x_u, y_u, y_l)
                if ok:
                    variants.append(new_cst)
            except Exception:
                continue

    variants = np.array(variants)

    # Deduplicate (remove near-identical shapes)
    if len(variants) > 100:
        unique = [variants[0]]
        for v in variants[1:]:
            diffs = [np.max(np.abs(v - u)) for u in unique[-50:]]
            if min(diffs) > 0.005:
                unique.append(v)
        variants = np.array(unique)

    print(f"  Camber variants: {len(variants)}")
    return variants


def run_xfoil_labeling(cst_array, re_values, alpha_values):
    """Run XFOIL on all shapes × all conditions."""
    print(f"\n[3/4] Running XFOIL labeling...")
    print(f"  Shapes: {len(cst_array)}")
    print(f"  Re: {re_values}")
    print(f"  Alpha: {alpha_values}")

    total_evals = len(cst_array) * len(re_values)
    print(f"  Total XFOIL runs: ~{total_evals}")

    from src.data.xfoil_runner import analyze_airfoil

    all_cst = []
    all_alpha = []
    all_re = []
    all_cl = []
    all_cd = []
    all_cm = []
    all_thickness = []

    start = time.time()
    n_converged = 0

    for i, cst in enumerate(cst_array):
        upper, lower = cst[:8], cst[8:]

        try:
            _, _, x_u, y_u, x_l, y_l = cst_to_coordinates(upper, lower, 100)
            props = compute_airfoil_properties(x_u, y_u, y_l)
            thickness = props['max_thickness']
        except Exception:
            continue

        for re in re_values:
            try:
                results, n_conv, _ = analyze_airfoil(
                    x_u, y_u, x_l, y_l,
                    reynolds_numbers=[re],
                    alpha_range=alpha_values,
                    max_iter=100,
                    timeout=15
                )

                if results:
                    for r in results:
                        if r['Cd'] > 0 and r['Cd'] < 0.5:
                            all_cst.append(cst.copy())
                            all_alpha.append(r['alpha'])
                            all_re.append(re)
                            all_cl.append(r['Cl'])
                            all_cd.append(r['Cd'])
                            all_cm.append(r['Cm'])
                            all_thickness.append(thickness)
                            n_converged += 1

            except Exception:
                continue

        # Progress
        if (i + 1) % 25 == 0 or i == len(cst_array) - 1:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (len(cst_array) - i - 1) / max(rate, 0.01)
            print(f"    {i+1}/{len(cst_array)} shapes | "
                  f"{n_converged} data points | "
                  f"{elapsed:.0f}s elapsed | ~{remaining:.0f}s left")

    data = {
        'cst_params': np.array(all_cst),
        'alpha': np.array(all_alpha),
        'reynolds': np.array(all_re),
        'cl': np.array(all_cl),
        'cd': np.array(all_cd),
        'cm': np.array(all_cm),
        'thickness': np.array(all_thickness),
    }

    print(f"\n  ✅ Total data points: {len(all_cl)}")
    if len(all_cl) > 0:
        print(f"  Cl range: [{min(all_cl):.3f}, {max(all_cl):.3f}]")
        print(f"  Cd range: [{min(all_cd):.5f}, {max(all_cd):.5f}]")
        print(f"  Cm range: [{min(all_cm):.3f}, {max(all_cm):.3f}]")

    return data


def main():
    print("=" * 65)
    print("  SCRIPT 1/3: CREATE CALIBRATION DATASET")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = 'checkpoints'

    # Load CVAE
    cvae = CVAE(n_cst=16, condition_dim=5, latent_dim=32)
    ckpt = torch.load(os.path.join(checkpoint_dir, 'generator_best.pt'),
                       map_location=device, weights_only=False)
    cvae.load_state_dict(ckpt['model_state_dict'])
    cvae = cvae.to(device)
    cvae.eval()
    print("  CVAE loaded ✓")

    # Load scaler
    with open(os.path.join(checkpoint_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    print("  Scaler loaded ✓")

    # Step 1: Base shapes
    base_shapes = generate_base_shapes(cvae, scaler, device, n_shapes=300)

    # Step 2: Camber variants
    camber_variants = create_camber_variants(base_shapes, n_target=600)

    # Combine all shapes
    all_shapes = np.vstack([base_shapes, camber_variants])
    print(f"\n  Total unique shapes: {len(all_shapes)}")

    # Step 3: XFOIL labeling
    re_values = [200000, 500000, 1000000]
    alpha_values = list(range(-4, 14, 2))  # -4 to 12 in steps of 2

    calib_data = run_xfoil_labeling(all_shapes, re_values, alpha_values)

    # Step 4: Save
    print(f"\n[4/4] Saving calibration data...")

    os.makedirs('data', exist_ok=True)
    save_path = 'data/calibration_data.pkl'

    with open(save_path, 'wb') as f:
        pickle.dump(calib_data, f)

    print(f"  Saved to {save_path}")
    print(f"  Total samples: {len(calib_data['cl'])}")

    print(f"\n{'=' * 65}")
    print(f"  SCRIPT 1/3 COMPLETE")
    print(f"  Next: python scripts/retrain_cvae.py")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()