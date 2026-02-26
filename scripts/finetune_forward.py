"""
╔══════════════════════════════════════════════════════════╗
║  SCRIPT 3/3: FINE-TUNE FORWARD MODEL                    ║
║                                                          ║
║  Trains forward model on ORIGINAL + GENERATED data       ║
║  so it predicts accurately on CVAE-generated shapes.     ║
║                                                          ║
║  Run: python scripts/finetune_forward.py                 ║
║  Time: ~15 minutes                                       ║
║  Requires: data/calibration_data.pkl (from Script 1)     ║
║  Output: checkpoints/forwardmodel_best.pt (overwritten)  ║
╚══════════════════════════════════════════════════════════╝
"""

import sys
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.forward_model import ForwardModel
from src.utils.config import Config
from src.data.preprocessing import load_dataset


def prepare_mixed_data(original_data, calib_data, scaler):
    """Prepare combined original + calibration data."""
    print("\n  Preparing mixed training data...")

    # Original data
    train_mask = original_data['train_mask'] & (original_data['cd'] > 0)
    val_mask = original_data['val_mask'] & (original_data['cd'] > 0)
    test_mask = original_data['test_mask'] & (original_data['cd'] > 0)

    def build_fwd(mask):
        cst = original_data['cst_params'][mask].astype(np.float32)
        alpha = original_data['alpha'][mask].astype(np.float32).reshape(-1, 1)
        log_re = np.log10(original_data['reynolds'][mask]).astype(np.float32).reshape(-1, 1)
        cl = original_data['cl'][mask].astype(np.float32).reshape(-1, 1)
        log_cd = np.log10(np.clip(original_data['cd'][mask], 1e-6, None)).astype(np.float32).reshape(-1, 1)
        cm = original_data['cm'][mask].astype(np.float32).reshape(-1, 1)

        inputs = np.hstack([cst, alpha, log_re])
        targets = np.hstack([cl, log_cd, cm])
        return inputs, targets

    train_in, train_tgt = build_fwd(train_mask)
    val_in, val_tgt = build_fwd(val_mask)
    test_in, test_tgt = build_fwd(test_mask)

    print(f"    Original — Train: {len(train_in)}, Val: {len(val_in)}, "
          f"Test: {len(test_in)}")

    # Add calibration data
    if calib_data is not None and len(calib_data['cl']) > 0:
        cal_cst = calib_data['cst_params'].astype(np.float32)
        cal_alpha = calib_data['alpha'].astype(np.float32).reshape(-1, 1)
        cal_log_re = np.log10(calib_data['reynolds']).astype(np.float32).reshape(-1, 1)
        cal_cl = calib_data['cl'].astype(np.float32).reshape(-1, 1)
        cal_log_cd = np.log10(np.clip(calib_data['cd'], 1e-6, None)).astype(np.float32).reshape(-1, 1)
        cal_cm = calib_data['cm'].astype(np.float32).reshape(-1, 1)

        cal_inputs = np.hstack([cal_cst, cal_alpha, cal_log_re])
        cal_targets = np.hstack([cal_cl, cal_log_cd, cal_cm])

        # 80/10/10 split for calibration data
        n = len(cal_inputs)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)

        idx = np.random.RandomState(42).permutation(n)
        cal_train_idx = idx[:n_train]
        cal_val_idx = idx[n_train:n_train + n_val]
        cal_test_idx = idx[n_train + n_val:]

        train_in = np.vstack([train_in, cal_inputs[cal_train_idx]])
        train_tgt = np.vstack([train_tgt, cal_targets[cal_train_idx]])
        val_in = np.vstack([val_in, cal_inputs[cal_val_idx]])
        val_tgt = np.vstack([val_tgt, cal_targets[cal_val_idx]])
        test_in = np.vstack([test_in, cal_inputs[cal_test_idx]])
        test_tgt = np.vstack([test_tgt, cal_targets[cal_test_idx]])

        print(f"    Added {n} calibration samples")
        print(f"    Combined — Train: {len(train_in)}, Val: {len(val_in)}, "
              f"Test: {len(test_in)}")

    # Normalize
    train_in_n = ((train_in - scaler['input_mean']) / scaler['input_std']).astype(np.float32)
    train_tgt_n = ((train_tgt - scaler['target_mean']) / scaler['target_std']).astype(np.float32)
    val_in_n = ((val_in - scaler['input_mean']) / scaler['input_std']).astype(np.float32)
    val_tgt_n = ((val_tgt - scaler['target_mean']) / scaler['target_std']).astype(np.float32)
    test_in_n = ((test_in - scaler['input_mean']) / scaler['input_std']).astype(np.float32)
    test_tgt_n = ((test_tgt - scaler['target_mean']) / scaler['target_std']).astype(np.float32)

    # Loaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_in_n), torch.from_numpy(train_tgt_n)),
        batch_size=512, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_in_n), torch.from_numpy(val_tgt_n)),
        batch_size=512, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_in_n), torch.from_numpy(test_tgt_n)),
        batch_size=512, shuffle=False
    )

    return train_loader, val_loader, test_loader, test_in, test_tgt


def train_forward(model, train_loader, val_loader, device,
                   epochs=200, lr=0.001, patience=30):
    """Train forward model from scratch on combined data."""
    print(f"\n  Training forward model for {epochs} epochs...")

    model = model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss weights — extra on Cd and Cm
    cl_w, cd_w, cm_w = 1.0, 3.0, 2.0

    best_val = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        n = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            pred = model(inputs)

            loss = (cl_w * nn.functional.mse_loss(pred[:, 0], targets[:, 0]) +
                    cd_w * nn.functional.mse_loss(pred[:, 1], targets[:, 1]) +
                    cm_w * nn.functional.mse_loss(pred[:, 2], targets[:, 2]))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n += 1

        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0
        vn = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                pred = model(inputs)
                loss = (cl_w * nn.functional.mse_loss(pred[:, 0], targets[:, 0]) +
                        cd_w * nn.functional.mse_loss(pred[:, 1], targets[:, 1]) +
                        cm_w * nn.functional.mse_loss(pred[:, 2], targets[:, 2]))
                val_loss += loss.item()
                vn += 1

        avg_train = train_loss / max(n, 1)
        avg_val = val_loss / max(vn, 1)

        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {avg_train:.6f} | Val: {avg_val:.6f} | "
                  f"Best: {best_val:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if no_improve >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    return model


def evaluate_model(model, test_loader, scaler, device, label=""):
    """Evaluate and print detailed metrics."""
    model.eval()

    all_pred = []
    all_true = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            pred = model(inputs.to(device)).cpu().numpy()
            all_pred.append(pred)
            all_true.append(targets.numpy())

    pred = np.vstack(all_pred)
    true = np.vstack(all_true)

    # Denormalize
    pred_phys = pred * scaler['target_std'] + scaler['target_mean']
    true_phys = true * scaler['target_std'] + scaler['target_mean']

    # R² scores
    from sklearn.metrics import r2_score, mean_absolute_error

    results = {}
    names = ['Cl', 'log10(Cd)', 'Cm']

    print(f"\n  {label} Results:")
    print(f"  {'Metric':<12s} {'R²':>8s} {'MAE':>10s} {'MAPE%':>8s}")
    print(f"  {'-' * 40}")

    for i, name in enumerate(names):
        r2 = r2_score(true[:, i], pred[:, i])
        mae = mean_absolute_error(true[:, i], pred[:, i])

        # Physical space errors
        if i == 1:  # Cd
            pred_cd = 10 ** np.clip(pred_phys[:, i], -5, 0)
            true_cd = 10 ** np.clip(true_phys[:, i], -5, 0)
            mape = np.mean(np.abs(pred_cd - true_cd) /
                           np.maximum(true_cd, 0.001)) * 100
        else:
            denom = np.maximum(np.abs(true_phys[:, i]), 0.01)
            mape = np.mean(np.abs(pred_phys[:, i] - true_phys[:, i]) /
                           denom) * 100

        print(f"  {name:<12s} {r2:>8.4f} {mae:>10.6f} {mape:>7.1f}%")
        results[name] = {'r2': r2, 'mae': mae, 'mape': mape}

    return results


def evaluate_on_generated_shapes(model, scaler, device, checkpoint_dir):
    """Test specifically on CVAE-generated shapes."""
    print(f"\n  Testing on CVAE-generated shapes...")

    from src.models.generator import CVAE
    from src.geometry.cst import cst_to_coordinates, validate_airfoil

    # Load new CVAE
    cvae = CVAE(n_cst=16, condition_dim=5, latent_dim=32)
    ckpt = torch.load(os.path.join(checkpoint_dir, 'generator_best.pt'),
                       map_location=device, weights_only=False)
    cvae.load_state_dict(ckpt['model_state_dict'])
    cvae = cvae.to(device)
    cvae.eval()

    # Generate shapes for different target Cl values
    test_cases = [
        {'cl': 0.3, 're': 500000, 'alpha': 2.0},
        {'cl': 0.7, 're': 300000, 'alpha': 5.0},
        {'cl': 0.9, 're': 200000, 'alpha': 5.0},
        {'cl': 1.2, 're': 1000000, 'alpha': 8.0},
    ]

    try:
        from src.data.xfoil_runner import analyze_airfoil
        xfoil_available = True
    except ImportError:
        xfoil_available = False
        print("  ⚠ XFOIL not available — skipping verification")
        return

    total_cl_err = []
    total_cd_err = []
    total_cm_err = []

    for case in test_cases:
        # Generate shapes
        cd_est = 0.01
        cond_raw = np.array([case['cl'], cd_est, np.log10(case['re']),
                              case['alpha'], 0.12], dtype=np.float32)
        cond_norm = (cond_raw - scaler['cond_mean']) / scaler['cond_std']
        cond_t = torch.from_numpy(cond_norm).unsqueeze(0).to(device)

        with torch.no_grad():
            z = torch.randn(10, cvae.latent_dim, device=device)
            cond_exp = cond_t.expand(10, -1)
            cst_norm = cvae.decode(z, cond_exp)

            cst_mean_t = torch.tensor(scaler['cst_mean'], device=device,
                                       dtype=torch.float32)
            cst_std_t = torch.tensor(scaler['cst_std'], device=device,
                                      dtype=torch.float32)
            cst_phys = (cst_norm * cst_std_t + cst_mean_t).cpu().numpy()

        # For each shape: forward predict + XFOIL
        for cst in cst_phys:
            upper, lower = cst[:8], cst[8:]

            try:
                _, _, x_u, y_u, x_l, y_l = cst_to_coordinates(upper, lower, 100)
                ok, _ = validate_airfoil(x_u, y_u, y_l)
                if not ok:
                    continue

                # Forward model prediction
                fwd_in = np.concatenate([cst, [case['alpha']],
                                          [np.log10(case['re'])]]).astype(np.float32)
                fwd_norm = (fwd_in - scaler['input_mean']) / scaler['input_std']

                with torch.no_grad():
                    pred = model(torch.from_numpy(fwd_norm).unsqueeze(0).to(device))
                    pred = pred.cpu().numpy()[0]

                pred_phys = pred * scaler['target_std'] + scaler['target_mean']
                pred_cl = pred_phys[0]
                pred_cd = 10 ** np.clip(pred_phys[1], -5, 0)
                pred_cm = pred_phys[2]

                # XFOIL
                results, _, _ = analyze_airfoil(
                    x_u, y_u, x_l, y_l,
                    reynolds_numbers=[case['re']],
                    alpha_range=[case['alpha']],
                    max_iter=100, timeout=15
                )

                if results:
                    r = min(results, key=lambda x: abs(x['alpha'] - case['alpha']))
                    xfoil_cl = r['Cl']
                    xfoil_cd = r['Cd']
                    xfoil_cm = r['Cm']

                    if xfoil_cd > 0:
                        cl_err = abs(pred_cl - xfoil_cl) / max(abs(xfoil_cl), 0.01) * 100
                        cd_err = abs(pred_cd - xfoil_cd) / max(xfoil_cd, 0.001) * 100
                        cm_err = abs(pred_cm - xfoil_cm) / max(abs(xfoil_cm), 0.001) * 100

                        total_cl_err.append(cl_err)
                        total_cd_err.append(cd_err)
                        total_cm_err.append(cm_err)

            except Exception:
                continue

    if total_cl_err:
        print(f"\n  Forward Model vs XFOIL on GENERATED shapes:")
        print(f"  {'Metric':<8s} {'Mean%':>8s} {'Median%':>8s} {'95th%':>8s}")
        print(f"  {'-'*35}")
        print(f"  {'Cl':<8s} {np.mean(total_cl_err):>8.1f} "
              f"{np.median(total_cl_err):>8.1f} "
              f"{np.percentile(total_cl_err, 95):>8.1f}")
        print(f"  {'Cd':<8s} {np.mean(total_cd_err):>8.1f} "
              f"{np.median(total_cd_err):>8.1f} "
              f"{np.percentile(total_cd_err, 95):>8.1f}")
        print(f"  {'Cm':<8s} {np.mean(total_cm_err):>8.1f} "
              f"{np.median(total_cm_err):>8.1f} "
              f"{np.percentile(total_cm_err, 95):>8.1f}")

        if np.mean(total_cd_err) < 5:
            print(f"\n  ⭐ TARGET MET: Mean Cd error < 5%!")
        elif np.mean(total_cd_err) < 20:
            print(f"\n  🟢 GOOD: Mean Cd error < 20%")
        elif np.mean(total_cd_err) < 50:
            print(f"\n  🟡 IMPROVED: Mean Cd error < 50%")
        else:
            print(f"\n  🟠 More work needed")


def main():
    print("=" * 65)
    print("  SCRIPT 3/3: FINE-TUNE FORWARD MODEL")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    checkpoint_dir = 'checkpoints'

    # Backup
    for fname in ['forwardmodel_best.pt']:
        src = os.path.join(checkpoint_dir, fname)
        bak = os.path.join(checkpoint_dir, 'forwardmodel_best_BACKUP.pt')
        if os.path.exists(src) and not os.path.exists(bak):
            shutil.copy2(src, bak)
            print(f"  Backed up {fname} ✓")

    # Load scaler
    with open(os.path.join(checkpoint_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    print(f"  Scaler loaded ✓")

    # Load original data
    config = Config("config.yaml")
    original_data = load_dataset(config.paths.processed_data)
    print(f"  Original data: {len(original_data['cl'])} samples ✓")

    # Load calibration data
    calib_path = 'data/calibration_data.pkl'
    calib_data = None
    if os.path.exists(calib_path):
        with open(calib_path, 'rb') as f:
            calib_data = pickle.load(f)
        print(f"  Calibration data: {len(calib_data['cl'])} samples ✓")
    else:
        print(f"  ⚠ No calibration data. Run Script 1 first!")

    # Prepare data
    train_loader, val_loader, test_loader, test_in, test_tgt = \
        prepare_mixed_data(original_data, calib_data, scaler)

    # Create fresh forward model
    model = ForwardModel(input_dim=18, hidden_dims=[256, 512, 256, 128],
                          dropout=0.1)
    print(f"  Created fresh forward model ✓")

    # Evaluate BEFORE (load old weights)
    print(f"\n{'=' * 65}")
    print(f"  BEFORE RETRAINING:")
    print(f"{'=' * 65}")

    old_model = ForwardModel(input_dim=18)
    for fname in ['forwardmodel_best_BACKUP.pt', 'forwardmodel_best.pt']:
        p = os.path.join(checkpoint_dir, fname)
        if os.path.exists(p):
            ckpt = torch.load(p, map_location=device, weights_only=False)
            old_model.load_state_dict(ckpt['model_state_dict'])
            break
    old_model = old_model.to(device)
    old_model.eval()
    evaluate_model(old_model, test_loader, scaler, device, "OLD Model")

    # Train
    print(f"\n{'=' * 65}")
    print(f"  RETRAINING:")
    print(f"{'=' * 65}")

    start = time.time()
    model = train_forward(model, train_loader, val_loader, device,
                           epochs=200, lr=0.001, patience=30)
    elapsed = time.time() - start

    # Evaluate AFTER
    print(f"\n{'=' * 65}")
    print(f"  AFTER RETRAINING:")
    print(f"{'=' * 65}")
    results = evaluate_model(model, test_loader, scaler, device, "NEW Model")

    # Save
    save_path = os.path.join(checkpoint_dir, 'forwardmodel_best.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': None,
        'retrained': True,
        'includes_calibration_data': calib_data is not None,
    }, save_path)
    print(f"\n  Saved to {save_path} ✓")

    # Test on generated shapes with XFOIL comparison
    print(f"\n{'=' * 65}")
    print(f"  GENERATED SHAPE VERIFICATION:")
    print(f"{'=' * 65}")
    evaluate_on_generated_shapes(model, scaler, device, checkpoint_dir)

    print(f"\n{'=' * 65}")
    print(f"  SCRIPT 3/3 COMPLETE ({elapsed/60:.1f} min)")
    print(f"{'=' * 65}")
    print(f"  Now test: streamlit run app.py")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()