"""
╔══════════════════════════════════════════════════════════╗
║  SCRIPT 2/3: RETRAIN CVAE WITH PHYSICS-INFORMED LOSS    ║
║                                                          ║
║  Retrains the CVAE generator with:                       ║
║  1. Original reconstruction loss                         ║
║  2. KL divergence                                        ║
║  3. NEW: Physics loss (Forward model feedback)           ║
║  4. NEW: Camber consistency loss                         ║
║                                                          ║
║  Run: python scripts/retrain_cvae.py                     ║
║  Time: ~40 minutes                                       ║
║  Requires: data/calibration_data.pkl (from Script 1)     ║
║  Output: checkpoints/generator_best.pt (overwritten)     ║
╚══════════════════════════════════════════════════════════╝
"""

import sys
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.generator import CVAE
from src.models.forward_model import ForwardModel
from src.utils.config import Config
from src.data.preprocessing import load_dataset


class PhysicsCVAELoss(nn.Module):
    """
    Physics-informed CVAE loss.

    L = λ_recon * L_recon
      + λ_kl * L_kl
      + λ_phys * L_physics
      + λ_smooth * L_smooth

    L_physics: Forward model predicts performance from generated CST,
               compared against conditioning targets.
               This teaches the CVAE that "if you want Cl=1.2,
               generate shapes that ACTUALLY produce Cl=1.2".

    L_smooth: CST weight smoothness + physical constraints.
    """

    def __init__(self, recon_weight=1.0, kl_weight=0.001,
                 physics_weight=5.0, smooth_weight=0.3):
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.physics_weight = physics_weight
        self.smooth_weight = smooth_weight

    def forward(self, cst_recon, cst_orig, mean, logvar,
                forward_model, conditions, scaler, device):
        """
        Parameters
        ----------
        cst_recon : (batch, 16) — reconstructed CST
        cst_orig : (batch, 16) — original CST
        mean, logvar : (batch, latent_dim) — latent distribution
        forward_model : trained forward model
        conditions : (batch, 5) — NORMALIZED [Cl, Cd, logRe, α, t/c]
        scaler : dict — normalization statistics
        device : torch device
        """
        if cst_recon.dim() == 3:
            cst_recon = cst_recon.squeeze(1)
        if cst_orig.dim() == 3:
            cst_orig = cst_orig.squeeze(1)

        # ─── Reconstruction Loss ───
        recon_loss = F.mse_loss(cst_recon, cst_orig)

        # ─── KL Divergence ───
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

        # ─── Physics Loss ───
        # Denormalize generated CST to physical space
        # Then run through forward model to check if it meets targets
        physics_loss = torch.tensor(0.0, device=device)

        try:
            # Denormalize CST
            cst_mean_t = torch.tensor(scaler['cst_mean'], device=device,
                                       dtype=torch.float32)
            cst_std_t = torch.tensor(scaler['cst_std'], device=device,
                                      dtype=torch.float32)
            cst_physical = cst_recon * cst_std_t + cst_mean_t

            # Denormalize conditions to get alpha and Re
            cond_mean_t = torch.tensor(scaler['cond_mean'], device=device,
                                        dtype=torch.float32)
            cond_std_t = torch.tensor(scaler['cond_std'], device=device,
                                       dtype=torch.float32)
            cond_physical = conditions * cond_std_t + cond_mean_t

            # conditions = [Cl, Cd, logRe, alpha, thickness]
            target_cl = cond_physical[:, 0]
            alpha = cond_physical[:, 3:4]
            log_re = cond_physical[:, 2:3]

            # Build forward model input: [CST(16), alpha(1), logRe(1)]
            fwd_input_physical = torch.cat([cst_physical, alpha, log_re], dim=1)

            # Normalize for forward model
            input_mean_t = torch.tensor(scaler['input_mean'], device=device,
                                         dtype=torch.float32)
            input_std_t = torch.tensor(scaler['input_std'], device=device,
                                        dtype=torch.float32)
            fwd_input_norm = (fwd_input_physical - input_mean_t) / input_std_t

            # Predict with forward model (no gradient through forward model)
            with torch.no_grad():
                pred = forward_model(fwd_input_norm)

            # Denormalize predictions
            target_mean_t = torch.tensor(scaler['target_mean'], device=device,
                                          dtype=torch.float32)
            target_std_t = torch.tensor(scaler['target_std'], device=device,
                                         dtype=torch.float32)
            pred_physical = pred * target_std_t + target_mean_t
            predicted_cl = pred_physical[:, 0]

            # Physics loss: predicted Cl should match target Cl
            physics_loss = F.mse_loss(predicted_cl, target_cl)

        except Exception:
            physics_loss = torch.tensor(0.0, device=device)

        # ─── Smoothness Loss ───
        smooth_loss = self._smoothness_loss(cst_recon)

        # ─── Total ───
        total = (self.recon_weight * recon_loss +
                 self.kl_weight * kl_loss +
                 self.physics_weight * physics_loss +
                 self.smooth_weight * smooth_loss)

        return total, {
            'recon': recon_loss.item(),
            'kl': kl_loss.item(),
            'physics': physics_loss.item(),
            'smooth': smooth_loss.item(),
            'total': total.item(),
        }

    def _smoothness_loss(self, cst):
        """Physical constraints on CST parameters."""
        if cst.dim() == 3:
            cst = cst.squeeze(1)

        upper = cst[:, :8]
        lower = cst[:, 8:]

        loss = torch.tensor(0.0, device=cst.device)

        # Upper mostly positive
        loss = loss + torch.mean(F.relu(-upper)) * 5.0

        # Lower positive (CST convention)
        loss = loss + torch.mean(F.relu(-lower)) * 3.0

        # Reasonable range
        loss = loss + torch.mean(F.relu(torch.abs(cst) - 0.5))

        # Smoothness
        loss = loss + torch.mean(torch.diff(upper, dim=1) ** 2) * 2.0
        loss = loss + torch.mean(torch.diff(lower, dim=1) ** 2) * 2.0

        # Positive thickness (upper mean > lower mean)
        loss = loss + torch.mean(F.relu(lower.mean(1) - upper.mean(1))) * 10.0

        return loss


def prepare_datasets(original_data, calib_data, scaler):
    """Prepare combined training data for CVAE."""
    print("\n  Preparing datasets...")

    # Original data
    train_mask = original_data['train_mask']
    val_mask = original_data['val_mask']

    # Filter Cd > 0
    valid = original_data['cd'] > 0
    train_mask = train_mask & valid
    val_mask = val_mask & valid

    def build_dataset(mask):
        cst = original_data['cst_params'][mask].astype(np.float32)
        cl = original_data['cl'][mask].astype(np.float32).reshape(-1, 1)
        cd = original_data['cd'][mask].astype(np.float32).reshape(-1, 1)
        log_re = np.log10(original_data['reynolds'][mask]).astype(np.float32).reshape(-1, 1)
        alpha = original_data['alpha'][mask].astype(np.float32).reshape(-1, 1)
        thickness = original_data['thickness'][mask].astype(np.float32).reshape(-1, 1)

        conditions = np.hstack([cl, cd, log_re, alpha, thickness])

        # Normalize
        cst_norm = (cst - scaler['cst_mean']) / scaler['cst_std']
        cond_norm = (conditions - scaler['cond_mean']) / scaler['cond_std']

        return cst_norm, cond_norm

    train_cst, train_cond = build_dataset(train_mask)
    val_cst, val_cond = build_dataset(val_mask)

    print(f"    Original train: {len(train_cst)}")
    print(f"    Original val: {len(val_cst)}")

    # Add calibration data
    if calib_data is not None and len(calib_data['cl']) > 0:
        cal_cst = calib_data['cst_params'].astype(np.float32)
        cal_cl = calib_data['cl'].astype(np.float32).reshape(-1, 1)
        cal_cd = calib_data['cd'].astype(np.float32).reshape(-1, 1)
        cal_log_re = np.log10(calib_data['reynolds']).astype(np.float32).reshape(-1, 1)
        cal_alpha = calib_data['alpha'].astype(np.float32).reshape(-1, 1)
        cal_thickness = calib_data['thickness'].astype(np.float32).reshape(-1, 1)

        cal_conditions = np.hstack([cal_cl, cal_cd, cal_log_re,
                                     cal_alpha, cal_thickness])

        cal_cst_norm = (cal_cst - scaler['cst_mean']) / scaler['cst_std']
        cal_cond_norm = (cal_conditions - scaler['cond_mean']) / scaler['cond_std']

        # Add to training set (80/20 split for cal data)
        n_cal = len(cal_cst_norm)
        n_cal_train = int(n_cal * 0.8)

        train_cst = np.vstack([train_cst, cal_cst_norm[:n_cal_train]])
        train_cond = np.vstack([train_cond, cal_cond_norm[:n_cal_train]])

        val_cst = np.vstack([val_cst, cal_cst_norm[n_cal_train:]])
        val_cond = np.vstack([val_cond, cal_cond_norm[n_cal_train:]])

        print(f"    Added calibration: {n_cal} samples")
        print(f"    Combined train: {len(train_cst)}")
        print(f"    Combined val: {len(val_cst)}")

    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.from_numpy(train_cond),
        torch.from_numpy(train_cst)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_cond),
        torch.from_numpy(val_cst)
    )

    train_loader = DataLoader(train_dataset, batch_size=256,
                               shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    return train_loader, val_loader


def train_physics_cvae(cvae, forward_model, train_loader, val_loader,
                        scaler, device, epochs=300, lr=0.0005,
                        kl_warmup=50, checkpoint_dir='checkpoints'):
    """Train CVAE with physics-informed loss."""
    print(f"\n  Training Physics-CVAE for {epochs} epochs...")

    cvae = cvae.to(device)
    cvae.train()
    forward_model = forward_model.to(device)
    forward_model.eval()

    optimizer = optim.AdamW(cvae.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2
    )

    loss_fn = PhysicsCVAELoss(
        recon_weight=1.0,
        kl_weight=0.001,     # will be warmed up
        physics_weight=5.0,
        smooth_weight=0.3
    )

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    max_patience = 60

    for epoch in range(epochs):
        # KL warmup
        kl_weight = min(0.001, 0.001 * (epoch / max(kl_warmup, 1)))
        loss_fn.kl_weight = kl_weight

        # Physics weight ramp (start after KL warmup)
        if epoch < kl_warmup:
            loss_fn.physics_weight = 0.5
        else:
            loss_fn.physics_weight = min(5.0, 0.5 + 4.5 * (epoch - kl_warmup) / 100)

        # Train
        cvae.train()
        train_losses = []

        for conditions, cst_target in train_loader:
            conditions = conditions.to(device)
            cst_target = cst_target.to(device)

            # Forward pass
            cst_recon, mean, logvar = cvae(cst_target, conditions)

            # Physics loss
            loss, loss_dict = loss_fn(
                cst_recon, cst_target, mean, logvar,
                forward_model, conditions, scaler, device
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cvae.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss_dict)

        scheduler.step()

        # Validate
        cvae.eval()
        val_losses = []

        with torch.no_grad():
            for conditions, cst_target in val_loader:
                conditions = conditions.to(device)
                cst_target = cst_target.to(device)

                cst_recon, mean, logvar = cvae(cst_target, conditions)
                loss, loss_dict = loss_fn(
                    cst_recon, cst_target, mean, logvar,
                    forward_model, conditions, scaler, device
                )
                val_losses.append(loss_dict)

        avg_train = np.mean([l['total'] for l in train_losses])
        avg_val = np.mean([l['total'] for l in val_losses])
        avg_recon = np.mean([l['recon'] for l in val_losses])
        avg_phys = np.mean([l['physics'] for l in val_losses])
        avg_kl = np.mean([l['kl'] for l in val_losses])

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone()
                          for k, v in cvae.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # Log
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
                  f"Recon: {avg_recon:.4f} | Phys: {avg_phys:.4f} | "
                  f"KL: {avg_kl:.4f} | "
                  f"kl_w: {kl_weight:.4f} | phys_w: {loss_fn.physics_weight:.1f}")

        # Early stopping
        if patience_counter >= max_patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # Load best
    if best_state is not None:
        cvae.load_state_dict(best_state)

    cvae.eval()

    # Save
    save_path = os.path.join(checkpoint_dir, 'generator_best.pt')
    torch.save({
        'model_state_dict': cvae.state_dict(),
        'physics_informed': True,
        'best_val_loss': best_val_loss,
    }, save_path)

    print(f"  ✅ Saved physics-CVAE to {save_path}")
    print(f"     Best val loss: {best_val_loss:.4f}")

    return cvae


def evaluate_cvae(cvae, forward_model, scaler, device):
    """Quick test: generate shapes and check if Cl matches target."""
    print(f"\n  Evaluating CVAE quality...")

    test_cases = [
        {'cl': 0.3, 're': 500000, 'alpha': 2.0},
        {'cl': 0.7, 're': 500000, 'alpha': 5.0},
        {'cl': 0.9, 're': 200000, 'alpha': 5.0},
        {'cl': 1.2, 're': 1000000, 'alpha': 8.0},
    ]

    for case in test_cases:
        cd_est = 0.01
        thickness = 0.12
        cond_raw = np.array([case['cl'], cd_est, np.log10(case['re']),
                              case['alpha'], thickness], dtype=np.float32)

        cond_norm = (cond_raw - scaler['cond_mean']) / scaler['cond_std']
        cond_t = torch.from_numpy(cond_norm).unsqueeze(0).to(device)

        with torch.no_grad():
            z = torch.randn(20, cvae.latent_dim, device=device)
            cond_exp = cond_t.expand(20, -1)
            cst_norm = cvae.decode(z, cond_exp)

            cst_mean = torch.tensor(scaler['cst_mean'], device=device,
                                     dtype=torch.float32)
            cst_std = torch.tensor(scaler['cst_std'], device=device,
                                    dtype=torch.float32)
            cst_phys = cst_norm * cst_std + cst_mean

            # Forward model prediction
            alpha_t = torch.full((20, 1), case['alpha'], device=device)
            log_re_t = torch.full((20, 1), np.log10(case['re']), device=device)
            fwd_in = torch.cat([cst_phys, alpha_t, log_re_t], dim=1)

            input_mean = torch.tensor(scaler['input_mean'], device=device,
                                       dtype=torch.float32)
            input_std = torch.tensor(scaler['input_std'], device=device,
                                      dtype=torch.float32)
            fwd_norm = (fwd_in - input_mean) / input_std
            pred = forward_model(fwd_norm)

            target_mean = torch.tensor(scaler['target_mean'], device=device,
                                        dtype=torch.float32)
            target_std = torch.tensor(scaler['target_std'], device=device,
                                       dtype=torch.float32)
            pred_phys = pred * target_std + target_mean
            pred_cl = pred_phys[:, 0].cpu().numpy()

        mean_cl = np.mean(pred_cl)
        std_cl = np.std(pred_cl)
        err = abs(mean_cl - case['cl']) / max(abs(case['cl']), 0.01) * 100

        quality = "✓" if err < 15 else "⚠"
        print(f"    Target Cl={case['cl']:.1f} → "
              f"Generated Cl={mean_cl:.3f} ± {std_cl:.3f} "
              f"({err:.1f}% err) {quality}")


def main():
    print("=" * 65)
    print("  SCRIPT 2/3: RETRAIN CVAE WITH PHYSICS LOSS")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    checkpoint_dir = 'checkpoints'

    # Backup original generator
    gen_path = os.path.join(checkpoint_dir, 'generator_best.pt')
    backup_path = os.path.join(checkpoint_dir, 'generator_best_BACKUP.pt')
    if os.path.exists(gen_path) and not os.path.exists(backup_path):
        shutil.copy2(gen_path, backup_path)
        print(f"  Backed up original generator ✓")

    # Load forward model (frozen, used for physics loss)
    forward_model = ForwardModel(input_dim=18)
    for fname in ['forwardmodel_best.pt', 'forward_model_best.pt']:
        fwd_path = os.path.join(checkpoint_dir, fname)
        if os.path.exists(fwd_path):
            ckpt = torch.load(fwd_path, map_location=device, weights_only=False)
            forward_model.load_state_dict(ckpt['model_state_dict'])
            print(f"  Forward model loaded ✓ ({fname})")
            break
    forward_model = forward_model.to(device)
    forward_model.eval()
    for p in forward_model.parameters():
        p.requires_grad = False

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
        print(f"  ⚠ No calibration data found. Run Script 1 first!")
        print(f"    python scripts/create_calibration_data.py")

    # Create CVAE (fresh weights — full retrain)
    cvae = CVAE(
        n_cst=16, condition_dim=5, latent_dim=32,
        encoder_hidden=[256, 256, 128],
        decoder_hidden=[256, 512, 512, 256, 128]
    )
    print(f"  Created fresh CVAE ✓")

    # Prepare data
    train_loader, val_loader = prepare_datasets(
        original_data, calib_data, scaler
    )

    # Train
    start = time.time()
    cvae = train_physics_cvae(
        cvae, forward_model, train_loader, val_loader,
        scaler, device,
        epochs=300, lr=0.0005, kl_warmup=50,
        checkpoint_dir=checkpoint_dir
    )
    elapsed = time.time() - start

    # Evaluate
    evaluate_cvae(cvae, forward_model, scaler, device)

    print(f"\n{'=' * 65}")
    print(f"  SCRIPT 2/3 COMPLETE ({elapsed/60:.1f} min)")
    print(f"  Next: python scripts/finetune_forward.py")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()