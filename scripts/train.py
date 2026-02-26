"""
╔══════════════════════════════════════════════════════════╗
║  COMPLETE TRAINING SCRIPT                                ║
║                                                          ║
║  Usage: python scripts/train.py                          ║
╚══════════════════════════════════════════════════════════╝
"""

import sys
import os
import pickle
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.helpers import set_seed, get_device, count_parameters
from src.utils.logger import setup_logger
from src.data.preprocessing import load_dataset
from src.data.dataset import ForwardDataset, GeneratorDataset, create_dataloaders
from src.models.forward_model import ForwardModel
from src.models.generator import CVAE
from src.models.losses import ForwardModelLoss, CVAELoss
from src.training.trainer import ForwardModelTrainer, CVAETrainer


def main():
    # Setup
    config = Config("config.yaml")
    logger = setup_logger("training", log_dir="logs")
    set_seed(config.project.random_seed)
    device = get_device()

    # Load Data
    logger.info("Loading dataset...")
    data = load_dataset(config.paths.processed_data)

    logger.info(f"Total samples: {len(data['cl']):,}")
    logger.info(f"Train: {data['train_mask'].sum():,}")
    logger.info(f"Val: {data['val_mask'].sum():,}")
    logger.info(f"Test: {data['test_mask'].sum():,}")

    # Quick data sanity check
    n_neg_cd = np.sum(data['cd'] <= 0)
    if n_neg_cd > 0:
        logger.warning(f"Found {n_neg_cd} non-positive Cd values. Filtering...")
        valid_mask = data['cd'] > 0
        for key in ['cst_params', 'cl', 'cd', 'cm', 'alpha',
                     'reynolds', 'thickness', 'airfoil_ids']:
            if key == 'cst_params':
                data[key] = data[key][valid_mask]
            else:
                data[key] = data[key][valid_mask]
        data['train_mask'] = data['train_mask'][valid_mask]
        data['val_mask'] = data['val_mask'][valid_mask]
        data['test_mask'] = data['test_mask'][valid_mask]
        logger.info(f"After filtering: {len(data['cl']):,} samples")

    checkpoint_dir = config.paths.checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ═══════════════════════════════════════
    # PART 1: TRAIN FORWARD MODEL
    # ═══════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("  PART 1: Training Forward Model")
    logger.info("=" * 60)

    train_fwd = ForwardDataset(
        cst_params=data['cst_params'][data['train_mask']],
        alpha=data['alpha'][data['train_mask']],
        reynolds=data['reynolds'][data['train_mask']],
        cl=data['cl'][data['train_mask']],
        cd=data['cd'][data['train_mask']],
        cm=data['cm'][data['train_mask']],
        normalize=True, scaler=None
    )

    val_fwd = ForwardDataset(
        cst_params=data['cst_params'][data['val_mask']],
        alpha=data['alpha'][data['val_mask']],
        reynolds=data['reynolds'][data['val_mask']],
        cl=data['cl'][data['val_mask']],
        cd=data['cd'][data['val_mask']],
        cm=data['cm'][data['val_mask']],
        normalize=True, scaler=train_fwd.scaler
    )

    test_fwd = ForwardDataset(
        cst_params=data['cst_params'][data['test_mask']],
        alpha=data['alpha'][data['test_mask']],
        reynolds=data['reynolds'][data['test_mask']],
        cl=data['cl'][data['test_mask']],
        cd=data['cd'][data['test_mask']],
        cm=data['cm'][data['test_mask']],
        normalize=True, scaler=train_fwd.scaler
    )

    # Save scaler early
    scaler = train_fwd.scaler.copy()

    batch_size_fwd = config.forward_model.training.batch_size
    train_loader_fwd, val_loader_fwd, test_loader_fwd = create_dataloaders(
        train_fwd, val_fwd, test_fwd,
        batch_size=batch_size_fwd, num_workers=0
    )

    forward_model = ForwardModel(
        input_dim=18,
        hidden_dims=config.forward_model.hidden_dims,
        dropout=config.forward_model.dropout
    )
    count_parameters(forward_model)

    fwd_loss = ForwardModelLoss(cl_weight=1.0, cd_weight=2.0, cm_weight=1.0)

    fwd_trainer = ForwardModelTrainer(
        model=forward_model, loss_fn=fwd_loss,
        device=device, config=config
    )

    fwd_history = fwd_trainer.train(
        train_loader=train_loader_fwd,
        val_loader=val_loader_fwd,
        max_epochs=config.forward_model.training.max_epochs,
        patience=config.forward_model.training.early_stopping_patience,
        checkpoint_dir=checkpoint_dir
    )

    # Evaluate forward model
    logger.info("\nEvaluating Forward Model on test set...")
    forward_model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader_fwd:
            inputs = inputs.to(device)
            preds = forward_model(inputs).cpu()
            all_preds.append(preds)
            all_targets.append(targets)

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    from sklearn.metrics import r2_score, mean_absolute_error

    logger.info("\n  Forward Model Test Results:")
    for i, name in enumerate(['Cl', 'log10(Cd)', 'Cm']):
        r2 = r2_score(all_targets[:, i], all_preds[:, i])
        mae = mean_absolute_error(all_targets[:, i], all_preds[:, i])
        logger.info(f"    {name:12s}: R² = {r2:.4f}, MAE = {mae:.4f}")

    # Check gate
    cl_r2 = r2_score(all_targets[:, 0], all_preds[:, 0])
    if cl_r2 < 0.90:
        logger.warning(f"  Cl R² = {cl_r2:.4f} is below 0.90. "
                       f"Generator may not work well.")
        logger.warning(f"  Consider: more data, bigger model, or longer training.")
    else:
        logger.info(f"  ✓ Forward Model PASSED gate (Cl R² = {cl_r2:.4f})")

    # ═══════════════════════════════════════
    # PART 2: TRAIN CVAE GENERATOR
    # ═══════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("  PART 2: Training CVAE Generator")
    logger.info("=" * 60)

    train_gen = GeneratorDataset(
        cst_params=data['cst_params'][data['train_mask']],
        cl=data['cl'][data['train_mask']],
        cd=data['cd'][data['train_mask']],
        reynolds=data['reynolds'][data['train_mask']],
        alpha=data['alpha'][data['train_mask']],
        thickness=data['thickness'][data['train_mask']],
        normalize=True, scaler=None
    )

    val_gen = GeneratorDataset(
        cst_params=data['cst_params'][data['val_mask']],
        cl=data['cl'][data['val_mask']],
        cd=data['cd'][data['val_mask']],
        reynolds=data['reynolds'][data['val_mask']],
        alpha=data['alpha'][data['val_mask']],
        thickness=data['thickness'][data['val_mask']],
        normalize=True, scaler=train_gen.scaler
    )

    # Save COMPLETE scaler
    scaler['cond_mean'] = train_gen.scaler['cond_mean']
    scaler['cond_std'] = train_gen.scaler['cond_std']
    scaler['cst_mean'] = train_gen.scaler['cst_mean']
    scaler['cst_std'] = train_gen.scaler['cst_std']

    scaler_path = os.path.join(checkpoint_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"  Scaler saved to {scaler_path}")

    batch_size_gen = config.generator.training.batch_size
    train_loader_gen, val_loader_gen, _ = create_dataloaders(
        train_gen, val_gen, val_gen,
        batch_size=batch_size_gen, num_workers=0
    )

    cvae = CVAE(
        n_cst=config.data.n_cst_total,
        condition_dim=5,
        latent_dim=config.generator.latent_dim,
        encoder_hidden=config.generator.encoder.hidden_dims,
        decoder_hidden=config.generator.decoder.hidden_dims
    )
    count_parameters(cvae)

    cvae_loss = CVAELoss(
        recon_weight=config.generator.training.loss_weights.reconstruction,
        kl_weight=config.generator.training.loss_weights.kl_divergence,
        perf_weight=config.generator.training.loss_weights.performance,
        physics_weight=config.generator.training.loss_weights.smoothness
    )

    cvae_trainer = CVAETrainer(
        cvae_model=cvae,
        loss_fn=cvae_loss,
        forward_model=forward_model,
        device=device,
        config=config
    )

    cvae_history = cvae_trainer.train(
        train_loader=train_loader_gen,
        val_loader=val_loader_gen,
        max_epochs=config.generator.training.max_epochs,
        patience=config.generator.training.early_stopping_patience,
        kl_warmup_epochs=config.generator.training.kl_warmup_epochs,
        checkpoint_dir=checkpoint_dir
    )

    # ═══════════════════════════════════════
    # PART 3: QUICK GENERATION TEST
    # ═══════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("  PART 3: Quick Generation Test")
    logger.info("=" * 60)

    try:
        from src.models.inference import AirfoilGenerator

        gen = AirfoilGenerator(checkpoint_dir=checkpoint_dir, device=device)

        test_cases = [
            {'Cl': 0.5, 'Re': 500000, 'alpha': 3.0},
            {'Cl': 1.0, 'Re': 500000, 'alpha': 5.0},
            {'Cl': 1.5, 'Re': 1000000, 'alpha': 8.0},
        ]

        os.makedirs('results/exported_airfoils', exist_ok=True)
        os.makedirs('results/figures', exist_ok=True)

        for case in test_cases:
            logger.info(f"\n  Testing: Cl={case['Cl']}, Re={case['Re']}, "
                        f"α={case['alpha']}°")

            designs = gen.generate(
                Cl=case['Cl'], Re=case['Re'], alpha=case['alpha'],
                n_designs=3, n_candidates=50,
                verify_xfoil=True  # Verify with YOUR XFOIL!
            )

            if designs:
                logger.info(f"  Generated {len(designs)} designs ✓")
                for d in designs:
                    logger.info(f"    {d}")
                    d.export_dat(
                        f"results/exported_airfoils/"
                        f"cl{case['Cl']}_design_{d.design_id}.dat"
                    )

                # Plot first design
                designs[0].plot(
                    save=(f"results/figures/"
                          f"generated_cl{case['Cl']}.png"),
                    show=False
                )
            else:
                logger.warning(f"  No valid designs for Cl={case['Cl']}")

    except Exception as e:
        logger.warning(f"  Generation test failed: {e}")
        import traceback
        traceback.print_exc()

    # ═══════════════════════════════════════
    # DONE
    # ═══════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("  TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"  Forward Model: {checkpoint_dir}/forward_model_best.pt")
    logger.info(f"  Generator:     {checkpoint_dir}/generator_best.pt")
    logger.info(f"  Scaler:        {checkpoint_dir}/scaler.pkl")
    logger.info(f"")
    logger.info(f"  To generate airfoils:")
    logger.info(f"    python scripts/generate_airfoil.py --cl 1.2 "
                f"--re 500000 --alpha 5 --plot")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()