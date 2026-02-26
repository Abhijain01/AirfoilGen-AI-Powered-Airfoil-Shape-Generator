"""
Training pipeline for Forward Model and CVAE Generator.
FIXED: Handles tensor dimensions correctly throughout.
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import numpy as np

from src.utils.helpers import EarlyStopping


class ForwardModelTrainer:
    """Trainer for the Forward Model."""
    
    def __init__(self, model, loss_fn, device='cpu', config=None):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        
        lr = 0.001
        wd = 0.0001
        if config is not None:
            try:
                lr = config.forward_model.training.learning_rate
                wd = config.forward_model.training.weight_decay
            except Exception:
                pass
        
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    def train(self, train_loader, val_loader, max_epochs=300,
              patience=30, checkpoint_dir='checkpoints'):
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.optimizer.param_groups[0]['lr'],
            epochs=max_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        history = {
            'train_loss': [], 'val_loss': [],
            'val_cl_loss': [], 'val_cd_loss': [], 'val_cm_loss': [],
            'lr': []
        }
        
        model_path = os.path.join(checkpoint_dir, 'forward_model_best.pt')
        
        print(f"\n{'='*60}")
        print(f"  TRAINING FORWARD MODEL")
        print(f"  Device: {self.device}")
        print(f"  Max epochs: {max_epochs}, Patience: {patience}")
        print(f"{'='*60}\n")
        
        for epoch in range(max_epochs):
            # Train
            self.model.train()
            train_losses = []
            
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                preds = self.model(inputs)
                loss, _ = self.loss_fn(preds, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                scheduler.step()
                
                train_losses.append(loss.item())
            
            # Validate
            self.model.eval()
            val_losses = []
            val_cl, val_cd, val_cm = [], [], []
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    preds = self.model(inputs)
                    loss, loss_dict = self.loss_fn(preds, targets)
                    
                    val_losses.append(loss.item())
                    val_cl.append(loss_dict['cl_loss'])
                    val_cd.append(loss_dict['cd_loss'])
                    val_cm.append(loss_dict['cm_loss'])
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_cl_loss'].append(np.mean(val_cl))
            history['val_cd_loss'].append(np.mean(val_cd))
            history['val_cm_loss'].append(np.mean(val_cm))
            history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                }, model_path)
                print(f"Epoch {epoch+1}/{max_epochs} | "
                      f"Train: {train_loss:.6f} | Val: {val_loss:.6f} ← BEST")
            else:
                patience_counter += 1
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{max_epochs} | "
                          f"Train: {train_loss:.6f} | Val: {val_loss:.6f}")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Load best
        checkpoint = torch.load(model_path, map_location=self.device,
                                weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n✓ Best model loaded (val_loss = {best_val_loss:.6f})")
        
        return history


class CVAETrainer:
    """Trainer for the CVAE Generator."""
    
    def __init__(self, cvae_model, loss_fn, forward_model=None,
                 device='cpu', config=None):
        self.model = cvae_model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        
        if forward_model is not None:
            self.forward_model = forward_model.to(device)
            self.forward_model.eval()
            for param in self.forward_model.parameters():
                param.requires_grad = False
        else:
            self.forward_model = None
        
        lr = 0.0005
        wd = 0.0001
        if config is not None:
            try:
                lr = config.generator.training.learning_rate
                wd = config.generator.training.weight_decay
            except Exception:
                pass
        
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
    
    def train(self, train_loader, val_loader, max_epochs=500,
              patience=50, kl_warmup_epochs=50,
              checkpoint_dir='checkpoints'):
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        scheduler = CosineAnnealingLR(self.optimizer, T_max=max_epochs, eta_min=1e-6)
        
        best_val_loss = float('inf')
        patience_counter = 0
        target_kl_weight = self.loss_fn.kl_weight
        
        history = {
            'train_loss': [], 'val_loss': [],
            'recon_loss': [], 'kl_loss': [],
            'perf_loss': [], 'physics_loss': [], 'lr': []
        }
        
        model_path = os.path.join(checkpoint_dir, 'generator_best.pt')
        
        print(f"\n{'='*60}")
        print(f"  TRAINING CVAE GENERATOR")
        print(f"  Device: {self.device}")
        print(f"  Max epochs: {max_epochs}, KL warmup: {kl_warmup_epochs}")
        print(f"  Forward model: {'YES' if self.forward_model else 'NO'}")
        print(f"{'='*60}\n")
        
        for epoch in range(max_epochs):
            # KL warmup
            if epoch < kl_warmup_epochs:
                self.loss_fn.kl_weight = target_kl_weight * (epoch / kl_warmup_epochs)
            else:
                self.loss_fn.kl_weight = target_kl_weight
            
            # ─── Train ───
            self.model.train()
            train_losses = []
            epoch_metrics = {'recon': [], 'kl': [], 'perf': [], 'physics': []}
            
            for batch in train_loader:
                # DataLoader returns (conditions, cst_targets)
                conditions = batch[0].to(self.device)
                cst_targets = batch[1].to(self.device)
                
                # SAFETY: Ensure 2D
                if conditions.dim() == 3:
                    conditions = conditions.squeeze(1)
                if cst_targets.dim() == 3:
                    cst_targets = cst_targets.squeeze(1)
                
                self.optimizer.zero_grad()
                
                # Forward through CVAE
                cst_recon, mean, logvar = self.model(cst_targets, conditions)
                
                # Compute loss
                loss, loss_dict = self.loss_fn(
                    cst_recon, cst_targets, mean, logvar,
                    forward_model=self.forward_model,
                    conditions=conditions
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_losses.append(loss.item())
                epoch_metrics['recon'].append(loss_dict['recon_loss'])
                epoch_metrics['kl'].append(loss_dict['kl_loss'])
                epoch_metrics['perf'].append(loss_dict['perf_loss'])
                epoch_metrics['physics'].append(loss_dict['physics_loss'])
            
            scheduler.step()
            
            # ─── Validate ───
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    conditions = batch[0].to(self.device)
                    cst_targets = batch[1].to(self.device)
                    
                    if conditions.dim() == 3:
                        conditions = conditions.squeeze(1)
                    if cst_targets.dim() == 3:
                        cst_targets = cst_targets.squeeze(1)
                    
                    cst_recon, mean, logvar = self.model(cst_targets, conditions)
                    loss, _ = self.loss_fn(
                        cst_recon, cst_targets, mean, logvar,
                        forward_model=self.forward_model,
                        conditions=conditions
                    )
                    val_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['recon_loss'].append(np.mean(epoch_metrics['recon']))
            history['kl_loss'].append(np.mean(epoch_metrics['kl']))
            history['perf_loss'].append(np.mean(epoch_metrics['perf']))
            history['physics_loss'].append(np.mean(epoch_metrics['physics']))
            history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{max_epochs} | "
                      f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
                      f"Recon: {np.mean(epoch_metrics['recon']):.5f} | "
                      f"KL: {np.mean(epoch_metrics['kl']):.4f} | "
                      f"KL_w: {self.loss_fn.kl_weight:.5f}")
            
            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                }, model_path)
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Load best
        checkpoint = torch.load(model_path, map_location=self.device,
                                weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n✓ Best CVAE loaded (val_loss = {best_val_loss:.6f})")
        
        return history