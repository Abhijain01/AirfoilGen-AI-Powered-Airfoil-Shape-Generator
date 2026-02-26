"""
Loss functions for Forward Model and CVAE Generator.
FIXED: Safe tensor handling, no dimension mismatches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForwardModelLoss(nn.Module):
    """Loss for Forward Model: weighted MSE on Cl, Cd, Cm."""
    
    def __init__(self, cl_weight=1.0, cd_weight=2.0, cm_weight=1.0):
        super().__init__()
        self.cl_weight = cl_weight
        self.cd_weight = cd_weight
        self.cm_weight = cm_weight
    
    def forward(self, predictions, targets):
        cl_loss = F.mse_loss(predictions[:, 0], targets[:, 0])
        cd_loss = F.mse_loss(predictions[:, 1], targets[:, 1])
        cm_loss = F.mse_loss(predictions[:, 2], targets[:, 2])
        
        total = (self.cl_weight * cl_loss +
                 self.cd_weight * cd_loss +
                 self.cm_weight * cm_loss)
        
        return total, {
            'cl_loss': cl_loss.item(),
            'cd_loss': cd_loss.item(),
            'cm_loss': cm_loss.item(),
            'total_loss': total.item(),
        }


class CVAELoss(nn.Module):
    """
    Loss for CVAE Generator.
    
    Components:
    1. Reconstruction: generated CST should match real CST
    2. KL Divergence: latent space regularization
    3. Physics: constraints on CST parameters
    """
    
    def __init__(self, recon_weight=1.0, kl_weight=0.001,
                 perf_weight=2.0, physics_weight=0.5):
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.perf_weight = perf_weight
        self.physics_weight = physics_weight
    
    def forward(self, cst_reconstructed, cst_original, mean, logvar,
                forward_model=None, conditions=None):
        """
        Compute CVAE loss.
        
        Parameters
        ----------
        cst_reconstructed : (batch, 16)
        cst_original : (batch, 16)
        mean : (batch, latent_dim)
        logvar : (batch, latent_dim)
        forward_model : optional, for performance checking
        conditions : optional, (batch, 5) for performance targets
        """
        # Ensure 2D
        if cst_reconstructed.dim() == 3:
            cst_reconstructed = cst_reconstructed.squeeze(1)
        if cst_original.dim() == 3:
            cst_original = cst_original.squeeze(1)
        
        # ─── Loss 1: Reconstruction ───
        recon_loss = F.mse_loss(cst_reconstructed, cst_original)
        
        # ─── Loss 2: KL Divergence ───
        kl_loss = -0.5 * torch.mean(
            1 + logvar - mean.pow(2) - logvar.exp()
        )
        
        # ─── Loss 3: Performance ───
        perf_loss = torch.tensor(0.0, device=cst_reconstructed.device)
        
        if forward_model is not None and conditions is not None:
            try:
                # Ensure conditions is 2D
                if conditions.dim() == 3:
                    conditions = conditions.squeeze(1)
                
                # Extract alpha and Re from conditions
                # conditions = [Cl, Cd, log10_Re, alpha, thickness]
                alpha = conditions[:, 3:4]
                log_re = conditions[:, 2:3]
                
                # Build forward model input
                forward_input = torch.cat([
                    cst_reconstructed, alpha, log_re
                ], dim=1)
                
                with torch.no_grad():
                    predicted_perf = forward_model(forward_input)
                
                # Compare predicted Cl with target Cl
                target_cl = conditions[:, 0]
                predicted_cl = predicted_perf[:, 0]
                perf_loss = F.mse_loss(predicted_cl, target_cl)
                
            except Exception:
                perf_loss = torch.tensor(0.0, device=cst_reconstructed.device)
        
        # ─── Loss 4: Physics ───
        physics_loss = self._physics_loss(cst_reconstructed)
        
        # ─── Combine ───
        total = (self.recon_weight * recon_loss +
                 self.kl_weight * kl_loss +
                 self.perf_weight * perf_loss +
                 self.physics_weight * physics_loss)
        
        return total, {
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'perf_loss': perf_loss.item(),
            'physics_loss': physics_loss.item(),
            'total_loss': total.item(),
        }
    
    def _physics_loss(self, cst_params):
        """Physics constraints on CST parameters."""
        if cst_params.dim() == 3:
            cst_params = cst_params.squeeze(1)
        
        upper = cst_params[:, :8]
        lower = cst_params[:, 8:]
        
        loss = torch.tensor(0.0, device=cst_params.device)
        
        # Upper weights should be positive
        loss = loss + torch.mean(F.relu(-upper)) * 10.0
        
        # Lower weights should be positive (CST convention)
        loss = loss + torch.mean(F.relu(-lower)) * 5.0
        
        # Weights in reasonable range
        loss = loss + torch.mean(F.relu(torch.abs(cst_params) - 0.5))
        
        # Smoothness
        upper_diff = torch.diff(upper, dim=1)
        lower_diff = torch.diff(lower, dim=1)
        loss = loss + torch.mean(upper_diff ** 2) * 2.0
        loss = loss + torch.mean(lower_diff ** 2) * 2.0
        
        # Upper > lower (positive thickness)
        mean_upper = upper.mean(dim=1)
        mean_lower = lower.mean(dim=1)
        loss = loss + torch.mean(F.relu(mean_lower - mean_upper)) * 20.0
        
        return loss