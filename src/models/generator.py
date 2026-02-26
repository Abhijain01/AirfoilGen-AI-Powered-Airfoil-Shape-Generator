"""
CVAE (Conditional Variational Autoencoder) — Airfoil Shape Generator

FIXED: Handles tensor dimension mismatches safely.

INPUT:  Desired performance conditions (Cl, Cd, Re, α, thickness)
OUTPUT: CST parameters (16 numbers) → convert to (x,y) coordinates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CVAE(nn.Module):
    """
    Complete Conditional Variational Autoencoder for airfoil generation.
    
    TRAINING:
      real_cst + conditions → Encoder → μ, σ → z → Decoder → reconstructed_cst
    
    GENERATION:
      random_z + desired_conditions → Decoder → new_cst → (x,y) coordinates
    """
    
    def __init__(self, n_cst=16, condition_dim=5, latent_dim=32,
                 encoder_hidden=None, decoder_hidden=None):
        super().__init__()
        
        self.n_cst = n_cst
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        
        # Default hidden dims
        if encoder_hidden is None:
            encoder_hidden = [256, 256, 128]
        if decoder_hidden is None:
            decoder_hidden = [256, 512, 512, 256, 128]
        
        # ═══════════════════════════════════════
        # ENCODER
        # ═══════════════════════════════════════
        # Input: [cst(n_cst) + conditions(condition_dim)]
        encoder_input_dim = n_cst + condition_dim
        
        encoder_layers = []
        prev_dim = encoder_input_dim
        for h_dim in encoder_hidden:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.SiLU())
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            prev_dim = h_dim
        
        self.encoder_backbone = nn.Sequential(*encoder_layers)
        self.fc_mean = nn.Linear(encoder_hidden[-1], latent_dim)
        self.fc_logvar = nn.Linear(encoder_hidden[-1], latent_dim)
        
        # Initialize logvar with small values for stable training
        nn.init.xavier_uniform_(self.fc_mean.weight)
        nn.init.zeros_(self.fc_mean.bias)
        nn.init.xavier_uniform_(self.fc_logvar.weight, gain=0.1)
        nn.init.constant_(self.fc_logvar.bias, -2.0)
        
        # ═══════════════════════════════════════
        # DECODER
        # ═══════════════════════════════════════
        # Input: [z(latent_dim) + conditions(condition_dim)]
        decoder_input_dim = latent_dim + condition_dim
        
        decoder_layers = []
        prev_dim = decoder_input_dim
        for h_dim in decoder_hidden:
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.SiLU())
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            prev_dim = h_dim
        
        decoder_layers.append(nn.Linear(decoder_hidden[-1], n_cst))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize decoder
        self._init_decoder_weights()
    
    def _init_decoder_weights(self):
        """Initialize decoder with small weights"""
        for module in self.decoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _ensure_2d(self, tensor):
        """
        Ensure tensor is 2D (batch, features).
        Fixes the dimension mismatch error.
        """
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 3:
            # (batch, 1, features) → (batch, features)
            tensor = tensor.squeeze(1)
        elif tensor.dim() > 3:
            # Flatten everything except batch
            tensor = tensor.view(tensor.size(0), -1)
        return tensor
    
    def encode(self, cst_params, conditions):
        """
        Encode real airfoil + conditions into latent distribution.
        
        Parameters
        ----------
        cst_params : torch.Tensor, shape (batch, n_cst)
            Real CST parameters
        conditions : torch.Tensor, shape (batch, condition_dim)
            Performance conditions [Cl, Cd, log10_Re, alpha, thickness]
        
        Returns
        -------
        mean : torch.Tensor, shape (batch, latent_dim)
        logvar : torch.Tensor, shape (batch, latent_dim)
        """
        # SAFETY: Ensure both are 2D
        cst_params = self._ensure_2d(cst_params)
        conditions = self._ensure_2d(conditions)
        
        # Concatenate
        inputs = torch.cat([cst_params, conditions], dim=1)
        
        # Forward through encoder
        h = self.encoder_backbone(inputs)
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10.0, max=2.0)
        
        return mean, logvar
    
    def decode(self, z, conditions):
        """
        Decode latent code + conditions into CST parameters.
        
        Parameters
        ----------
        z : torch.Tensor, shape (batch, latent_dim)
            Latent code
        conditions : torch.Tensor, shape (batch, condition_dim)
            Performance conditions
        
        Returns
        -------
        cst_params : torch.Tensor, shape (batch, n_cst)
            Generated CST parameters
        """
        # SAFETY: Ensure both are 2D
        z = self._ensure_2d(z)
        conditions = self._ensure_2d(conditions)
        
        # Concatenate and decode
        inputs = torch.cat([z, conditions], dim=1)
        cst_params = self.decoder(inputs)
        
        return cst_params
    
    def reparameterize(self, mean, logvar):
        """
        Reparameterization trick: z = μ + σ × ε, where ε ~ N(0,1)
        
        This allows gradients to flow through the sampling operation.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z
    
    def forward(self, cst_params, conditions):
        """
        Training forward pass.
        
        Parameters
        ----------
        cst_params : torch.Tensor
            Real CST parameters (target shapes)
        conditions : torch.Tensor
            Performance conditions
        
        Returns
        -------
        cst_reconstructed : torch.Tensor
        mean : torch.Tensor
        logvar : torch.Tensor
        """
        # Ensure correct shapes
        cst_params = self._ensure_2d(cst_params)
        conditions = self._ensure_2d(conditions)
        
        # Encode
        mean, logvar = self.encode(cst_params, conditions)
        
        # Sample latent
        z = self.reparameterize(mean, logvar)
        
        # Decode
        cst_reconstructed = self.decode(z, conditions)
        
        return cst_reconstructed, mean, logvar
    
    def generate(self, conditions, n_samples=1, device='cpu'):
        """
        Generate NEW airfoil shapes from desired conditions.
        
        Parameters
        ----------
        conditions : torch.Tensor, shape (condition_dim,) or (1, condition_dim)
            Desired conditions [Cl, Cd, log10_Re, alpha, thickness]
        n_samples : int
            Number of different designs to generate
        device : str or torch.device
            Device to use
        
        Returns
        -------
        cst_params : torch.Tensor, shape (n_samples, n_cst)
        """
        self.eval()
        
        with torch.no_grad():
            # Ensure conditions is 2D
            if isinstance(conditions, np.ndarray):
                conditions = torch.from_numpy(conditions.astype(np.float32))
            
            conditions = self._ensure_2d(conditions).to(device)
            
            # Expand conditions for all samples
            if conditions.size(0) == 1:
                conditions = conditions.expand(n_samples, -1)
            elif conditions.size(0) != n_samples:
                conditions = conditions[:1].expand(n_samples, -1)
            
            # Sample random latent codes from N(0, 1)
            z = torch.randn(n_samples, self.latent_dim, device=device)
            
            # Decode to CST parameters
            cst_params = self.decode(z, conditions)
        
        return cst_params
    
    def generate_diverse(self, conditions, n_samples=50, n_select=10,
                         device='cpu'):
        """
        Generate diverse designs using farthest-point sampling.
        """
        all_cst = self.generate(conditions, n_samples=n_samples, device=device)
        
        if n_samples <= n_select:
            return all_cst
        
        # Farthest-point sampling for diversity
        selected_indices = [0]
        
        for _ in range(n_select - 1):
            selected = all_cst[selected_indices]
            distances = torch.cdist(all_cst, selected).min(dim=1)[0]
            distances[selected_indices] = -1  # exclude already selected
            next_idx = distances.argmax().item()
            selected_indices.append(next_idx)
        
        return all_cst[selected_indices]