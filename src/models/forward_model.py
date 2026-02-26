"""
Forward Model — Predicts aerodynamic performance from airfoil shape.

INPUT:  CST parameters (16) + alpha (1) + log10(Re) (1) = 18 features
OUTPUT: Cl (1) + log10(Cd) (1) + Cm (1) = 3 targets

No changes from previous version except added numerical safety.
"""

import torch
import torch.nn as nn
import numpy as np


class ForwardModel(nn.Module):
    """
    Predicts Cl, Cd, Cm from CST parameters + flow conditions.
    """

    def __init__(self, input_dim=18, hidden_dims=None, dropout=0.1):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 512, 256, 128]

        # Shared Backbone
        layers = []
        prev_dim = input_dim

        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.SiLU())
            layers.append(nn.BatchNorm1d(h_dim))
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)

        backbone_out_dim = hidden_dims[-1]

        if input_dim != backbone_out_dim:
            self.residual_proj = nn.Linear(input_dim, backbone_out_dim)
        else:
            self.residual_proj = nn.Identity()

        # Cl Head
        self.cl_head = nn.Sequential(
            nn.Linear(backbone_out_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

        # Cd Head
        self.cd_head = nn.Sequential(
            nn.Linear(backbone_out_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

        # Cm Head
        self.cm_head = nn.Sequential(
            nn.Linear(backbone_out_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, 18)

        Returns
        -------
        output : torch.Tensor, shape (batch_size, 3)
            [Cl, log10(Cd), Cm]
        """
        features = self.backbone(x)
        residual = self.residual_proj(x)
        features = features + residual

        cl = self.cl_head(features)
        cd = self.cd_head(features)
        cm = self.cm_head(features)

        output = torch.cat([cl, cd, cm], dim=1)
        return output

    def predict_physical(self, cst_params, alpha_deg, reynolds, scaler=None):
        """
        User-friendly prediction in physical units.
        Returns dict with Cl, Cd, Cm values.
        """
        self.eval()

        cst_params = np.atleast_2d(cst_params).astype(np.float32)
        alpha = np.atleast_1d(alpha_deg).astype(np.float32).reshape(-1, 1)
        log_re = np.log10(np.atleast_1d(reynolds).astype(np.float32)).reshape(-1, 1)

        n = max(len(cst_params), len(alpha), len(log_re))
        if len(cst_params) == 1:
            cst_params = np.repeat(cst_params, n, axis=0)
        if len(alpha) == 1:
            alpha = np.repeat(alpha, n, axis=0)
        if len(log_re) == 1:
            log_re = np.repeat(log_re, n, axis=0)

        x = np.hstack([cst_params, alpha, log_re])

        if scaler is not None:
            x = (x - scaler['input_mean']) / scaler['input_std']

        x_tensor = torch.from_numpy(x)

        with torch.no_grad():
            output = self.forward(x_tensor).numpy()

        if scaler is not None:
            output = output * scaler['target_std'] + scaler['target_mean']

        # SAFETY: Clamp Cd to prevent negative values after denormalization
        cd_values = 10.0 ** np.clip(output[:, 1], -5, 0)

        return {
            'Cl': output[:, 0],
            'Cd': cd_values,
            'Cm': output[:, 2],
        }