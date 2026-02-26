"""
PyTorch Dataset classes for airfoil generator training.

Two datasets:
  1. ForwardDataset: (CST + conditions) → (Cl, Cd, Cm)
     Used to train the performance predictor

  2. GeneratorDataset: (Cl, Cd, Re, α, t/c) → (CST parameters)
     Used to train the CVAE generator
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import os


class ForwardDataset(Dataset):
    """
    Dataset for Forward Model training.
    
    Input:  CST parameters (16) + alpha (1) + log10(Re) (1) = 18 features
    Target: Cl (1) + log10(Cd) (1) + Cm (1) = 3 outputs
    
    The forward model predicts: given a shape + conditions → performance
    """
    
    def __init__(self, cst_params, alpha, reynolds, cl, cd, cm,
                 normalize=True, scaler=None):
        """
        Parameters
        ----------
        cst_params : numpy array, shape (N, 16)
            CST parameters for each data point
        alpha : numpy array, shape (N,)
            Angle of attack in degrees
        reynolds : numpy array, shape (N,)
            Reynolds number
        cl : numpy array, shape (N,)
            Lift coefficient
        cd : numpy array, shape (N,)
            Drag coefficient
        cm : numpy array, shape (N,)
            Moment coefficient
        normalize : bool
            Whether to normalize data
        scaler : dict or None
            Pre-computed normalization parameters (for val/test sets)
        """
        super().__init__()
        
        # Convert to float32
        self.cst_params = np.asarray(cst_params, dtype=np.float32)
        self.alpha = np.asarray(alpha, dtype=np.float32).reshape(-1, 1)
        self.log_re = np.log10(np.asarray(reynolds, dtype=np.float32)).reshape(-1, 1)
        self.cl = np.asarray(cl, dtype=np.float32).reshape(-1, 1)
        self.log_cd = np.log10(np.clip(np.asarray(cd, dtype=np.float32), 1e-6, None)).reshape(-1, 1)
        self.cm = np.asarray(cm, dtype=np.float32).reshape(-1, 1)
        
        # Combine inputs: [CST(16), alpha(1), log10_Re(1)] = 18 features
        self.inputs = np.hstack([self.cst_params, self.alpha, self.log_re])
        
        # Combine targets: [Cl, log10(Cd), Cm] = 3 outputs
        self.targets = np.hstack([self.cl, self.log_cd, self.cm])
        
        # Normalize
        self.normalize = normalize
        if normalize:
            if scaler is None:
                # Compute normalization from THIS data (training set)
                self.scaler = {
                    'input_mean': self.inputs.mean(axis=0),
                    'input_std': self.inputs.std(axis=0) + 1e-8,
                    'target_mean': self.targets.mean(axis=0),
                    'target_std': self.targets.std(axis=0) + 1e-8,
                }
            else:
                # Use pre-computed normalization (validation/test set)
                self.scaler = scaler
            
            self.inputs = (self.inputs - self.scaler['input_mean']) / self.scaler['input_std']
            self.targets = (self.targets - self.scaler['target_mean']) / self.scaler['target_std']
        else:
            self.scaler = None
        
        # Convert to tensors
        self.inputs = torch.from_numpy(self.inputs)
        self.targets = torch.from_numpy(self.targets)
        
        print(f"[DATASET] ForwardDataset: {len(self)} samples, "
              f"input_dim={self.inputs.shape[1]}, target_dim={self.targets.shape[1]}")
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
    
    def denormalize_targets(self, normalized_targets):
        """Convert normalized predictions back to physical values"""
        if self.scaler is None:
            return normalized_targets
        
        if isinstance(normalized_targets, torch.Tensor):
            mean = torch.tensor(self.scaler['target_mean'], device=normalized_targets.device)
            std = torch.tensor(self.scaler['target_std'], device=normalized_targets.device)
        else:
            mean = self.scaler['target_mean']
            std = self.scaler['target_std']
        
        return normalized_targets * std + mean


class GeneratorDataset(Dataset):
    """
    Dataset for CVAE Generator training.
    
    Input (conditions):  Cl (1) + Cd (1) + log10(Re) (1) + alpha (1) + t/c (1) = 5
    Target (shape):      CST parameters (16)
    
    The generator learns: given desired performance → produce shape parameters
    """
    
    def __init__(self, cst_params, cl, cd, reynolds, alpha, thickness,
                 normalize=True, scaler=None):
        """
        Parameters
        ----------
        cst_params : numpy array, shape (N, 16)
            CST parameters (this is what the generator should OUTPUT)
        cl : numpy array, shape (N,)
            Lift coefficient
        cd : numpy array, shape (N,)
            Drag coefficient
        reynolds : numpy array, shape (N,)
            Reynolds number
        alpha : numpy array, shape (N,)
            Angle of attack in degrees
        thickness : numpy array, shape (N,)
            Maximum thickness ratio
        normalize : bool
            Whether to normalize
        scaler : dict or None
            Pre-computed normalization parameters
        """
        super().__init__()
        
        # Convert to float32
        self.cst_params = np.asarray(cst_params, dtype=np.float32)
        self.cl = np.asarray(cl, dtype=np.float32).reshape(-1, 1)
        self.cd = np.asarray(cd, dtype=np.float32).reshape(-1, 1)
        self.log_re = np.log10(np.asarray(reynolds, dtype=np.float32)).reshape(-1, 1)
        self.alpha = np.asarray(alpha, dtype=np.float32).reshape(-1, 1)
        self.thickness = np.asarray(thickness, dtype=np.float32).reshape(-1, 1)
        
        # Conditions: [Cl, Cd, log10_Re, alpha, t/c] = 5 features
        self.conditions = np.hstack([
            self.cl, self.cd, self.log_re, self.alpha, self.thickness
        ])
        
        # Target: CST parameters (16)
        self.targets = self.cst_params.copy()
        
        # Normalize
        self.normalize = normalize
        if normalize:
            if scaler is None:
                self.scaler = {
                    'cond_mean': self.conditions.mean(axis=0),
                    'cond_std': self.conditions.std(axis=0) + 1e-8,
                    'cst_mean': self.targets.mean(axis=0),
                    'cst_std': self.targets.std(axis=0) + 1e-8,
                }
            else:
                self.scaler = scaler
            
            self.conditions = (self.conditions - self.scaler['cond_mean']) / self.scaler['cond_std']
            self.targets = (self.targets - self.scaler['cst_mean']) / self.scaler['cst_std']
        else:
            self.scaler = None
        
        # Convert to tensors
        self.conditions = torch.from_numpy(self.conditions)
        self.targets = torch.from_numpy(self.targets)
        
        print(f"[DATASET] GeneratorDataset: {len(self)} samples, "
              f"condition_dim={self.conditions.shape[1]}, "
              f"target_dim={self.targets.shape[1]}")
    
    def __len__(self):
        return len(self.conditions)
    
    def __getitem__(self, idx):
        return self.conditions[idx], self.targets[idx]
    
    def denormalize_cst(self, normalized_cst):
        """Convert normalized CST parameters back to actual values"""
        if self.scaler is None:
            return normalized_cst
        
        if isinstance(normalized_cst, torch.Tensor):
            mean = torch.tensor(self.scaler['cst_mean'], device=normalized_cst.device)
            std = torch.tensor(self.scaler['cst_std'], device=normalized_cst.device)
        else:
            mean = self.scaler['cst_mean']
            std = self.scaler['cst_std']
        
        return normalized_cst * std + mean
    
    def denormalize_conditions(self, normalized_cond):
        """Convert normalized conditions back to actual values"""
        if self.scaler is None:
            return normalized_cond
        
        if isinstance(normalized_cond, torch.Tensor):
            mean = torch.tensor(self.scaler['cond_mean'], device=normalized_cond.device)
            std = torch.tensor(self.scaler['cond_std'], device=normalized_cond.device)
        else:
            mean = self.scaler['cond_mean']
            std = self.scaler['cond_std']
        
        return normalized_cond * std + mean


def create_dataloaders(train_dataset, val_dataset, test_dataset,
                       batch_size=256, num_workers=0):
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    
    Parameters
    ----------
    train_dataset, val_dataset, test_dataset : Dataset
        PyTorch datasets
    batch_size : int
        Batch size (default 256)
    num_workers : int
        Number of data loading workers (0 for Windows compatibility)
    
    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    print(f"[DATALOADER] Train: {len(train_loader)} batches, "
          f"Val: {len(val_loader)} batches, "
          f"Test: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader