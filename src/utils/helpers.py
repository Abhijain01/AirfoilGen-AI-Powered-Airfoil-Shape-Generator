"""
General helper functions used across the project
"""
import os
import random
import numpy as np
import torch
import json
from datetime import datetime


def set_seed(seed=42):
    """
    Set random seed for reproducibility EVERYWHERE.
    The 1% always ensure reproducibility.

    Call this at the start of every script.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[SEED] All random seeds set to {seed}")


def get_device():
    """Get best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[DEVICE] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"[DEVICE] Using CPU")
    return device


def count_parameters(model):
    """Count trainable parameters in a model"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Total parameters: {total:,}")
    print(f"[MODEL] Trainable parameters: {trainable:,}")
    return trainable


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
    }
    torch.save(checkpoint, path)
    print(f"[CHECKPOINT] Saved to {path} (epoch {epoch}, loss {loss:.6f})")


def load_checkpoint(model, optimizer, path):
    """Load model checkpoint"""
    if not os.path.exists(path):
        print(f"[CHECKPOINT] No checkpoint found at {path}")
        return 0, float('inf')

    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"[CHECKPOINT] Loaded from {path} (epoch {epoch}, loss {loss:.6f})")
    return epoch, loss


def save_results(results, path):
    """Save results dictionary to JSON"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean_results = {k: convert(v) for k, v in results.items()}

    with open(path, 'w') as f:
        json.dump(clean_results, f, indent=2)
    print(f"[RESULTS] Saved to {path}")


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Stops training when validation loss doesn't improve for 'patience' epochs.
    """
    def __init__(self, patience=30, min_delta=1e-5, path='checkpoints/best_model.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def __call__(self, val_loss, model, optimizer, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss, self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"[EARLY STOPPING] No improvement for {self.patience} epochs. "
                      f"Best loss: {self.best_loss:.6f}")

    def reset(self):
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False