
import torch
import numpy as np
import os
import sys
from sklearn.metrics import r2_score, mean_absolute_error

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.models.forward_model import ForwardModel
from src.data.dataset import ForwardDataset, create_dataloaders
from src.data.preprocessing import load_dataset
from src.utils.helpers import get_device

def evaluate():
    device = get_device()
    print(f"Device: {device}")

    # Load Data
    data_path = os.path.join(project_root, 'data', 'processed')
    print(f"Loading data from {data_path}...")
    data = load_dataset(data_path)
    
    # Filter
    valid = data['cd'] > 0
    for key in data:
        if isinstance(data[key], np.ndarray) and len(data[key]) == len(valid):
            data[key] = data[key][valid]

    # Create Test Dataset (same logic as notebook)
    # We need the scaler from training data first
    train_fwd = ForwardDataset(
        cst_params=data['cst_params'][data['train_mask']],
        alpha=data['alpha'][data['train_mask']],
        reynolds=data['reynolds'][data['train_mask']],
        cl=data['cl'][data['train_mask']],
        cd=data['cd'][data['train_mask']],
        cm=data['cm'][data['train_mask']],
        normalize=True, scaler=None
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

    test_loader = torch.utils.data.DataLoader(test_fwd, batch_size=1024, shuffle=False)

    # Load Model
    model = ForwardModel(input_dim=18, hidden_dims=[256, 512, 256, 128], dropout=0.1)
    checkpoint_path = os.path.join(project_root, 'checkpoints', 'forwardmodel_best.pt')
    
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found!")
        return

    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Predict
    all_preds = []
    all_targets = []

    print("Evaluating...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            preds = model(inputs).cpu()
            all_preds.append(preds)
            all_targets.append(targets)

    preds_np = torch.cat(all_preds).numpy()
    targets_np = torch.cat(all_targets).numpy()

    # Metrics
    names = ['Cl', 'log10(Cd)', 'Cm']
    print("\n" + "="*30)
    print("MODEL ACCURACY (Test Set)")
    print("="*30)
    
    metrics = {}
    for i, name in enumerate(names):
        r2 = r2_score(targets_np[:, i], preds_np[:, i])
        mae = mean_absolute_error(targets_np[:, i], preds_np[:, i])
        metrics[name] = {'R2': r2, 'MAE': mae}
        print(f"{name:10s} | R² = {r2:.4f} | MAE = {mae:.5f}")
    print("="*30)

if __name__ == "__main__":
    evaluate()
