"""
COMPREHENSIVE DIAGNOSTIC — Find the exact normalization bug
Run: python diagnose.py
"""

import sys
import os
import pickle
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.forward_model import ForwardModel
from src.utils.config import Config

print("=" * 70)
print("  DIAGNOSTIC: Finding the normalization bug")
print("=" * 70)

# ══════════════════════════════════════════
# 1. LOAD SCALER AND CHECK CONSISTENCY
# ══════════════════════════════════════════
print("\n\n[1] SCALER CONSISTENCY CHECK")
print("-" * 70)

with open('checkpoints/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

input_mean = np.array(scaler['input_mean'])
input_std = np.array(scaler['input_std'])
cst_mean = np.array(scaler['cst_mean'])
cst_std = np.array(scaler['cst_std'])
target_mean = np.array(scaler['target_mean'])
target_std = np.array(scaler['target_std'])
cond_mean = np.array(scaler['cond_mean'])
cond_std = np.array(scaler['cond_std'])

# The critical check: do CST normalizations match?
fwd_cst_mean = input_mean[:16]
fwd_cst_std = input_std[:16]

print(f"\n  Forward model CST mean (input_mean[:16]):")
print(f"    {np.round(fwd_cst_mean, 6)}")
print(f"\n  Generator CST mean (cst_mean):")
print(f"    {np.round(cst_mean, 6)}")
print(f"\n  DIFFERENCE (should be ~0):")
diff_mean = fwd_cst_mean - cst_mean
print(f"    {np.round(diff_mean, 6)}")
print(f"    Max absolute difference: {np.max(np.abs(diff_mean)):.8f}")

if np.max(np.abs(diff_mean)) > 0.001:
    print(f"\n  [X] CST MEANS DO NOT MATCH -- THIS IS THE BUG")
elif np.max(np.abs(fwd_cst_std - cst_std)) > 0.001:
    print(f"\n  [X] CST STDS DO NOT MATCH -- THIS IS THE BUG")
else:
    print(f"\n  [OK] CST statistics match -- bug is elsewhere")

print(f"\n\n[5] TESTING ON ACTUAL TRAINING DATA")
print("-" * 70)

try:
    config = Config("config.yaml")
    from src.data.preprocessing import load_dataset
    
    data = load_dataset(config.paths.processed_data)
    print(f"  Loaded {len(data['cl'])} samples")
    
    np.random.seed(42)
    indices = np.random.choice(len(data['cl']), 20, replace=False)
    
    forward_model = ForwardModel(input_dim=18)
    checkpoint = torch.load('checkpoints/forwardmodel_best.pt', map_location='cpu', weights_only=False)
    forward_model.load_state_dict(checkpoint['model_state_dict'])
    forward_model.eval()

    cd_errors = []
    
    for idx in indices:
        cst = data['cst_params'][idx]
        alpha = data['alpha'][idx]
        re = data['reynolds'][idx]
        true_cd = data['cd'][idx]
        
        fi = np.concatenate([cst, [alpha], [np.log10(re)]]).astype(np.float32)
        fi_norm = (fi - input_mean) / input_std
        
        with torch.no_grad():
            p = forward_model(torch.from_numpy(fi_norm).unsqueeze(0)).numpy()[0]
        p = p * target_std + target_mean
        
        pred_cd = 10 ** np.clip(p[1], -5, 0)
        cd_err = abs(pred_cd - true_cd) / max(true_cd, 0.001) * 100
        cd_errors.append(cd_err)
        
    mean_cd_err = np.mean(cd_errors)
    print(f"\n  Mean Cd error on REAL data: {mean_cd_err:.1f}%")
except Exception as e:
    print(f"Error: {e}")
