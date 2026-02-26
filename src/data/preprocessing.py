"""
Complete data preprocessing pipeline.

Steps:
  1. Generate/collect airfoil shapes
  2. Fit CST parameters
  3. Run XFOIL for performance data
  4. Create training pairs
  5. Split into train/val/test
  6. Save everything
"""

import numpy as np
import pandas as pd
import h5py
import os
import pickle
from tqdm import tqdm
from sklearn.model_selection import GroupKFold

from src.geometry.cst import (
    cst_to_coordinates,
    coordinates_to_cst,
    compute_airfoil_properties,
    generate_random_cst,
    generate_x_cosine,
    validate_airfoil
)
from src.geometry.naca import generate_naca_family
from src.data.xfoil_runner import batch_analyze
from src.utils.helpers import set_seed


def run_full_pipeline(config, output_dir="data/processed"):
    """
    Run the COMPLETE data generation and processing pipeline.
    
    This is the main function that creates ALL training data.
    
    Parameters
    ----------
    config : Config
        Configuration object (from config.yaml)
    output_dir : str
        Where to save processed data
    
    Returns
    -------
    data : dict
        Complete dataset ready for model training
    """
    set_seed(config.project.random_seed)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("  AIRFOIL DATA PIPELINE")
    print("=" * 60)
    
    # ═══════════════════════════════════════
    # STEP 1: Generate airfoil shapes
    # ═══════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STEP 1: Generate Airfoil Shapes")
    print("=" * 60)
    
    all_airfoils = []
    
    # 1a. NACA 4-digit family
    print("\n[STEP 1a] Generating NACA 4-digit airfoils...")
    naca_airfoils = generate_naca_family(n_points=100)
    all_airfoils.extend(naca_airfoils)
    print(f"  Generated: {len(naca_airfoils)} NACA airfoils")
    
    # 1b. Random CST airfoils
    print("\n[STEP 1b] Generating random CST airfoils...")
    n_random = config.data.sources.n_random_cst
    random_cst_params, valid_mask = generate_random_cst(
        n_airfoils=n_random,
        n_weights=config.data.n_cst_upper,
        seed=config.project.random_seed
    )
    
    x = generate_x_cosine(100)
    for i in range(len(random_cst_params)):
        cst_u = random_cst_params[i, :8]
        cst_l = random_cst_params[i, 8:]
        _, _, _, y_upper, _, y_lower = cst_to_coordinates(cst_u, cst_l, 100)
        
        all_airfoils.append({
            'name': f'RandomCST_{i:04d}',
            'x': x,
            'y_upper': y_upper,
            'y_lower': y_lower,
        })
    
    print(f"  Generated: {len(random_cst_params)} random CST airfoils")
    print(f"  Total airfoils: {len(all_airfoils)}")
    
    # ═══════════════════════════════════════
    # STEP 2: Fit CST parameters to ALL airfoils
    # ═══════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STEP 2: Fit CST Parameters")
    print("=" * 60)
    
    n_weights = config.data.n_cst_upper
    cst_data = []
    valid_airfoils = []
    
    for airfoil in tqdm(all_airfoils, desc="Fitting CST"):
        try:
            cst_upper, cst_lower, err_upper, err_lower = coordinates_to_cst(
                airfoil['x'], airfoil['y_upper'],
                airfoil['x'], airfoil['y_lower'],
                n_weights=n_weights
            )
            
            # Verify fit quality
            total_error = err_upper + err_lower
            if total_error < 0.01:  # good fit
                # Compute properties
                props = compute_airfoil_properties(
                    airfoil['x'], airfoil['y_upper'], airfoil['y_lower']
                )
                
                cst_data.append({
                    'name': airfoil['name'],
                    'cst_upper': cst_upper,
                    'cst_lower': cst_lower,
                    'fit_error': total_error,
                    **props
                })
                valid_airfoils.append(airfoil)
            
        except Exception as e:
            continue
    
    print(f"  Valid airfoils after CST fitting: {len(cst_data)}")
    
    # ═══════════════════════════════════════
    # STEP 3: Run XFOIL analysis
    # ═══════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STEP 3: XFOIL Analysis")
    print("=" * 60)
    
    Re_list = config.data.xfoil.reynolds_numbers
    alpha_range = np.arange(
        config.data.xfoil.alpha_min,
        config.data.xfoil.alpha_max + 0.1,
        config.data.xfoil.alpha_step
    )
    
    all_results, xfoil_summary = batch_analyze(
        valid_airfoils,
        reynolds_numbers=Re_list,
        alpha_range=alpha_range,
        n_crit=config.data.xfoil.n_crit,
        max_iter=config.data.xfoil.max_iterations
    )
    
    print(f"  Total converged data points: {len(all_results)}")
    
    # ═══════════════════════════════════════
    # STEP 4: Create training pairs
    # ═══════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STEP 4: Create Training Pairs")
    print("=" * 60)
    
    # Create a lookup from airfoil name to CST parameters
    cst_lookup = {}
    for item in cst_data:
        cst_lookup[item['name']] = item
    
    # Build training arrays
    cst_params_list = []
    cl_list = []
    cd_list = []
    cm_list = []
    re_list = []
    alpha_list = []
    thickness_list = []
    airfoil_names = []
    airfoil_ids = []
    
    # Map airfoil name to ID (for splitting)
    name_to_id = {}
    current_id = 0
    
    for result in all_results:
        name = result['airfoil_name']
        
        if name not in cst_lookup:
            continue
        
        cst_info = cst_lookup[name]
        
        # Assign airfoil ID
        if name not in name_to_id:
            name_to_id[name] = current_id
            current_id += 1
        
        # Collect data
        cst_combined = np.concatenate([cst_info['cst_upper'], cst_info['cst_lower']])
        cst_params_list.append(cst_combined)
        cl_list.append(result['Cl'])
        cd_list.append(result['Cd'])
        cm_list.append(result['Cm'])
        re_list.append(result['Re'])
        alpha_list.append(result['alpha'])
        thickness_list.append(cst_info['max_thickness'])
        airfoil_names.append(name)
        airfoil_ids.append(name_to_id[name])
    
    # Convert to arrays
    cst_params_array = np.array(cst_params_list, dtype=np.float32)
    cl_array = np.array(cl_list, dtype=np.float32)
    cd_array = np.array(cd_list, dtype=np.float32)
    cm_array = np.array(cm_list, dtype=np.float32)
    re_array = np.array(re_list, dtype=np.float32)
    alpha_array = np.array(alpha_list, dtype=np.float32)
    thickness_array = np.array(thickness_list, dtype=np.float32)
    airfoil_id_array = np.array(airfoil_ids, dtype=np.int32)
    
    print(f"  Total training pairs: {len(cl_array)}")
    print(f"  Unique airfoils: {len(name_to_id)}")
    print(f"  CST params shape: {cst_params_array.shape}")
    print(f"  Cl range: [{cl_array.min():.3f}, {cl_array.max():.3f}]")
    print(f"  Cd range: [{cd_array.min():.5f}, {cd_array.max():.5f}]")
    print(f"  Re range: [{re_array.min():.0f}, {re_array.max():.0f}]")
    
    # ═══════════════════════════════════════
    # STEP 5: Train/Val/Test Split
    # ═══════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STEP 5: Train/Val/Test Split (BY AIRFOIL)")
    print("=" * 60)
    
    unique_ids = np.unique(airfoil_id_array)
    n_airfoils = len(unique_ids)
    
    # Shuffle airfoil IDs
    rng = np.random.RandomState(config.project.random_seed)
    rng.shuffle(unique_ids)
    
    # Split
    n_train = int(n_airfoils * config.data.split.train_fraction)
    n_val = int(n_airfoils * config.data.split.val_fraction)
    
    train_ids = set(unique_ids[:n_train])
    val_ids = set(unique_ids[n_train:n_train + n_val])
    test_ids = set(unique_ids[n_train + n_val:])
    
    train_mask = np.array([aid in train_ids for aid in airfoil_id_array])
    val_mask = np.array([aid in val_ids for aid in airfoil_id_array])
    test_mask = np.array([aid in test_ids for aid in airfoil_id_array])
    
    print(f"  Train: {train_mask.sum()} points ({len(train_ids)} airfoils)")
    print(f"  Val:   {val_mask.sum()} points ({len(val_ids)} airfoils)")
    print(f"  Test:  {test_mask.sum()} points ({len(test_ids)} airfoils)")
    
    # Verify no overlap
    assert len(train_ids & val_ids) == 0, "LEAK: train/val overlap!"
    assert len(train_ids & test_ids) == 0, "LEAK: train/test overlap!"
    assert len(val_ids & test_ids) == 0, "LEAK: val/test overlap!"
    print("  ✓ No data leakage detected")
    
    # ═══════════════════════════════════════
    # STEP 6: Save everything
    # ═══════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STEP 6: Save Data")
    print("=" * 60)
    
    # Save as HDF5
    h5_path = os.path.join(output_dir, "dataset.h5")
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('cst_params', data=cst_params_array)
        f.create_dataset('cl', data=cl_array)
        f.create_dataset('cd', data=cd_array)
        f.create_dataset('cm', data=cm_array)
        f.create_dataset('reynolds', data=re_array)
        f.create_dataset('alpha', data=alpha_array)
        f.create_dataset('thickness', data=thickness_array)
        f.create_dataset('airfoil_ids', data=airfoil_id_array)
        f.create_dataset('train_mask', data=train_mask)
        f.create_dataset('val_mask', data=val_mask)
        f.create_dataset('test_mask', data=test_mask)
        
        # Metadata
        f.attrs['n_samples'] = len(cl_array)
        f.attrs['n_airfoils'] = len(name_to_id)
        f.attrs['n_cst_params'] = cst_params_array.shape[1]
        f.attrs['created'] = str(np.datetime64('now'))
    
    print(f"  Saved dataset to: {h5_path}")
    print(f"  File size: {os.path.getsize(h5_path) / 1024 / 1024:.1f} MB")
    
    # Save name mapping
    name_map_path = os.path.join(output_dir, "airfoil_names.pkl")
    with open(name_map_path, 'wb') as f:
        pickle.dump(name_to_id, f)
    
    print(f"  Saved name mapping to: {name_map_path}")
    
    # ═══════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE ✓")
    print("=" * 60)
    print(f"  Total airfoils:      {len(name_to_id)}")
    print(f"  Total data points:   {len(cl_array)}")
    print(f"  Train data points:   {train_mask.sum()}")
    print(f"  Val data points:     {val_mask.sum()}")
    print(f"  Test data points:    {test_mask.sum()}")
    print(f"  Output file:         {h5_path}")
    print("=" * 60)
    
    return {
        'cst_params': cst_params_array,
        'cl': cl_array,
        'cd': cd_array,
        'cm': cm_array,
        'reynolds': re_array,
        'alpha': alpha_array,
        'thickness': thickness_array,
        'airfoil_ids': airfoil_id_array,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'name_to_id': name_to_id,
    }


def load_dataset(data_dir="data/processed"):
    """
    Load pre-processed dataset from disk.
    
    Parameters
    ----------
    data_dir : str
        Directory containing dataset.h5
    
    Returns
    -------
    data : dict
        All arrays and masks
    """
    h5_path = os.path.join(data_dir, "dataset.h5")
    
    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"Dataset not found at {h5_path}\n"
            f"Run the data pipeline first: python scripts/generate_data.py"
        )
    
    with h5py.File(h5_path, 'r') as f:
        data = {
            'cst_params': f['cst_params'][:],
            'cl': f['cl'][:],
            'cd': f['cd'][:],
            'cm': f['cm'][:],
            'reynolds': f['reynolds'][:],
            'alpha': f['alpha'][:],
            'thickness': f['thickness'][:],
            'airfoil_ids': f['airfoil_ids'][:],
            'train_mask': f['train_mask'][:],
            'val_mask': f['val_mask'][:],
            'test_mask': f['test_mask'][:],
        }
    
    print(f"[DATA] Loaded dataset from {h5_path}")
    print(f"  Samples: {len(data['cl'])}")
    print(f"  Train: {data['train_mask'].sum()}")
    print(f"  Val: {data['val_mask'].sum()}")
    print(f"  Test: {data['test_mask'].sum()}")
    
    return data