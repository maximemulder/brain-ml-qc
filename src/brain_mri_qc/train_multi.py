import os
import torch
import torch.fft
import pandas as pd
import numpy as np
import re
from pathlib import Path
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    Orientationd, ScaleIntensityd, Resized, ToTensord
)
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import resnet18

# --- 1. PHYSICS-INFORMED LOSS MODULE ---
def compute_physics_loss(images, outputs):
    """
    Penalizes high-frequency noise typical of motion/spikes.
    If model predicts "Good" (1) but frequency variance is high, penalize.
    """
    k_space = torch.fft.fftn(images, dim=(-3, -2, -1))
    k_mag = torch.abs(torch.fft.fftshift(k_space))
    
    # Calculate spectral variance normalized by mean magnitude
    spectral_variance = torch.var(k_mag, dim=(-3, -2, -1))
    norm_variance = spectral_variance / (torch.mean(k_mag) + 1e-6)
    
    # Higher 'preds' (closer to 1) means model thinks it's a good scan.
    # We penalize high noise * confidence that it is "Good".
    preds = torch.sigmoid(outputs).squeeze()
    phy_penalty = norm_variance * preds 
    
    return phy_penalty.mean() * 0.01

'''
# --- 2. DATA PREPARATION (SYMLINK AWARE) ---
def prepare_mri_data(data_dir, csv_path, val_size=0.2):
    df = pd.read_csv(csv_path)
    # Order: [artifact, motion, bias_field, spike, noise]
    label_columns = ['artifact', 'motion', 'bias_field', 'spike', 'noise']
    label_map = {row['filename']: row[label_columns].values.astype(float).tolist() 
                 for _, row in df.iterrows()}

    data_path = Path(data_dir)
    # rglob("*") follows symlinks and searches recursively
    all_files = list(data_path.rglob("*.nii.gz"))
    
    print(f"Checking {len(all_files)} files (including symbolic links)...")
    
    subjects = {}
    valid_count = 0
    
    for f in all_files:
        # Resolve symlink to the actual physical file
        resolved_f = f.resolve()
        basename = f.name
        
        if not resolved_f.exists():
            print(f"Warning: Broken link skipped: {f}")
            continue
            
        if basename not in label_map: 
            continue
            
        # Grouping by subject ID to prevent data leakage in split
        sub_id = basename.split("-")[0]
        if sub_id not in subjects: subjects[sub_id] = []
        
        subjects[sub_id].append({"image": str(resolved_f), "label": label_map[basename]})
        valid_count += 1
    
    print(f"Successfully mapped {valid_count} images to {len(subjects)} subjects.")
    
    sub_ids = list(subjects.keys())
    train_ids, val_ids = train_test_split(sub_ids, test_size=val_size, random_state=42)
    
    return ([item for sid in train_ids for item in subjects[sid]], 
            [item for sid in val_ids for item in subjects[sid]])
'''
import re

def prepare_abide_data(data_root, tsv_path, train_subsites, val_subsite):
    df = pd.read_csv(tsv_path, sep='\t')
    
    # 1. Standardize Labels (1=Good, 0=Bad)
    def binarize_label(row):
        if pd.isna(row['score']) or str(row['confidence']).lower() == 'exclude':
            return 0.0
        try:
            return 1.0 if float(row['score']) == 1.0 else 0.0
        except ValueError:
            return 0.0

    df['qc_label'] = df.apply(binarize_label, axis=1)
    
    # 2. Create the label map with zero-padding to match folder names
    # ABIDE folders are usually 7 digits or 5 digits; we'll strip leading zeros 
    # during matching to be safe.
    label_map = {str(int(sid)): label for sid, label in zip(df['subject_id'], df['qc_label'])}

    train_files, val_files = [], []
    root_path = Path(data_root)
    
    print(f"Searching in: {root_path.resolve()}")

    # 3. Use rglob to find mprage files regardless of depth
    all_mprage = list(root_path.rglob("mprage.nii.gz"))
    
    if not all_mprage:
        print("CRITICAL: No 'mprage.nii.gz' files found. Check your file names or root path.")
        return [], []

    for scan_path in all_mprage:
        # 4. Robust Subject ID Extraction
        # Look for a part of the path that is a long number (5-7 digits)
        sub_id_match = re.search(r'(\d{5,7})', str(scan_path))
        if not sub_id_match:
            continue
            
        # Convert to string integer to match our label_map (removes leading zeros)
        sub_id_str = str(int(sub_id_match.group(1)))
        
        if sub_id_str in label_map:
            data_item = {
                "image": str(scan_path.resolve()), 
                "label": [label_map[sub_id_str]]
            }
            
            # 5. Site matching
            # Check which training/val site folder this file belongs to
            path_str = str(scan_path)
            if any(site in path_str for site in train_subsites):
                train_files.append(data_item)
            elif val_subsite in path_str:
                val_files.append(data_item)

    print(f"Successfully mapped {len(train_files)} training and {len(val_files)} validation samples.")
    return train_files, val_files

# --- 3. CONFIGURATION & EXECUTION ---
DATA_DIR = "/brain-ml-qc/files/ABIDE1/extracted" 
TSV_PATH = "/brain-ml-qc/files/ABIDE1/labels.tsv"
TRAIN_SITES = ["NYU_a", "NYU_b", "NYU_c"]
VAL_SITE = "NYU_d"

train_data, val_data = prepare_abide_data(DATA_DIR, TSV_PATH, TRAIN_SITES, VAL_SITE)

transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image"], spatial_size=(128, 128, 128)),
    ToTensord(keys=["image", "label"]),
])

# --- 4. DATALOADERS & MODEL ---
train_loader = DataLoader(CacheDataset(train_data, transforms, cache_rate=1.0), batch_size=4, shuffle=True)
val_loader = DataLoader(CacheDataset(val_data, transforms, cache_rate=1.0), batch_size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(spatial_dims=3, n_input_channels=1, num_classes=1).to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --- 5. TRAINING LOOP ---
def run_train(epochs=20):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            bce_loss = criterion(outputs, labels)
            phy_loss = compute_physics_loss(inputs, outputs)
            
            total_loss = bce_loss + phy_loss
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            
        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for v_batch in val_loader:
                v_inputs, v_labels = v_batch["image"].to(device), v_batch["label"].to(device)
                v_out = model(v_inputs)
                pred = (torch.sigmoid(v_out) > 0.5).float()
                correct += (pred == v_labels).item()
        
        val_acc = correct / len(val_data) if len(val_data) > 0 else 0
        print(f"Epoch {epoch+1} | Total Loss: {epoch_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2%}")

if __name__ == "__main__":
    run_train()