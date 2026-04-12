import os
import glob
import torch
import torch.fft
import pandas as pd
import numpy as np
from pathlib import Path
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    Orientationd, ScaleIntensityd, Resized, ToTensord
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import resnet18

# --- 1. PHYSICS-INFORMED LOSS MODULE ---
def compute_physics_loss(images, labels, outputs):
    """
    Penalizes the model if it misses frequency-domain signatures of spikes/motion.
    """
    # Transform images to k-space (Frequency Domain)
    k_space = torch.fft.fftn(images, dim=(-3, -2, -1))
    k_mag = torch.abs(torch.fft.fftshift(k_space))
    
    # Spectral variance as a proxy for 'unnatural' frequency noise
    spectral_variance = torch.var(k_mag, dim=(-3, -2, -1))
    
    # Logic: if model predicts low 'spike' (index 3) but spectral variance is high, penalize
    preds = torch.sigmoid(outputs)
    spike_penalty = spectral_variance * (1 - preds[:, 3])
    
    return spike_penalty.mean() * 0.05

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

# --- 3. CONFIG & INITIALIZATION ---
data_root = "/brain-ml-qc/files/dataset/images" 
csv_path = "/brain-ml-qc/files/dataset/labels.csv"
checkpoint_path = "best_resnet_qc_physics.pth"

train_files, val_files = prepare_mri_data(data_root, csv_path)

transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"), 
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image"], spatial_size=(128, 128, 128)), 
    ToTensord(keys=["image", "label"]),
])

train_loader = DataLoader(Dataset(data=train_files, transform=transforms), batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(Dataset(data=val_files, transform=transforms), batch_size=4, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelqc = resnet18(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=5).to(device)

if os.path.exists(checkpoint_path):
    print(f"Loading existing checkpoint: {checkpoint_path}")
    modelqc.load_state_dict(torch.load(checkpoint_path, map_location=device))

optimizer = Adam(modelqc.parameters(), lr=1e-4, weight_decay=1e-5)
loss_function = BCEWithLogitsLoss(reduction='none') # 'none' to allow manual weighting per label

# --- 4. TRAINING FUNCTION ---
def run_training(model, epochs=50):
    best_artifact_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Weighted Loss: Focus heavily on 'artifact' (Index 0)
            # Other labels act as auxiliary guidance
            raw_loss = loss_function(outputs, labels)
            weights = torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0]).to(device) # Artifact is 2x more important
            weighted_loss = (raw_loss * weights).mean()
            
            # Add Physics Loss
            phy_loss = compute_physics_loss(inputs, labels, outputs)
            total_loss = weighted_loss + phy_loss
            
            total_loss.backward()
            optimizer.step()
            
            # Stats for the primary output: 'artifact'
            batch_preds = (outputs > 0).float()
            artifact_acc = (batch_preds[:, 0] == labels[:, 0]).float().mean().item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Total Loss: {total_loss.item():.4f} | Artifact Acc: {artifact_acc:.2%}")

        # --- Validation ---
        model.eval()
        correct_artifact, total_samples = 0, 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
                outputs = model(inputs)
                preds = (outputs > 0).float()
                
                # Metric that determines 'Good' vs 'Poor' scan
                correct_artifact += (preds[:, 0] == labels[:, 0]).sum().item()
                total_samples += labels.size(0)
        
        val_acc = correct_artifact / total_samples
        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"Current Validation Artifact Acc: {val_acc:.2%}")

        if val_acc > best_artifact_acc:
            best_artifact_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> [SAVED] New Best Accuracy: {best_artifact_acc:.2%}\n")

if __name__ == "__main__":
    try:
        run_training(modelqc)
    except KeyboardInterrupt:
        print("\nTraining stopped manually.")