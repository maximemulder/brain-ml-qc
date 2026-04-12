import os
import glob
import torch
import sys
import pandas as pd
import numpy as np
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    Orientationd, ScaleIntensityd, Resized, ToTensord
)
from monai.data import Dataset, DataLoader
# Importing the 3D ResNet from MONAI
from monai.networks.nets import ResNet, resnet50

# 1. DATA PREPARATION (Same logic, ensured label order)
def prepare_mri_data(data_dir, csv_path, val_size=0.2):
    df = pd.read_csv(csv_path)
    # Order: [artifact, motion, bias_field, spike, noise]
    label_columns = ['artifact', 'motion', 'bias_field', 'spike', 'noise']
    label_map = {row['filename']: row[label_columns].values.astype(float).tolist() 
                 for _, row in df.iterrows()}

    all_files = glob.glob(os.path.join(data_dir, "*.nii.gz"))
    subjects = {}
    
    for f in all_files:
        basename = os.path.basename(f)
        if basename not in label_map: continue
        sub_id = basename.split("-")[0]
        if sub_id not in subjects: subjects[sub_id] = []
        subjects[sub_id].append({"image": f, "label": label_map[basename]})
    
    sub_ids = list(subjects.keys())
    train_ids, val_ids = train_test_split(sub_ids, test_size=val_size, random_state=42)
    return ([item for sid in train_ids for item in subjects[sid]], 
            [item for sid in val_ids for item in subjects[sid]])

# --- Config ---
data_root = "/brain-ml-qc/files/dataset/images" 
csv_path = "/brain-ml-qc/files/dataset/labels.csv"
train_files, val_files = prepare_mri_data(data_root, csv_path)

transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"), 
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image"], spatial_size=(128, 128, 128)), 
    ToTensord(keys=["image", "label"]),
])

train_loader = DataLoader(Dataset(data=train_files, transform=transforms), batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(Dataset(data=val_files, transform=transforms), batch_size=4, shuffle=False, num_workers=4)

# --- 3. MODEL INITIALIZATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelqc = resnet50(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=5).to(device)

# --- LOAD MODEL CODE ---
if os.path.exists("best_resnet_qc_model.pth"):
    print(f"Loading existing checkpoint: best_resnet_qc_model.pth")
    # map_location ensures it loads correctly even if trained on a different GPU/CPU
    modelqc.load_state_dict(torch.load("best_resnet_qc_model.pth", map_location=device))
else:
    print("No checkpoint found. Starting from scratch.")

optimizer = Adam(modelqc.parameters(), lr=1e-4)
loss_function = BCEWithLogitsLoss()

# 5. TRAINING LOOP WITH SPECIFIC ARTIFACT TRACKING
# --- 4. TRAINING LOOP ---
def run_training(model, epochs=50):
    best_artifact_acc = 0.0  # Track best accuracy for the main QC task
    
    for epoch in range(epochs):
        modelqc.train()
        epoch_loss = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            
            optimizer.zero_grad() # Clear gradients before forward pass
            outputs = modelqc(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Print batch stats
            batch_preds = (outputs > 0).float()
            batch_acc = (batch_preds[:, 0] == labels[:, 0]).float().mean().item()
            print(f"Epoch {epoch+1} | Batch {batch_idx+1} | Loss: {loss.item():.4f} | Artif-Acc: {batch_acc:.2%}")
            
            epoch_loss += loss.item()
            
        # Validation
        modelqc.eval()
        val_loss, total = 0, 0
        correct_artifact, correct_all = 0, 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
                outputs = modelqc(inputs)
                val_loss += loss_function(outputs, labels).item()
                
                preds = (outputs > 0).float()
                correct_artifact += (preds[:, 0] == labels[:, 0]).sum().item()
                correct_all += (preds == labels).sum().item()
                total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        artifact_acc = correct_artifact / total
        mean_acc = correct_all / (total * 5)
        
        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"Val Loss: {avg_val_loss:.4f} | Artifact Acc: {artifact_acc:.2%} | Mean Multi-label Acc: {mean_acc:.2%}")

        # Save based on Best Artifact Accuracy
        if artifact_acc > best_artifact_acc:
            best_artifact_acc = artifact_acc
            torch.save(model.state_dict(), "best_resnet_qc_model.pth")
            print(f"  --> New Best Artifact Accuracy! Model saved.\n")
        else:
            print(f"  --> Best Artifact Acc so far: {best_artifact_acc:.2%}\n")

if __name__ == "__main__":
    run_training(modelqc)