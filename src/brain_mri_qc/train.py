import os
import glob
import torch
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import MRIQuality3D
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    Orientationd, ScaleIntensityd, Resized, ToTensord, RandRotate90d, RandFlipd
)
from monai.data import Dataset, DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from sklearn.model_selection import train_test_split

# 1. DATA PREPARATION WITH SUBJECT-AWARE SPLITTING
def prepare_mri_data(data_dir, val_size=0.2):
    # glob follows symlinks and finds the .nii.gz files
    all_files = glob.glob(os.path.join(data_dir, "*.nii.gz"))
    
    subjects = {}
    for f in all_files:
        # Get the filename of the link (e.g., IXI002-..._clean.nii.gz)
        basename = os.path.basename(f)
        
        # Resolve the symbolic link to the actual physical path
        # This is good practice to ensure the link isn't broken
        actual_path = os.path.realpath(f)
        if not os.path.exists(actual_path):
            print(f"Warning: Skipping broken link {f}")
            continue

        # Extract Subject ID (e.g., IXI002)
        sub_id = basename.split("-")[0] 
        
        if sub_id not in subjects:
            subjects[sub_id] = []
        
        # Determine label based on the filename
        # Using 'basename' is safer here because the link name contains the status
        label = 1 if "_clean" in basename else 0
        
        subjects[sub_id].append({
            "image": f,       # MONAI will follow the symlink automatically
            "label": [label]
        })
    
    # Split based on unique Subject IDs
    sub_ids = list(subjects.keys())
    if not sub_ids:
        raise ValueError(f"No valid MRI files found in {data_dir}!")

    train_ids, val_ids = train_test_split(sub_ids, test_size=val_size, random_state=42)
    
    train_files = [item for sid in train_ids for item in subjects[sid]]
    val_files = [item for sid in val_ids for item in subjects[sid]]
    
    return train_files, val_files

# --- Path Configuration ---
data_root = "/brain-ml-qc/files/dataset/images" 
train_files, val_files = prepare_mri_data(data_root)

# 2. DEFINE VOLUMETRIC TRANSFORMS
transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"), 
    ScaleIntensityd(keys=["image"]),
    RandRotate90d(keys=["image"], prob=0.5, spatial_axes=[0, 1]),
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
    Resized(keys=["image"], spatial_size=(128, 128, 128)), 
    ToTensord(keys=["image", "label"]),
])

# 3. CREATE LOADERS
train_ds = Dataset(data=train_files, transform=transforms)
val_ds = Dataset(data=val_files, transform=transforms)

# num_workers=4 is standard for 2026 to speed up NIfTI decompression
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)

# 4. INITIALIZE MODEL, OPTIMIZER, AND LOSS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MRIQuality3D().to(device)

optimizer = Adam(model.parameters(), lr=1e-4)
loss_function = BCEWithLogitsLoss() 

# 5. TRAINING AND VALIDATION LOOP
def run_training(model, epochs=50):
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        epoch_loss = 0
        for batch_data in train_loader:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].float().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            print(f"  [Epoch {epoch+1}] Batch Loss: {loss.item():.4f}")
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        # --- Validation Phase ---
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_data in val_loader:
                inputs = batch_data["image"].to(device)
                labels = batch_data["label"].float().to(device)
                
                outputs = model(inputs)
                val_loss += loss_function(outputs, labels).item()
                
                # Accuracy check: Logit > 0 means Probability > 0.5
                preds = (outputs > 0).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  [Train] Loss: {avg_train_loss:.4f}")
        print(f"  [Val]   Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2%}")

        # Save checkpoint if it's the best performing model on validation set
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_mri_qc_model.pth")
            print("  --> Best model saved.")

# --- Execute ---
if __name__ == "__main__":
    print(f"Starting training: {len(train_files)} training samples, {len(val_files)} validation samples.")
    run_training(model)