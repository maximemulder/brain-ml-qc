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
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
import logging

# Set up logging to both console and a file
log_file = "training_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler() # This keeps printing to your terminal too
    ]
)
logger = logging.getLogger(__name__)

def compute_physics_loss(images, outputs):
    # 1. Transform to Fourier Domain
    k_space = torch.fft.fftn(images, dim=(-3, -2, -1))
    k_mag = torch.abs(torch.fft.fftshift(k_space))
    
    # 2. Compute Spectral Variance (Signal for artifacts/noise)
    spectral_variance = torch.var(k_mag, dim=(-3, -2, -1))
    norm_variance = spectral_variance / (torch.mean(k_mag) + 1e-6)
    
    # 3. Get probability of being "Good"
    preds = torch.sigmoid(outputs).squeeze()
    
    # 4. The Penalty: High Variance * High Probability of "Good"
    # If variance is high, this term is large. To minimize it, 
    # the model MUST drive 'preds' toward 0 (Bad).
    phy_penalty = preds * norm_variance
    
    return phy_penalty.mean()

def focal_loss(outputs, targets, alpha=0.25, gamma=1.5):
    probs = torch.sigmoid(outputs)
    ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
    p_t = probs * targets + (1 - probs) * (1 - targets)
    loss = alpha * (1 - p_t)**gamma * ce_loss
    return loss.mean()

# --- 2. DATA PREPARATION ---
def prepare_abide_data(data_root, tsv_path, train_subsites, val_subsite):
    df = pd.read_csv(tsv_path, sep='\t')
    
    def binarize_label(row):
        if pd.isna(row['score']) or str(row['confidence']).lower() == 'exclude':
            return None
        try:
            val = float(row['score'])
            if val == 1.0: return 1.0   # Good
            elif val == -1.0: return 0.0 # Bad
            elif val == 0.0: return 1.0  # Marginal mapped to Good
        except: return None
        return None

    df['qc_label'] = df.apply(binarize_label, axis=1)
    df_clean = df.dropna(subset=['qc_label'])
    label_map = {str(int(sid)): label for sid, label in zip(df_clean['subject_id'], df_clean['qc_label'])}

    train_files, val_files = [], []
    root_path = Path(data_root)
    all_mprage = list(root_path.rglob("mprage.nii.gz"))

    for scan_path in all_mprage:
        sub_id_match = re.search(r'(\d{5,7})', str(scan_path))
        if not sub_id_match: continue
        sub_id_str = str(int(sub_id_match.group(1)))
        
        if sub_id_str in label_map:
            data_item = {"image": str(scan_path.resolve()), "label": [label_map[sub_id_str]]}
            path_str_upper = str(scan_path).upper()
            if any(site.upper() in path_str_upper for site in train_subsites):
                train_files.append(data_item)
            elif any(site.upper() in path_str_upper for site in val_subsite):
                val_files.append(data_item)

    # Calculate sampling weights for balance
    labels = [f["label"][0] for f in train_files]
    class_counts = np.bincount(labels) # [count_bad, count_good]
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[int(l)] for l in labels]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    print(f"Stats: Good={int(class_counts[1])}, Bad={int(class_counts[0])}")
    print(f"Oversampling enabled to force 50/50 batches.")
    return train_files, val_files, sampler

# --- 3. CONFIGURATION & LOADING ---
DATA_DIR = "/brain-ml-qc/files/ABIDE1/extracted" 
TSV_PATH = "/brain-ml-qc/files/ABIDE1/labels.tsv"
TRAIN_SITES = ["NYU_a", "NYU_b", "SDSU", "USM"]
VAL_SITE = ["KKI", "UM"]

train_data, val_data, train_sampler = prepare_abide_data(DATA_DIR, TSV_PATH, TRAIN_SITES, VAL_SITE)

transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image"], spatial_size=(128, 128, 128)),
    ToTensord(keys=["image", "label"]),
])

# Use the sampler for training; shuffle must be False when using a sampler
train_loader = DataLoader(CacheDataset(train_data, transforms, cache_rate=1.0), 
                          batch_size=4, sampler=train_sampler)
val_loader = DataLoader(CacheDataset(val_data, transforms, cache_rate=1.0), 
                        batch_size=1)

# --- 4. MODEL & LOSS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(spatial_dims=3, n_input_channels=1, num_classes=1).to(device)

# Using neutral weight because Sampler handles the balance
# criterion = torch.nn.BCEWithLogitsLoss()
criterion = focal_loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) # Slightly higher LR to break local minima

# --- 5. TRAINING LOOP ---
def run_train(epochs=100):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # Physics weight warmup: Start at 0, slowly increase after epoch 5
        phy_weight = 0.0001
        
        for batch in train_loader:
            inputs, labels = batch["image"].to(device), batch["label"].to(device).float()
            optimizer.zero_grad()
            
            outputs = model(inputs)
            bce_loss = criterion(outputs, labels)
            
            # Compute physics loss
            phy_loss = compute_physics_loss(inputs, outputs) if phy_weight > 0 else torch.tensor(0.0).to(device)
            
            total_loss = bce_loss + (phy_weight * phy_loss)
            
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        # --- VALIDATION ---
        model.eval()
        val_correct = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for v_batch in val_loader:
                v_inputs, v_labels = v_batch["image"].to(device), v_batch["label"].to(device).float()
                v_out = model(v_inputs)
                pred = (torch.sigmoid(v_out) > 0.5).float()
                
                val_correct += (pred == v_labels).sum().item()
                all_preds.extend(pred.cpu().numpy().flatten())
                all_labels.extend(v_labels.cpu().numpy().flatten())

        # Metric Reporting
        val_acc = val_correct / len(val_data)
        report = classification_report(all_labels, all_preds, target_names=['Bad (0)', 'Good (1)'], zero_division=0)
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

        print(f"\n--- Epoch {epoch+1:02d} ---")
        print(f"Loss: {epoch_loss/len(train_loader):.4f} | PHY Weight: {phy_weight}")
        print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp} | Acc: {val_acc:.2%}")
        if (epoch + 1) % 5 == 0: # Print full report every 5 epochs
            print(report)
        print("-" * 40)
        # Log formatted results
        logger.info(f"Epoch {epoch+1:02d} | Loss: {epoch_loss/len(train_loader):.4f} | PHY Weight: {phy_weight}")
        logger.info(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp} | Acc: {val_acc:.2%}")
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"\nDetailed Report:\n{report}")
        logger.info("-" * 40)

if __name__ == "__main__":
    run_train()