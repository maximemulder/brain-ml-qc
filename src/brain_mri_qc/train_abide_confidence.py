import os
import torch
import torch.fft
import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    Orientationd, ScaleIntensityd, Resized, ToTensord
)
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import resnet18
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# --- 0. LOGGING SETUP ---
log_file = "logs/training_log_confidence.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 1. LOSS FUNCTIONS ---

def compute_physics_loss(images, outputs):
    """
    Penalizes high-frequency noise typical of motion/spikes.
    The penalty is higher if the model predicts 'Good' for a noisy image.
    """
    k_space = torch.fft.fftn(images, dim=(-3, -2, -1))
    k_mag = torch.abs(torch.fft.fftshift(k_space))
    
    spectral_variance = torch.var(k_mag, dim=(-3, -2, -1))
    norm_variance = spectral_variance / (torch.mean(k_mag) + 1e-6)
    
    preds = torch.sigmoid(outputs).squeeze()
    if preds.ndim == 0: preds = preds.unsqueeze(0) # Handle batch size 1
    
    phy_penalty = preds * norm_variance
    return phy_penalty.mean()

def focal_loss(outputs, targets, weights=None, alpha=0.25, gamma=1.5):
    """
    Focal Loss to handle hard examples, with optional per-sample weights (Confidence).
    """
    probs = torch.sigmoid(outputs)
    ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
    p_t = probs * targets + (1 - probs) * (1 - targets)
    loss = alpha * (1 - p_t)**gamma * ce_loss
    
    if weights is not None:
        loss = loss * weights
        
    return loss.mean()

def prepare_abide_data(data_root, tsv_path, train_subsites, val_subsites):
    """
    Refined data loader that handles nested folders (NYU_a/NYU),
    extracts rater confidence, and ensures balanced sampling.
    """
    df = pd.read_csv(tsv_path, sep='\t')
    
    # Mapping Rater Confidence to numerical weights for the loss function
    # High confidence = full weight; lower confidence = reduced impact on loss
    conf_map = {"high": 1.0, "medium": 0.5, "low": 0.25}

    def binarize_label(row):
        # Filter out excludes or missing scores immediately
        if pd.isna(row['score']) or str(row['confidence']).lower() == 'exclude':
            return None
        try:
            val = float(row['score'])
            # 1.0 = Good (Score 1 or 0), 0.0 = Bad (Score -1)
            return 1.0 if val >= 0.0 else 0.0
        except (ValueError, TypeError):
            return None

    df['qc_label'] = df.apply(binarize_label, axis=1)
    # Default to 0.25 (low) if confidence is missing but scan is valid
    df['conf_weight'] = df['confidence'].str.lower().map(conf_map).fillna(0.25)
    
    df_clean = df.dropna(subset=['qc_label'])
    
    # Map Subject ID to (Label, Confidence Weight)
    # Standardization to str(int(sid)) is crucial for matching '0050952' to '50952'
    label_info = {
        str(int(sid)): (label, weight) 
        for sid, label, weight in zip(df_clean['subject_id'], df_clean['qc_label'], df_clean['conf_weight'])
    }

    train_files, val_files = [], []
    root_path = Path(data_root)
    
    # Use a slightly more flexible glob in case of prefixing, though rglob is generally robust
    all_mprage = list(root_path.rglob("mprage.nii.gz"))

    for scan_path in all_mprage:
        # Get absolute path and upper case for reliable string matching
        full_path_str = str(scan_path.resolve())
        path_str_upper = full_path_str.upper()

        # Extract 5-7 digit Subject ID from the directory name or filename
        sub_id_match = re.search(r'(\d{5,7})', full_path_str)
        if not sub_id_match:
            continue
            
        sub_id_str = str(int(sub_id_match.group(1)))
        
        # Check if ID exists in our QC label database
        if sub_id_str in label_info:
            label, weight = label_info[sub_id_str]
            data_item = {
                "image": full_path_str, 
                "label": [label], 
                "conf": [weight]
            }
            
            # Assignment based on site keywords in the path string
            # This handles nested structures like NYU_a/NYU/
            is_train = any(site.upper() in path_str_upper for site in train_subsites)
            is_val = any(site.upper() in path_str_upper for site in val_subsites)

            if is_train:
                train_files.append(data_item)
            elif is_val:
                val_files.append(data_item)

    # --- Safety Check for num_samples=0 ---
    if not train_files:
        print(f"ERROR: No training files matched. Found {len(all_mprage)} NIfTIs on disk.")
        print(f"Site keywords searched: {train_subsites}")
        if all_mprage:
            print(f"Check this example path: {all_mprage[0]}")
        return [], [], None

    # Calculate Balanced Sampling Weights for Training
    # This ensures that 'Bad' scans (usually the minority) are seen as often as 'Good' scans
    labels = [int(f["label"][0]) for f in train_files]
    class_counts = np.bincount(labels)
    
    if len(class_counts) < 2:
        # Fallback if the selected sites only contain one class
        class_weights = torch.tensor([1.0, 1.0])
    else:
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        
    sample_weights = [class_weights[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    print(f"Success: {len(train_files)} Train, {len(val_files)} Val samples loaded.")
    return train_files, val_files, sampler

# --- 3. CONFIGURATION & LOADING ---
DATA_DIR = "/brain-ml-qc/files/ABIDE1/extracted" 
TSV_PATH = "/brain-ml-qc/files/ABIDE1/labels.tsv"
TRAIN_SITES = ["NYU_a", "NYU_b", "SDSU", "USM", "CMU_a", "CMU_b"]
VAL_SITE = ["KKI", "UM"]

train_data, val_data, train_sampler = prepare_abide_data(DATA_DIR, TSV_PATH, TRAIN_SITES, VAL_SITE)

transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image"], spatial_size=(128, 128, 128)),
    ToTensord(keys=["image", "label", "conf"]),
])

train_loader = DataLoader(CacheDataset(train_data, transforms, cache_rate=1.0), 
                          batch_size=4, sampler=train_sampler)
val_loader = DataLoader(CacheDataset(val_data, transforms, cache_rate=1.0), 
                        batch_size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(spatial_dims=3, n_input_channels=1, num_classes=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# --- 4. TRAINING LOOP ---

def run_train(epochs=50):
    best_acc = 0.0
    phy_weight = 0.0001 # Start with a small constant weight

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device).float()
            conf_weights = batch["conf"].to(device).float()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Confidence-Weighted Focal Loss
            bce_loss = focal_loss(outputs, labels, weights=conf_weights)
            
            # Physics-Informed Penalty
            phy_loss = compute_physics_loss(inputs, outputs)
            
            total_loss = bce_loss + (phy_weight * phy_loss)
            
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        # --- VALIDATION ---
        model.eval()
        best_f1_bad = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for v_batch in val_loader:
                v_inputs = v_batch["image"].to(device)
                v_labels = v_batch["label"].to(device).float()
                
                v_out = model(v_inputs)
                pred = (torch.sigmoid(v_out) > 0.5).float()
                
                all_preds.extend(pred.cpu().numpy().flatten())
                all_labels.extend(v_labels.cpu().numpy().flatten())

        # Calculate Detailed Metrics
        # precision[0] is for Bad, precision[1] is for Good
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, labels=[0, 1], zero_division=0
        )
        val_acc = np.mean(np.array(all_labels) == np.array(all_preds))
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

        # Save Best Model based on Bad Class F1 (The priority for QC)
        if f1[0] > best_f1_bad or val_acc > best_acc:
            best_f1_bad = f1[0]
            best_acc = val_acc
            torch.save(model.state_dict(), "models/best_resnet18_qc_w_conf.pth")
            logger.info(f"*** NEW BEST MODEL (Bad F1: {f1[0]:.4f}) SAVED ***")

        # LOGGING FOR PAPER TABLE
        logger.info(f"Epoch {epoch+1:03d} | Total Loss: {epoch_loss/len(train_loader):.4f}")
        logger.info(f"Acc: {val_acc:.4f} | CM: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        # Format metrics specifically for your paper's Results sub-columns
        logger.info(f"{'Metric':<10} | {'Bad (0)':<10} | {'Good (1)':<10}")
        logger.info(f"{'-'*35}")
        logger.info(f"{'Precision':<10} | {precision[0]:.4f} {'':<5} | {precision[1]:.4f}")
        logger.info(f"{'Recall':<10} | {recall[0]:.4f} {'':<5} | {recall[1]:.4f}")
        logger.info(f"{'F1-Score':<10} | {f1[0]:.4f} {'':<5} | {f1[1]:.4f}")
        logger.info(f"{'='*50}")

if __name__ == "__main__":
    run_train()