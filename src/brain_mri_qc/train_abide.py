#!/usr/bin/env python
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.fft
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import resnet18
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, Orientationd, Resized, ScaleIntensityd, ToTensord
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import WeightedRandomSampler

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
            if val == 1.0:
                return 1.0   # Good
            elif val == -1.0:
                return 0.0 # Bad
            elif val == 0.0:
                return 1.0  # Marginal mapped to Good
        except ValueError:
            return None
        return None

    df['qc_label'] = df.apply(binarize_label, axis=1)
    df_clean = df.dropna(subset=['qc_label'])
    label_map = {str(int(sid)): label for sid, label in zip(df_clean['subject_id'], df_clean['qc_label'])}

    train_files, val_files = [], []
    root_path = Path(data_root)
    all_mprage = list(root_path.rglob("mprage.nii.gz"))

    for scan_path in all_mprage:
        sub_id_match = re.search(r'(\d{5,7})', str(scan_path))
        if not sub_id_match:
            continue
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
    sample_weights = [class_weights[int(label)] for label in labels]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    print(f"Stats: Good={int(class_counts[1])}, Bad={int(class_counts[0])}")
    print("Oversampling enabled to force 50/50 batches.")
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
def run_train(epochs=50):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            inputs, labels = batch["image"].to(device), batch["label"].to(device).float()
            optimizer.zero_grad()

            outputs = model(inputs)
            bce_loss = criterion(outputs, labels)

            total_loss = bce_loss

            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        # --- VALIDATION ---
        model.eval()
        all_preds, all_labels = [], []
        best_f1_bad = 0.0

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
        if f1[0] > best_f1_bad:
            best_f1_bad = f1[0]
            torch.save(model.state_dict(), "best_resnet18_qc_wo_conf.pth")
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
