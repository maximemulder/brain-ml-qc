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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler

# --- 1. LOGGING SETUP ---
log_file = "training_log_synthetic.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s', # Clean format for easier copy-pasting into LaTeX/Excel
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 2. LOSS FUNCTIONS ---
def compute_physics_loss(images, outputs):
    k_space = torch.fft.fftn(images, dim=(-3, -2, -1))
    k_mag = torch.abs(torch.fft.fftshift(k_space))
    spectral_variance = torch.var(k_mag, dim=(-3, -2, -1))
    norm_variance = spectral_variance / (torch.mean(k_mag) + 1e-6)
    preds = torch.sigmoid(outputs).squeeze()
    if preds.ndim == 0:
        preds = preds.unsqueeze(0)
    phy_penalty = preds * norm_variance
    return phy_penalty.mean()

def focal_loss(outputs, targets, alpha=0.25, gamma=1.5):
    probs = torch.sigmoid(outputs)
    ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
    p_t = probs * targets + (1 - probs) * (1 - targets)
    loss = alpha * (1 - p_t)**gamma * ce_loss
    return loss.mean()

# --- 3. DATA PREPARATION ---
# (Keeping your original logic for data loading)
def prepare_synthetic_data(data_dir, csv_path, val_size=0.2):
    df = pd.read_csv(csv_path)
    label_columns = ['artifact', 'motion', 'bias_field', 'spike', 'noise']
    label_map = {row['filename']: [0.0 if row[label_columns].any() else 1.0] for _, row in df.iterrows()}
    all_files = list(Path(data_dir).rglob("*.nii.gz"))
    subjects = {}
    for f in all_files:
        if f.name in label_map:
            sub_id = f.name.split("-")[0]
            if sub_id not in subjects:
                subjects[sub_id] = []
            subjects[sub_id].append({"image": str(f.resolve()), "label": label_map[f.name]})
    tr_ids, val_ids = train_test_split(list(subjects.keys()), test_size=val_size, random_state=42)
    return [i for sid in tr_ids for i in subjects[sid]], [i for sid in val_ids for i in subjects[sid]]

def prepare_abide_data(data_root, tsv_path, site_list):
    df = pd.read_csv(tsv_path, sep='\t')
    def binarize(row):
        if pd.isna(row['score']) or str(row['confidence']).lower() == 'exclude':
            return None
        return 1.0 if float(row['score']) >= 0.0 else 0.0
    df['qc_label'] = df.apply(binarize, axis=1)
    label_info = {str(int(sid)): [label] for sid, label in zip(df['subject_id'], df['qc_label']) if not np.isnan(label)}
    matched_files = []
    for scan in Path(data_root).rglob("mprage.nii.gz"):
        path_str = str(scan.resolve())
        if any(site.upper() in path_str.upper() for site in site_list):
            match = re.search(r'(\d{5,7})', path_str)
            if match:
                sid = str(int(match.group(1)))
                if sid in label_info:
                    matched_files.append({"image": path_str, "label": label_info[sid]})
    return matched_files

# --- 4. EXECUTION ---
SYN_DIR = "/brain-ml-qc/files/SYNTHETIC/images"
SYN_CSV = "/brain-ml-qc/files/SYNTHETIC/labels.csv"
ABIDE_DIR = "/brain-ml-qc/files/ABIDE1/extracted"
ABIDE_TSV = "/brain-ml-qc/files/ABIDE1/labels.tsv"

train_syn, val_syn = prepare_synthetic_data(SYN_DIR, SYN_CSV)
val_abide = prepare_abide_data(ABIDE_DIR, ABIDE_TSV, ["KKI", "UM"])

transforms = Compose([
    LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"), ScaleIntensityd(keys=["image"]),
    Resized(keys=["image"], spatial_size=(128, 128, 128)), ToTensord(keys=["image", "label"]),
])

train_labels = [int(f["label"][0]) for f in train_syn]
class_weights = 1. / torch.tensor(np.bincount(train_labels), dtype=torch.float)
sampler = WeightedRandomSampler([class_weights[label] for label in train_labels], len(train_labels), replacement=True)

train_loader = DataLoader(CacheDataset(train_syn, transforms, cache_rate=1.0), batch_size=4, sampler=sampler)
val_loader_syn = DataLoader(CacheDataset(val_syn, transforms, cache_rate=1.0), batch_size=1)
val_loader_abide = DataLoader(CacheDataset(val_abide, transforms, cache_rate=1.0), batch_size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(spatial_dims=3, n_input_channels=1, num_classes=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

def run_train(epochs=50):
    best_f1_abide = 0.0
    best_acc_abide = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            inputs, labels = batch["image"].to(device), batch["label"].to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_val = focal_loss(outputs, labels) + (0.0001 * compute_physics_loss(inputs, outputs))
            loss_val.backward()
            optimizer.step()
            epoch_loss += loss_val.item()

        # Validation Function for Comprehensive Table Metrics
        def get_table_metrics(loader, name):
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for b in loader:
                    out = model(b["image"].to(device))
                    pred = (torch.sigmoid(out) > 0.5).float()
                    y_true.extend(b["label"].numpy().flatten())
                    y_pred.extend(pred.cpu().numpy().flatten())

            acc = accuracy_score(y_true, y_pred) * 100
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)

            logger.info(f"\n--- {name} Results (Epoch {epoch+1}) ---")
            logger.info(f"{'Metric':<15} | {'Class 0 (Bad)':<15} | {'Class 1 (Good)':<15}")
            logger.info("-" * 50)
            logger.info(f"{'Accuracy (%)':<15} | {acc:.2f}")
            logger.info(f"{'Precision':<15} | {prec[0]:.4f} {'':<10} | {prec[1]:.4f}")
            logger.info(f"{'Recall':<15} | {rec[0]:.4f} {'':<10} | {rec[1]:.4f}")
            logger.info(f"{'F1-Score':<15} | {f1[0]:.4f} {'':<10} | {f1[1]:.4f}")

            # Simple line format for Table 2 copy-pasting
            logger.info(f"TABLE ROW: {name} | {acc:.1f}% | {prec[0]:.2f}/{prec[1]:.2f} | {rec[0]:.2f}/{rec[1]:.2f} | {f1[0]:.2f}/{f1[1]:.2f}")
            return acc, f1[0]

        logger.info(f"\n{'='*20} EPOCH {epoch+1} {'='*20}")
        get_table_metrics(val_loader_syn, "Synthetic-Only")
        val_acc_abide, f1_bad_abide = get_table_metrics(val_loader_abide, "ABIDE-Real")

        if f1_bad_abide > best_f1_abide or val_acc_abide > best_acc_abide:
            best_f1_abide = f1_bad_abide
            best_acc_abide = val_acc_abide
            torch.save(model.state_dict(), "best_qc_model_synthesized.pth")
            logger.info(">>> Best ABIDE F1(Bad) improved. Model saved.")

if __name__ == "__main__":
    run_train()
