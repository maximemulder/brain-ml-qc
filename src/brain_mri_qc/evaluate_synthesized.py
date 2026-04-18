#!/usr/bin/env python
import os
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import resnet18
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, Orientationd, Resized, ScaleIntensityd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from brain_mri_qc.train_abide_freq import prepare_abide_data

def evaluate_and_visualize(model_path, val_data, transforms, device, save_dir="qc_results"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. Load Model
    model = resnet18(spatial_dims=3, n_input_channels=1, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Prepare Data
    ds = CacheDataset(data=val_data, transform=transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    # Accumulators for Metrics
    all_gt = []
    all_pred = []

    print(f"\n{'Index':<5} | {'Ground Truth':<12} | {'Prediction':<10} | {'Prob'}")
    print("-" * 55)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            inputs = batch["image"].to(device)
            gt_label = batch["label"].item() # Get as scalar

            logit = model(inputs)
            prob = torch.sigmoid(logit).item()
            pred_label = 1.0 if prob > 0.5 else 0.0

            # Store for metrics
            all_gt.append(gt_label)
            all_pred.append(pred_label)

            # Extract image for visualization
            img_data = inputs.cpu().numpy()[0, 0]
            mid_slice = img_data[:, :, img_data.shape[2] // 2]

            # Output
            pred_str = "Good" if pred_label == 1 else "Bad"
            gt_str = "Good" if gt_label == 1 else "Bad"
            print(f"{i:<5} | {gt_str:<12} | {pred_str:<10} | {prob:.4f}")

            # Plotting (Simplified)
            plt.figure(figsize=(5, 5))
            plt.imshow(mid_slice, cmap='gray')
            color = 'green' if gt_label == pred_label else 'red'
            plt.title(f"Pred: {pred_str} | Actual: {gt_str}", color=color)
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, f"result_{i}.png"))
            plt.close()

    # --- 3. PRINT QUANTITATIVE METRICS ---
    print("\n" + "="*30)
    print("      FINAL PERFORMANCE")
    print("="*30)
    
    # Global Accuracy
    acc = accuracy_score(all_gt, all_pred)
    print(f"Overall Accuracy: {acc*100:.2f}%")
    
    # Detailed Report (Precision, Recall, F1 per class)
    # 0 = Bad, 1 = Good
    print("\nClassification Report:")
    print(classification_report(all_gt, all_pred, target_names=['Bad', 'Good']))
    
    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(all_gt, all_pred))
    print("="*30)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- RE-USE YOUR CONFIG ---
    DATA_DIR = "/brain-ml-qc/files/ABIDE1/extracted"
    TSV_PATH = "/brain-ml-qc/files/ABIDE1/labels.tsv"
    TRAIN_SITES = ["NYU_a", "NYU_b", "SDSU", "USM", "CMU_a", "CMU_b"]
    VAL_SITE = ["KKI", "UM"]

    # --- GENERATE THE LISTS ---
    # This recreates the val_data list of dictionaries
    _, val_data, _ = prepare_abide_data(DATA_DIR, TSV_PATH, TRAIN_SITES, VAL_SITE)

    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(128, 128, 128)),
    ])

    # --- RUN EVALUATION ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_and_visualize("models/best_qc_model_synthesized.pth", val_data, transforms, device)
