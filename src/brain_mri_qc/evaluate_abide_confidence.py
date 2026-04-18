import os
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
from monai.networks.nets import resnet18
from monai.data import DataLoader, CacheDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Orientationd, ScaleIntensityd, Resized, ToTensord
)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from brain_mri_qc.train_abide_freq import prepare_abide_data

def evaluate_and_visualize(model_path, val_data, transforms, device):
    # 1. Load Model
    model = resnet18(spatial_dims=3, n_input_channels=1, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 2. Prepare Data
    ds = CacheDataset(data=val_data, transform=transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    print(f"{'Subject ID':<15} | {'Ground Truth':<12} | {'Prediction':<10} | {'Prob (Good)'}")
    print("-" * 60)

    results = []

def evaluate_and_visualize(model_path, val_data, transforms, device, save_dir="qc_results"):
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. Load Model
    model = resnet18(spatial_dims=3, n_input_channels=1, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Prepare Data
    ds = CacheDataset(data=val_data, transform=transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    print(f"{'Index':<5} | {'Ground Truth':<12} | {'Prediction':<10} | {'Prob'}")
    print("-" * 50)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            inputs = batch["image"].to(device)
            gt_label = batch["label"] # Convert tensor to int

            logit = model(inputs)
            prob = torch.sigmoid(logit) # Convert to Python float

            # Extract image for visualization
            img_data = inputs.cpu().numpy()[0, 0]
            mid_slice = img_data[:, :, img_data.shape[2] // 2]

            # 1. Determine Label
            pred_label = 1.0 if prob > 0.5 else 0.0

            # 2. Calculate Confidence Score (0% to 100%)
            confidence_score = abs(prob - 0.5) * 2

            # 3. Text output
            pred_str = "Good" if pred_label == 1 else "Bad"
            gt_str = "Good" if gt_label == 1 else "Bad"
            print(f"{i} | {gt_str} | {pred_str} | {prob}")

            # 4. Plotting
            plt.figure(figsize=(6, 6))
            plt.imshow(mid_slice, cmap='gray')

            # Set color: Green if correct, Red if wrong
            color = 'green' if gt_label == pred_label else 'red'

            plt.title(f"Pred: {pred_str} ({confidence_score}-Conf)\nActual: {gt_str}",
                      color=color, fontsize=12)
            plt.axis('off')

            # Save the figure
            save_path = os.path.join(save_dir, f"result_{i}_gt_{gt_str}.png")
            plt.savefig(save_path, bbox_inches='tight')

            # Optional: Show in notebook/console
            # plt.show()
            plt.close() # Close to free up memory

if __name__ == "__main__":
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
    evaluate_and_visualize("models/best_resnet18_qc_w_conf.pth", val_data, transforms, device)