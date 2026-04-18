#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from monai.data import DataLoader, Dataset
from monai.networks.nets import resnet18
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, Orientationd, Resized, ScaleIntensityd
from monai.transforms.utility.dictionary import Lambdad

from brain_mri_qc.utils import is_3d_nifti_file, is_nifti_file, print_error_exit


def predict_scans(model_path, scan_paths, transforms, device, verbose=True):
    """
    Run prediction on one or multiple scans.

    Args:
        model_path: Path to the trained model
        scan_paths: List of paths to NIfTI files (can be single path or list)
        transforms: MONAI transforms to apply
        device: torch device
        verbose: Whether to print detailed results

    Returns:
        List of dictionaries containing prediction results for each scan
    """
    # Convert single path to list for uniform handling
    if isinstance(scan_paths, (str, Path)):
        scan_paths = [str(scan_paths)]
    else:
        scan_paths = [str(p) for p in scan_paths]

    # 1. Load Model
    model = resnet18(spatial_dims=3, n_input_channels=1, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Prepare data list (dummy labels since we only care about prediction)
    data_list = [{"image": path, "label": 0} for path in scan_paths]

    # 3. Apply transforms
    ds = Dataset(data=data_list, transform=transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    results = []

    # 4. Run inference
    with torch.no_grad():
        for i, batch in enumerate(loader):
            inputs = batch["image"].to(device)

            # Get model output
            logit = model(inputs)

            prob_good = torch.sigmoid(logit).item()
            prob_bad = 1 - prob_good

            predicted_class = "Good" if prob_good > 0.5 else "Bad"
            confidence = abs(prob_good - 0.5) * 2

            result = {
                "scan_path": scan_paths[i],
                "scan_name": str(Path(scan_paths[i]).relative_to(Path(scan_paths[i]).anchor)),
                "prob_good": prob_good,
                "prob_bad": prob_bad,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "logit": logit.item()
            }
            results.append(result)

            # Print results if verbose
            if verbose:
                print(f"\nScan {i+1}: {result['scan_name']}")
                print(f"  Probability Good: {prob_good:.4f} ({prob_good*100:.2f}%)")
                print(f"  Probability Bad:  {prob_bad:.4f} ({prob_bad*100:.2f}%)")
                print(f"  Prediction: {predicted_class}")
                print(f"  Confidence: {confidence:.4f} ({confidence*100:.2f}%)")

    return results

def visualize_prediction(scan_path, prediction_result, save_path=None):
    """
    Visualize a scan with prediction overlay.

    Args:
        scan_path: Path to the original NIfTI file
        prediction_result: Dictionary from predict_scans
        save_path: Optional path to save the visualization
    """
    import nibabel as nib

    img = nib.load(scan_path)
    img_data = img.get_fdata()

    # Take middle slice
    mid_slice = img_data[:, :, img_data.shape[2] // 2]

    plt.figure(figsize=(8, 8))
    plt.imshow(mid_slice, cmap='gray')

    # Set title with prediction info
    title = f"Prediction: {prediction_result['predicted_class']}\n"
    title += f"Prob Good: {prediction_result['prob_good']:.3f} | "
    title += f"Prob Bad: {prediction_result['prob_bad']:.3f}\n"
    title += f"Confidence: {prediction_result['confidence']:.3f}"

    plt.title(title, fontsize=12)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Run QC prediction on brain MRI scans')
    parser.add_argument('input_path', type=str,
                       help='Path to a single NIfTI file or directory containing NIfTI files')
    parser.add_argument('--model', type=str, default='models/best_qc_model_synthesized.pth',
                       help='Path to the trained model (default: models/best_qc_model_synthesized.pth)')
    parser.add_argument('--output_csv', type=str, default=None,
                       help='Save results to CSV file (optional)')
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualization of the first scan (or all if used with --all)')
    parser.add_argument('--all', action='store_true',
                       help='Process all NIfTI files in directory (requires input_path to be directory)')
    parser.add_argument('--recursive', action='store_true',
                       help='Search recursively for NIfTI files in subdirectories')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output, only show final results')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'], help='Device to use (default: auto-detect)')

    args = parser.parse_args()

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transforms (same as training)
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS", labels=None),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(128, 128, 128)),
    ])

    # Collect scan paths
    input_path = Path(args.input_path)
    scan_paths = []

    if input_path.is_file():
        # Single file
        if is_nifti_file(input_path):
            scan_paths = [input_path]
        else:
            print_error_exit(f"{input_path} is not a NIfTI file (.nii or .nii.gz)")
    elif input_path.is_dir():
        # Directory
        pattern = "**/*.nii*" if args.recursive else "*.nii*"
        scan_paths = list(input_path.glob(pattern))
        # Filter for .nii and .nii.gz
        scan_paths = [p for p in scan_paths if is_nifti_file(p)]
        print(f"Found {len(scan_paths)} total NIfTI files")
        scan_paths = [p for p in scan_paths if is_3d_nifti_file(p)]
        print(f"Keeping {len(scan_paths)} 3D scans (skipping 4D/time series)")

        if not scan_paths:
            print_error_exit(f"No NIfTI files found in {input_path}")

        if not args.all and len(scan_paths) > 1:
            print(f"Found {len(scan_paths)} files in directory. Use --all to process all, or specify a single file.")
            print("First 5 files found:")
            for p in scan_paths[:5]:
                print(f"  - {p}")
            sys.exit(1)
    else:
        print_error_exit(f"{input_path} does not exist")

    # Run prediction
    print(f"\n{'='*50}")
    print(f"PREDICTING ON {len(scan_paths)} SCAN(S)")
    print(f"{'='*50}")

    results = predict_scans(
        model_path=args.model,
        scan_paths=scan_paths,
        transforms=transforms,
        device=device,
        verbose=not args.quiet
    )

    # Print summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['scan_name']}: {result['predicted_class']} "
              f"(Good: {result['prob_good']:.3f}, Bad: {result['prob_bad']:.3f})")

    # Save to CSV if requested
    if args.output_csv:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(args.output_csv, index=False)
        print(f"\nResults saved to {args.output_csv}")

    # Visualize if requested
    if args.visualize:
        if len(results) == 1 or args.all:
            for i, result in enumerate(results):
                vis_path = f"visualization_{i+1}_{result['scan_name'].replace('.nii', '').replace('.gz', '')}.png"
                visualize_prediction(result['scan_path'], result, save_path=vis_path)
        else:
            # Visualize first result only
            visualize_prediction(results[0]['scan_path'], results[0],
                               save_path=f"visualization_{results[0]['scan_name'].replace('.nii', '').replace('.gz', '')}.png")

    # Return results for potential scripting use
    return results

if __name__ == '__main__':
    results = main()
