#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

from brain_mri_qc.utils import format_int_string, print_error_exit, print_warning


def process_subject(input_dir: Path, output_dir: Path, site: str, subject_id: str):
    """
    Process a single subject: copy and rename files to BIDS structure.
    Removes trailing '00' from subject IDs.
    """

    # Remove trailing '00' from subject ID if present
    bids_subject_id = format_int_string(subject_id)

    # Subject directory in BIDS
    subject_bids_dir = output_dir / f'sub-{bids_subject_id}'
    subject_bids_dir.mkdir(parents=True, exist_ok=True)

    # Path to subject data in input
    subject_input_dir = input_dir / site / subject_id / 'session_1'

    if not subject_input_dir.exists():
        print_warning(f"{subject_input_dir} does not exist, skipping")
        return False

    # Process anatomical image
    anat_input = subject_input_dir / 'anat_1' / 'mprage.nii.gz'
    if anat_input.exists():
        anat_output_dir = subject_bids_dir / 'anat'
        anat_output_dir.mkdir(parents=True, exist_ok=True)
        anat_output = anat_output_dir / f'sub-{bids_subject_id}_T1w.nii.gz'
        shutil.copy2(anat_input, anat_output)
        print(f"  Copied anatomical: {anat_output}")
    else:
        print_warning(f"  No anatomical image found for {site}/{subject_id}")

    # Process resting state functional image
    func_input = subject_input_dir / 'rest_1' / 'rest.nii.gz'
    if func_input.exists():
        func_output_dir = subject_bids_dir / 'func'
        func_output_dir.mkdir(parents=True, exist_ok=True)
        func_output = func_output_dir / f'sub-{bids_subject_id}_task-rest_bold.nii.gz'
        shutil.copy2(func_input, func_output)
        print(f"  Copied functional: {func_output}")
    else:
        print_warning(f"  No functional image found for {site}/{subject_id}")

    return True

def create_dataset_description(output_dir):
    """Create minimal dataset_description.json"""

    description = {
        "Name": "ABIDE dataset converted to BIDS",
        "BIDSVersion": "1.0.0"
    }
    desc_file = output_dir / 'dataset_description.json'
    with open(desc_file, 'w') as f:
        json.dump(description, f, indent=4)
    print(f"Created {desc_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert ABIDE dataset to BIDS format')

    parser.add_argument('input_dir',
        type=Path,
        help='Path to ABIDE directory')
    parser.add_argument('output_dir',
        type=Path,
        help='Output BIDS directory')

    parser.add_argument('--sites',
        required=True,
        help='Comma-separated list of sites to process')

    args =  parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    sites = [site.strip() for site in args.sites.split(',')]

    # Check if input directory exists
    if not input_dir.exists():
        print_error_exit(f"Input directory {input_dir} does not exist")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each site
    for site in sites:
        site_dir = input_dir / site
        if not site_dir.exists():
            print(f"Warning: Site {site} not found in {input_dir}, skipping")
            continue

        print(f"\nProcessing site: {site}")

        # Get all subject directories (should be numeric IDs)
        subject_dirs = [d for d in site_dir.iterdir() if d.is_dir()]

        for subject_dir in subject_dirs:
            subject_id = subject_dir.name
            print(f"  Processing subject: {subject_id}")
            process_subject(input_dir, output_dir, site, subject_id)

    # Create minimal BIDS dataset description
    create_dataset_description(output_dir)

    print(f"\nDone! BIDS dataset created in {output_dir}")

if __name__ == "__main__":
    main()
