#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def check_docker():
    """Check if Docker is installed and accessible"""
    try:
        # Run docker --version to check if Docker is available
        result = subprocess.run(
            ['docker', '--version'],
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0:
            print("Error: Docker is not accessible")
            print("Make sure Docker is installed and you have proper permissions")
            print("\nPossible issues:")
            print("  1. Docker not installed")
            print("  2. Docker daemon not running")
            print("  3. User not in docker group (sudo may be required)")
            return False

        # Parse version to confirm it's working
        version = result.stdout.strip()
        print(f"✓ Docker found: {version}")
        return True

    except FileNotFoundError:
        print("Error: Docker command not found")
        print("Please install Docker: https://docs.docker.com/get-docker/")
        return False
    except Exception as e:
        print(f"Error checking Docker: {e}")
        return False

def check_paths(bids_dir: Path, output_dir: Path):
    """Check if input paths exist and are valid"""

    # Check BIDS directory
    if not bids_dir.exists():
        print(f"Error: BIDS directory not found: {bids_dir}")
        return False

    # Check for dataset_description.json (minimal BIDS requirement)
    dataset_desc = bids_dir / 'dataset_description.json'
    if not dataset_desc.exists():
        print(f"Warning: {dataset_desc} not found")
        print("This may not be a valid BIDS dataset")

    # Check for at least one subject
    subject_dirs = list(bids_dir.glob('sub-*'))
    if not subject_dirs:
        print(f"Warning: No subject directories found in {bids_dir}")
        print("Make sure the BIDS dataset has subjects (sub-<ID> directories)")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory ready: {output_dir}")

    return True

def run_mriqc(bids_dir: Path, output_dir: Path):
    """Run MRIQC using Docker"""

    # Convert to absolute paths
    bids_dir_abs = bids_dir.absolute()
    output_dir_abs = output_dir.absolute()

    # Docker command
    docker_cmd = [
        'docker', 'run', '-it', '--rm',
        '-v', f'{bids_dir_abs}:/data:ro',
        '-v', f'{output_dir_abs}:/out',
        'nipreps/mriqc:latest',
        '/data', '/out', 'participant'
    ]

    print("\nRunning MRIQC with command:")
    print(' '.join(docker_cmd))
    print("\nThis may take a while depending on the dataset size...")
    print("MRIQC will process all subjects and generate quality reports\n")

    try:
        # Run the command
        result = subprocess.run(
            docker_cmd,
            check=False
        )

        if result.returncode == 0:
            print("\n✓ MRIQC completed successfully!")
            print(f"Results saved in: {output_dir_abs}")
            print(f"  - HTML reports: {output_dir_abs}/reports/")
            print(f"  - Quality metrics: {output_dir_abs}/mriqc_*.json")
            return True
        else:
            print(f"\n✗ MRIQC failed with exit code {result.returncode}")
            return False

    except KeyboardInterrupt:
        print("\n\nMRIQC interrupted by user")
        return False
    except Exception as e:
        print(f"\nError running MRIQC: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Run MRIQC on BIDS dataset using Docker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  run-mri-qc /data/abide-bids /data/abide-mriqc
  run-mri-qc ./bids_dataset ./qc_output
        """
    )

    parser.add_argument('bids_dir',
        type=Path,
        help='Path to BIDS dataset directory')
    parser.add_argument('output_dir',
        type=Path,
        help='Output directory for MRIQC results')

    args = parser.parse_args()

    # Check Docker availability
    print("Checking Docker installation...")
    if not check_docker():
        print("\nTroubleshooting:")
        print("  - Install Docker: https://docs.docker.com/get-docker/")
        print("  - Start Docker daemon: 'sudo systemctl start docker' (Linux)")
        print("  - Add user to docker group: 'sudo usermod -aG docker $USER'")
        print("  - Log out and back in for group changes to take effect")
        sys.exit(1)

    # Check paths
    print("\nChecking input paths...")
    if not check_paths(args.bids_dir, args.output_dir):
        sys.exit(1)

    # Run MRIQC
    success = run_mriqc(args.bids_dir, args.output_dir)

    if success:
        print("\n✓ Processing complete!")
        sys.exit(0)
    else:
        print("\n✗ Processing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
