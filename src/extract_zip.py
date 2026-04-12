import tarfile
import os
from pathlib import Path

def extract_abide_data(input_dir: str, output_base_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_base_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all .tgz files in the input directory
    tgz_files = list(input_path.glob("*.tgz"))
    
    if not tgz_files:
        print(f"No .tgz files found in {input_dir}")
        return

    print(f"Found {len(tgz_files)} files. Starting extraction and cleanup...")

    for i, file in enumerate(tgz_files, 1):
        site_name = file.stem.replace(".tar", "") 
        target_dir = output_path / site_name
        target_dir.mkdir(exist_ok=True)

        print(f"[{i}/{len(tgz_files)}] Extracting {file.name}...")
        
        try:
            with tarfile.open(file, "r:gz") as tar:
                tar.extractall(path=target_dir)
            
            # --- STORAGE CLEANUP LOGIC ---
            # Only delete if the line above didn't raise an exception
            print(f"  Successfully extracted. Removing {file.name} to save space...")
            file.unlink() 
            
        except Exception as e:
            print(f"  Error extracting {file.name}: {e}")
            print(f"  Keeping {file.name} for manual check.")

    print("\nDone! All datasets extracted and zip files cleaned up.")

if __name__ == "__main__":
    # Updated paths
    SOURCE_DIR = "/brain-ml-qc/files/ABIDE1"
    DEST_DIR = "/brain-ml-qc/files/ABIDE1/extracted"
    
    extract_abide_data(SOURCE_DIR, DEST_DIR)