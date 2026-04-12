#!/usr/bin/env python
import argparse
import time
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path
import sys
import urllib3
import os

import requests
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from brain_mri_qc.abide import ABIDE_1, ABIDE_2, Link
from brain_mri_qc.utils import format_size, format_size_difference, print_error, print_error_exit

@dataclass
class FileInfo:
    name: str
    size: int

def login(username: str, password: str) -> requests.Session:
    """Log in to NITRC and return an authenticated session."""
    login_url = "https://www.nitrc.org/account/login.php"
    login_data = {
        'form_loginname': username,
        'form_pw': password,
        'return_to': '',
        'login': 'Login'
    }

    session = requests.Session()
    try:
        response = session.post(login_url, data=login_data, allow_redirects=True)
        if response.status_code == 200:
            print(f"Successfully logged in as {username}")
            return session
        else:
            print_error_exit(f"Login failed for {username}")
    except requests.RequestException as e:
        print_error_exit(f"✗ Login error: {e}")

def get_file_info(session: requests.Session, link: Link) -> FileInfo:
    """Get file name and size from NITRC using a HEAD request."""
    try:
        response = session.head(link.url, allow_redirects=True, timeout=30)
        response.raise_for_status()

        content_length = response.headers.get('Content-Length')
        if content_length is None:
            print_error_exit(f"Missing content length from {link.url}")

        size = int(content_length)
        name = Path(response.url).name
        return FileInfo(name, size)
    except requests.RequestException as e:
        print_error_exit(f"Could not get file info from {link.url}: {e}")

def download_file(session: requests.Session, link: Link, file_info: FileInfo, output_dir: Path):
    """Download a file with resume/overwrite support."""
    output_path = output_dir / file_info.name

    if output_path.exists():
        actual_size = output_path.stat().st_size
        if actual_size == file_info.size:
            print("  File already exists with correct size, skipping...")
            return True # Success
        else:
            print(f"  Size mismatch: {file_info.name} ({format_size_difference(file_info.size, actual_size)})")
            output_path.unlink()

    try:
        # Added a 30s timeout to prevent hanging on dead connections
        response = session.get(link.url, stream=True, allow_redirects=True, timeout=30)
        response.raise_for_status()

        output_dir.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024): # 1MB chunks for efficiency
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress = (downloaded / file_info.size) * 100
                    print(f"\r  Progress: {progress:.1f}% ({downloaded / (1024**3):.2f} GB / {file_info.size / (1024**3):.2f} GB)", end='')

        print() # New line after progress bar

        if downloaded != file_info.size:
            print_error("Download size mismatch. Cleaning up...")
            if output_path.exists(): output_path.unlink()
            return False

        print("  Download success")
        return True

    except (requests.RequestException, Exception) as e:
        print_error(f"\n  Error during download: {e}")
        if output_path.exists(): output_path.unlink()
        return False

def main():
    parser = argparse.ArgumentParser(description="Download script for the ABIDE dataset scan data")
    parser.add_argument('dataset', choices=['abide-i', 'abide-ii'], help="The ABIDE dataset to download")
    parser.add_argument('output', type=Path, help="The directory in which to place the downloaded files")
    parser.add_argument('--username', help="NITRC username")
    parser.add_argument('--password', help="NITRC password")

    args = parser.parse_args()

    username = args.username or input("Enter your NITRC username: ")
    password = args.password or getpass("Enter your NITRC password: ")
    output_dir = args.output

    session = login(username, password)
    session.verify = False
    urllib3.disable_warnings(category=urllib3.exceptions.InsecureRequestWarning)

    dataset = ABIDE_1 if args.dataset == 'abide-i' else ABIDE_2
    
    # Flatten the list of links for easier progress tracking
    all_links = [link for institution in dataset for link in institution.links]
    total_files = len(all_links)

    for idx, link in enumerate(all_links, start=1):
        file_info = get_file_info(session, link)
        print(f"[{idx}/{total_files}] Processing {file_info.name} ({format_size(file_info.size)})")

        max_retries = 5
        success = False
        
        for attempt in range(max_retries):
            if download_file(session, link, file_info, output_dir):
                success = True
                break
            else:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"  Retrying in {wait_time}s (Attempt {attempt + 2}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print_error(f"  Failed to download {file_info.name} after {max_retries} attempts.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(0)
