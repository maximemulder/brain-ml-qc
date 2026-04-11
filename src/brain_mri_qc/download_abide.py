#!/usr/bin/env python
import argparse
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path
import sys
import urllib3

import requests

from brain_mri_qc.abide import ABIDE_1, ABIDE_2, Link
from brain_mri_qc.utils import format_size, format_size_difference, print_error, print_error_exit


@dataclass
class FileInfo:
    name: str
    size: int


def login(username: str, password: str) -> requests.Session:
    """
    Log in to NITRC and return an authenticated session.
    """

    login_url = "https://www.nitrc.org/account/login.php"

    login_data = {
        'form_loginname': username,
        'form_pw': password,
        'return_to': '',
        'login': 'Login'
    }

    session = requests.Session()
    # session.verify = False

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
    """
    Get a file and size from NITRC using HEAD request.
    """

    try:
        # Send HEAD request that follows redirects
        response = session.head(link.url, allow_redirects=True)
        response.raise_for_status()

        # Get file size from Content-Length header

        content_length = response.headers.get('Content-Length')
        if content_length is None:
            print_error_exit(f"Missing content length from {link.url}")

        size = int(content_length)

        # Get filename from the final URL after redirect
        name = Path(response.url).name

        return FileInfo(name, size)
    except requests.RequestException as e:
        print_error_exit(f"Could not get file info from {link.url}: {e}")


def download_file(session: requests.Session, link: Link, file_info: FileInfo, output_dir: Path):
    """
    Download a file with incremental resume support based on filename and size.
    """

    output_path = output_dir / file_info.name

    # Check if file already exists with correct size
    if output_path.exists():
        actual_size = output_path.stat().st_size
        if actual_size == file_info.size:
            print("  File already exists with correct size, skipping...")
            return
        else:
            print(f"  File exists but size mismatch: {file_info.name} ({format_size_difference(file_info.size, actual_size)})")
            print("  Re-downloading...")

            output_path.unlink()

    try:
        response = session.get(link.url, stream=True, allow_redirects=True)
        response.raise_for_status()

        output_dir.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Progress indicator
                    progress = (downloaded / file_info.size) * 100
                    print(f"\r  Progress: {progress:.1f}% ({downloaded / (1024**3):.2f} GB / {file_info.size / (1024**3):.2f} GB)", end='')

        # New line after progress
        print()

        # Verify download
        if downloaded != file_info.size:
            print_error("Download size mismatch. Cleaning up...")
            output_path.unlink()
            sys.exit(-1)

        print("  Download success")

    except requests.RequestException as e:
        print_error(f"Error downloading {link.url}: {e}")
        # Clean up partial download
        if output_path.exists():
            output_path.unlink()
        sys.exit(-1)


def main():
    parser = argparse.ArgumentParser(description="Download script for the ABIDE dataset scan data")

    parser.add_argument('dataset',
        choices=['abide-i', 'abide-ii'],
        help="The ABIDE dataset to download")

    parser.add_argument('output',
        type=Path,
        help="The directory in which to place the downloaded files")

    parser.add_argument('--username',
        help="NITRC username")

    parser.add_argument('--password',
        help="NITRC password (if not provided, will prompt securely)")

    args = parser.parse_args()

    username = args.username
    if not username:
        username = input("Enter your NITRC username: ")

    password = args.password
    if not password:
        password = getpass("Enter your NITRC password: ")

    output_dir = args.output

    session = login(username, password)

    # For some reasons the NITRC SSL ceritificates do not work for the redirects.
    session.verify = False
    urllib3.disable_warnings(category=urllib3.exceptions.InsecureRequestWarning)

    match args.dataset:
        case 'abide-i':
            dataset = ABIDE_1
        case 'abide-ii':
            dataset = ABIDE_2

    total = sum((len(institution.links) for institution in dataset))

    for i, institution in enumerate(dataset, start=1):
        for j, link in enumerate(institution.links):
            file_info = get_file_info(session, link)
            print(f"Downloading {file_info.name} ({format_size(file_info.size)}) ({i + j}/{total})")
            download_file(session, link, file_info, output_dir)


if __name__ == '__main__':
    main()
