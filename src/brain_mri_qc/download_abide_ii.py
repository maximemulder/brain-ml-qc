#!/usr/bin/env python
import argparse
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path
import sys
import urllib3

import requests

from brain_mri_qc.utils import format_size, format_size_difference, print_error, print_error_exit


@dataclass
class Link:
    name: str
    url: str


@dataclass
class Institution:
    name: str
    links: list[Link]


@dataclass
class FileInfo:
    name: str
    size: int


# Links extracted from: https://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html
institutions = [
    Institution(
        name="Barrow Neurological Institute",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9066")]
    ),
    Institution(
        name="Erasmus University Medical CenterRotterdam",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9064")]
    ),
    Institution(
        name="ETH Zürich",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9090")]
    ),
    Institution(
        name="Georgetown University",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9068")]
    ),
    Institution(
        name="Indiana University",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9070")]
    ),
    Institution(
        name="Institut Pasteur and Robert Debré Hospital",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9072")]
    ),
    Institution(
        name="Katholieke Universiteit Leuven",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9086")]
    ),
    Institution(
        name="Kennedy Krieger Institute",
        links=[
            Link(name="Scan Data A", url="https://www.nitrc.org/frs/downloadlink.php/9098"),
            Link(name="Scan Data B", url="https://www.nitrc.org/frs/downloadlink.php/9100"),
            Link(name="Scan Data C", url="https://www.nitrc.org/frs/downloadlink.php/9101"),
            Link(name="Scan Data D", url="https://www.nitrc.org/frs/downloadlink.php/9102")
        ]
    ),
    Institution(
        name="NYU Langone Medical Center:Sample 1",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9074")]
    ),
    Institution(
        name="NYU Langone Medical Center:Sample 2",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9062")]
    ),
    Institution(
        name="Olin Neuropsychiatry Research Center, Institute of Living at Hartford Hospital",
        links=[
            Link(name="Scan Data A", url="https://www.nitrc.org/frs/downloadlink.php/9104"),
            Link(name="Scan Data B", url="https://www.nitrc.org/frs/downloadlink.php/9105"),
            Link(name="Scan Data C", url="https://www.nitrc.org/frs/downloadlink.php/9106"),
            Link(name="Scan Data D", url="https://www.nitrc.org/frs/downloadlink.php/9107")
        ]
    ),
    Institution(
        name="Oregon Health and Science University",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9076")]
    ),
    Institution(
        name="Trinity Centre for Health Sciences",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9080")]
    ),
    Institution(
        name="San Diego State University",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9078")]
    ),
    Institution(
        name="Stanford University",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9937")]
    ),
    Institution(
        name="University of California Davis",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9082")]
    ),
    Institution(
        name="University of California Los Angeles",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9084")]
    ),
    Institution(
        name="University of Miami",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9935")]
    ),
    Institution(
        name="University of Utah School of Medicine",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9088")]
    ),
    Institution(
        name="University of California Los Angeles: Longitudinal Sample",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9093")]
    ),
    Institution(
        name="University of Pittsburgh School of Medicine: Longitudinal Sample",
        links=[Link(name="Scan Data", url="https://www.nitrc.org/frs/downloadlink.php/9095")]
    )
]

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

        print("  Download succesful")

    except requests.RequestException as e:
        print_error(f"Error downloading {link.url}: {e}")
        # Clean up partial download
        if output_path.exists():
            output_path.unlink()
        sys.exit(-1)


def main():
    parser = argparse.ArgumentParser(description="Download script for the ABIDE II scan data")
    parser.add_argument('--username', help="NITRC username")
    parser.add_argument('--password', help="NITRC password (if not provided, will prompt securely)")
    parser.add_argument('output', type=Path, help="The directory in which to place the downloaded files")
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

    total = sum((len(institution.links) for institution in institutions))

    for i, institution in enumerate(institutions, start=1):
        for j, link in enumerate(institution.links):
            file_info = get_file_info(session, link)
            print(f"Downloading {file_info.name} ({format_size(file_info.size)}) ({i + j}/{total})")
            download_file(session, link, file_info, output_dir)


if __name__ == '__main__':
    main()
