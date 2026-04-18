from pathlib import Path
import sys
from collections.abc import Sequence
from typing import Never

import nibabel as nib
import numpy as np


def print_warning(message: str):
    """
    Print a warning message.
    """

    print(f"WARNING: {message}", file=sys.stderr)


def print_error(message: str):
    """
    Print an error message.
    """

    print(f"ERROR: {message}", file=sys.stderr)


def print_error_exit(message: str) -> Never:
    """
    Print an error message and exit the program.
    """

    print_error(message)
    sys.exit(-1)


def format_size(size: int) -> str:
    """
    Format a file size as a string.
    """

    return f"{size / (1024**3):.2f} GB"


def format_size_difference(expected_size: int, actual_size: int) -> str:
    """
    Format a file size difference as a stirng.
    """

    return f"expected {format_size(expected_size)}, found: {format_size(actual_size)}"


def normal_variance(values: Sequence[float], min: float, max: float) -> float:
    """
    Get the normalized variance of a list of values within a given range.
    """

    # Population variance (ddof=0)
    var = np.var(values, ddof=0)

    max_var = ((min - max) ** 2) / 4.0
    consensus = 1 - (var / max_var)

    # Clamp to [0,1] to avoid tiny floating point errors
    return float(np.clip(consensus, 0.0, 1.0))


def is_nifti_file(path: Path):
    """
    Check if path is a NIfTI file using its extension.
    """

    return path.name.endswith('.nii') or path.name.endswith('.nii.gz')


def is_3d_nifti_file(path: Path):
    """
    Check if a file is a 3D NIfTI file (instead of a 4D one).
    """

    try:
        img = nib.load(path)
        # Check if it's 3D (len(shape) == 3) or 4D but with first dimension 1 (squeezable)
        if len(img.shape) == 3:
            return True
        elif len(img.shape) == 4 and img.shape[3] == 1:
            return True  # 4D with single volume, can be squeezed
        else:
            print(f"Skipping {path.name}: {img.shape} dimensions (not 3D T1)")
            return False
    except Exception as e:
        print(f"Warning: Could not read {path.name}: {e}")
        return False
