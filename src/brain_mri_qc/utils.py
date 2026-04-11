import sys
from typing import Never


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
