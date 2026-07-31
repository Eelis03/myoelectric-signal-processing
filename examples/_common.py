"""Shared wiring for the example scripts.

The example scripts contain no algorithm logic. They parse two arguments, call the
pipeline and analysis layers, print the tables, and optionally write a figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_OUTPUT = Path("outputs")


def parse_arguments(description: str) -> argparse.Namespace:
    """Parse the two arguments that every example script accepts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run with reduced settings, used by the integration tests",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="directory for figures, created if absent",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="skip figure generation",
    )
    return parser.parse_args()


def heading(title: str) -> None:
    """Print a section heading."""
    print()
    print(title)
    print("=" * len(title))
