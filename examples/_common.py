"""Shared wiring for the example scripts.

The example scripts contain no algorithm logic. They parse two arguments, call the
pipeline and analysis layers, print the tables, and optionally write a figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_OUTPUT = Path("outputs")

# The tracked figures live at a fixed place in the repository, so the script that
# regenerates them defaults to an absolute path derived from its own location rather
# than to a path relative to the working directory. Regenerating the figures must not
# depend on which directory the command was typed in.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRACKED_FIGURES = REPOSITORY_ROOT / "docs" / "figures"


def parse_arguments(description: str, default_outdir: Path = DEFAULT_OUTPUT) -> argparse.Namespace:
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
        default=default_outdir,
        help=f"directory for figures, created if absent (default {default_outdir})",
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
