"""Integration tier: every example script runs to completion under reduced settings.

The scripts are run as subprocesses so that the test exercises the same entry point a
reader would use, including the argument parsing and the import of ``_common`` from the
examples directory. Reduced settings are selected with ``--quick``.

One script is also run with figure generation enabled, writing into a temporary
directory, so that the figure code is covered without leaving files in the repository.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SCRIPTS = (
    "generate_signal.py",
    "filter_design.py",
    "onset_benchmark.py",
    "feature_report.py",
    "fatigue_demo.py",
    "amplitude_latency.py",
)


def _run(script: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXAMPLES / script), *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


def test_every_script_is_listed() -> None:
    """The integration tier covers every script in the examples directory."""
    found = {path.name for path in EXAMPLES.glob("*.py") if not path.name.startswith("_")}
    assert found == set(SCRIPTS)


@pytest.mark.parametrize("script", SCRIPTS)
def test_example_script_runs_to_completion(script: str) -> None:
    """The script exits cleanly and writes something to standard output."""
    completed = _run(script, "--quick", "--no-figures")
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip()


def test_one_script_produces_its_figure(tmp_path: Path) -> None:
    """Figure generation runs and writes a file, without leaving it in the repository."""
    completed = _run("fatigue_demo.py", "--quick", "--outdir", str(tmp_path))
    assert completed.returncode == 0, completed.stderr
    written = list(tmp_path.glob("*.png"))
    assert len(written) == 1
    assert written[0].stat().st_size > 0
