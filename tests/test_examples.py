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
    "make_figures.py",
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


def test_the_figure_script_writes_every_tracked_figure_inside_the_budget(
    tmp_path: Path,
) -> None:
    """The regeneration command produces exactly the tracked set and reports its size.

    Run into a temporary directory rather than over the tracked figures, because the
    tracked files are a committed snapshot and a test must not rewrite them. What is
    checked is the number of files, that each is a PNG, and that the total is inside
    the budget the repository is held to, which is the failure mode a size regression
    would take.
    """
    completed = _run("make_figures.py", "--quick", "--outdir", str(tmp_path))
    assert completed.returncode == 0, completed.stderr
    written = sorted(tmp_path.glob("*.png"))
    assert [path.name for path in written] == [
        "amplitude-latency-ripple.png",
        "fatigue-median-frequency.png",
        "onset-detectors.png",
    ]
    assert all(path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" for path in written)
    assert sum(path.stat().st_size for path in written) <= 250 * 1024
    assert "of a 250 KB budget" in completed.stdout


def test_the_tracked_figures_are_committed_and_inside_the_budget() -> None:
    """The snapshot in the repository is what the README embeds, so it is checked here.

    The check is on the set of files and on their total size, never on their bytes.
    Matplotlib output is not byte reproducible across platforms or across its own
    releases, so a byte comparison would fail on a runner that changed nothing.
    """
    figures = sorted((EXAMPLES.parent / "docs" / "figures").glob("*.png"))
    assert [path.name for path in figures] == [
        "amplitude-latency-ripple.png",
        "fatigue-median-frequency.png",
        "onset-detectors.png",
    ]
    assert sum(path.stat().st_size for path in figures) <= 250 * 1024
