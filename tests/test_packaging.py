"""The package declares its own types under PEP 561.

Passing mypy in strict mode says nothing about what an installed copy delivers. Without
a ``py.typed`` marker inside the package directory, every type checker treats the
package as untyped and every annotation in it is invisible to anything that depends on
it, so the guarantee stops at this repository's own boundary.
"""

from __future__ import annotations

from pathlib import Path

import myoelectric


def _package_directory() -> Path:
    """Directory that ``import myoelectric`` actually resolves to."""
    module_file = myoelectric.__file__
    assert module_file is not None, "myoelectric must be a regular package with a file"
    return Path(module_file).resolve().parent


def test_py_typed_marker_sits_inside_the_package_directory() -> None:
    """The marker is a sibling of ``__init__.py``, which is where PEP 561 requires it."""
    package = _package_directory()
    assert (package / "__init__.py").is_file()
    assert (package / "py.typed").is_file()


def test_py_typed_marker_is_empty() -> None:
    """PEP 561 defines the marker by its presence, so it carries no content."""
    assert (_package_directory() / "py.typed").read_bytes() == b""


def test_py_typed_marker_ships_with_the_wheel() -> None:
    """The wheel build includes the whole package directory, marker included.

    Hatchling packages ``src/myoelectric`` as a directory, so a file inside it is
    included without needing to be listed. This test pins the configuration that makes
    that true, because a change to ``packages`` that switched to an explicit include
    list would silently drop the marker from the distribution while leaving every test
    above passing on the source tree.
    """
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert 'packages = ["src/myoelectric"]' in text
