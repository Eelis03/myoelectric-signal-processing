"""Fixtures shared by the test suite.

Measurement helpers and the tolerance derivations live in :mod:`tests.helpers`.
"""

from __future__ import annotations

import pytest

from tests.helpers import SAMPLE_RATE_HZ


@pytest.fixture
def sample_rate_hz() -> float:
    """Sample rate used by every test that does not need another one."""
    return SAMPLE_RATE_HZ
