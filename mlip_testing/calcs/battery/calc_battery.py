"""run calculations for electrode-electrolyte interface and battery system."""

from __future__ import annotations

import pytest

from mlip_testing.calcs.config import MLIPS


@pytest.mark.slow
@pytest.mark.parametrize("mlip", MLIPS)
def test_battery(mlip):
    """Run calculations required for battery tests."""
    # Run MD...
    assert mlip
