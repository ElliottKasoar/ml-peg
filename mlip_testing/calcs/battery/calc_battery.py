"""Run calculations for electrode-electrolyte interface and battery system."""

from __future__ import annotations

from pathlib import Path

from ase import units
from janus_core.calculations.single_point import SinglePoint
import pytest

from mlip_testing.calcs.config import MLIPS

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"
DENS_FACT = (units.m / 1.0e2) ** 3 / units.mol


@pytest.mark.parametrize("mlip", MLIPS)
def test_intra_inter(mlip):
    """Run calculations required for intra/intermolecule comparison."""
    struct_paths = (DATA_PATH / "intra_inter").glob("*.xyz")

    for struct_path in struct_paths:
        SinglePoint(
            struct=struct_path,
            arch="mace_mp",
            model_path=mlip,
            write_results=True,
            file_prefix=OUT_PATH / "intra_inter" / f"{struct_path.stem}-{mlip}",
        ).run()


@pytest.mark.slow
@pytest.mark.parametrize("mlip", MLIPS)
def test_battery(mlip):
    """Run calculations required for battery tests."""
    # Run MD...
    assert mlip
