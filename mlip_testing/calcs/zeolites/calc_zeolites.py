"""Run calculations for BCC metals tests."""

from __future__ import annotations

from pathlib import Path

from ase.io import read
from janus_core.calculations.md import NPT
from janus_core.calculations.neb import NEB
import pytest
from yaml import safe_load

from mlip_testing.calcs.config import MLIPS

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"


@pytest.fixture(scope="module")
def structs():
    """Structures to run calculations on."""
    return [read(struct_path) for struct_path in DATA_PATH.glob("*.cif")]


@pytest.fixture(scope="module")
def md_config():
    """Load configuration file for simulation."""
    with open(DATA_PATH / "npt.yml", encoding="utf8") as file:
        return safe_load(file)


@pytest.fixture(scope="module")
def neb_config():
    """Load configuration file for simulation."""
    with open(DATA_PATH / "neb.yml", encoding="utf8") as file:
        return safe_load(file)


@pytest.mark.slow
@pytest.mark.parametrize("mlip", MLIPS)
def test_md(structs, md_config, mlip):
    """Run molecular dynamics simulations for structures."""
    for key in ("struct", "arch", "model", "file_prefix", "tracker"):
        md_config.pop(key, None)
    for struct in structs:
        npt = NPT(
            struct=struct.copy(),
            arch="mace_mp",
            model_path=mlip,
            file_prefix=OUT_PATH / f"{struct.get_chemical_formula()}-{mlip}-md",
            **md_config,
        )
        npt.run()


@pytest.mark.slow
@pytest.mark.parametrize("mlip", MLIPS)
def test_neb(structs, neb_config, mlip):
    """Run NEB."""
    for key in (
        "init_struct",
        "final_struct",
        "arch",
        "model",
        "file_prefix",
        "tracker",
    ):
        neb_config.pop(key, None)
    for struct in structs:
        neb = NEB(
            init_struct=DATA_PATH / "MOR-Al-20H2O-acetone-stateA.cif",
            final_struct=DATA_PATH / "MOR-Al-20H2O-acetone-stateB.cif",
            arch="mace_mp",
            model_path=mlip,
            file_prefix=OUT_PATH / f"{struct.get_chemical_formula()}-{mlip}-neb",
            **neb_config,
        )
        neb.run()
