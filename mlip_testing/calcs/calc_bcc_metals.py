"""Run calculations for BCC metals tests."""

from __future__ import annotations

from pathlib import Path

from ase.build import bulk, make_supercell
from janus_core.calculations.eos import EoS
from janus_core.calculations.geom_opt import GeomOpt
from janus_core.calculations.single_point import SinglePoint
import numpy as np
import pytest

DATA_PATH = Path(__file__).parent / "data" / "bcc_metals"
OUT_PATH = Path(__file__).parent / "outputs" / "bcc_metals"

MLIPS = ("medium", "medium-mpa-0", "medium-omat-0")

# @pytest.fixture(scope="session")
# def mlips():
#     """MLIPs to run calculations for."""
#     return ("medium", "medium-mpa-0", "medium-omat-0")


@pytest.fixture(scope="module")
def structs():
    """Structures to run calculations on."""
    return [bulk(metal) for metal in ("W", "Mo", "Nb")]
    # return list(DATA_PATH.glob("*.cif"))


@pytest.fixture(scope="module")
def relax_structs(structs):
    """Run geometry optimisation on all structures."""
    relaxed_structs = {}
    for struct in structs:
        for mlip in MLIPS:
            geomopt = GeomOpt(
                struct=struct,
                arch="mace_mp",
                model_path=mlip,
                write_results=True,
                write_traj=True,
                file_prefix=OUT_PATH / f"{struct.get_chemical_formula()}-{mlip}",
            )
            geomopt.run()
            relaxed_structs[f"{struct.get_chemical_formula()}-{mlip}"] = geomopt.struct
    return relaxed_structs


@pytest.mark.parametrize("mlip", MLIPS)
def test_lattice_consts(relax_structs, mlip):
    """Run calculations required for lattice constants."""
    # Only need relaxed structures
    assert relax_structs


@pytest.mark.parametrize("mlip", MLIPS)
def test_eos(structs, relax_structs, mlip):
    """Run calculations required for energy-volume curves."""
    for struct in structs:
        eos = EoS(
            struct=relax_structs[f"{struct.get_chemical_formula()}-{mlip}"],
            arch="mace_mp",
            model_path=mlip,
            minimize=True,
            minimize_all=True,
            minimize_kwargs={"steps": 50},
            write_results=True,
            write_structures=True,
            plot_to_file=True,
            file_prefix=OUT_PATH / f"{struct.get_chemical_formula()}-{mlip}",
        )
        eos.run()


@pytest.mark.parametrize("mlip", MLIPS)
def test_vacancy(structs, relax_structs, mlip):
    """Run calculations required to get vacancy energies."""
    for struct in structs:
        struct_supercell = make_supercell(
            relax_structs[f"{struct.get_chemical_formula()}-{mlip}"], np.eye(3) * 5
        )

        single_point = SinglePoint(
            struct=struct_supercell,
            arch="mace_mp",
            model_path=mlip,
            write_results=True,
            file_prefix=OUT_PATH / f"{struct.get_chemical_formula()}-{mlip}-supercell",
        )
        single_point.run()

        del struct_supercell[0]
        geomopt = GeomOpt(
            struct=struct_supercell,
            fmax=1e-5,
            filter_func=None,
            arch="mace_mp",
            model_path=mlip,
            write_results=True,
            write_traj=True,
            file_prefix=OUT_PATH / f"{struct.get_chemical_formula()}-{mlip}-vacancy",
        )
        geomopt.run()
