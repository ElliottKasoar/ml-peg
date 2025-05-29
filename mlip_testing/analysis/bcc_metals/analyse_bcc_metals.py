"""Analyse results from BCC metals tests."""

from __future__ import annotations

from pathlib import Path

from ase.io import read
import numpy as np
import pytest

from mlip_testing.analysis.utils.decorators import build_table, plot_bar
from mlip_testing.calcs.config import MLIPS

OUT_PATH = Path(__file__).parent.parent.parent / "calcs" / "bcc_metals" / "outputs"
ELEMENTS = ("W", "Mo", "Nb")


@pytest.fixture
@plot_bar(labels=ELEMENTS, filename="lattice_bar.json")
def get_lattice_consts():
    """Compare lattice constants."""
    results = {"ref": []} | {mlip: [] for mlip in MLIPS}

    # Reference data for "W", "Mo", "Nb"
    results["ref"] = [3.185, 3.163, 3.322]

    for mlip in MLIPS:
        for element in ELEMENTS:
            struct = read(OUT_PATH / f"{element}-{mlip}-opt.extxyz")
            results[mlip].append(struct.cell.get_bravais_lattice().vars()["a"])

    return results


@pytest.fixture
def lattice_consts_errors(get_lattice_consts):
    """Get metric for lattice constant errors."""
    mean_errors = {}
    for mlip in MLIPS:
        errors = []

        for ref_result, mlip_result in zip(
            get_lattice_consts["ref"], get_lattice_consts[mlip], strict=True
        ):
            if ref_result and mlip_result:
                errors.append(abs(mlip_result - ref_result))
            else:
                errors.append(0.0)

        mean_errors[mlip] = np.mean(errors)

    return mean_errors


@pytest.fixture
# @plot_scatter
def get_eos():
    """Compare energy-volume curves."""
    pytest.skip("Reference data missing")
    results = {"ref": []} | {mlip: [] for mlip in MLIPS}
    for mlip in MLIPS:
        for element in ELEMENTS:
            output_file = OUT_PATH / f"{element}-{mlip}-eos-fit.dat"
            with open(output_file, encoding="utf8") as f:
                data = f.readlines()
                _, e_0, _ = tuple(float(x) for x in data[1].split())
                results[mlip].append(e_0 / 2)

    return results


@pytest.fixture
# @plot_parity(plot_combined=True)
@plot_bar(labels=ELEMENTS, filename="vacancy_bar.json")
def get_vacancy_energies():
    """Compare vacancy energies."""
    results = {"ref": []} | {mlip: [] for mlip in MLIPS}

    # Reference data for "W", "Mo", "Nb"
    results["ref"] = [3.223, 2.787, 2.646]

    for mlip in MLIPS:
        for element in ELEMENTS:
            struct_supercell = read(
                OUT_PATH / f"{element}-{mlip}-supercell-results.extxyz"
            )
            supercell_energy = struct_supercell.get_potential_energy()

            struct_vacancy = read(OUT_PATH / f"{element}-{mlip}-vacancy-opt.extxyz")
            vacancy_energy = struct_vacancy.get_potential_energy()
            results[mlip].append(
                vacancy_energy
                - (len(struct_vacancy) * supercell_energy / len(struct_supercell))
            )
    return results


@pytest.fixture
def vacancy_energies_errors(get_vacancy_energies):
    """Get metric for vacancy energy errors."""
    mean_errors = {}
    for mlip in MLIPS:
        errors = []

        for ref_result, mlip_result in zip(
            get_vacancy_energies["ref"], get_vacancy_energies[mlip], strict=True
        ):
            if ref_result and mlip_result:
                errors.append(abs(mlip_result - ref_result))
            else:
                errors.append(0.0)

        mean_errors[mlip] = np.mean(errors)

    return mean_errors


@pytest.fixture
@build_table()
def metrics(vacancy_energies_errors, lattice_consts_errors):
    """Get all metrics."""
    return {
        "Vacancy energy": vacancy_energies_errors,
        "Lattice constants": lattice_consts_errors,
    }


def test_bcc_metals(metrics, coeffs=(1, 1)):
    """Combine all metrics."""
    results = {}
    for mlip in MLIPS:
        results[mlip] = 0
        for metric, coeff in zip(metrics, coeffs, strict=True):
            results[mlip] += metrics[metric][mlip] * coeff
