"""Helper functions for MOF EoS test."""

from __future__ import annotations

from csv import reader
from pathlib import Path

from ase.io import read
import numpy as np

REF_FILE = Path(
    "/Users/elliottkasoar/Documents/PSDI/data/mof_paper/BulkModulus_data/DFT.csv"
)


def get_mofs():
    """Get list of MOFs."""
    mofs = []
    with open(REF_FILE) as csv_file:
        csv_reader = reader(csv_file, delimiter=",")
        next(csv_reader)  # Skip headers
        for row in csv_reader:
            mofs.append(row[0])

    return {
        mof: Path(f"/Users/elliottkasoar/Documents/PSDI/data/mof_paper/{mof}.cif")
        for mof in mofs
    }


def get_mof_files():
    """Get list of filepaths to MOFs."""
    mofs = get_mofs()

    return [
        Path(f"/Users/elliottkasoar/Documents/PSDI/data/mof_paper/{mof}.cif")
        for mof in mofs
    ]


def get_bulk_moduli(mlips):
    """Get bulk moduli from reference and MLIP calculations."""
    bulk_moduli = {"ref": []} | {mlip: [] for mlip in mlips}
    mofs = get_mofs()

    # Get reference data
    with open(REF_FILE) as csv_file:
        csv_reader = reader(csv_file, delimiter=",")
        next(csv_reader)  # Skip headers
        for row in csv_reader:
            bulk_moduli["ref"].append(float(row[2]))

    # Get MLIP data
    for mlip in mlips:
        for mof in mofs:
            output_file = Path(
                f"/Users/elliottkasoar/Documents/PSDI/mlip-testing/mof_eos_results/{mof}-{mlip}-eos-fit.dat"
            )
            if not output_file.exists():
                bulk_moduli[mlip].append(None)
                continue
            with open(output_file, encoding="utf8") as f:
                data = f.readlines()
                bulk_modulus, _, _ = tuple(float(x) for x in data[1].split())

            bulk_moduli[mlip].append(bulk_modulus)

    return bulk_moduli


def get_V_0s(mlips):
    """Get reference and calculated V_Os."""
    V_0s = {"ref": []} | {mlip: [] for mlip in mlips}
    mofs = get_mofs()

    struct_files = (
        Path(f"/Users/elliottkasoar/Documents/PSDI/data/mof_paper/{mof}.cif")
        for mof in mofs
    )
    for struct_file in struct_files:
        struct = read(struct_file)
        V_0s["ref"].append(struct.get_volume())

    # Get MLIP data
    for mlip in mlips:
        for mof in mofs:
            output_file = Path(
                f"/Users/elliottkasoar/Documents/PSDI/mlip-testing/mof_eos_results/{mof}-{mlip}-eos-fit.dat"
            )
            if not output_file.exists():
                V_0s[mlip].append(None)
                continue
            with open(output_file, encoding="utf8") as f:
                data = f.readlines()
                _, _, V_0 = tuple(float(x) for x in data[1].split())
            V_0s[mlip].append(V_0)

    return V_0s


def get_E_0s(mlips):
    """Get reference and calculated E_Os."""
    E_0s = {"ref": []} | {mlip: [] for mlip in mlips}
    mofs = get_mofs()
    formulae = []

    for struct_file in get_mof_files():
        struct = read(struct_file)
        formulae.append(struct.get_chemical_formula(mode="metal"))

    for formula in formulae:
        found_value = False
        with open(
            "/Users/elliottkasoar/Downloads/13147324/qmof_database/qmof_database/qmof.csv"
        ) as csv_file:
            csv_reader = reader(csv_file, delimiter=",")
            next(csv_reader)  # Skip headers
            for row in csv_reader:
                if row[2] == formula:
                    E_0s["ref"].append(float(row[28]))
                    found_value = True
                    break
        if not found_value:
            E_0s["ref"].append(None)

    # Get MLIP data
    for mlip in mlips:
        for mof in mofs:
            output_file = Path(
                f"/Users/elliottkasoar/Documents/PSDI/mlip-testing/mof_eos_results/{mof}-{mlip}-eos-fit.dat"
            )
            if not output_file.exists():
                E_0s[mlip].append(None)
                continue
            with open(output_file, encoding="utf8") as f:
                data = f.readlines()
                _, E_0, _ = tuple(float(x) for x in data[1].split())
            E_0s[mlip].append(E_0)

    return E_0s


def get_mean_errors(mlips, results):
    """Calculate mean error for single metric for each MLIP."""
    mean_errors = {}
    for mlip in mlips:
        errors = []

        for ref_result, mlip_result in zip(results["ref"], results[mlip], strict=True):
            if ref_result and mlip_result:
                errors.append(abs(mlip_result - ref_result))
            else:
                errors.append(0.0)

        mean_errors[mlip] = np.mean(errors)

    return mean_errors


def combine_eos_metrics(mlips, metrics):
    """Combine metrics linearly to calculate test score."""
    score = dict.fromkeys(mlips, 0)
    for mlip in mlips:
        for metric, value in metrics.items():
            if metric == "B":
                score[mlip] += 0.9 * value[mlip]
            else:
                score[mlip] += 0.05 * value[mlip]
    return score
