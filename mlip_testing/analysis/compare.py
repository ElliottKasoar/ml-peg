"""File comparisons."""

from __future__ import annotations

from ase.io import read
import numpy as np


def flatten(lst):
    return [x for xs in lst for x in xs]


def get_energies(ref_files, mlip_files, config):
    ref_energies = []
    mlip_energies = {}

    # Get reference energies
    for file in ref_files:
        energies = []
        key = config["compare"]["reference_key"]
        for atoms in read(file, index=":"):
            energies.append(atoms.info[key])
        ref_energies.append(energies)

    for mlip in mlip_files:
        mlip_energies[mlip] = []
        for file in mlip_files[mlip]:
            energies = []
            key = config["compare"]["mlip"][mlip]
            for atoms in read(file, index=":"):
                energies.append(atoms.info[key])
            mlip_energies[mlip].append(energies)

    return ref_energies, mlip_energies


def compare_energies(ref_files, mlip_files, config):
    ref_energies, mlip_energies = get_energies(ref_files, mlip_files, config)

    flat_ref_energies = flatten(ref_energies)
    results = {}

    for mlip in mlip_energies:
        flat_energies = flatten(mlip_energies[mlip])
        results[mlip] = np.mean(
            np.subtract(np.array(flat_ref_energies), np.array(flat_energies)) ** 2
        )

    return results
