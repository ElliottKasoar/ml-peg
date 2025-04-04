"""File comparisons."""

from __future__ import annotations

from typing import Any

from ase.io import read
from janus_core.helpers.janus_types import PathLike
import numpy as np


def flatten(lst: list) -> list:
    """
    Flatten nested list.

    Parameters
    ----------
    lst
        List to be flattened.

    Returns
    -------
    list
        Flattened list.
    """
    return [x for xs in lst for x in xs]


def get_energies(
    ref_files: list, mlip_files: dict[str, PathLike], config: dict[str, Any]
) -> tuple[list[float], dict[str, float]]:
    """
    Get reference and MLIP energies.

    Parameters
    ----------
    ref_files
        List of reference files.
    mlip_files
        Dictionary of MLIP files, labelled by their keys.
    config
        Loaded configuration.

    Returns
    -------
    tuple[list[float], dict[str, float]]
        Reference and MLIP energies.
    """
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


def compare_energies(
    ref_files: list, mlip_files: list, config: dict[str, Any]
) -> dict[str, Any]:
    """
    Compare reference and MLIP energies.

    Parameters
    ----------
    ref_files
        Reference files.
    mlip_files
        MLIP files.
    config
        Loaded configuration.

    Returns
    -------
    dict[str, Any]
        Mean difference in energies for each MLIP.
    """
    ref_energies, mlip_energies = get_energies(ref_files, mlip_files, config)

    flat_ref_energies = flatten(ref_energies)
    results = {}

    for mlip in mlip_energies:
        flat_energies = flatten(mlip_energies[mlip])
        results[mlip] = np.mean(
            np.subtract(np.array(flat_ref_energies), np.array(flat_energies)) ** 2
        )

    return results
