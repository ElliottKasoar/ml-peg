"""Results analysis module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.io import read
from janus_core.helpers.janus_types import PathLike
import matplotlib.pyplot as plt
from mlip_tesitng.analysis import compare, convert
import numpy as np
from yaml import safe_load


def get_config(config_file: PathLike) -> dict[str, Any]:
    """
    Load configuration file.

    Parameters
    ----------
    config_file
        File containing configuration to be loaded.

    Returns
    -------
    dict[str, Any]
        Loaded configuration.
    """
    with open("analysis_config.yml", encoding="utf8") as file:
        return safe_load(file)


def get_files(config: dict[str, Any]) -> tuple[list, dict[str, PathLike]]:
    """
    Get reference and MLIP files.

    Parameters
    ----------
    config
        Loaded configuration file.

    Returns
    -------
    tuple[list, list]
        Reference and MLIP files.
    """
    # Get reference files
    ref_files = list(Path().glob(config["reference_files"]["pattern"]))

    # Use converter functions to predict MLIP files
    mlip_files = {}
    for mlip in config["mlip_files"]:
        converter = getattr(convert, config["mlip_files"][mlip]["convert"])
        mlip_files[mlip] = converter(ref_files)

    # Test which files were found and missing
    missing_files = []
    found_count = 0
    for mlip in mlip_files:
        for file in mlip_files[mlip]:
            if not file.exists():
                missing_files.append(file)
            else:
                found_count += 1

    print("Files missing:")
    print(missing_files)
    print(f"Found {found_count} results files")

    return ref_files, mlip_files


def get_energies(
    ref_files: list, mlip_files: dict[str, PathLike], config: dict[str, Any]
) -> tuple[list, dict[str, Any]]:
    """
    Get energies for reference and MLIPs.

    Parameters
    ----------
    ref_files
        List of reference data files.
    mlip_files
        Dictionary of MLIP data files.
    config
        Loaded configuration file.

    Returns
    -------
    tuple[list, dict[str, Any]]
        Reference energies and dictionary of MLIP energies.
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


def plot_energies(
    ref_energies: list, mlip_energies: dict[str, Any], lim=tuple[float, float] | None
) -> None:
    """
    Plot scatter plot of energies.

    Parameters
    ----------
    ref_energies
        Reference energies to plot.
    mlip_energies
        MLIP energies to plot.
    lim
        Limits for axes.
    """
    for mlip in mlip_energies:
        plt.scatter(
            np.array(flatten(ref_energies)),
            np.array(flatten(mlip_energies[mlip])),
            label=mlip,
        )

    # if not lim:
    #     lim = (-np.max(flatten(ref_energies)), np.max(flatten(ref_energies)))

    x = np.arange(-900, 100)
    y = np.arange(-900, 100)
    plt.plot(x, y, "--")
    plt.xlim(lim)
    plt.ylim(lim)

    plt.savefig("test.svg")
    plt.legend()
    plt.show()


config = get_config()
ref_files, mlip_files = get_files(config)
ref_energies, mlip_energies = get_energies(ref_files, mlip_files, config)

plot_energies(ref_energies, mlip_energies)
plot_energies(ref_energies, mlip_energies, (-150, 10))
plot_energies(ref_energies, mlip_energies, (-850, -830))

comparison = getattr(compare, config["compare"]["compare"])
print(comparison(ref_files, mlip_files, config))
