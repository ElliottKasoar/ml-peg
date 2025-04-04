"""Results analysis module."""

from __future__ import annotations

from pathlib import Path

from ase.io import read
import matplotlib.pyplot as plt
import numpy as np
from yaml import safe_load

from janus_core.analysis import compare, convert


def get_config():
    with open("analysis_config.yml", encoding="utf8") as file:
        config = safe_load(file)
    return config


def get_files(config):
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


def flatten(lst):
    return [x for xs in lst for x in xs]


def plot_energies(ref_energies, mlip_energies, lim=None):
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
