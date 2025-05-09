"""Functions for homonuclear diatomics test."""

from __future__ import annotations

from pathlib import Path

from ase.io import read
import numpy as np


def get_structs():
    """Get list of all input structures."""
    return list(Path().glob("../../data/homonuclear/*.xyz"))


def get_output_file(mlip):
    """Get output files for all structures for an MLIP."""
    structs = get_structs()
    if mlip in ("mace_mp", "mace_mpa"):
        return [
            f"../../data/homonuclear/{struct.stem}_{mlip}-results.extxyz"
            for struct in structs
        ]
    raise ValueError("Invalid mlip")


def get_energy_values(mlips):
    """Get energy metric values for all structures for each MLIP."""
    values = {"ref": []} | {mlip: [] for mlip in mlips}

    for mlip in mlips:
        output_files = get_output_file(mlip)
        for output_file in output_files:
            try:
                structs = read(output_file, index=":")
            except FileNotFoundError:
                pass
            dists = []
            energies = []
            for struct in structs:
                dists.append(struct.get_distance(0, 1))
                energies.append(struct.get_potential_energy())

            values[mlip].append(
                np.tanh(np.nanmax(energies))
                * (len(energies) - np.argmax(energies))
                / len(energies)
            )
    return values


def get_energy_metric(mlips, results):
    """Get combined energy metric for each MLIP."""
    means = {}
    for mlip in mlips:
        means[mlip] = np.mean(results[mlip])

    return means


def combine_metrics(mlips, metrics):
    """Combine metrics linearly to calculate test score."""
    score = dict.fromkeys(mlips, 0)
    for mlip in mlips:
        for metric, value in metrics.items():
            score[mlip] += value[mlip]
    return score
