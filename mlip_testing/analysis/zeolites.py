"""Helper functions for Zeolites test."""

from __future__ import annotations

from pathlib import Path

from ase.io import read
import numpy as np


def get_md_results(mlip, molecule):
    """Get MD results file for zeolite + molecule for an MLIP."""
    if mlip == "mace_mpa":
        return Path(
            f"/Users/elliottkasoar/Documents/PSDI/data/mace_mp_paper/MOR-Al-300-mpa-{molecule}-traj.extxyz"
        )
    if mlip == "mace_mpa_d3":
        return Path(
            f"/Users/elliottkasoar/Documents/PSDI/data/mace_mp_paper/MOR-Al-{molecule}-traj.extxyz"
        )
    raise ValueError("Missing MLIP")


def get_neb_results(mlip):
    """Get NEB results file for zeolite + acetone + water."""
    if mlip == "mace_mpa":
        return Path(
            "/Users/elliottkasoar/Documents/PSDI/data/mace_mp_paper/acetone-keto-enol/MOR-Al-20H2O-acetone-stateA-neb-results-mpa.dat"
        )
    if mlip == "mace_mpa_d3":
        return Path(
            "/Users/elliottkasoar/Documents/PSDI/data/mace_mp_paper/acetone-keto-enol/MOR-Al-20H2O-acetone-stateA-neb-results-d3.dat"
        )
    raise ValueError("Missing MLIP")


def get_formamide_values(mlips):
    """Get values for zeolites with formamide."""
    values = {"ref": []} | {mlip: [] for mlip in mlips}

    OB = 95
    HB = 144
    Al = 141
    O = 289

    # Order of values:
    # O - HB dist
    # OB - HB dist
    # Al - OB dist

    # Reference data
    values["ref"] = [1.1066, 1.3775, 1.8138]

    # Get MLIP data
    for mlip in mlips:
        output_file = get_md_results(mlip, "formamide")
        structs = read(output_file, index=":")

        dists_O_HB = []
        dists_OB_HB = []
        dists_Al_OB = []

        for struct in structs:
            dists_O_HB.append(struct.get_distance(O, HB, mic=True))
            dists_OB_HB.append(struct.get_distance(OB, HB, mic=True))
            dists_Al_OB.append(struct.get_distance(Al, OB, mic=True))

        av_O_HB = np.average(dists_O_HB)
        av_OB_HB = np.average(dists_OB_HB)
        av_Al_OB = np.average(dists_Al_OB)

        values[mlip] = [av_O_HB, av_OB_HB, av_Al_OB]

    return values


def get_acetone_values(mlips):
    """Get values for zeolites with acetone."""
    values = {"ref": []} | {mlip: [] for mlip in mlips}

    OB = 95
    HB = 144
    Al = 141
    O = 294

    # Order of values:
    # O-HB dist
    # OB-HB dist
    # Al-OB dist
    # OB-HB-O angle

    # Reference data
    values["ref"] = [1.3528, 1.1111, 1.8582, 174.0]

    # Get MLIP data
    for mlip in mlips:
        output_file = get_md_results(mlip, "acetone")
        structs = read(output_file, index=":")

        dists_O_HB = []
        dists_OB_HB = []
        dists_Al_OB = []
        angs_OB_HB_O = []

        for struct in structs:
            dists_O_HB.append(struct.get_distance(O, HB, mic=True))
            dists_OB_HB.append(struct.get_distance(OB, HB, mic=True))
            dists_Al_OB.append(struct.get_distance(Al, OB, mic=True))
            angs_OB_HB_O.append(struct.get_angle(OB, HB, O, mic=True))

        av_O_HB = np.average(dists_O_HB)
        av_OB_HB = np.average(dists_OB_HB)
        av_Al_OB = np.average(dists_Al_OB)
        av_OB_HB_O = np.average(angs_OB_HB_O)

        values[mlip] = [av_O_HB, av_OB_HB, av_Al_OB, av_OB_HB_O]

    return values


def get_nh3_values(mlips):
    """Get values for zeolites with nh3."""
    values = {"ref": []} | {mlip: [] for mlip in mlips}

    OB = 95
    HB = 144
    Al = 141
    N = 289

    # Order of values:
    # N-HB dist
    # OB-HB dist
    # Al-OB dist

    # Reference dat
    values["ref"] = [1.0710, 1.6864, 1.7904]

    # Get MLIP data
    for mlip in mlips:
        output_file = get_md_results(mlip, "nh3")
        structs = read(output_file, index=":")

        dists_N_HB = []
        dists_OB_HB = []
        dists_Al_OB = []

        for struct in structs:
            dists_N_HB.append(struct.get_distance(N, HB, mic=True))
            dists_OB_HB.append(struct.get_distance(OB, HB, mic=True))
            dists_Al_OB.append(struct.get_distance(Al, OB, mic=True))

        av_N_HB = np.average(dists_N_HB)
        av_OB_HB = np.average(dists_OB_HB)
        av_Al_OB = np.average(dists_Al_OB)

        values[mlip] = [av_N_HB, av_OB_HB, av_Al_OB]

    return values


def get_keto_enol_values(mlips):
    values = {"ref": []} | {mlip: [] for mlip in mlips}

    # Reference data
    values["ref"] = [1.6695]

    # Get MLIP data
    for mlip in mlips:
        output_file = get_neb_results(mlip)
        with open(output_file, encoding="utf8") as f:
            data = f.readlines()
            barrier, _, _ = tuple(float(x) for x in data[1].split())
        values[mlip] = [barrier]

    return values


def get_rel_errors(mlips, results):
    """Calculate mean relative error for single metric for each MLIP."""
    mean_errors = {}
    for mlip in mlips:
        errors = []

        for ref_result, mlip_result in zip(results["ref"], results[mlip], strict=True):
            if ref_result and mlip_result:
                errors.append((mlip_result - ref_result) / ref_result)
            else:
                errors.append(0.0)

        mean_errors[mlip] = np.mean(errors)

    return mean_errors


def combine_zeolite_metrics(mlips, metrics):
    """Combine metrics linearly to calculate test score."""
    score = dict.fromkeys(mlips, 0)
    for mlip in mlips:
        for metric, value in metrics.items():
            score[mlip] += 0.25 * value[mlip]
    return score
