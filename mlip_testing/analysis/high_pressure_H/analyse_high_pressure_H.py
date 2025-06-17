"""Analyse results from high pressure H tests."""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.calculators.calculator import PropertyNotImplementedError
from ase.io import read
from janus_core.helpers.stats import Stats
import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator
from scipy.spatial.distance import directed_hausdorff

from mlip_testing.analysis.utils.decorators import build_table, plot_scatter
from mlip_testing.analysis.utils.utils import rmse
from mlip_testing.calcs.config import MLIPS

OUT_PATH = Path(__file__).parent.parent.parent / "calcs" / "high_pressure_H" / "outputs"
REF_FILE = OUT_PATH / "Hpres.xyz"
DENS_FACT = (units.m / 1.0e2) ** 3 / units.mol


@pytest.fixture(scope="module")
def ref_data():
    """Get reference data."""
    return read(REF_FILE, index=":")


@pytest.fixture(scope="module")
def ref_density(ref_data):
    """Read reference density."""
    return [
        np.sum(atoms.get_masses()) / atoms.get_volume() * DENS_FACT
        for atoms in ref_data
    ]


@pytest.fixture(scope="module")
def traj_data():
    """Get paths to trajectory data."""
    return {mlip: OUT_PATH / f"H-{mlip}-traj.extxyz" for mlip in MLIPS}


@pytest.fixture(scope="module")
def stats_data():
    """Get paths to trajectory data."""
    return {mlip: OUT_PATH / f"H-{mlip}-stats.dat" for mlip in MLIPS}


@pytest.fixture(scope="module")
def static_data():
    """Get paths to static MD data."""
    return {mlip: OUT_PATH / f"H-static-{mlip}-results.extxyz" for mlip in MLIPS}


@pytest.fixture
# @plot_scatter()
def get_static_energy(ref_data, static_data, ref_density):
    """Get data for static potential energy."""
    results = {"ref": []} | {mlip: [] for mlip in MLIPS}

    # Read reference data and extract energy
    results["ref"] = [ref_density[::50]]
    results["ref"].append(
        [atoms.get_potential_energy() / len(atoms) for atoms in ref_data[::50]]
    )

    # Read MLIP data and extract energy
    for mlip in MLIPS:
        results[mlip].append(ref_density)
        results[mlip].append(
            [
                atoms.calc.results["energy"] / len(atoms)
                for atoms in read(static_data[mlip], index=":")
            ]
        )

    return results


@pytest.fixture
# @plot_scatter()
def get_static_pressure(ref_data, static_data, ref_density):
    """Get data for static pressure results."""
    results = {"ref": []} | {mlip: [] for mlip in MLIPS}

    # Read reference data and extract pressure
    results["ref"] = [ref_density[::50]]
    results["ref"].append([atoms.info["pressure_static"] for atoms in ref_data[::50]])

    # Read MLIP data and extract pressure
    for mlip in MLIPS:
        results[mlip] = [ref_density[::50]]
        results[mlip].append(
            [
                -np.trace(atoms.get_stress(include_ideal_gas=True, voigt=False))
                / 3
                / units.GPa
                for atoms in read(static_data[mlip], index=":")
            ]
        )

    return results


@pytest.fixture
# @plot_scatter()
def get_static_force(ref_data, static_data, ref_density):
    """Get data for static force results."""
    results = {"ref": []} | {mlip: [] for mlip in MLIPS}

    # Read reference data and extract force
    results["ref"] = [ref_density[::50]]
    results["ref"].append(
        [np.linalg.norm(atoms.get_forces()) for atoms in ref_data[::50]]
    )

    # Read MLIP data and extract forces
    for mlip in MLIPS:
        results[mlip] = [ref_density[::50]]
        results[mlip].append(
            [
                np.linalg.norm(atoms.calc.results["forces"])
                for atoms in read(static_data[mlip], index=":")
            ]
        )

    return results


@pytest.fixture
def static_energy(get_static_energy):
    """Get metric for static energy."""
    results = {}
    for mlip in MLIPS:
        results[mlip] = rmse(get_static_energy["ref"][1], get_static_energy[mlip][1])

    return results


@pytest.fixture
def static_pressure(get_static_pressure):
    """Get metric for static stress."""
    results = {}
    for mlip in MLIPS:
        results[mlip] = rmse(
            get_static_pressure["ref"][1], get_static_pressure[mlip][1]
        )

    return results


@pytest.fixture
def static_force(get_static_force):
    """Get metric for static force."""
    results = {}
    for mlip in MLIPS:
        results[mlip] = rmse(get_static_force["ref"][1], get_static_force[mlip][1])

    return results


def stress_grad_idx(stresses, lim=0.2):
    """Get index for maximum stress gradient."""
    stress_grad = np.gradient(stresses, axis=0)
    stress_grad = abs(stress_grad)  # absolute value to check the limit

    hits = np.nonzero(np.any(stress_grad >= lim, axis=1))[0]
    if len(hits) > 0:
        return np.min(hits)
    return len(stresses)


def stress_stability(file):
    """Get index for stress stability limit for a file."""
    stresses = []
    for struct in read(file, index=":"):
        try:
            stresses.append(struct.get_stress())
        except PropertyNotImplementedError:
            break
    return stress_grad_idx(stresses)


def max_force_idx(forces, lim=6):
    """Get index for maximum force stability."""
    forces_mags = np.linalg.norm(forces, axis=2)
    max_forces = np.max(forces_mags, axis=1)

    hits = np.nonzero(max_forces >= lim)[0]
    if len(hits) > 0:
        return np.min(hits)
    return len(forces)


def force_stability(file):
    """Get index for force stability limit for a file."""
    forces = []
    for struct in read(file, index=":"):
        try:
            forces.append(struct.get_forces())
        except PropertyNotImplementedError:
            break
    return max_force_idx(forces)


@pytest.fixture(scope="module")
def get_stability(traj_data):
    """Get max indicies for MD stability."""
    stability = {}

    for mlip in MLIPS:
        file = traj_data[mlip]
        stability[mlip] = {"force": force_stability(file)}
        stability[mlip]["stress"] = stress_stability(file)

    return stability


@pytest.fixture
def stability(traj_data, get_stability):
    """Get metric for MD stability."""
    results = {}
    for mlip in MLIPS:
        file = traj_data[mlip]
        max_step = read(file, index="-1").info["step"]
        force_idx = get_stability[mlip]["force"]
        stress_idx = get_stability[mlip]["stress"]

        results[mlip] = (
            0.5
            * (
                read(file, index=f"{force_idx}").info["step"]
                + read(file, index=f"{stress_idx}").info["step"]
            )
            / max_step
        )
    return results


@pytest.fixture
@plot_scatter()
def get_dynamic_pressure(ref_data, stats_data, ref_density):
    """Get data for dynamic pressure."""
    results = {"ref": []} | {mlip: [] for mlip in MLIPS}

    # Read reference data and dynamic pressure
    results["ref"] = [ref_density]
    results["ref"].append([atoms.info["pressure_dynamic"] for atoms in ref_data])

    # Read MLIP data and extract dynamic pressure
    for mlip in MLIPS:
        data = Stats(stats_data[mlip])
        results[mlip] = [data[7], data[9]]

    return results


def clean_densities(densities, pressure):
    """Sort and remove duplicate densities."""
    # Sort by density
    sort = np.argsort(densities)
    densities = np.array(densities)[sort]
    pressure = pressure[sort]

    # Group repeated densities
    unique_densities, inv_idx = np.unique(densities, return_inverse=True)
    avg_pressure = np.zeros_like(unique_densities, dtype=float)

    for i in len(unique_densities):
        avg_pressure[i] = np.mean(pressure[inv_idx == i])
    return unique_densities, avg_pressure


@pytest.fixture
def dynamic_pressure(get_stability, get_dynamic_pressure, ref_data, ref_density):
    """Get metric for dynamic pressure."""
    results = {}

    # Sort reference data by density
    ref_pressure = np.array([atoms.info["pressure_dynamic"] for atoms in ref_data])
    ref_density, ref_pressure = clean_densities(ref_density, ref_pressure)

    ref_interpolator = PchipInterpolator(ref_density, ref_pressure)

    for mlip in MLIPS:
        idx = max(get_stability[mlip]["force"], get_stability[mlip]["stress"])

        mlip_density = get_dynamic_pressure[mlip][0][5:idx]
        mlip_pressure = get_dynamic_pressure[mlip][1][5:idx]
        mlip_density, mlip_pressure = clean_densities(mlip_density, mlip_pressure)

        mlip_interpolator = PchipInterpolator(mlip_density, mlip_pressure)

        # Prepare common evaluation grid and interpolate
        common_densities = np.linspace(
            max(min(mlip_density), min(ref_density)),
            min(max(mlip_density), max(ref_density)),
            500,
        )

        ref_interpolated_pressure = ref_interpolator(common_densities)
        mlip_interpolated_pressure = mlip_interpolator(common_densities)

        # Measure distance
        mlip_points = np.column_stack((common_densities, mlip_interpolated_pressure))
        ref_points = np.column_stack((common_densities, ref_interpolated_pressure))

        results[mlip] = directed_hausdorff(ref_points, mlip_points)[0]

    return results


@pytest.fixture
@build_table("high_pressure_H_table.json")
def metrics(dynamic_pressure, stability, static_energy, static_force, static_pressure):
    """Get all metrics."""
    return {
        "Dynamic Pressure": dynamic_pressure,
        "Stability": stability,
        "Static Energy": static_energy,
        "Static Force": static_force,
        "Static Pressure": static_pressure,
    }


def test_high_pressure_h(metrics, coeffs=(1, 1, 0.01, 0.01, 0.01)):
    """Combine all metrics."""
    results = {}
    for mlip in MLIPS:
        results[mlip] = 0
        for metric, coeff in zip(metrics, coeffs, strict=True):
            results[mlip] += metrics[metric][mlip] * coeff
