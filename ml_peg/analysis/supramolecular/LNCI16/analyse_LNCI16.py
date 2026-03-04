"""Analyse LNCI16 benchmark."""

from __future__ import annotations

from pathlib import Path
import shutil

from ase.data import chemical_symbols
from ase.io import read
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "supramolecular" / "LNCI16" / "outputs"
OUT_PATH = APP_ROOT / "data" / "supramolecular" / "LNCI16"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def get_info() -> dict[str, list[str]]:
    """
    Get structure information for LNCI16 systems.

    dict[str, list[str]]
        Dictationary of info returned from first non-empty model directory.
    """
    info = {
        "systems": [],
        "charges": [],
        "atoms_counts": [],
        "is_charged": [],
        "filters": [],
    }

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            xyz_files = sorted(model_dir.glob("*.xyz"))
            if xyz_files:
                element_filter = np.zeros(
                    (len(xyz_files), len(chemical_symbols)), dtype=bool
                )

                for xyz_file in xyz_files:
                    atoms = read(xyz_file)
                    info["systems"].append(
                        atoms.info.get("system", f"system_{xyz_file.stem}")
                    )
                    info["atoms_counts"].append(len(atoms))

                    charge = atoms.info.get("complex_charge", 0)
                    info["charges"].append(charge)
                    info["is_charged"].append(charge != 0)

                    elements = set(atoms.get_chemical_symbols())
                return info
    return info


INFO = get_info()


def compute_lnci16_mae(energies: dict[str, list]) -> dict[str, float | None]:
    """
    Compute mean absolute error (MAE) for each model.

    Parameters
    ----------
    energies
        Interaction energies with keys ``ref`` and one key per model.

    Returns
    -------
    dict[str, float | None]
        MAE value for each model. Returns ``None`` for models with no data.
    """
    results: dict[str, float | None] = {}
    ref_values = energies["ref"]
    for model_name in MODELS:
        model_values = energies[model_name]
        if ref_values and model_values:
            results[model_name] = mae(ref_values, model_values)
        else:
            results[model_name] = None
    return results


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_interaction_energies.json",
    title="LNCI16 Interaction Energies",
    x_label="Predicted interaction energy / kcal/mol",
    y_label="Reference interaction energy / kcal/mol",
    hoverdata={
        "System": INFO["systems"],
        "Complex Atoms": INFO["atoms_counts"],
        "Charge": INFO["charges"],
    },
)
def interaction_energies() -> dict[str, list]:
    """
    Get interaction energies for all LNCI16 systems.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted interaction energies.
    """
    interaction_energy_results = {"ref": []} | {mlip: [] for mlip in MODELS}
    reference_is_stored = False

    for model_name in MODELS:
        model_output_dir = CALC_PATH / model_name

        if not model_output_dir.exists():
            interaction_energy_results[model_name] = []
            continue

        xyz_files = sorted(model_output_dir.glob("*.xyz"))
        if not xyz_files:
            interaction_energy_results[model_name] = []
            continue

        model_energies = []
        ref_energies = []

        for xyz_file in xyz_files:
            atoms = read(xyz_file)
            model_energies.append(atoms.info["E_int_model_kcal"])
            if not reference_is_stored:
                ref_energies.append(atoms.info["E_int_ref_kcal"])

        interaction_energy_results[model_name] = model_energies

        # Store reference energies (only once)
        if not reference_is_stored:
            interaction_energy_results["ref"] = ref_energies
            reference_is_stored = True

        # Copy individual structure files to app data directory
        structs_dir = OUT_PATH / model_name
        structs_dir.mkdir(parents=True, exist_ok=True)

        for i, xyz_file in enumerate(xyz_files):
            shutil.copy(xyz_file, structs_dir / f"{i}.xyz")

    return interaction_energy_results


@pytest.fixture
def lnci16_mae(interaction_energies) -> dict[str, float]:
    """
    Get mean absolute error for interaction energies.

    Parameters
    ----------
    interaction_energies
        Dictionary of reference and predicted interaction energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted interaction energy errors for all models.
    """
    return compute_lnci16_mae(interaction_energies)


@pytest.fixture
@build_table(
    filename=OUT_PATH / "lnci16_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(lnci16_mae: dict[str, float]) -> dict[str, dict]:
    """
    Get all LNCI16 metrics.

    Parameters
    ----------
    lnci16_mae
        Mean absolute errors for all systems.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": lnci16_mae,
    }


def test_lnci16(metrics: dict[str, dict]) -> None:
    """
    Run LNCI16 test.

    Parameters
    ----------
    metrics
        All LNCI16 metrics.
    """
    return
