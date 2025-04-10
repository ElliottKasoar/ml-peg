"""Utility functions."""

from __future__ import annotations

from typing import Any

from janus_core.helpers.utils import PathLike
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
    with open(config_file, encoding="utf8") as file:
        return safe_load(file)
