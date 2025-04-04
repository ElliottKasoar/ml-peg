"""File conversions."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from janus_core.helpers.janus_types import PathLike


def convert_mace_files(reference_files: Sequence[PathLike]) -> list[Path]:
    """
    Convert list of reference files to corresponding MACE files.

    Parameters
    ----------
    reference_files
        Sequence of reference files to find corresponding files for.

    Returns
    -------
    list[Path]
        List of files corresponding to the reference files.
    """
    return [Path(file) for file in reference_files]


def convert_sevennet_files(reference_files: Sequence[PathLike]) -> list[Path]:
    """
    Convert list of reference files to corresponding SevenNet files.

    Parameters
    ----------
    reference_files
        Sequence of reference files to find corresponding files for.

    Returns
    -------
    list[Path]
        List of files corresponding to the reference files.
    """
    return [
        Path(file).with_name(file.stem + "-sevennet" + Path(file).suffix)
        for file in reference_files
    ]
