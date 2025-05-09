"""Generate homonuclear diatomic data."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import write
from janus_core.calculations.single_point import SinglePoint
import numpy as np

# for i in (1, 2, 3, 4, 5, 6, 7, 8, 59):
for i in np.arange(95):
    struct = Atoms((i, i))

    structs = []
    for j in np.arange(0.1, 6.1, 0.1):
        struct.positions[1][0] = j
        structs.append(struct.copy())

    filename = f"./mlip_testing/data/homonuclear/{struct.get_chemical_formula()}.xyz"
    write(filename, structs)

data_files = Path().glob("./mlip_testing/data/homonuclear/*.xyz")

for data_file in data_files:
    try:
        sp = SinglePoint(
            struct=data_file,
            arch="mace_mp",
            model_path="medium",
            file_prefix=f"./mlip_testing/data/homonuclear/{data_file.stem}_mace_mp",
            write_results=True,
        )
        sp.run()
        sp = SinglePoint(
            struct=data_file,
            arch="mace_mp",
            model_path="medium-mpa-0",
            file_prefix=f"mlip_testing/data/homonuclear/{data_file.stem}_mace_mpa",
            write_results=True,
        )
        sp.run()
    except ValueError:
        continue
