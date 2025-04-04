"""Run test calculations."""

from __future__ import annotations

from pathlib import Path

from janus_core.calculations.single_point import SinglePoint

files = (
    p.resolve()
    for p in Path().glob("tests/data/*")
    if p.suffix in (".cif", ".xyz", ".extxyz")
)

for file in files:
    out = Path(f"./split_results/{file.stem}-mace.extxyz")
    sp = SinglePoint(
        struct=file, arch="mace_mp", write_results=True, write_kwargs={"filename": out}
    )
    sp.run()

files = Path().glob("./split_results/*extxyz")

for file in files:
    out = Path(f"./split_results/{file.stem}-sevennet.extxyz")
    sp = SinglePoint(
        struct=file, arch="sevennet", write_results=True, write_kwargs={"filename": out}
    )
    sp.run()
