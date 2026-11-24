"""Intake (FR-01): expand files and folders into a sorted list of files."""
from pathlib import Path


def collect_files(inputs: list[str | Path]) -> list[Path]:
    files: set[Path] = set()
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files.update(p for p in path.rglob("*")
                         if p.is_file() and not p.name.startswith((".", "~$")))
        elif path.is_file():
            files.add(path)
        else:
            raise FileNotFoundError(f"Input not found: {path}")
    return sorted(p.resolve() for p in files)
