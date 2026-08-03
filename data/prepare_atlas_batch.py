"""
Batch wrapper for prepare_atlas.py.

It scans a datasets directory, keeps only non-.zip protein folders,
and runs prepare_atlas.py for each discovered folder name.

Example:
  python data/prepare_atlas_batch.py \
    --datasets_dir /path/to/ATLAS_1k \
    --selection "protein and backbone" \
    --property rog
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def discover_protein_folders(datasets_dir: Path) -> list[Path]:
    """Return sorted protein folders (directories only, excluding .zip names)."""
    if not datasets_dir.exists():
        raise FileNotFoundError(f"Datasets directory not found: {datasets_dir}")

    folders: list[Path] = []
    for entry in sorted(datasets_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.endswith(".zip"):
            continue
        folders.append(entry)

    return folders


def has_required_files(folder: Path) -> bool:
    """A valid protein folder must contain at least one .xtc and one .pdb file."""
    return any(folder.glob("*.xtc")) and any(folder.glob("*.pdb"))


def run_prepare_for_folder(
    prepare_script: Path,
    folder_name: str,
    datasets_dir: Path,
    selection: str,
    prop: str,
    output_name: str,
) -> int:
    cmd = [
        sys.executable,
        str(prepare_script),
        folder_name,
        "--selection",
        selection,
        "--property",
        prop,
        "--datasets_dir",
        str(datasets_dir),
        "--output",
        output_name,
    ]

    print(f"\n[run] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run prepare_atlas.py for every non-.zip protein directory in a datasets folder."
    )
    parser.add_argument(
        "--datasets_dir",
        required=True,
        help="Path to folder containing protein directories (e.g., ATLAS_1k).",
    )
    parser.add_argument(
        "--selection",
        default="protein and backbone",
        help='MDTraj selection string (default: "protein and backbone").',
    )
    parser.add_argument(
        "--property",
        default="rog",
        choices=["rog", "sasa", "none"],
        help="Property used by prepare_atlas.py (default: rog).",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip proteins whose output graph pickle already exists in data/graphs/.",
    )

    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir).expanduser().resolve()
    prepare_script = Path(__file__).with_name("prepare_atlas.py").resolve()
    graphs_dir = Path(__file__).with_name("graphs").resolve()

    if not prepare_script.exists():
        print(f"[error] Could not find prepare script: {prepare_script}")
        return 1

    try:
        protein_folders = discover_protein_folders(datasets_dir)
    except FileNotFoundError as exc:
        print(f"[error] {exc}")
        return 1

    if not protein_folders:
        print("[warn] No protein directories found.")
        return 0

    print(f"[info] Found {len(protein_folders)} candidate protein folders")

    successes: list[str] = []
    skipped: list[str] = []
    failures: list[str] = []

    for folder in protein_folders:
        protein_name = folder.name
        output_name = f"{protein_name}_graphs.pkl"
        output_path = graphs_dir / output_name

        if not has_required_files(folder):
            print(f"[skip] {protein_name}: missing .xtc or .pdb")
            skipped.append(protein_name)
            continue

        if args.skip_existing and output_path.exists():
            print(f"[skip] {protein_name}: output exists at {output_path}")
            skipped.append(protein_name)
            continue

        rc = run_prepare_for_folder(
            prepare_script=prepare_script,
            folder_name=protein_name,
            datasets_dir=datasets_dir,
            selection=args.selection,
            prop=args.property,
            output_name=output_name,
        )

        if rc == 0:
            successes.append(protein_name)
        else:
            failures.append(protein_name)

    print("\n===== Batch Summary =====")
    print(f"[done] success: {len(successes)}")
    print(f"[done] skipped: {len(skipped)}")
    print(f"[done] failed:  {len(failures)}")

    if failures:
        print("[failed proteins] " + ", ".join(failures))
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
