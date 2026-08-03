"""
Batch-train ProtSCAPE sequentially across graph datasets.

This script discovers *_graphs.pkl files and runs train.py one protein at a time,
so you do not need to edit config files between runs.

Example:
  python train_all_proteins.py \
    --config configs/config.yaml \
        --graphs_dir data/graphs \
        --protein_ids 6h86 7jfl
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SUFFIX = "_graphs.pkl"


def discover_graph_pickles(graphs_dir: Path) -> list[Path]:
    if not graphs_dir.exists():
        raise FileNotFoundError(f"Graphs directory not found: {graphs_dir}")

    files = sorted(p for p in graphs_dir.glob(f"*{SUFFIX}") if p.is_file())
    return files


def protein_name_from_graph(graph_path: Path) -> str:
    name = graph_path.name
    if not name.endswith(SUFFIX):
        raise ValueError(f"Unexpected graph filename format: {name}")
    return name[: -len(SUFFIX)]


def _normalize_id(value: str) -> str:
    return value.strip().lower()


def _pdb_id_from_protein_name(protein_name: str) -> str:
    # Supports names like 6h86_A by taking the token before the first underscore.
    return protein_name.split("_", 1)[0].lower()


def run_one_training(
    python_exec: str,
    train_script: Path,
    config_path: Path,
    protein_name: str,
    pkl_path: Path,
) -> int:
    cmd = [
        python_exec,
        str(train_script),
        "--config",
        str(config_path),
        "--protein",
        protein_name,
        "--pkl_path",
        str(pkl_path),
    ]

    print(f"\n[run] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train ProtSCAPE sequentially on all graph pickles in a folder."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Training config used for every run.",
    )
    parser.add_argument(
        "--graphs_dir",
        default="data/graphs",
        help="Directory containing *_graphs.pkl files.",
    )
    parser.add_argument(
        "--python_exec",
        default=sys.executable,
        help="Python executable to use (default: current interpreter).",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately if any protein training fails.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running training.",
    )
    parser.add_argument(
        "--max_proteins",
        type=int,
        default=None,
        help="Train only the first N proteins after sorting graph files.",
    )
    parser.add_argument(
        "--protein_ids",
        nargs="+",
        default=None,
        help=(
            "Train only these protein/PDB IDs (e.g. 6h86 7jfl). "
            "Matches either full protein name (e.g. 6h86_A) or PDB prefix."
        ),
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    train_script = (repo_root / "train.py").resolve()
    config_path = (repo_root / args.config).resolve()
    graphs_dir = (repo_root / args.graphs_dir).resolve()

    if not train_script.exists():
        print(f"[error] train.py not found at {train_script}")
        return 1
    if not config_path.exists():
        print(f"[error] config file not found at {config_path}")
        return 1

    try:
        graph_files = discover_graph_pickles(graphs_dir)
    except FileNotFoundError as exc:
        print(f"[error] {exc}")
        return 1

    if not graph_files:
        print(f"[warn] No {SUFFIX} files found in {graphs_dir}")
        return 0

    if args.protein_ids and args.max_proteins is not None:
        print("[error] Use either --protein_ids or --max_proteins, not both.")
        return 1

    total_graphs = len(graph_files)
    if args.protein_ids:
        requested_ids = {_normalize_id(pid) for pid in args.protein_ids}
        filtered_graphs: list[Path] = []
        for graph_path in graph_files:
            protein_name = protein_name_from_graph(graph_path)
            protein_name_norm = _normalize_id(protein_name)
            pdb_id_norm = _pdb_id_from_protein_name(protein_name)
            if protein_name_norm in requested_ids or pdb_id_norm in requested_ids:
                filtered_graphs.append(graph_path)

        graph_files = filtered_graphs

        if not graph_files:
            print(
                "[error] No graph files matched --protein_ids: "
                + ", ".join(sorted(requested_ids))
            )
            return 1
    elif args.max_proteins is not None:
        if args.max_proteins <= 0:
            print("[error] --max_proteins must be a positive integer")
            return 1
        graph_files = graph_files[: args.max_proteins]

    print(f"[info] Found {total_graphs} graph files in {graphs_dir}")
    print(f"[info] Selected {len(graph_files)} proteins for this run")

    successes: list[str] = []
    failures: list[str] = []

    for graph_path in graph_files:
        protein_name = protein_name_from_graph(graph_path)

        if args.dry_run:
            print(
                f"[dry_run] {args.python_exec} {train_script} --config {config_path} "
                f"--protein {protein_name} --pkl_path {graph_path}"
            )
            continue

        rc = run_one_training(
            python_exec=args.python_exec,
            train_script=train_script,
            config_path=config_path,
            protein_name=protein_name,
            pkl_path=graph_path,
        )

        if rc == 0:
            successes.append(protein_name)
            print(f"[ok] {protein_name}")
        else:
            failures.append(protein_name)
            print(f"[failed] {protein_name} (exit={rc})")
            if args.stop_on_error:
                break

    print("\n===== Batch Training Summary =====")
    print(f"[done] successes: {len(successes)}")
    print(f"[done] failures:  {len(failures)}")

    if failures:
        print("[failed proteins] " + ", ".join(failures))
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())