"""
Download and prepare Alanine Dipeptide dataset from MD-AlanineDipeptide repository.

This script downloads trajectory data from:
https://github.com/sbrodehl/MD-AlanineDipeptide

The dataset contains molecular dynamics simulations of alanine dipeptide,
which exhibits two major conformational states (C7eq and C7ax) corresponding
to local minima in the energy landscape.

Usage:
    python download_alanine_dipeptide.py [--output_dir <DIR>]
"""

import os
import urllib.request
import tarfile
import shutil
from pathlib import Path
import argparse


GITHUB_BASE = "https://github.com/sbrodehl/MD-AlanineDipeptide/raw/master"

# Files to download
FILES = {
    "trajectory": "trajectory-1.dcd",  # Main trajectory
    "topology": "alanine-dipeptide.pdb",  # Structure file
}

# Alternative: if you want to download the full dataset archive
ARCHIVE_URL = "https://github.com/sbrodehl/MD-AlanineDipeptide/archive/refs/heads/master.zip"


def download_file(url: str, dest_path: str) -> None:
    """Download a file from URL to destination."""
    print(f"Downloading {url} -> {dest_path}")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"✓ Downloaded {dest_path}")
    except Exception as e:
        print(f"✗ Failed to download {url}: {e}")
        raise


def download_alanine_dipeptide(output_dir: str = "datasets/alanine_dipeptide") -> None:
    """
    Download Alanine Dipeptide MD trajectory data.
    
    Args:
        output_dir: Directory to save downloaded files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[info] Downloading Alanine Dipeptide dataset to {output_dir}")
    
    # Download individual files
    for file_type, filename in FILES.items():
        url = f"{GITHUB_BASE}/{filename}"
        dest = output_path / filename
        
        if dest.exists():
            print(f"[skip] {dest} already exists")
            continue
        
        try:
            download_file(url, str(dest))
        except Exception as e:
            print(f"[error] Could not download {filename}: {e}")
            print(f"[info] You may need to manually download from:")
            print(f"       {GITHUB_BASE}/{filename}")
            continue
    
    # Create a README
    readme_path = output_path / "README.txt"
    with open(readme_path, "w") as f:
        f.write("Alanine Dipeptide MD Trajectory Dataset\n")
        f.write("=" * 60 + "\n\n")
        f.write("Source: https://github.com/sbrodehl/MD-AlanineDipeptide\n\n")
        f.write("Description:\n")
        f.write("  Molecular dynamics simulation of alanine dipeptide in vacuum.\n")
        f.write("  The system exhibits two major conformational states:\n")
        f.write("    - C7eq (extended): Phi ~ -80°, Psi ~ 80°\n")
        f.write("    - C7ax (compact): Phi ~ 80°, Psi ~ -80°\n\n")
        f.write("Files:\n")
        f.write(f"  - {FILES['topology']}: PDB structure file\n")
        f.write(f"  - {FILES['trajectory']}: DCD trajectory file\n\n")
        f.write("Expected energy landscape:\n")
        f.write("  Two distinct local minima separated by an energy barrier,\n")
        f.write("  making this an ideal test system for path generation methods.\n")
    
    print(f"\n[done] Alanine Dipeptide dataset downloaded to {output_dir}")
    print(f"[next] Run prepare_alanine_dipeptide.py to generate graph data")


def main():
    parser = argparse.ArgumentParser(
        description="Download Alanine Dipeptide MD trajectory dataset"
    )
    parser.add_argument(
        "--output_dir",
        default="datasets/alanine_dipeptide",
        help="Output directory for downloaded files (default: datasets/alanine_dipeptide)"
    )
    
    args = parser.parse_args()
    
    try:
        download_alanine_dipeptide(args.output_dir)
        return 0
    except Exception as e:
        print(f"[error] Failed to download dataset: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
