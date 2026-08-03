#!/usr/bin/env python3
"""
Script to download molecular dynamics trajectories from the ATLAS database.
Downloads 10,000 frames of protein-only MD simulations.

Usage:
    python download_atlas.py <PDB_ID> [--chain <CHAIN_ID>] [--output_dir <OUTPUT_DIR>]

Example:
    python download_atlas.py 1MBN
    python download_atlas.py 1MBN --chain A --output_dir ./md_data/
"""

import argparse
import os
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
import json
import tarfile
import zipfile


ATLAS_API_BASE_URL = "https://www.dsimb.inserm.fr/ATLAS/api"


def _infer_archive_type(content_type: str, content_disposition: str, url: str):
    """Infer archive type from headers/URL. Returns one of: tar, tar.gz, zip, or None."""
    ct = (content_type or "").lower()
    cd = (content_disposition or "").lower()
    u = (url or "").lower()

    joined = " ".join([ct, cd, u])
    if any(x in joined for x in [".tar.gz", "application/gzip", "application/x-gzip", "x-tar"]):
        return "tar.gz"
    if ".tar" in joined or "application/x-tar" in joined:
        return "tar"
    if ".zip" in joined or "application/zip" in joined or "application/x-zip-compressed" in joined:
        return "zip"
    return None


def _default_filename_for_archive(base_name: str, archive_type: str) -> str:
    if archive_type == "tar.gz":
        return f"{base_name}.tar.gz"
    if archive_type == "tar":
        return f"{base_name}.tar"
    if archive_type == "zip":
        return f"{base_name}.zip"
    return base_name


def _extract_archive(archive_path: str, extract_dir: str, archive_type: str) -> bool:
    try:
        os.makedirs(extract_dir, exist_ok=True)
        if archive_type in ("tar", "tar.gz"):
            mode = "r:gz" if archive_type == "tar.gz" else "r:"
            with tarfile.open(archive_path, mode) as tf:
                tf.extractall(extract_dir)
        elif archive_type == "zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(extract_dir)
        else:
            return False
        return True
    except Exception as e:
        print(f"[warn] Could not extract archive {archive_path}: {e}")
        return False


def get_available_chains(pdb_id: str) -> list:
    """
    Query the ATLAS API to get available chains for a given PDB ID.
    
    Args:
        pdb_id: The PDB identifier (e.g., '1MBN')
        
    Returns:
        List of available chain identifiers
    """
    try:
        # First, fetch metadata to see available chains
        url = f"{ATLAS_API_BASE_URL}/ATLAS/metadata/{pdb_id}"
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urlopen(req, timeout=30) as response:
            metadata = json.loads(response.read().decode('utf-8'))
            if isinstance(metadata, dict) and 'chains' in metadata:
                return list(metadata['chains'].keys())
            # If structure is different, try to extract chains from the response
            if isinstance(metadata, dict):
                # Try to infer available chains from the response structure
                print(f"Metadata response: {list(metadata.keys())}")
                # Default to just the PDB ID without chain for now
                return [pdb_id.upper()]
    except (HTTPError, URLError) as e:
        print(f"Warning: Could not fetch metadata for {pdb_id}: {e}")
        # Default to trying without specific chain
        return [pdb_id.upper()]


def download_protein_trajectory(pdb_id: str, chain_id: str = None, output_dir: str = None) -> bool:
    """
    Download protein trajectory from ATLAS for a given PDB ID and chain.
    
    Args:
        pdb_id: The PDB identifier (e.g., '1MBN')
        chain_id: The chain identifier (e.g., 'A'). If None, will try to get the main chain.
        output_dir: Directory to save the file. If None, uses current directory.
        
    Returns:
        True if successful, False otherwise
    """
    pdb_id = pdb_id
    
    # Determine the pdb_chain identifier
    if chain_id:
        pdb_chain = f"{pdb_id.lower()}_{chain_id.upper()}"
    else:
        pdb_chain = pdb_id
    
    # Set output directory
    if output_dir is None:
        output_dir = "."
    else:
        output_dir = str(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    # Construct the download URL for protein trajectory
    url = f"{ATLAS_API_BASE_URL}/ATLAS/protein/{pdb_chain}"
    
    base_name = f"{pdb_chain}_protein"
    
    print(f"Downloading MD trajectory for {pdb_chain}...")
    print(f"URL: {url}")
    
    try:
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        
        with urlopen(req, timeout=300) as response:
            content_type = response.headers.get('Content-Type', '')
            content_disposition = response.headers.get('Content-Disposition', '')

            archive_type = _infer_archive_type(content_type, content_disposition, url)
            output_filename = _default_filename_for_archive(base_name, archive_type)
            output_path = os.path.join(output_dir, output_filename)

            total_size = response.headers.get('Content-Length')
            if total_size:
                total_size = int(total_size)
                print(f"File size: {total_size / (1024**3):.2f} GB")

            print(f"Content-Type: {content_type or 'unknown'}")
            if content_disposition:
                print(f"Content-Disposition: {content_disposition}")
            if archive_type:
                print(f"Detected archive payload: {archive_type}")
            
            downloaded = 0
            chunk_size = 8192 * 16  # 128 KB chunks
            
            with open(output_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size:
                        percent = (downloaded / total_size) * 100
                        print(f"Downloaded: {downloaded / (1024**3):.2f} GB / {total_size / (1024**3):.2f} GB ({percent:.1f}%)", end='\r')
        
        print(f"\n✓ Successfully downloaded to {output_path}")

        if archive_type:
            extract_dir = os.path.join(output_dir, base_name)
            if _extract_archive(output_path, extract_dir, archive_type):
                print(f"✓ Extracted archive to {extract_dir}")
            else:
                print("[warn] Download succeeded but extraction failed. Archive kept as-is.")

        return True
        
    except HTTPError as e:
        print(f"\n✗ HTTP Error {e.code}: {e.reason}")
        if e.code == 404:
            print(f"  PDB ID '{pdb_chain}' not found in ATLAS database")
        return False
    except URLError as e:
        print(f"\n✗ Connection Error: {e.reason}")
        return False
    except Exception as e:
        print(f"\n✗ Error downloading file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download molecular dynamics trajectories from the ATLAS database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_atlas.py 1MBN
  python download_atlas.py 1MBN --chain A
  python download_atlas.py 1MBN --chain A --output_dir ./data/
        """
    )
    
    parser.add_argument('pdb_id', help='PDB identifier (e.g., 1MBN)')
    parser.add_argument('--chain', '-c', default=None, help='Chain identifier (e.g., A). If not specified, downloads the main entry.')
    parser.add_argument('--output_dir', '-o', default="/nfs/roberts/pi/pi_sk2433/shared/ProtSCAPE_2026_MDSimulations", help='Output directory for the downloaded file. Default: current directory')
    parser.add_argument('--check-chains', action='store_true', help='Check available chains for the PDB ID without downloading')
    
    args = parser.parse_args()
    
    if args.check_chains:
        print(f"Checking available chains for {args.pdb_id}...")
        chains = get_available_chains(args.pdb_id)
        print(f"Available chains: {', '.join(chains)}")
        return
    
    success = download_protein_trajectory(
        pdb_id=args.pdb_id,
        chain_id=args.chain,
        output_dir=args.output_dir
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
