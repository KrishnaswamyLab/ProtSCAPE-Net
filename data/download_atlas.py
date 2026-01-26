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


ATLAS_API_BASE_URL = "https://www.dsimb.inserm.fr/ATLAS/api"


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
        pdb_chain = f"{pdb_id}{chain_id.upper()}"
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
    
    # Determine output filename
    output_filename = f"{pdb_chain}_protein.xtc"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Downloading MD trajectory for {pdb_chain}...")
    print(f"URL: {url}")
    
    try:
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        
        with urlopen(req, timeout=300) as response:
            total_size = response.headers.get('Content-Length')
            if total_size:
                total_size = int(total_size)
                print(f"File size: {total_size / (1024**3):.2f} GB")
            
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
    parser.add_argument('--output_dir', '-o', default=None, help='Output directory for the downloaded file. Default: current directory')
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
