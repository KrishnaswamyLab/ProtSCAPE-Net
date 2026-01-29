"""
Convert BPTI PDB files from multiple directories to a single XTC trajectory.

This script collects all PDB files from BPTI subdirectories (0 to 2 us, 2 to 4 us, etc.)
and converts them into a single .xtc trajectory file.

Usage:
    python dihedral_analysis.py
"""

import os
import glob
import subprocess
from shutil import which


def de_shaw_filesort(filename):
	"""Sort DE Shaw files by frame number."""
	base = os.path.basename(filename)
	fname = os.path.splitext(base)[0]
	frame = fname.split("-")[1]
	return int(frame)


def convert_pdbs_to_xtc(parent_dir, prot_name, traj_dirs, output_file):
	"""
	Convert all PDB files from multiple directories to a single XTC file.
	
	Args:
		parent_dir: Parent directory containing protein subdirectories
		prot_name: Protein name (e.g., "BPTI")
		traj_dirs: List of trajectory directory names
		output_file: Output XTC filename
	"""
	if which('mdconvert') is None:
		raise Exception("mdconvert utility not found. Fix: pip install mdtraj")
	
	# Collect all PDB files from all trajectory directories
	all_pdb_files = []
	
	print(f"[info] Collecting PDB files from {len(traj_dirs)} directories...")
	for traj_dir in traj_dirs:
		traj_path = os.path.join(parent_dir, prot_name, traj_dir)
		pdb_files = sorted(
			glob.glob(os.path.join(traj_path, "*.pdb")),
			key=de_shaw_filesort
		)
		print(f"[info] Found {len(pdb_files)} PDB files in {traj_dir}")
		all_pdb_files.extend(pdb_files)
	
	if not all_pdb_files:
		raise FileNotFoundError(f"No PDB files found in {parent_dir}/{prot_name}")
	
	print(f"[info] Total PDB files: {len(all_pdb_files)}")
	
	# Convert to XTC using mdconvert
	print(f"[info] Converting PDB files to {output_file}...")
	
	# Call mdconvert directly with all files
	cmd = ['mdconvert', '-f', '-o', output_file] + all_pdb_files
	print(f"[cmd] mdconvert -f -o {output_file} <{len(all_pdb_files)} PDB files>")
	
	try:
		result = subprocess.run(cmd, capture_output=True, text=True, check=True)
		if result.stdout:
			print(result.stdout)
		if result.stderr:
			print(result.stderr)
		exit_code = 0
	except subprocess.CalledProcessError as e:
		print(f"[error] mdconvert failed with exit code {e.returncode}")
		if e.stdout:
			print(e.stdout)
		if e.stderr:
			print(e.stderr)
		exit_code = e.returncode
	
	if exit_code == 0:
		print(f"[done] Successfully created {output_file}")
		if os.path.exists(output_file):
			size_mb = os.path.getsize(output_file) / (1024 * 1024)
			print(f"[info] Output file size: {size_mb:.2f} MB")
	else:
		print(f"[error] Conversion failed with exit code {exit_code}")
		return exit_code
	
	return 0


def main():
	# Configuration
	parent_dir = "/gpfs/gibbs/pi/krishnaswamy_smita/de_shaw"
	prot_name = "Ubiquitin"
	traj_dirs = ["0 to 2 us", "2 to 4 us", "4 to 6 us", "6 to 8 us", "8 to 10 us"]
	
	# Output directory and file
	output_dir = "datasets/Ubiquitin"
	os.makedirs(output_dir, exist_ok=True)
	output_file = os.path.join(output_dir, "Ubiquitin_full_trajectory.xtc")
	
	# Also get the first PDB as topology reference
	first_dir = os.path.join(parent_dir, prot_name, traj_dirs[0])
	pdb_files = sorted(glob.glob(os.path.join(first_dir, "*.pdb")), key=de_shaw_filesort)
	if pdb_files:
		topology_file = pdb_files[0]
		topology_output = os.path.join(output_dir, "Ubiquitin_topology.pdb")
		if not os.path.exists(topology_output):
			os.system(f"cp {topology_file} {topology_output}")
			print(f"[info] Copied topology file to {topology_output}")
	
	# Convert all PDBs to single XTC
	exit_code = convert_pdbs_to_xtc(parent_dir, prot_name, traj_dirs, output_file)
	
	return exit_code


if __name__ == "__main__":
	exit(main())
