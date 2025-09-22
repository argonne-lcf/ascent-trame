#!/usr/bin/env python3
"""
LBM-CFD PVD Creator from VTS Files
Creates ParaView Data (PVD) files for both vorticity and velocity visualization.
Auto-detects available VTS files and creates appropriate PVD collections.
"""

import os
import re
import argparse
import sys
from pathlib import Path

def extract_timestep(filename):
    """Extract timestep from VTS filename."""
    match = re.search(r't(\d+)\.vts$', filename)
    return int(match.group(1)) if match else None

def create_pvd_file(vts_files, output_file, data_type):
    """Create a PVD file from a list of VTS files."""
    if not vts_files:
        return False
    
    # Sort files by timestep
    vts_files.sort(key=lambda x: extract_timestep(x[0]) or 0)
    
    with open(output_file, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')
        
        for filename, timestep in vts_files:
            f.write(f'    <DataSet timestep="{float(timestep)}" group="" part="0" file="{filename}"/>\n')
        
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')
    
    timesteps = [timestep for _, timestep in vts_files]
    print(f"  ✓ {data_type.capitalize()}: {len(vts_files)} files (t={min(timesteps)}-{max(timesteps)})")
    return True

def scan_vts_files(input_dir):
    """Scan directory for VTS files and categorize them."""
    vorticity_files = []
    velocity_files = []
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.vts'):
            timestep = extract_timestep(filename)
            if timestep is not None:
                if filename.startswith('simulation_state_'):
                    vorticity_files.append((filename, timestep))
                elif filename.startswith('velocity_vectors_'):
                    velocity_files.append((filename, timestep))
    
    return vorticity_files, velocity_files

def main():
    parser = argparse.ArgumentParser(
        description="Create PVD files from VTS files for LBM-CFD visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_pvd_from_vts.py --input-dir ./paraview
  python create_pvd_from_vts.py --input-dir . --vorticity-output custom_vorticity.pvd
        """
    )
    
    parser.add_argument('--input-dir', '-i', 
                       default='.', 
                       help='Directory containing VTS files (default: current directory)')
    
    parser.add_argument('--vorticity-output', 
                       default='lbm_vorticity.pvd',
                       help='Output filename for vorticity PVD (default: lbm_vorticity.pvd)')
    
    parser.add_argument('--velocity-output', 
                       default='velocity_vectors.pvd',
                       help='Output filename for velocity PVD (default: velocity_vectors.pvd)')
    
    args = parser.parse_args()
    
    # Convert to absolute path
    input_dir = os.path.abspath(args.input_dir)
    
    if not os.path.isdir(input_dir):
        print(f"Error: Directory '{input_dir}' does not exist")
        sys.exit(1)
    
    # Scan for VTS files
    vorticity_files, velocity_files = scan_vts_files(input_dir)
    
    created_files = 0
    
    # Create vorticity PVD if files exist
    if vorticity_files:
        vorticity_path = os.path.join(input_dir, args.vorticity_output)
        if create_pvd_file(vorticity_files, vorticity_path, "vorticity"):
            created_files += 1
    
    # Create velocity PVD if files exist  
    if velocity_files:
        velocity_path = os.path.join(input_dir, args.velocity_output)
        if create_pvd_file(velocity_files, velocity_path, "velocity"):
            created_files += 1
    
if __name__ == "__main__":
    main()
