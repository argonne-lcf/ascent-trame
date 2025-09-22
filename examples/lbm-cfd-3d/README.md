# LBM-CFD-3D

### Lattice-Boltzmann Method Computational Fluid Dynamics Simulation

- 3D LBM CFD simulation parallelized using MPI
- Supports multiple lattice models (D3Q15, D3Q19, D3Q27)
- Generates ParaView-compatible VTS files for visualizing velocity and vorticity fields
- Utilizes adaptive time-stepping for improved accuracy and efficiency
- Outputs the physical parameters and simulation details during execution, as well as a timing summary at the end

### Time Stepping

This LBM-CFD implementation uses adaptive time stepping based on stability conditions:

- **CFL Condition**: Ensures fluid doesn't move more than one grid cell per time step
- **Diffusion Stability**: Prevents numerical instabilities in viscous flows  
- **Automatic Selection**: The simulation automatically chooses the smaller (more restrictive) time step
- **Simulated time:** Varies automatically based on grid resolution and flow parameters

## Building and Running

Note:  Directions for running the simulation are based on Cray MPI systems on Crux

### Prerequisites

- **C++ Compiler**: mpic++
- **MPI**: OpenMPI/MPICH
- **ParaView**: For visualization (optional)
- **Python**: For ParaView file generation (optional)

### Command-Line Arguments

When running the executable directly, use these command-line arguments. One lattice model must be specified

- `--d3q15` - Use D3Q15 lattice model
- `--d3q19` - Use D3Q19 lattice model
- `--d3q27` - Use D3Q27 lattice model
- `--output-velocity` - Enable velocity vector output to VTS files
- `--output-vorticity` - Enable vorticity output to VTS files

### Run simulation (using command line arguments)
```
cd examples/lbm-cfd-3d
make 
mpiexec -n <num_procs> -ppn <procs_per_node> ./bin/lbmcfd3d --<lattice_model> --<vts_output_arguments>
```

### Makefile Commands

- `make all` - Compile application (default)
- `make run` - Run simulation
- `make pvd` - Generate ParaView files
- `make complete` - Build + Run + Generate PVD
- `make clean` - Remove compiled files
- `make help` - Display makefile usage information

### Alternative: Complete Workflow with ParaView Output Files

```
cd examples/lbm-cfd-3d
make complete N=<num_procs> PPN=<procs_per_node> LATTICE=<lattice_model> OUTPUT_VELOCITY=1 OUTPUT_VORTICITY=1
```

## Visualization with ParaView

The simulation generates VTS files containing vorticity and/or velocity data based on user-defined arguments. The files can be found in the `paraview/` directory. Use `make pvd` after running the simulation to create PVD files

There are two options to perform the visualization:
### Option #1: Local Visualization

To visualize the results, follow these steps:
1. Download the generated VTS/PVD files
2. Launch ParaView locally
3. In ParaView, go to File > Open and select  VTS/PVD files
4. Apply filters and settings to visualize the data

### Option #2: Remote Visualization using ParaView Modules*
Copy the name of the assigned compute node, and run the following in a new terminal:
```
ssh -L 11111:<node_name>:11111 <username>@<hostname>.alcf.anl.gov
```
Back on the compute node, start the ParaView server:
```
module use /soft/modulefiles
module load visualization/paraview/paraview-5.13.3
pvserver --server-port=11111
```
Then connect to localhost:11111 in ParaView's "Connect" dialog box

#


*Make sure locally installed version of ParaView matches the version used on the compute node

*More detailed instructions for the remote ParaView connection can be found here: https://docs.alcf.anl.gov/polaris/visualization/paraview-manual-launch/#setting-up-paraview
