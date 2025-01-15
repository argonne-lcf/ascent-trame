#!/bin/bash -l

# Setup script to build ascent-trame dependencies and examples
# on the Crux HPC: https://docs.alcf.anl.gov/crux/hardware-overview/machine-overview/

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
ROOT_DIR=$(dirname "$SCRIPT_DIR")
# TODO: combine into one for readability?

BUILD_DIR=$SCRIPT_DIR/build-crux
VENV_DIR=$BUILD_DIR/python-venv
ASCENT_INSTALL_DIR=$BUILD_DIR/ascent/scripts/build_ascent/install
ASCENT_CONFIG_MK=$ASCENT_INSTALL_DIR/ascent-checkout/share/ascent/ascent_config.mk
# TODO: add cmake directory?

EXAMPLES_DIR=$ROOT_DIR/examples
LBMCFD_DIR="$EXAMPLES_DIR/lbm-cfd"
NEKIBM_DIR="$EXAMPLES_DIR/nekIBM-ascent"
NEKIBM_TOOLS_DIR="$NEKIBM_DIR/tools"
SOURCEME_PATH="$NEKIBM_DIR/sourceme"
MAKENEK_PATH="$NEKIBM_DIR/bin/makenek"
BLOODFLOW_DIR="$EXAMPLES_DIR/bloodflow"

function update_submodules() {
    echo "Updating submodules..."
    git submodule update --init --recursive
}

function load_modules() {
    echo "Loading required modules for crux..."
    module use /soft/modulefiles
    module load spack-pe-base cmake PrgEnv-gnu cray-python
}

function setup_venv() {
    echo "Creating a new Python virtual environment..."
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install pip setuptools wheel numpy opencv-python trame trame-vuetify trame-rca pandas matplotlib --upgrade
    pip install --no-binary :all: --compile mpi4py
}

function build_ascent() {
    echo "Cloning Ascent..."
    git clone https://github.com/alpine-dav/ascent.git --recursive "$BUILD_DIR/ascent"
    cd "$BUILD_DIR/ascent/scripts/build_ascent/"

    cat << EOF > build_ascent_crux.sh
#!/bin/bash -l
set -eu -o pipefail

# Load modules
module use /soft/modulefiles
module load spack-pe-base cmake PrgEnv-gnu cray-python

# Activate Python venv before building Ascent
source "$VENV_DIR/bin/activate"

env enable_mpi=ON enable_python=ON enable_openmp=ON ./build_ascent.sh
EOF

    echo "Building Ascent..."
    chmod +x build_ascent_crux.sh
    ./build_ascent_crux.sh

    echo "Fixing linking issues in ascent_config.mk..."
    replace_line() {
        local file=$1
        local line_number=$2
        local new_line=$3
        local escaped_new_line=$(printf '%s\n' "$new_line" | sed 's/[&/\]/\\&/g')
        sed -i "${line_number}s/.*/${escaped_new_line}/" "$file"
    }

    replace_line "$ASCENT_CONFIG_MK" 90 "ASCENT_UMPIRE_RPATH_FLAGS_VALUE = -Wl,-rpath,\$(ASCENT_UMPIRE_DIR)/lib64"
    replace_line "$ASCENT_CONFIG_MK" 215 "ASCENT_UMPIRE_LIB_FLAGS = \$(if \$(ASCENT_UMPIRE_DIR),-L \$(ASCENT_UMPIRE_DIR)/lib64 -lumpire)"
    replace_line "$ASCENT_CONFIG_MK" 220 "ASCENT_CAMP_LIB_FLAGS = \$(if \$(ASCENT_CAMP_DIR),-L \$(ASCENT_CAMP_DIR)/lib -lcamp)"
}

# LBM-CFD Example
function setup_lbm_cfd() {
    echo "Building LBM-CFD example..."
    cd "$LBMCFD_DIR"
    env ASCENT_DIR="$ASCENT_INSTALL_DIR/ascent-checkout" make

    cat << EOF > "$LBMCFD_DIR/run-crux.sh"
#!/bin/bash -l

# Load required modules
module use /soft/modulefiles
module load spack-pe-base
module load cmake
module load PrgEnv-gnu
module load cray-python

# MPI and OpenMP Configuration
NNODES=\$(wc -l < \$PBS_NODEFILE)
NRANKS_PER_NODE=32
NTHREADS=2
NDEPTH=2
NTOTRANKS=\$((NNODES * NRANKS_PER_NODE))

echo "NUM_OF_NODES= \${NNODES}"
echo "TOTAL_NUM_RANKS= \${NTOTRANKS}"
echo "RANKS_PER_NODE= \${NRANKS_PER_NODE}"
echo "THREADS_PER_RANK= \${NTHREADS}"

# Activate Python Virtual Environment
source "$VENV_DIR/bin/activate"

# Set Python and Ascent Paths
PYTHON_SITE_PKG="\${PYTHON_VENV_DIR}/lib/python3.11/site-packages"
ASCENT_DIR="$ASCENT_INSTALL_DIR"
export PYTHONPATH="\$PYTHONPATH:\$PYTHON_SITE_PKG:\$ASCENT_DIR/ascent-checkout/python-modules/:\$ASCENT_DIR/conduit-v0.9.2/python-modules/"

# Start Trame Server in the Background
nohup python trame/trame_app.py --host 0.0.0.0 --port 8888 --server --timeout 0 > trame.log 2>&1 &
TRAME_PID=\$!
trap "kill \$TRAME_PID" EXIT

echo "---------------------------------------------------------------------------------"
echo "To access the Trame server, copy and run this command in a local terminal:"
echo "ssh -v -N -L 8888:\$(hostname):8888 \$USER@crux.alcf.anl.gov"
echo "---------------------------------------------------------------------------------"

# Pause for SSH command copying
sleep 5

# MPI Arguments
MPI_ARGS="-n \${NTOTRANKS} --ppn \${NRANKS_PER_NODE} --depth=\${NDEPTH} --cpu-bind depth"
OMP_ARGS="--env OMP_NUM_THREADS=\${NTHREADS} --env OMP_PROC_BIND=true --env OMP_PLACES=cores"
mpiexec \${MPI_ARGS} \${OMP_ARGS} ./bin/lbmcfd
EOF

    chmod +x "$LBMCFD_DIR/run-crux.sh"
}

# NekIBM Example
function setup_nekibm() {
    echo "Setting up nekIBM-ascent example..."

    echo "Creating sourceme file..."
    rm -f "$SOURCEME_PATH"
    cat << EOF > "$SOURCEME_PATH"
# Modules required for nekIBM-ascent
module use /soft/modulefiles
module load spack-pe-base cmake PrgEnv-gnu cray-python
source $VENV_DIR/bin/activate
EOF

    # Update makefile.template and makenek
    sed -i "86s|include /path/to/ascent_config.mk|include ${ASCENT_CONFIG_MK}|" ${NEKIBM_DIR}/core/makefile.template
    sed -i "29s|ASCENT_DIR=\"/path/to/ascent-checkout/lib/cmake/ascent\"|ASCENT_DIR=\"${ASCENT_INSTALL_DIR}/ascent-checkout/lib/cmake/ascent\"|" ${MAKENEK_PATH}

    # Build tools
    cd "$NEKIBM_TOOLS_DIR"
    ./maketools all

    # Run makenek in lidar_case directory with "uniform" argument
    echo "Running makenek..."

    cd "$NEKIBM_DIR/lidar_case"
    "$MAKENEK_PATH" uniform
}

function setup_bloodflow() {
    echo "Setting up nekIBM-ascent example..."

    cd "$BLOODFLOW_DIR"
    mkdir -p build
    cd build
    env Ascent_DIR="${ASCENT_INSTALL_DIR}/ascent-checkout/lib/cmake/ascent" cmake ..
    make

    cat << EOF > "$BLOODFLOW_DIR/build/examples/bidirectionalSingleCell/run-crux.sh"
#!/bin/bash -l

# Load required modules
module use /soft/modulefiles
module load spack-pe-base
module load cmake
module load PrgEnv-gnu
module load cray-python

# MPI and OpenMP Configuration
NNODES=\$(wc -l < \$PBS_NODEFILE)
NRANKS_PER_NODE=32
NTHREADS=2
NDEPTH=2
NTOTRANKS=\$((NNODES * NRANKS_PER_NODE))

echo "NUM_OF_NODES= \${NNODES}"
echo "TOTAL_NUM_RANKS= \${NTOTRANKS}"
echo "RANKS_PER_NODE= \${NRANKS_PER_NODE}"
echo "THREADS_PER_RANK= \${NTHREADS}"

# Activate Python Virtual Environment
source "$VENV_DIR/bin/activate"

# Set Python and Ascent Paths
PYTHON_SITE_PKG="\${PYTHON_VENV_DIR}/lib/python3.11/site-packages"
ASCENT_DIR="$ASCENT_INSTALL_DIR"
export PYTHONPATH="\$PYTHONPATH:\$PYTHON_SITE_PKG:\$ASCENT_DIR/ascent-checkout/python-modules/:\$ASCENT_DIR/conduit-v0.9.2/python-modules/"

# Start Trame Server in the Background
nohup python trame/trame_app.py --host 0.0.0.0 --port 8888 --server --timeout 0 > trame.log 2>&1 &
TRAME_PID=\$!
trap "kill \$TRAME_PID" EXIT

echo "---------------------------------------------------------------------------------"
echo "To access the Trame server, copy and run this command in a local terminal:"
echo "ssh -v -N -L 8888:\$(hostname):8888 \$USER@crux.alcf.anl.gov"
echo "---------------------------------------------------------------------------------"

# Pause for SSH command copying
sleep 5

# MPI Arguments
MPI_ARGS="-n \${NTOTRANKS} --ppn \${NRANKS_PER_NODE} --depth=\${NDEPTH} --cpu-bind depth"
OMP_ARGS="--env OMP_NUM_THREADS=\${NTHREADS} --env OMP_PROC_BIND=true --env OMP_PLACES=cores"
mpiexec -n 1 ./bidirectionalCellFlow in.lmp4cell 10 1
EOF
    chmod +x "$BLOODFLOW_DIR/build/examples/bidirectionalSingleCell/run-crux.sh"
}

# Main Workflow
echo "Setting up Ascent-Trame examples on crux"
echo "This script assumes that you are running from a login node, since it requires internet access to download dependencies"

# TODO: Make the user hit enter to continue, acknowlding that they must be on a login node

echo "Preparing dependencies..."
mkdir -p "$BUILD_DIR"
update_submodules
load_modules
setup_venv
build_ascent

# Build examples
setup_lbm_cfd
setup_nekibm
setup_bloodflow

echo "Setup complete!"
