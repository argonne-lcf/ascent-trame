#!/bin/bash -l

# Setup script to build ascent-trame dependencies and examples
# on the Crux HPC: https://docs.alcf.anl.gov/crux/hardware-overview/machine-overview/

# Constants
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
ROOT_DIR=$SCRIPT_DIR/../
BUILD_DIR=$SCRIPT_DIR/build-crux
VENV_DIR=$BUILD_DIR/python-venv
EXAMPLES_DIR=$ROOT_DIR/examples
LBMCFD_DIR=$EXAMPLES_DIR/lbm-cfd
ASCENT_INSTALL_DIR=$BUILD_DIR/ascent/scripts/build_ascent/install
ASCENT_CONFIG_MK=$ASCENT_INSTALL_DIR/ascent-checkout/share/ascent/ascent_config.mk

# Functions
function load_modules() {
    echo "Loading required modules..."
    module use /soft/modulefiles
    module load spack-pe-base
    module load cmake
    module load PrgEnv-gnu
    module load cray-python
}

function setup_venv() {
    echo "Setting up Python virtual environment..."
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install pip setuptools numpy opencv-python trame trame-vuetify trame-rca --upgrade
    pip install --no-binary :all: --compile mpi4py
}

function build_ascent() {
    echo "Cloning and building Ascent..."
    git clone https://github.com/alpine-dav/ascent.git --recursive "$BUILD_DIR/ascent"
    cd "$BUILD_DIR/ascent/scripts/build_ascent/"

    cat << EOF > build_ascent_crux.sh
#!/bin/bash -l
set -eu -o pipefail

# Load modules
module use /soft/modulefiles
module load spack-pe-base
module load cmake
module load PrgEnv-gnu
module load cray-python

# Activate Python venv before building Ascent
source "$VENV_DIR/bin/activate"

env enable_mpi=ON enable_python=ON enable_openmp=ON ./build_ascent.sh
EOF

    chmod +x build_ascent_crux.sh
    ./build_ascent_crux.sh
}

function fix_ascent_config() {
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

function build_examples() {
    echo "Building LBM-CFD example..."
    if [ ! -d "$LBMCFD_DIR" ]; then
        echo "Error: $LBMCFD_DIR does not exist. Ensure the examples are present."
        exit 1
    fi
    cd "$LBMCFD_DIR"
    env ASCENT_DIR="$ASCENT_INSTALL_DIR/ascent-checkout" make
}

function create_run_script() {
    echo "Creating run script..."
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

    chmod +x "$ROOT_DIR/run-crux.sh"
}

# Main Workflow
echo "Starting setup..."
load_modules
mkdir -p "$BUILD_DIR"
setup_venv
build_ascent
fix_ascent_config
build_examples
create_run_script
echo "Setup complete!"

