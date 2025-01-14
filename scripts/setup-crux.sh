#!/bin/bash -l

# Setup script to build ascent-trame dependencies and examples
# on the Crux HPC: https://docs.alcf.anl.gov/crux/hardware-overview/machine-overview/

# Constants
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
ROOT_DIR=$(dirname "$SCRIPT_DIR")
BUILD_DIR=$SCRIPT_DIR/build-crux
VENV_DIR=$BUILD_DIR/python-venv
EXAMPLES_DIR=$ROOT_DIR/examples
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

# LBM-CFD Example
function setup_lbm_cfd() {
    local lbm_cfd_dir="$EXAMPLES_DIR/lbm-cfd"
    echo "Building LBM-CFD example..."

    if [ ! -d "$lbm_cfd_dir" ]; then
        echo "Error: $lbm_cfd_dir does not exist. Ensure the example is present."
        exit 1
    fi

    cd "$lbm_cfd_dir"
    env ASCENT_DIR="$ASCENT_INSTALL_DIR/ascent-checkout" make
}

# NekIBM Example
function setup_nekibm() {
    local nekibm_dir="$EXAMPLES_DIR/nekIBM-ascent"
    local tools_dir="$nekibm_dir/tools"
    local sourceme_file="$nekibm_dir/sourceme"
    local makenek_file="$nekibm_dir/bin/makenek"

    echo "Setting up nekIBM-ascent example..."

    # Create sourceme if it doesn't exist
    if [ ! -f "$sourceme_file" ]; then
        echo "Creating $sourceme_file..."
        cat << EOF > "$sourceme_file"
# Modules required for nekIBM-ascent
module use /soft/modulefiles
module load spack-pe-base cmake PrgEnv-gnu cray-python
source $VENV_DIR/bin/activate
EOF
    else
        echo "$sourceme_file already exists. Skipping creation."
    fi

    # Update makefile.template and makenek
    sed -i "86s|include /path/to/ascent_config.mk|include ${ASCENT_CONFIG_MK}|" ${nekibm_dir}/core/makefile.template
    sed -i "29s|ASCENT_DIR=\"/path/to/ascent-checkout/lib/cmake/ascent\"|ASCENT_DIR=\"${ASCENT_INSTALL_DIR}/ascent-checkout/lib/cmake/ascent\"|" ${nekibm_dir}/bin/makenek

    # Build tools
    cd "$tools_dir"
    ./maketools all

    # Run makenek in lidar_case directory with "uniform" argument
    local lidar_case_dir="$nekibm_dir/lidar_case"
    echo "Running makenek with 'uniform' in $lidar_case_dir..."
    if [ ! -f "$makenek_file" ]; then
        echo "Error: makenek script not found at $makenek_file"
        exit 1
    fi

    cd "$lidar_case_dir"
    "$makenek_file" uniform
}

# Main Workflow
echo "Starting setup..."
load_modules
mkdir -p "$BUILD_DIR"
setup_venv
build_ascent
fix_ascent_config

# Build examples
setup_lbm_cfd
setup_nekibm

echo "Setup complete!"
