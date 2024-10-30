# Ascent-Trame
Bridge for accessing Ascent extracts in a Trame application


### MPI
Install desired version of MPI (e.g. OpenMPI, MPICH, etc.)


### Python Virtual Environment
```
cd lbm-cfd
python -m venv ./.venv
source .venv/bin/activate
pip install setuptools
pip install numpy
pip install opencv-python
pip install trame
pip install trame-vuetify
pip install trame-rca
pip install --no-binary :all: --compile mpi4py
```


### Ascent Install and Build (with MPI)
```
git clone --recursive https://github.com/alpine-dav/ascent.git
cd ascent
```

Edit "./scripts/build_ascent/build_ascent.sh" to change mfem version from 4.6 to 4.7

Make sure that the Python virtual environment created in the previous step is activated

```
env enable_python=ON enable_mpi=ON prefix=<ascent_install_dir> ./scripts/build_ascent/build_ascent.sh
```

