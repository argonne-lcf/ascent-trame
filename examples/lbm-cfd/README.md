# LBM-CFD

Lattice-Boltzmann Method 2D Computational Fluid Dynamics Simulation

Code is parallelized using MPI


### Build LBM-CFD application
With Ascent support:
```
env ASCENT_DIR=<ascent_install_dir>/install/ascent-checkout make
```

Without Ascent support:
```
make
```


### Run Trame server
Make sure that the Python virtual environment created for Ascent install is activated

```
python trame_app.py --host 0.0.0.0 --port <port> --server --timeout 0
```


### Run LBM-CFD application
Make sure that the Python virtual environment created for Ascent install is activated

Ensure Trame server is up and running, and you have opened the web page to the Trame application

Then run the following (note: you may need to specify different versions for Python and Conduit - check your install)
```
PYTHON_SITE_PKG="<python_virtual_env_path>/lib/python3.12/site-packages"
ASCENT_DIR="<ascent_install_dir>/install"
export PYTHONPATH=$PYTHONPATH:PYTHON_SITE_PKG:$ASCENT_DIR/ascent-checkout/python-modules/:$ASCENT_DIR/conduit-v0.9.2/python-modules/

mpiexec -np <num_procs> ./bin/lbmcfd
```

