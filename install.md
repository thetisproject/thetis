# Installing COFS on Ubuntu 15.10

### Install build essentials etc

    sudo apt-get install -y build-essential python-dev git-core \
    mercurial python-pip libopenmpi-dev openmpi-bin libblas-dev \
    liblapack-dev gfortran curl cmake cmake-data libjsoncpp0v5 \
    libspatialindex-c4v5 libspatialindex-dev libspatialindex4v5 \
    pkg-config virtualenv zlib1g-dev \
    ipython python-scipy python-matplotlib

    pip install --user cachetools

### Install with firedrake-install

    curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install

```
    export PETSC_CONFIGURE_OPTIONS="--download-metis --download-parmetis --download-netcdf --download-hdf5"
    python firedrake-install --developer --log --minimal_petsc
```

- Activate virtualenv

```
    source /home/tuomas/sources/firedrake/bin/activate
```

- Install COFS with `pip` in editable mode

```
    pip install -e /path/to/cofs/repo
```

- Install `pytest` in the same environment

```
    pip install --ignore-installed pytest
```

### Issues

If `import h5py` fails with error `ValueError: numpy.dtype has the wrong size, try recompiling`, rebuild `h5py` in the virtualenv:

    cd /home/tuomas/sources/firedrake/src/h5py-2.5.0/
    source /home/tuomas/sources/firedrake/bin/activate
    export CC=mpicc
    python setup.py configure --hdf5=/home/tuomas/sources/firedrake/local/lib/python2.7/site-packages/petsc/
    python setup.py build
    ~/sources/firedrake/bin/python setup.py install
