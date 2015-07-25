# Installing COFS on Ubuntu 14.04

## Install Ubuntu 14.04

Either do a clean install or install in a virtual machine.

## Preliminaries

### Install build essentials etc

    sudo apt-get install -y build-essential python-dev git-core \
    mercurial python-pip libopenmpi-dev openmpi-bin libblas-dev \
    liblapack-dev gfortran

    sudo apt-get install swig
    sudo apt-get install gmsh
    sudo apt-get install ipython python-scipy python-matplotlib
    pip install --user cachetools

### Update python modules

    sudo pip install "Cython>=0.20" decorator "numpy>=1.6" "mpi4py>=1.3.1"

## Install PETSc

    PETSC_CONFIGURE_OPTIONS="--download-ctetgen --download-triangle --download-chaco" \
    pip install --user https://bitbucket.org/mapdes/petsc/get/firedrake.tar.bz2

    unset PETSC_DIR
    unset PETSC_ARCH

## Install petsc4py

    pip install --user git+https://bitbucket.org/mapdes/petsc4py.git@firedrake#egg=petsc4py

## Install PyOP2

    mkdir -p ~/src/PyOP2
    cd ~/src/PyOP2
    git clone git://github.com/OP2/PyOP2.git .
    python setup.py install --user

## Install COFFEE

    mkdir -p ~/src/COFFEE
    cd ~/src/COFFEE
    git clone https://github.com/coneoproject/COFFEE.git .
    python setup.py install --user

## install FEniCS components

    mkdir -p ~/src/fenics/ffc
    cd ~/src/fenics/ffc
    git clone https://bitbucket.org/mapdes/ffc.git .
    git checkout fd_bendy
    python setup.py install --user

    mkdir -p ~/src/fenics/ufl
    cd ~/src/fenics/ufl
    git clone https://bitbucket.org/mapdes/ufl.git .
    git checkout fd_bendy
    python setup.py install --user

    mkdir -p ~/src/fenics/fiat
    cd ~/src/fenics/fiat
    git clone https://bitbucket.org/mapdes/fiat.git .
    python setup.py install --user

    mkdir -p ~/src/fenics/instant
    cd ~/src/fenics/instant
    git clone https://bitbucket.org/fenics-project/instant.git .
    python setup.py install --user

## install firedrake

    git clone https://github.com/firedrakeproject/firedrake.git .
    git remote add myfork https://github.com/tkarna/firedrake.git
    git pull myfork
    git checkout -b tensorelem_facetdofs_bendy myfork/tensorelem_facetdofs_bendy
    python setup.py install --user

## install COFS

    mkdir -p ~/src/cofs
    cd ~/src/cofs
    git clone https://tkarna@bitbucket.org/tkarna/cofs.git .
    python setup.py install --user
