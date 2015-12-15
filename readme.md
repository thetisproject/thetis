# COFS - Coastal Ocean Flow Solver

Finite element flow solver for simulating coastal and estuarine flows.

This project is licensed under the terms of the MIT license.

## Installation

- Install firedrake with all its dependencies
- COFS currently needs the following branches:
    - PyOP2: `master`
    - COFFEE: `master`
    - ffc: `cofs`
    - ufl: `cofs`
    - firedrake: `cofs`

### Installation with firedrake-install

- Install to a virtualenv in developer mode with correct branches

```
    export PETSC_CONFIGURE_OPTIONS="--download-metis --download-parmetis --download-netcdf --download-hdf5"
    python firedrake-install --developer --log --minimal_petsc \
    --package_branch PyOP2 master \
    --package_branch COFFEE master \
    --package_branch ffc cofs \
    --package_branch ufl cofs \
    --package_branch firedrake cofs
```

- Activate virtualenv
- Install COFS with `pip` in editable mode

```
    pip install -e /path/to/cofs/repo
```

- Install `pytest` in the same environment

```
    pip install --ignore-installed pytest
```
