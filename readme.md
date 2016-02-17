# Thetis

Finite element flow solver for simulating coastal and estuarine flows.

This project is licensed under the terms of the MIT license.

## Installation

- Install firedrake with all its dependencies

### Installation with firedrake-install

- Install to a virtualenv in developer mode with correct branches

```
    export PETSC_CONFIGURE_OPTIONS="--download-metis --download-parmetis --download-netcdf --download-hdf5"
    python firedrake-install --developer --log --minimal_petsc
```

### Installing Thetis

- Activate virtualenv
- Install Thetis with `pip` in editable mode

```
    pip install -e /path/to/thetis/repo
```

- Install `pytest` and `colorama` in the same environment

```
    pip install --ignore-installed pytest colorama
```


