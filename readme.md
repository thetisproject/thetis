# COFS - Coastal Ocean Flow Solver

Finite element flow solver for simulating coastal and estuarine flows.

This project is licensed under the terms of the MIT license.

## Installation

- Install firedrake with all its dependencies
- COFS currently needs the following branches:
    - PyOP2: `multiple_top_bottom_masks`
    - COFFEE: `enhance-rewriter-mode3-logflops`
    - ffc: `fd_bendy`
    - ufl: `fd_bendy`
    - firedrake: `extruded_geometric_bcs` merged with `bendy_changes`

### Installation with firedrake-install

- Install to a virtualenv in developer mode with correct branches

```
    python firedrake-install --developer --log --minimal_petsc \
    --package_branch PyOP2 multiple_top_bottom_masks \
    --package_branch COFFEE enhance-rewriter-mode3-logflops \
    --package_branch ffc fd_bendy \
    --package_branch ufl fd_bendy \
    --package_branch firedrake extruded_geometric_bcs
```

- Merge firedrake branch with `bendy_changes`
- Activate virtualenv
- Install COFS with `pip` in editable mode

```
    pip install -e /path/to/cofs/repo
```
