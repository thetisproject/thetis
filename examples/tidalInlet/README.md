# Tidal Inlet

This example reproduces the idealised **tidal inlet** test case of
[Warner et al. (2008)](https://doi.org/10.1016/j.cageo.2008.02.012) (also used in other studies, e.g. [Wu et al., 2011](https://doi.org/10.1007/s11802-011-1788-3); [Bosboom et al., 2020](https://doi.org/10.1016/j.coastaleng.2020.103662)) and uses it
to demonstrate **combined wave–current sediment transport** in Thetis. In
particular, it exercises the [Van Rijn (2007)](https://doi.org/10.1061/(ASCE)0733-9429(2007)133:6(649))
intra-wave bedload formulation for combined wave–current flows, together with
radiation-stress (wave excess-momentum) forcing in the shallow-water momentum
equations.

The waves themselves are not solved by Thetis, but are pre-computed offline
with the **WAVEWATCH III (WW3)** spectral wave model and supplied to Thetis as a
time series of wave fields on the same mesh. A ready-to-use forcing file
(`ww3inlet9mc_12h.nc` with 4 records only) is shipped with the example, and the rest
explains how to regenerate it from scratch if need to change the wave
conditions or run a different experiment.

## The experiment

The domain is a rectangular basin **15 km wide × 14 km long**. A thin
land barrier runs across the middle of the domain and is cut by a single
narrow **inlet channel** (of 2 km wide). The barrier separates an open-ocean
region from a shallow basin, and water exchanges between the two
only through the inlet — the classic geometry for studying ebb/flood deltas and
inlet morphodynamics.

- **Bathymetry** (see `bathymetry_utils.py`): a flat 4 m shelf over the inner
  basin (`y ≤ 6800 m`) that ramps linearly down to 15 m at the open boundary
  (`y = 14000 m`) at the northern end.
- **Tidal forcing**: a single semi-diurnal (M2-like) constituent imposed as an
  elevation boundary condition at the open ocean boundary
  (amplitude ≈ 1 m, period 12 h).
- **Wave forcing**: significant wave height, peak frequency, mean period,
  direction, near-bed orbital velocity and the radiation-stress tensor, read
  from the WW3 output and interpolated in time at every Thetis step.
- **Sediment transport**: suspended load + bedload with bed evolution (Exner),
  using the Van Rijn (2007) bedload closure so that the waves actively reshape
  the inlet and its deltas.

The mesh (`inlet_v1_4.1.msh`, Gmsh format 4.1, 5079 nodes) carries
physical groups that Thetis can use for boundary conditioning:
`openboundary` (tag 2) - north edge, `coastline` (tag 3) and `Fluid_Domain` (tag 1).

## Files in this example

| File                            | Purpose                                                                          |
| ------------------------------- | -------------------------------------------------------------------------------- |
| `inlet.py`                    | Main Thetis runner: bathymetry, tide, wave forcing, sediment model and time loop |
| `bathymetry_utils.py`         | Analytical Warner et al. (2008) bathymetry profile                               |
| `generate_tidalinlet_mesh.py` | Builds `inlet_v1_4.1.msh` with Gmsh (run only if the mesh is missing)          |
| `plot_results.py`             | Post-processing / visualisation of the output results                           |
| `inlet_v1_4.1.msh`            | The computational mesh.                                                          |
| `ww3inlet9mc_12h.nc`          | Pre-computed WW3 wave forcing read by `inlet.py`                               |

## Running the Thetis simulation

If the mesh is not present, generate it first (requires `gmsh`):

```bash
python generate_tidalinlet_mesh.py
```

Then run the model (in parallel with MPI, or drop `mpirun` for serial):

```bash
mpirun -n 4 python inlet.py
python plot_results.py
```

By default `inlet.py` reads the bundled `ww3inlet9mc_12h.nc`. To use your own
wave forcing, point the `ww3waves` variable near the top of `inlet.py` at the
NetCDF file you produced with the workflow below or by any other approach reproducing wave fields (e.g. SWAN))

---

## Reproducing the WW3 wave dataset

The wave forcing is generated with **WAVEWATCH III v6.07.1**. The whole process
is: build the model -> generate boundary spectra -> run the WW3
pre-processors, solver, and then post-process and concatenate the output into a
single NetCDF file that Thetis reads.

### 1. Build WW3 v6.07.1

Download the source from the official release:
[https://github.com/NOAA-EMC/WW3/releases/tag/6.07.1](https://github.com/NOAA-EMC/WW3/releases/tag/6.07.1).

WW3 is configured through a **switch file** (the physics/parallelism namelist).
Create `model/bin/switch_tidalinlet` containing exactly:

```
F90 NOGRB NC4 DIST MPI PR3 UQ FLX0 LN1 ST0 NL0 BT1 DB1 TR0 BS0 IC0 IS0 REF0 XX0 WNT1 WNX1 CRT1 CRX1 TRKNC O0 O1 O2 O4 O5 O6 O7
```

The important choices here are `NC4` (NetCDF output, required by the
pre/post-processors), `MPI`/`DIST` (parallel build), and the unstructured-grid
propagation scheme (`PR3 UQ`). `ST0 NL0 REF0` switch the source terms off for a
clean, prescribed-spectrum boundary experiment; use e.g. `ST4 NL1 REF1` if need
to activate wind input, nonlinear interactions and reflection, and more (see WW3 manual)

From `model/bin`, build with the standard WW3 helper scripts:

```bash
cd WW3/model/bin
./w3_clean                                 # clean-up, only if a previous build exists
./w3_setup -c Gnu -s tidalinlet ../        # compiler = Gnu, switch = tidalinlet
./w3_automake                              # compiles the dependency libs + executables
```

After a successful build the executables (`ww3_grid`, `ww3_bounc`, `ww3_shel`,
`ww3_ounf`, …) are generated in `model/exe`, and templates `.inp` can be found
in `model/inp`.

> Compiler/MPI/NetCDF paths are platform specific. Use whatever Fortran
> compiler key the machine requires (`-c Gnu`, `-c Intel`, …) ensuring a
> NetCDF-enabled `mpif90` is on the `PATH` before running `w3_automake`.

### 2. Generate the boundary spectra

The open-boundary forcing is a set of per-node JONSWAP directional spectra
written in WW3 BOUNC NetCDF format by `gen_ww3_bc_spectra.py` (in the `waves/`
directory of the case):

```bash
python gen_ww3_bc_spectra.py
```

This writes one `id_<n>_spec.nc` file per boundary point into `harmonics/`,
plus a matching `ww3_bounc.inp` that lists them. Edit the configuration block at
the bottom of the script to change the number of spectra, the significant wave
height / peak period, the mean direction, or to switch on a tidal modulation of
the wave height (`TIDAL_SIGNAL = True`). Make sure that the resulting wave direction is southward.

> **Mesh format note:** WW3 v6.07.1 reads Gmsh **v2.2** meshes, whereas Thetis
> uses the v4.1 mesh in this experiment. Export/convert the mesh to v2.2 for the WW3 grid step
> (e.g. open in Gmsh → *Export* → *Version 2 ASCII*). The boundary node IDs
> listed in `ww3_grid.inp` correspond to this v2.2 mesh (manually extracted using Blender/Gmsh node ids)

### 3. Set up a run directory

Create a run folder (e.g. `examples/tidalInlet/waves/runner`) and copy the four input
templates into it, alongside the v2.2 mesh and the `harmonics/` spectra:

```
waves/
├── runner/
│   ├── ww3_grid.inp     # grid + boundary-point definition
│   ├── ww3_bounc.inp    # assembles the spectra into the nest file
│   ├── ww3_shel.inp     # run controls (times, output fields)
│   ├── ww3_ounf.inp     # NetCDF field post-processing
│   └── tidalinlet.msh   # v2.2 mesh referenced by ww3_grid.inp
└── harmonics/           # id_*_spec.nc produced in step 2
```

`ww3_bounc.inp` references the spectra as `../harmonics/id_*_spec.nc`, so run the
WW3 executables from inside `runner/`. Ready-to-edit copies of all four `.inp`
files are provided with the case under `waves/runner/`.

### 4. Run the WW3 workflow

From the run directory, execute the steps in order (assuming the `exe/` folder
is on your `PATH`, or call the binaries by full path):

```bash
# 1. Build the model-definition file (grid, spectral discretisation, BC points)
ww3_grid          # -> mod_def.ww3

# 2. Process the boundary spectra into the nesting file
ww3_bounc         # -> nest.ww3
		  # ensure that ww3_bounc.inp path to spectral files is correct relative to running directory

# 3. Run the wave model (parallel)
mpirun -n 4 ww3_shel        # ...or simply `ww3_shel` for a serial run
                            # -> out_grd.ww3, restart*.ww3, log.ww3

# 4. Convert the gridded output to NetCDF
ww3_ounf          # -> ww3.YYYYMMDD.nc  (one file per output day - this format 
	     	  #  			 is defined in ww3_ounf.inp)
```

`ww3_shel.inp` controls the simulated period, time step and the list of output
fields. For this case it requests
`SXY HS DIR T01 T02 UBR FP DEP` — i.e. the radiation-stress tensor, significant
wave height, direction, mean periods, near-bed orbital velocity, peak frequency
and depth, which provide necessary fields for `inlet.py` needs.

### 5. Concatenate and supply to Thetis

`If ww3_ounf` writes one NetCDF file per day (`ww3.20180101.nc`, `ww3.20180102.nc`,
…). Merge them along the time axis into a single file with
[NCO](https://nco.sourceforge.net/)'s `ncrcat`:

```bash
ncrcat -h ww3.2018010*.nc ww3inlet.nc
```

Finally, point the `ww3waves` variable in `inlet.py` at the resulting file and
re-run the Thetis simulation. The KD-tree node matching in `inlet.py` maps the
WW3 fields onto the Thetis mesh automatically, so no further alignment is
needed.

## References

1. Warner, J.C., Sherwood, C.R., Signell, R.P., Harris, C.K., Arango, H.G.
   (2008). *Development of a three-dimensional, regional, coupled wave,
   current, and sediment-transport model.* Computers & Geosciences, 34(10),
   1284–1306. [https://doi.org/10.1016/j.cageo.2008.02.012](https://doi.org/10.1016/j.cageo.2008.02.012)
2. van Rijn, L.C. (2007). *Unified View of Sediment Transport by Currents and
   Waves. II: Suspended Transport.* Journal of Hydraulic Engineering, 133(6),
   649–667. [https://doi.org/10.1061/(ASCE)0733-9429(2007)133:6(649)](https://doi.org/10.1061/(ASCE)0733-9429(2007)133:6(649))
3. The WAVEWATCH III® Development Group (2019). *User manual and system
   documentation of WAVEWATCH III version 6.07.* NOAA/NWS/NCEP/MMAB.
   [https://github.com/NOAA-EMC/WW3/releases/tag/6.07.1](https://github.com/NOAA-EMC/WW3/releases/tag/6.07.1)
4. Wu, L., Chen, C., Guo, P., Shi, M., Qi, J., and Ge, J. (2011). *A FVCOM-based
   unstructured grid wave, current, sediment transport model, I. Model description
   and validation.* Journal of Ocean University of China, 10, 1–8.
   [https://doi.org/10.1007/s11802-011-1788-3](https://doi.org/10.1007/s11802-011-1788-3)
5. Bosboom, J., Mol, M., Reniers, A., Stive, M., and de Valk, C. (2020). *Optimal
   sediment transport for morphodynamic model validation.* Coastal Engineering,
   158, 103662.
   [https://doi.org/10.1016/j.coastaleng.2020.103662](https://doi.org/10.1016/j.coastaleng.2020.103662)
