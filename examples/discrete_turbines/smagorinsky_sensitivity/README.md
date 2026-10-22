# Single-turbine Smagorinsky sensitivity

Run `./run_sensitivity.sh` from this directory for 50 m depth, or
`./run_sensitivity.sh 30` for 30 m depth.  It starts each case from a
uniform 3 m/s velocity, evolves a single AR2000 turbine wake for 600 s, and
writes centreline transects at `y = 150 m`.  The 1,200 m domain gives more
than 38D downstream of the turbine.  Each transect is the mean of the final
five 20 s timesteps (t = 520--600 s).

The cases are constant 1 m2/s viscosity (no Smagorinsky), then unbounded
Smagorinsky viscosity with `C_s = 0.1`, `1`, `5`, and `10`.  The resulting
`results/velocity_deficit.png` plots `1 - u/3` against downstream distance
from the turbine in rotor diameters.  Results are written under
`results/depth_<depth>m/`.  The constant-viscosity curve is dashed
black and drawn on top when it overlaps another case.

The shell script generates `single_turbine.msh`; Physical Surface 100 contains
the turbine bump.  It accepts an alternate Firedrake interpreter through
`FIREDRAKE_PYTHON`.
