"""
Tidal Inlet
=======================
Simulates the tidal inlet test case from Warner et al. (2008) [https://doi.org/10.1016/j.cageo.2008.02.012].

Description:
    This example demonstrates the use of wave-current forcing in the sediment transport module.
    It tests the implmentation of Leo Van Rijn, 2007 [10.1061/(ASCE)0733-9429(2007)133:6(649)]
    bedload transport formulation for combined wave-current flows,
    if the following options are set to True:
        use_vanRijn_2007_bedload = True (if false, fallws back to default Meyer-Peter Müller - only currents)
        wave_curr_inter = True (this will activate the excess momentum flux terms in momentum eq.)

Usage:
    The inlet.py can be run given the presence of the mesh and wave forcing files as:
        mpirun -n <num_procs> python inlet.py
    Reusults visualization:
        python plot_results.py

    If the mesh file is not present, run (required gmsh):
        python generate_tidalinlet_mesh.py
        -> export in 4.1 .msh format to handle physical groups (default save format)
    If wave frocing is not present, or a different wave forcing is desired,
        the WW3 model can be set-up and run using v6.07.1 version:
            [https://github.com/NOAA-EMC/WW3/releases/tag/6.07.1]

        requirements: switch file (physics namelist) with the following settings:



author: Seimur Shirinov
date: June 2025
"""

from thetis import *

import numpy as np
from scipy.spatial import cKDTree
import os
import netCDF4 as nc
from bathymetry_utils import generate_bathymetry

# Setup zones
sim_tz = timezone.pytz.utc
coord_system = coordsys.UTMCoordinateSystem(utm_zone=30)


op2.init(log_level=INFO)

# turn OFF reverse Cuthill-McKee reordering to optimize mesh inside Firedrake
# parameters["reorder_meshes"] = False


# FUNCTIONS:
def interpolate_at_dt(wavefield, time_array, dt_seconds):
    """
    linearly interpolates wave fields to a time step given in seconds

    params:
    - wavefield: np.ndarray of shape (n_times, n_nodes)
    - time_array: xarray DataArray or np.ndarray of datetime64[ns], shape (n_times,)
    - dt_seconds: float, seconds from the first time step (beginning of the simulation)
    output:
    - wavefield_interp: np.ndarray of shape (n_nodes,) representing interpolated values
    """

    # Convert datetime64[ns] to seconds since the first timestamp
    time_seconds = (time_array - time_array[0]) / np.timedelta64(1, 's')
    time_seconds = np.asarray(time_seconds)

    # Check if dt matches an existing time exactly
    if dt_seconds in time_seconds:
        idx = np.where(time_seconds == dt_seconds)[0][0]
        return wavefield[idx]

    # otherwise find bounding indices for interpolation
    if dt_seconds < time_seconds[0] or dt_seconds > time_seconds[-1]:
        raise ValueError("dt is outside the time range of the dataset.")

    idx_before = np.searchsorted(time_seconds, dt_seconds) - 1
    idx_after = idx_before + 1

    t0 = time_seconds[idx_before]
    t1 = time_seconds[idx_after]
    w = (dt_seconds - t0) / (t1 - t0)

    # interpolation
    wavefield_interp = (1 - w) * wavefield[idx_before] + w * wavefield[idx_after]
    return wavefield_interp


# MESH
# ---------------------------------------------
# mesh2dfile = './inlet_v1_2.2.msh'
mesh2dfile = 'inlet_v1_4.1.msh'
mesh2d = Mesh(mesh2dfile)

# define function spaces
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
vectorP1_2d = VectorFunctionSpace(mesh2d, 'DG', 1)


# BATHYMETRY
# ---------------------------------------------
# Replicates the Warner et al. (2008) tidal inlet profile:
#   flat shelf h=4 m for y ≤ 6800 m, then linear ramp to h=15 m at y=14000 m
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.dat.data[:] = generate_bathymetry(mesh2d, P1_2d).dat.data[:]

# Wave Fields
# ---------------------------------------------
ww3waves = 'ww3inlet9mc_12h.nc'
_ds = nc.Dataset(ww3waves)

# Extract coordinates and variables as plain numpy arrays
x_nc = _ds.variables['x'][:]
y_nc = _ds.variables['y'][:]
coords_nc = np.column_stack((x_nc, y_nc))
sxx = _ds.variables['sxx'][:]
syy = _ds.variables['syy'][:]
sxy = _ds.variables['sxy'][:]
hs_w = _ds.variables['hs'][:]
dir_w = _ds.variables['dir'][:]
p_m = _ds.variables['t01'][:]
f_p = _ds.variables['fp'][:]
uubr = _ds.variables['uubr'][:]
vubr = _ds.variables['vubr'][:]

# Convert CF time to numpy datetime64 (same dtype as xarray, so interpolate_at_dt is unchanged)
_tv = _ds.variables['time']
wave_time = np.array([
    np.datetime64(t.isoformat())
    for t in nc.num2date(_tv[:], _tv.units, getattr(_tv, 'calendar', 'standard'))
])


# match NetCDF nodes to mesh nodes
# using KDTree to find closest matching node
tree = cKDTree(coords_nc)
_, idx_match = tree.query(mesh2d.coordinates.dat.data, k=1)

# reorder stress fields to match mesh node order
sxx_aligned = sxx[:, idx_match]
syy_aligned = syy[:, idx_match]
sxy_aligned = sxy[:, idx_match]
# similarly with other fields:
hs_w_aligned = hs_w[:, idx_match]
dir_w_aligned = dir_w[:, idx_match]
p_m_aligned = p_m[:, idx_match]
f_p_aligned = f_p[:, idx_match]
uubr_aligned = uubr[:, idx_match]
vubr_aligned = vubr[:, idx_match]
u_mag_aligned = np.sqrt(uubr_aligned**2 + vubr_aligned**2)

# Create functions for radiation stresses and wave parameters:
rad_stress_2d = Function(vectorP1_2d, name='rad_stress_2d')
wave_height_2d = Function(P1_2d, name='wave_height_2d')
wave_peak_freq_2d = Function(P1_2d, name='wave_peak_freq_2d')
wave_dir_2d = Function(P1_2d, name='wave_dir_2d')
wave_orbital_vel_2d = Function(P1_2d, name='wave_orbital_vel_2d')
wave_mean_period_2d = Function(P1_2d, name='wave_mean_period_2d')

# def update_wave_forcing(t_new,rad_stress_2d,P1_2d,vectorP1_2d,solver_obj):


def update_wave_forcing(t_new, P1_2d, solver_obj):

    #     # Interpolate
    solver_obj.fields.wave_height_2d.dat.data[:] = interpolate_at_dt(hs_w_aligned, wave_time, t_new)
    solver_obj.fields.wave_peak_freq_2d.dat.data[:] = interpolate_at_dt(f_p_aligned, wave_time, t_new)
    solver_obj.fields.wave_dir_2d.dat.data[:] = interpolate_at_dt(dir_w_aligned, wave_time, t_new)
    solver_obj.fields.wave_orbital_vel_2d.dat.data[:] = interpolate_at_dt(u_mag_aligned, wave_time, t_new)
    solver_obj.fields.wave_mean_period_2d.dat.data[:] = interpolate_at_dt(p_m_aligned, wave_time, t_new)

    sxx_int = Function(P1_2d, name="sxx_int")
    sxy_int = Function(P1_2d, name="sxy_int")
    syy_int = Function(P1_2d, name="syy_int")

    sxx_int.dat.data[:] = interpolate_at_dt(sxx_aligned, wave_time, t_new)
    sxy_int.dat.data[:] = interpolate_at_dt(sxy_aligned, wave_time, t_new)
    syy_int.dat.data[:] = interpolate_at_dt(syy_aligned, wave_time, t_new)

    sxx_dx = Function(P1_2d).interpolate(sxx_int.dx(0))
    syy_dy = Function(P1_2d).interpolate(syy_int.dx(1))
    sxy_dx = Function(P1_2d).interpolate(sxy_int.dx(0))
    sxy_dy = Function(P1_2d).interpolate(sxy_int.dx(1))

    s_x = Function(P1_2d).interpolate(-sxx_dx - sxy_dy)
    s_y = Function(P1_2d).interpolate(-syy_dy - sxy_dx)

    solver_obj.fields.rad_stress_2d.interpolate(as_vector([s_x, s_y]))

    # return rad_stress_2d
    return


# TIMING
# ---------------------------------------------
# Simulation window:
timestep = 20       # -> time-stepping (if not adaptive)
t_end = 3600 * 48   # -> total run time in sec.: 2 days
t_export = 600      # -> export interval in sec: every 10min


# Copied from other test cases, perhaps not needed here
if os.getenv('THETIS_REGRESSION_TEST') is not None:
    # when run as a pytest test, only run 5 timesteps
    # and test the gradient
    t_end = 5*timestep

print(t_end)


# Solver
# ---------------------------------------------
outputdir = f'outputs_sed_{t_export}s'

temp_const = 18.0
salt_ocean = 35.0
viscosity_hydro = Constant(5*10**(-2))
average_size = 1 * 10**(-4)
ksp = Constant(3*average_size)


# # define solver object, passing a mesh and a bathymetry
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)

options = solver_obj.options
options.output_directory = outputdir
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.swe_timestepper_type = 'CrankNicolson'  # stable
# options.swe_timestepper_type = 'BackwardEuler' # 'ForwardEuler' # -> unstable # BackwardEuler - stable
# options.swe_timestepper_type = 'SSPRK33'
# options.swe_timestepper_type = 'DIRK22'
options.check_volume_conservation_2d = True
options.element_family = 'dg-dg'  # the stable element family for this test case
# options.element_family = 'rt-dg' # worked too
options.swe_timestepper_options.implicitness_theta = 0.5
options.swe_timestepper_options.use_semi_implicit_linearization = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'bathymetry_2d', 'sediment_2d',
                            'rad_stress_2d', 'wave_height_2d', 'wave_orbital_vel_2d',
                            'wave_dir_2d']
options.timestep = timestep

# options.swe_timestepper_options.use_automatic_timestep = True
# options.set_timestepper_type('CrankNicolson', implicitness_theta=1.0)
# Many schemes do not have automatic timestepping mech.
if hasattr(options.swe_timestepper_options, 'use_automatic_timestep'):
    options.swe_timestepper_options.use_automatic_timestep = True

# options.swe_timestepper_options.use_semi_implicit_linearization = False # this is forbidden if automatic timestepper = True

# Chose (uncomment) only one of the following friction formulations:
# ---------------------------------------------
# # Manning drag coeff:
# manning_2d = Function(P1_2d, name="Manning coefficient")
# manning_2d.assign(5.0e-03)
# options.manning_drag_coefficient = manning_2d

# # Linear drag coeff:
# options.linear_drag_coefficient = Constant(0.003)
# options.nikuradse_bed_roughness = Constant(ksp)

# Quadratic drag coefficient:
quadratic_drag = Function(P1_2d, name="Quadratic drag coefficient")
quadratic_drag.assign(2.5e-03)
options.quadratic_drag_coefficient = quadratic_drag

# horiz viscosity:
options.horizontal_viscosity = Constant(0.002)
# options.use_smagorinsky_viscosity = False

# Wave terms:
options.wave_curr_inter = True


# Sediments
# -------------
MORPHOFAC = 1.0
# below two options activate wave forcing and the use of van Rijn intra-wave bedload transport formulation
options.sediment_model_options.van_Rijn_bedload = True
options.sediment_model_options.wave_forcing = True
options.sediment_model_options.solve_suspended_sediment = True
options.sediment_model_options.use_bedload = True
options.sediment_model_options.solve_exner = True
options.sediment_model_options.use_angle_correction = True
options.sediment_model_options.use_slope_mag_correction = True
options.sediment_model_options.use_secondary_current = False  # if true will not use van_Rijn_bedload
# if solve_suspended_sediment is True then correction should be true and backwards
options.sediment_model_options.use_advective_velocity_correction = True  # if false leads to Nonetype * Function -> error
options.sediment_model_options.morphological_viscosity = Constant(1e-6)
options.sediment_model_options.average_sediment_size = Constant(average_size)
# options.sediment_model_options.bed_reference_height = Constant(average_size*2.5)
options.sediment_model_options.bed_reference_height = Constant(0.03)
options.sediment_model_options.morphological_acceleration_factor = Constant(MORPHOFAC)
options.sediment_model_options.horizontal_diffusivity = Constant(0.002)  # < - just like in shyfem

# options.sediment_model_options.sediment_timestepper_options.use_automatic_timestep = True
# options.sediment_model_options.exner_timestepper_options.use_automatic_timestep = True

# Tidal forcing
# ---------------------------------------------
tidal_amplitude = 1.03  # meters
tidal_period = 43200    # seconds
phase_seconds = 10800   # seconds
ramp_duration = phase_seconds
ramp_duration = 4 * 3600  # delay in tidal evolution time
# ramp_duration = 1.0
phase_radians = (2 * pi / tidal_period) * phase_seconds
t = np.linspace(0, t_end, int(timestep))  # seconds
tidal_elev = Constant(0.0)

RAMPUP = False

# Forcing fields at each time step
# ---------------------------------------------


def update_forcings(t_new,):

    if RAMPUP:
        # Ramp-up:
        tidal_elev.assign(tanh(t_new / ramp_duration) * sin(2*pi/tidal_period * (t_new + phase_seconds)) * tidal_amplitude)
    else:
        # No ramp-up:
        # offset -> to align with shyfem tidal signal output
        offset = 1600
        tidal_elev.assign(tidal_amplitude * sin(2 * pi * (t_new + phase_seconds + offset) / tidal_period))

        # zval = zref+ampli*sin(2.d0*pi*(rit+phase)/period)

    # update wave forcing:
    update_wave_forcing(t_new, P1_2d, solver_obj)


# Define the boundary conditions for the SWE
# ---------------------------------------------
# boundary condtitions are defined for each external boundary using their ID.
# Ids taken as physical groups from mesh - predefined in gmsh
openboundary = 2
coastline = 3
domain_surf = 1
swe_bnd = {}
ocean_salt = Constant(salt_ocean)
bnd_ocean_salt = {'value': ocean_salt}
solver_obj.bnd_functions['salt'] = {domain_surf: bnd_ocean_salt, coastline: bnd_ocean_salt, openboundary: bnd_ocean_salt}


freeslip_bc = {'un': Constant(0.0)}
swe_bnd[coastline] = freeslip_bc  # imposes constant norm vel all along the coastline
# swe_bnd[domain_surf] = freeslip_bc
# swe_bnd[openboundary] = freeslip_bc

# OB coundition
swe_bnd[openboundary] = {'flux': Constant(0.0), 'elev': tidal_elev}

# pass BCs to the solver:
solver_obj.bnd_functions['shallow_water'] = swe_bnd

elev_init = Function(P1_2d)
if RAMPUP:
    elev_init.assign(0.0)
else:
    elev_init.assign(tidal_amplitude)
solver_obj.assign_initial_conditions(elev=elev_init, uv=as_vector((1e-5, 0.0)))
solver_obj.iterate(update_forcings=update_forcings)
