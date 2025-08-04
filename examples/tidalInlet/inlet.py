"""
Tidal Inlet
=======================
Simulates the tidal inlet test case from Warner et al. (2008).

author: Seimur Shirinov
date: June 2025
"""

from thetis import *

import meshio
import numpy as np
from scipy.spatial import cKDTree
import os, sys
import xarray as xr

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
# bathymetry data (.msh format):
bathymetry_data='tidalInlet_bathymetry_data.msh'
# mesh with physical groups:
mesh2dfile='tidalinlet.msh'
mesh2d = Mesh(mesh2dfile)

# define function spaces
# P1_2d = FunctionSpace(mesh2d, 'CG', 1)
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
vectorP1_2d = VectorFunctionSpace(mesh2d, 'DG', 1)


# BATHYMETRY
# ---------------------------------------------
# define function space for bathymetry - wrapps P1_2d object
bathymetry_2d = Function(P1_2d, name='Bathymetry')
# bathymetric data at nodes from the mesh itself - added in pre-processing already
mesh = meshio.read(bathymetry_data)
# node coordinates
points = mesh.points  # shape: (num_nodes, 3)

# RE-ALIGN bathymetry with mesh
# ---------------------------------------------
# Given that the mesh verticies numbering gets shaffled by Mesh(), we need to re-align the bathymetry
# using kdtree find nearest neighbors from coords to points and thus re-align them:
tree = cKDTree(points[:, :2])
distances, indices = tree.query(mesh2d.coordinates.dat.data, k=1)

# final bathymetry
bathymetry_2d.dat.data[:] = points[indices, 2]



# Wave Fields
# ---------------------------------------------
ww3waves = '/Users/sshirinov/mac/cmcc/shyfem/cases/inlet/tideinlet3/ww3waves.nc'
ds = xr.open_dataset(ww3waves)

# Extract coordinates and variables
x_nc = ds['x'].values  # shape: (node,)
y_nc = ds['y'].values
coords_nc = np.column_stack((x_nc, y_nc))
sxx = ds['sxx'].values  # shape: (time, node)
syy = ds['syy'].values
sxy = ds['sxy'].values
hs_w    = ds['hs'].values
dir_w   = ds['dir'].values
p_m     = ds['t01'].values
f_p     = ds['fp'].values
uubr    = ds['uubr'].values
vubr    = ds['vubr'].values

mesh_rad = meshio.read(mesh2dfile)

# Get mesh points and connectivity
points_mesh = mesh_rad.points[:, :2]  # only x and y
cells = mesh_rad.cells_dict.get("triangle")
if cells is None:
    raise ValueError("Expected triangular mesh cells")

# match NetCDF nodes to mesh nodes 
# using KDTree to find closest matching node
tree = cKDTree(coords_nc)
_, idx_match = tree.query(mesh2d.coordinates.dat.data, k=1)

# reorder stress fields to match mesh node order
sxx_aligned = sxx[:, idx_match]
syy_aligned = syy[:, idx_match]
sxy_aligned = sxy[:, idx_match]
# similarly with other fields:
hs_w_aligned    = hs_w [:,idx_match] 
dir_w_aligned   = dir_w[:,idx_match]
p_m_aligned     = p_m  [:,idx_match]  
f_p_aligned     = f_p  [:,idx_match]  
uubr_aligned    = uubr [:,idx_match] 
vubr_aligned    = vubr [:,idx_match] 
u_mag_aligned   = np.sqrt(uubr_aligned**2 + vubr_aligned**2)

# Create functions for radiation stresses and wave parameters:
rad_stress_2d = Function(vectorP1_2d, name='rad_stress_2d')
wave_height_2d = Function(P1_2d, name = 'wave_height_2d')
wave_peak_freq_2d = Function(P1_2d, name = 'wave_peak_freq_2d')
wave_dir_2d = Function(P1_2d, name = 'wave_dir_2d')
wave_orbital_vel_2d = Function(P1_2d, name = 'wave_orbital_vel_2d') 
wave_mean_period_2d = Function(P1_2d, name = 'wave_mean_period_2d') 

# def update_wave_forcing(t_new,rad_stress_2d,P1_2d,vectorP1_2d,solver_obj):
def update_wave_forcing(t_new,P1_2d,solver_obj):

    #     # Interpolate
    solver_obj.fields.wave_height_2d.dat.data[:] = interpolate_at_dt(hs_w_aligned, ds.time, t_new)
    solver_obj.fields.wave_peak_freq_2d.dat.data[:] = interpolate_at_dt(f_p_aligned, ds.time, t_new)
    solver_obj.fields.wave_dir_2d.dat.data[:] = interpolate_at_dt(dir_w_aligned, ds.time, t_new)
    solver_obj.fields.wave_orbital_vel_2d.dat.data[:] = interpolate_at_dt(u_mag_aligned, ds.time, t_new)
    solver_obj.fields.wave_mean_period_2d.dat.data[:] = interpolate_at_dt(p_m_aligned, ds.time, t_new)

    sxx_int = Function(P1_2d, name="sxx_int")
    sxy_int = Function(P1_2d, name="sxy_int")
    syy_int = Function(P1_2d, name="syy_int")

    sxx_int.dat.data[:] = interpolate_at_dt(sxx_aligned, ds.time, t_new)
    sxy_int.dat.data[:] = interpolate_at_dt(sxy_aligned, ds.time, t_new)
    syy_int.dat.data[:] = interpolate_at_dt(syy_aligned, ds.time, t_new)

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
timestep = 60.    # -> time-stepping (if not adaptive)
t_end = 3600 * 49 # -> total run time in sec.: 2 days + 1h 
# t_export = 3600.0 # -> export interval in sec.
t_export = 1800.0 # -> export interval in sec.


# set-up tides:
# tidal_amplitude = 0.1
tidal_amplitude = 1.0
tidal_period = 43200 # 12 hours

# Copied from other test cases, perhaps not needed here
if os.getenv('THETIS_REGRESSION_TEST') is not None:
    # when run as a pytest test, only run 5 timesteps
    # and test the gradient
    t_end = 5*timestep
    test_gradient = True  # test gradient using Taylor test (see below)
    optimise = False  # skip actual gradient based optimisation
else:
    test_gradient = False
    optimise = True

print(t_end)


# Solver
# ---------------------------------------------
# outputdir = 'outputs_hydro'
outputdir = 'outputs_sed'

temp_const = 18.0
salt_ocean = 35.0
viscosity_hydro = Constant(5*10**(-2))
# average_size = 2 * 10**(-4)
average_size = 1 * 10**(-4)
ksp = Constant(3*average_size)


# # define solver object, passing a mesh and a bathymetry
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)

options = solver_obj.options
options.output_directory = outputdir
options.simulation_export_time = t_export
options.simulation_end_time = t_end
# options.cfl_2d = Constant(0.8)
options.swe_timestepper_type = 'CrankNicolson' # stable
# options.swe_timestepper_type = 'ForwardEuler' # -> unstable # BackwardEuler - stable
options.check_volume_conservation_2d = True
options.element_family = 'dg-dg' # the stable element family for this test case
# options.element_family = 'rt-dg' # worked too
options.swe_timestepper_options.implicitness_theta = 0.5
options.swe_timestepper_options.use_semi_implicit_linearization = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'bathymetry_2d','sediment_2d',
                            'rad_stress_2d','wave_height_2d','wave_orbital_vel_2d',
                            'wave_dir_2d'] # 
options.timestep = timestep


# Chose only one of the following friction formulations: 
# ---------------------------------------------
# # Manning drag coeff:
# manning_2d = Function(P1_2d, name="Manning coefficient")
# manning_2d.assign(5.0e-03)
# options.manning_drag_coefficient = manning_2d

# # Linear drag coeff:
# options.linear_drag_coefficient = Constant(0.003)
# options.nikuradse_bed_roughness = Constant(ksp)

# Using quadratic drag coefficient:
quadratic_drag = Function(P1_2d, name="Quadratic drag coefficient")
quadratic_drag.assign(2.5e-03)
options.quadratic_drag_coefficient = quadratic_drag

# horiz viscosity:
options.horizontal_viscosity = Constant(0.002) 
# Wave terms:
options.wave_curr_inter = True


# Sediments
# -------------
# below two options activate wave forcing and the use of van Rijn intra-wave bedload transport formulation
options.sediment_model_options.van_Rijn_bedload = True
options.sediment_model_options.wave_forcing = True
options.sediment_model_options.solve_suspended_sediment = True
options.sediment_model_options.use_bedload = True
options.sediment_model_options.solve_exner = True
options.sediment_model_options.use_angle_correction = True
options.sediment_model_options.use_slope_mag_correction = True
options.sediment_model_options.use_secondary_current = True
# if solve_suspended_sediment is True then correction should be true and backwards
options.sediment_model_options.use_advective_velocity_correction = True # if false leads to Nonetype * Function -> error
options.sediment_model_options.morphological_viscosity = Constant(1e-6) 
options.sediment_model_options.average_sediment_size = Constant(average_size)
options.sediment_model_options.bed_reference_height = Constant(0.03)
options.sediment_model_options.morphological_acceleration_factor = Constant(1.0)
options.sediment_model_options.horizontal_diffusivity = Constant(0.002) #< - just like in shyfem


# Tidal forcing
# ---------------------------------------------
# tidal_amplitude = 0.5  # meters
tidal_amplitude = 1.0  # meters !! as in the paper !
tidal_period = 43200   # seconds (12 hours)
phase_seconds = 10800  # seconds
ramp_duration = phase_seconds
ramp_duration = 4 * 3600 # delay in tidal evolution time
# phase_radians = (2 * np.pi / tidal_period) * phase_seconds
phase_radians = (2 * pi / tidal_period) * phase_seconds
t = np.linspace(0, t_end, int(timestep))  # seconds
tidal_elev = Constant(0.0)


# Forcing fields at each time step
# ---------------------------------------------
def update_forcings(t_new,):
    # print(f" updating forcing at t: {t_new}")
    # no ramp-up:
    # tidal_elev.assign(tidal_amplitude * sin((2 * pi / tidal_period) * (t_new + phase_seconds))) 
    # with ramp-up:
    tidal_elev.assign(tanh(t_new / ramp_duration) * sin(2*pi/tidal_period * (t_new + phase_seconds)) * tidal_amplitude)

    # update wave forcing:
    update_wave_forcing(t_new,P1_2d,solver_obj)


# Define the boundary conditions for the SWE
# ---------------------------------------------
# boundary condtitions are defined for each external boundary using their ID. 
# Ids taken as physical groups from mesh - predefined
openboundary = 2
coastline = 3
domain_surf = 1
swe_bnd = {}
ocean_salt = Constant(salt_ocean)
bnd_ocean_salt = {'value': ocean_salt}
solver_obj.bnd_functions['salt'] = {domain_surf: bnd_ocean_salt, coastline: bnd_ocean_salt, openboundary: bnd_ocean_salt}


freeslip_bc = {'un': Constant(0.0)}
swe_bnd[coastline] = freeslip_bc # this works-> imposes constant norm vel all along the coastline (free slip condition)
# swe_bnd[domain_surf] = freeslip_bc
# swe_bnd[openboundary] = freeslip_bc

swe_bnd[openboundary] = {'flux': Constant(0.0), 'elev': tidal_elev}

# BCs are now complete:
solver_obj.bnd_functions['shallow_water'] = swe_bnd

elev_init = Function(P1_2d)
elev_init.assign(0.0)
solver_obj.assign_initial_conditions(elev=elev_init, uv=as_vector((1e-5, 0.0)))
solver_obj.iterate(update_forcings=update_forcings)


