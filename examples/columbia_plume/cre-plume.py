"""
Columbia river plume simulation
===============================

Simulates the Columbia river plume with realistic tides and wind, atmospheric
pressure and river discharge forcings.

Initial and boundary conditions for salinity and temperature are obtained from
NCOM hindcast simulation. Tides are computed from the FES2004 tidal
models. Atmospheric forcings are from the NAM model forecasts. Bathymetry is a
CMOP composite for Columbia river and the adjacent shelf.

The forcing data are loaded from subdirectories:

- bathymetry
    ./bathymetry_300m.npz
- tides
    ./forcings/tide.fes2004.nc
- atmospheric data
    ./forcings/atm/nam/nam_air.local.YYYY_MM_DD.nc
- NCOM ocean data
    ./forcings/ncom/model_h.nc
    ./forcings/ncom/model_lat.nc
    ./forcings/ncom/model_ang.nc
    ./forcings/ncom/model_lon.nc
    ./forcings/ncom/model_zm.nc
    ./forcings/ncom/YYYY/t3d/t3d.glb8_2f_YYYYMMDD00.nc
    ./forcings/ncom/YYYY/s3d/s3d.glb8_2f_YYYYMMDD00.nc

"""
from thetis import *
from bathymetry import *
from tidal_forcing import TidalBoundaryForcing
from diagnostics import *
from ncom_forcing import NCOMInterpolator
from atm_forcing import *
comm = COMM_WORLD

rho0 = 1000.0
# set physical constants
physical_constants['rho0'].assign(rho0)
physical_constants['z0_friction'].assign(0.005)

reso_str = 'coarse'
meshfile = {
    'coarse': 'mesh_cre-plume_02_coarse.msh',
    'normal': 'mesh_cre-plume_02_normal.msh',
}
dt_select = {
    'coarse': 30.,
    'normal': 15.,
}
dt = dt_select[reso_str]
zgrid_params = {
    # nlayers, surf_elem_height, max_z_stretch
    'coarse': (9, 5.0, 4.0),
    'normal': (20, 0.25, 4.0),
}
nlayers, surf_elem_height, max_z_stretch = zgrid_params[reso_str]
outputdir = 'outputs_{:}_dt{:}'.format(reso_str, int(dt))
mesh2d = Mesh(meshfile[reso_str])
print_output('Loaded mesh ' + mesh2d.name)

north_bnd_id = 2
west_bnd_id = 5
south_bnd_id = 7
river_bnd_id = 6

if reso_str == 'coarse':
    north_bnd_id = 2
    west_bnd_id = 4
    south_bnd_id = 6
    river_bnd_id = 5

nnodes = comm.allreduce(mesh2d.topology.num_vertices(), MPI.SUM)
ntriangles = comm.allreduce(mesh2d.topology.num_cells(), MPI.SUM)
nprisms = ntriangles*nlayers

sim_tz = timezone.FixedTimeZone(-8, 'PST')
init_date = datetime.datetime(2006, 5, 10, tzinfo=sim_tz)

t_end = 52*24*3600.
t_export = 900.

# interpolate bathymetry and smooth it
bathymetry_2d = get_bathymetry('bathymetry_utm.nc', mesh2d, project=False)
bathymetry_2d = smooth_bathymetry(
    bathymetry_2d, delta_sigma=1.0, bg_diff=0,
    alpha=1e2, exponent=2.5,
    minimum_depth=3.5, niter=30)
bathymetry_2d = smooth_bathymetry_at_bnd(bathymetry_2d,
                                         [north_bnd_id, south_bnd_id])

# 3d mesh vertical stretch factor
z_stretch_fact_2d = Function(bathymetry_2d.function_space(), name='z_stretch')
# 1.0 (sigma mesh) in shallow areas, 4.0 in deep ocean
z_stretch_fact_2d.project(-ln(surf_elem_height/bathymetry_2d)/ln(nlayers))
z_stretch_fact_2d.dat.data[z_stretch_fact_2d.dat.data < 1.0] = 1.0
z_stretch_fact_2d.dat.data[z_stretch_fact_2d.dat.data > max_z_stretch] = max_z_stretch

coriolis_f, coriolis_beta = beta_plane_coriolis_params(46.25)
q_river = 5000.
salt_river = 0.0
salt_ocean_surface = 32.0
salt_ocean_bottom = 34.0
temp_river = 15.0
temp_ocean_surface = 13.0
temp_ocean_bottom = 8.0
reynolds_number = 160.0

u_scale = 3.0
w_scale = 1e-3
delta_x = 2e3
nu_scale = u_scale * delta_x / reynolds_number

simple_barotropic = False  # for debugging

# create solver
extrude_options = {
    'z_stretch_fact': z_stretch_fact_2d,
}

solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, nlayers,
                               extrude_options=extrude_options)
options = solver_obj.options
options.element_family = 'dg-dg'
options.timestepper_type = 'SSPRK22'
options.solve_salinity = not simple_barotropic
options.solve_temperature = not simple_barotropic
options.use_implicit_vertical_diffusion = True  # not simple_barotropic
options.use_bottom_friction = True  # not simple_barotropic
options.use_turbulence = True  # not simple_barotropic
options.use_turbulence_advection = False  # not simple_barotropic
options.use_smooth_eddy_viscosity = False
options.turbulence_model_type = 'gls'
options.use_baroclinic_formulation = not simple_barotropic
options.lax_friedrichs_velocity_scaling_factor = Constant(1.0)
options.lax_friedrichs_tracer_scaling_factor = Constant(1.0)
options.vertical_viscosity = Constant(2e-5)
options.vertical_diffusivity = Constant(2e-5)
options.horizontal_viscosity = Constant(2.0)
options.horizontal_diffusivity = Constant(2.0)
options.use_quadratic_pressure = True
options.use_limiter_for_tracers = True
options.use_limiter_for_velocity = True
options.use_smagorinsky_viscosity = True
options.smagorinsky_coefficient = Constant(1.0/np.sqrt(reynolds_number))
options.coriolis_frequency = Constant(coriolis_f)
options.simulation_export_time = t_export
options.simulation_end_time = t_end
if dt is not None:
    options.timestepper_options.use_automatic_timestep = False
    options.timestep = dt
options.output_directory = outputdir
options.horizontal_velocity_scale = Constant(u_scale)
options.vertical_velocity_scale = Constant(w_scale)
options.horizontal_viscosity_scale = Constant(nu_scale)
options.check_salinity_overshoot = True
options.check_temperature_overshoot = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                            'w_3d', 'w_mesh_3d', 'salt_3d', 'temp_3d',
                            'uv_dav_2d', 'uv_dav_3d', 'baroc_head_3d',
                            'density_3d',
                            'smag_visc_3d',
                            'eddy_visc_3d', 'shear_freq_3d',
                            'buoy_freq_3d', 'tke_3d', 'psi_3d',
                            'eps_3d', 'len_3d',
                            'int_pg_3d', 'hcc_metric_3d']
options.fields_to_export_hdf5 = ['uv_2d', 'elev_2d', 'uv_3d',
                                 'salt_3d', 'temp_3d', 'tke_3d', 'psi_3d']
options.equation_of_state_type = 'full'

solver_obj.create_function_spaces()

# atm forcing
wind_stress_3d = Function(solver_obj.function_spaces.P1v, name='wind stress')
wind_stress_2d = Function(solver_obj.function_spaces.P1v_2d, name='wind stress')
atm_pressure_2d = Function(solver_obj.function_spaces.P1_2d, name='atm pressure')
options.wind_stress = wind_stress_3d
options.atmospheric_pressure = atm_pressure_2d
copy_wind_stress_to_3d = ExpandFunctionTo3d(wind_stress_2d, wind_stress_3d)
atm_pattern = 'forcings/atm/nam/nam_air.local.{year}_*_*.nc'.format(year=init_date.year)
atm_interp = ATMInterpolator(
    solver_obj.function_spaces.P1_2d,
    wind_stress_2d, atm_pressure_2d, atm_pattern, init_date)
atm_interp.set_fields(0.0)

# ocean initial conditions
salt_bnd_3d = Function(solver_obj.function_spaces.P1, name='NCOM salinity')
temp_bnd_3d = Function(solver_obj.function_spaces.P1, name='NCOM temperature')
oce_bnd_interp = NCOMInterpolator(
    solver_obj.function_spaces.P1, [salt_bnd_3d, temp_bnd_3d],
    ['Salinity', 'Temperature'], ['s3d', 't3d'],
    'forcings/ncom/{year:04d}/{fieldstr:}/{fieldstr:}.glb8_2f_{year:04d}{month:02d}{day:02d}00.nc',
    init_date
)
oce_bnd_interp.set_fields(0.0)

# tides
elev_tide_2d = Function(bathymetry_2d.function_space(), name='Boundary elevation')
bnd_time = Constant(0)

ramp_t = 12*3600.
elev_ramp = conditional(le(bnd_time, ramp_t), bnd_time/ramp_t, 1.0)
elev_bnd_expr = elev_ramp*elev_tide_2d

bnd_elev_updater = TidalBoundaryForcing(
    elev_tide_2d, init_date,
    boundary_ids=[north_bnd_id, west_bnd_id, south_bnd_id])

# river flux
river_flux_interp = interpolation.NetCDFTimeSeriesInterpolator(
    'forcings/stations/beaverarmy/flux_*.nc',
    ['flux'], init_date, scalars=[-1.0])
river_flux_const = Constant(river_flux_interp(0)[0])

river_swe_funcs = {'flux': river_flux_const}
tide_elev_funcs = {'elev': elev_bnd_expr}
zero_elev_funcs = {'elev': Constant(0)}
open_uv_funcs = {'symm': None}
zero_uv_funcs = {'uv': Constant((0, 0, 0))}
bnd_river_salt = {'value': Constant(salt_river)}
ocean_salt_funcs = {'value': salt_bnd_3d}
bnd_river_temp = {'value': Constant(temp_river)}
ocean_temp_funcs = {'value': temp_bnd_3d}
solver_obj.bnd_functions['shallow_water'] = {
    river_bnd_id: river_swe_funcs,
    south_bnd_id: tide_elev_funcs,
    north_bnd_id: tide_elev_funcs,
    west_bnd_id: tide_elev_funcs,
}
solver_obj.bnd_functions['momentum'] = {
    river_bnd_id: open_uv_funcs,
    south_bnd_id: zero_uv_funcs,
    north_bnd_id: zero_uv_funcs,
    west_bnd_id: zero_uv_funcs,
}
solver_obj.bnd_functions['salt'] = {
    river_bnd_id: bnd_river_salt,
    south_bnd_id: ocean_salt_funcs,
    north_bnd_id: ocean_salt_funcs,
    west_bnd_id: ocean_salt_funcs,
}
solver_obj.bnd_functions['temp'] = {
    river_bnd_id: bnd_river_temp,
    south_bnd_id: ocean_temp_funcs,
    north_bnd_id: ocean_temp_funcs,
    west_bnd_id: ocean_temp_funcs,
}

solver_obj.create_fields()

# add relaxation terms for temperature and salinity
# dT/dt ... - 1/tau*(T_relax - T) = 0
t_tracer_relax = 12.*3600.  # time scale
lx_relax = 10e3  # distance scale from bnd
mask_tracer_relax_2d = solver_obj.function_spaces.P1_2d.get_work_function()
get_boundary_relaxation_field(mask_tracer_relax_2d,
                              [north_bnd_id, west_bnd_id, south_bnd_id],
                              lx_relax, scalar=1.0, cutoff=0.02)
mask_tracer_relax_3d = Function(solver_obj.function_spaces.P1,
                                name='mask_temp_relax_3d')
ExpandFunctionTo3d(mask_tracer_relax_2d, mask_tracer_relax_3d).solve()
solver_obj.function_spaces.P1_2d.restore_work_function(mask_tracer_relax_2d)
# File('mask.pvd').write(mask_tracer_relax_3d)
f_rel = mask_tracer_relax_3d/t_tracer_relax
options.temperature_source_3d = f_rel*(temp_bnd_3d - solver_obj.fields.temp_3d)
options.salinity_source_3d = f_rel*(salt_bnd_3d - solver_obj.fields.salt_3d)

solver_obj.create_equations()

station_list = [
    ('tpoin', ['elev_2d'], 440659., 5117484., None),
    ('dsdma', ['salt_3d', 'temp_3d'], 426349., 5119564., -7.30),
    ('red26', ['salt_3d', 'temp_3d'], 426607., 5117537., -7.50),
    ('sandi', ['salt_3d', 'temp_3d'], 424296., 5122980., -7.90),
    ('tansy', ['salt_3d', 'temp_3d'], 429120., 5115500., -8.40),
    ('rino', ['salt_3d', 'temp_3d'], 400032., 5143472., 'prof'),
    ('rice', ['salt_3d', 'temp_3d'], 407717., 5113304., 'prof'),
    ('riso', ['salt_3d', 'temp_3d'], 414871., 5100523., 'prof'),
    ('ogi01', ['salt_3d', 'temp_3d'], 402180., 5099093., 'prof'),
    ('red26', ['salt_3d', 'temp_3d'], 426607., 5117537., 'prof'),
]

for name, varlist, x, y, z in station_list:

    def _append_callback(cls, *args, **kwargs):
        kwargs.setdefault('append_to_log', False)
        cb = cls(solver_obj, *args, **kwargs)
        solver_obj.add_callback(cb)

    if z is None:
        _append_callback(TimeSeriesCallback2D, varlist, x, y, name)
    elif z == 'prof':
        _append_callback(VerticalProfileCallback, varlist, x, y, name)
    else:
        _append_callback(TimeSeriesCallback3D, varlist, x, y, z, name)

hcc_obj = Mesh3DConsistencyCalculator(solver_obj)
hcc_obj.solve()

print_output('Running CRE plume with options:')
print_output('Resolution: {:}'.format(reso_str))
print_output('Reynolds number: {:}'.format(reynolds_number))
print_output('Horizontal viscosity: {:}'.format(nu_scale))
print_output('Exporting to {:}'.format(outputdir))

# set init salt in estuary to zero
xyz = solver_obj.mesh.coordinates
salt_bnd_3d.interpolate(conditional(ge(xyz[0], 427500.), 0.0, salt_bnd_3d))

solver_obj.assign_initial_conditions(salt=salt_bnd_3d, temp=temp_bnd_3d)


def update_forcings(t):
    bnd_time.assign(t)
    bnd_elev_updater.set_tidal_field(t)
    river_flux_const.assign(river_flux_interp(t)[0])
    oce_bnd_interp.set_fields(t)
    atm_interp.set_fields(t)
    copy_wind_stress_to_3d.solve()


out_atm_pressure = File(options.output_directory + '/AtmPressure2d.pvd')
out_wind_stress = File(options.output_directory + '/WindStress2d.pvd')


def export_atm_fields():
    out_atm_pressure.write(atm_pressure_2d)
    out_wind_stress.write(wind_stress_2d)


solver_obj.iterate(update_forcings=update_forcings, export_func=export_atm_fields)
