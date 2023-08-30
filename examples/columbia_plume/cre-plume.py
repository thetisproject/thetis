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
import thetis.coordsys as coordsys
from bathymetry import *
from tidal_forcing import TPXOTidalBoundaryForcing
from ncom_forcing import NCOMInterpolator
from thetis.forcing import *
comm = COMM_WORLD

# define model coordinate system
COORDSYS = coordsys.UTM_ZONE10


rho0 = 1000.0
# set physical constants
physical_constants['rho0'].assign(rho0)

reso_str = 'coarse'
meshfile = {
    'coarse': 'mesh_cre-plume_03_coarse.msh',
    'normal': 'mesh_cre-plume_03_normal.msh',
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

sim_tz = timezone.FixedTimeZone(-8, 'PST')
init_date = datetime.datetime(2006, 5, 1, tzinfo=sim_tz)
end_date = datetime.datetime(2006, 7, 2, tzinfo=sim_tz)

t_end = (end_date - init_date).total_seconds()
t_export = 900.

# interpolate bathymetry and smooth it
bathymetry_2d = get_bathymetry('bathymetry_utm_large.nc', mesh2d, project=False)
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
salt_river = 0.0
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
options.bottom_roughness = Constant(0.005)
options.use_turbulence = True  # not simple_barotropic
options.use_turbulence_advection = False  # not simple_barotropic
options.turbulence_model_type = 'gls'
options.use_baroclinic_formulation = not simple_barotropic
options.use_lax_friedrichs_velocity = True
options.use_lax_friedrichs_tracer = False
options.lax_friedrichs_velocity_scaling_factor = Constant(1.0)
options.vertical_viscosity = Constant(2e-5)
options.vertical_diffusivity = Constant(2e-5)
options.horizontal_viscosity = Constant(1.0)
options.horizontal_diffusivity = Constant(1.0)
options.use_quadratic_pressure = True
options.use_limiter_for_tracers = True
options.use_limiter_for_velocity = True
options.use_smagorinsky_viscosity = True
options.smagorinsky_coefficient = Constant(1.0/numpy.sqrt(reynolds_number))
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
                            'w_3d', 'salt_3d', 'temp_3d',
                            'density_3d',
                            'smag_visc_3d',
                            'eddy_visc_3d', 'shear_freq_3d',
                            'buoy_freq_3d', 'tke_3d', 'psi_3d',
                            'eps_3d', 'len_3d',
                            'int_pg_3d', 'baroc_head_3d']
options.fields_to_export_hdf5 = []
options.equation_of_state_type = 'full'

solver_obj.create_function_spaces()

# additional diffusion at ocean boundary
viscosity_bnd_2d = solver_obj.function_spaces.P1_2d.get_work_function()
viscosity_bnd_3d = Function(solver_obj.function_spaces.P1, name='visc_bnd_3d')
visc_bnd_dist = 60e3
visc_bnd_value = 80.0
get_boundary_relaxation_field(viscosity_bnd_2d,
                              [north_bnd_id, west_bnd_id, south_bnd_id],
                              visc_bnd_dist, scalar=visc_bnd_value)
viscosity_bnd_2d.assign(viscosity_bnd_2d + options.horizontal_viscosity)
ExpandFunctionTo3d(viscosity_bnd_2d, viscosity_bnd_3d).solve()
# File('bnd_visc.pvd').write(viscosity_bnd_2d)
solver_obj.function_spaces.P1_2d.restore_work_function(viscosity_bnd_2d)
options.horizontal_viscosity = viscosity_bnd_3d

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
    wind_stress_2d, atm_pressure_2d, COORDSYS, atm_pattern, init_date)
atm_interp.set_fields(0.0)

solver_obj.create_fields()

# ocean initial conditions
salt_bnd_3d = Function(solver_obj.function_spaces.P1DG, name='NCOM salinity')
temp_bnd_3d = Function(solver_obj.function_spaces.P1DG, name='NCOM temperature')
uvel_bnd_3d = Function(solver_obj.function_spaces.P1DG, name='NCOM u velocity')
vvel_bnd_3d = Function(solver_obj.function_spaces.P1DG, name='NCOM v velocity')
elev_bnd_2d = Function(solver_obj.function_spaces.P1DG_2d, name='NCOM water elevation')
density_bnd_3d = Function(solver_obj.function_spaces.P1DG, name='NCOM density')
baroc_head_bnd_3d = Function(solver_obj.function_spaces.P1DG, name='NCOM baroclinic head')
ncom_vel_mask_3d = Function(solver_obj.function_spaces.P1DG, name='NCOM velocity mask')

uv_bnd_3d = Function(solver_obj.function_spaces.P1DGv, name='NCOM velocity')
uv_bnd_2d = Function(solver_obj.function_spaces.P1DGv_2d, name='NCOM velocity')
uv_bnd_dav_3d = Function(solver_obj.function_spaces.P1DGv, name='NCOM depth averaged velocity')

oce_bnd_interp = NCOMInterpolator(
    solver_obj.function_spaces.P1DG_2d, solver_obj.function_spaces.P1DG,
    [salt_bnd_3d, temp_bnd_3d, uvel_bnd_3d, vvel_bnd_3d, elev_bnd_2d],
    ['Salinity', 'Temperature', 'U_Velocity', 'V_Velocity', 'Surface_Elevation'],
    ['s3d', 't3d', 'u3d', 'v3d', 'ssh'],
    COORDSYS, 'forcings/ncom',
    '{year:04d}/{fieldstr:}/{fieldstr:}.glb8_2f_{year:04d}{month:02d}{day:02d}00.nc',
    init_date,
)


def interp_ocean_bnd(time):
    oce_bnd_interp.set_fields(time)
    uvel_bnd_3d.interpolate(uvel_bnd_3d*ncom_vel_mask_3d)
    vvel_bnd_3d.interpolate(vvel_bnd_3d*ncom_vel_mask_3d)


# tides
elev_tide_2d = Function(solver_obj.function_spaces.P1_2d, name='Tidal elevation')
UV_tide_2d = Function(solver_obj.function_spaces.P1v_2d, name='Tidal transport')
UV_tide_3d = Function(solver_obj.function_spaces.P1v, name='Tidal transport')
bnd_time = Constant(0)

ramp_t = 12*3600.
elev_ramp = conditional(le(bnd_time, ramp_t), bnd_time/ramp_t, 1.0)
bnd_elev_expr_2d = elev_ramp*(elev_tide_2d + elev_bnd_2d)
depth_2d = solver_obj.fields.bathymetry_2d + solver_obj.fields.elev_cg_2d
depth_3d = solver_obj.fields.bathymetry_3d + solver_obj.fields.elev_cg_2d.view_3d
tide_uv_expr_2d = elev_ramp*UV_tide_2d/depth_2d
tide_uv_expr_3d = elev_ramp*UV_tide_3d/depth_3d

tide_bnd_interp = TPXOTidalBoundaryForcing(
    elev_tide_2d, init_date, COORDSYS,
    uv_field=UV_tide_2d, data_dir='forcings',
    boundary_ids=[north_bnd_id, west_bnd_id, south_bnd_id])

# ramp up bnd baroclinicity
bnd_baroc_head_expr = elev_ramp*baroc_head_bnd_3d + (1-elev_ramp)*solver_obj.fields.baroc_head_3d

# river temperature and volume flux
river_flux_interp = interpolation.NetCDFTimeSeriesInterpolator(
    'forcings/stations/beaverarmy/flux_*.nc',
    ['flux'], init_date, scalars=[-1.0])
river_flux_const = Constant(river_flux_interp(0)[0])
river_temp_interp = interpolation.NetCDFTimeSeriesInterpolator(
    'forcings/stations/beaverarmy/temp_*.nc',
    ['temp'], init_date)
river_temp_const = Constant(river_temp_interp(0)[0])

river_swe_funcs = {'flux': river_flux_const}
ocean_tide_funcs = {'elev': bnd_elev_expr_2d, 'uv': uv_bnd_2d + tide_uv_expr_2d}
west_tide_funcs = {'elev': bnd_elev_expr_2d, 'uv': tide_uv_expr_2d}
open_uv_funcs = {'symm': None}
bnd_river_salt = {'value': Constant(salt_river)}
uv_bnd_sum_3d = uv_bnd_3d + uv_bnd_dav_3d + tide_uv_expr_3d
ocean_salt_funcs = {'value': salt_bnd_3d, 'uv': uv_bnd_sum_3d}
west_salt_funcs = {'value': salt_bnd_3d, 'uv': tide_uv_expr_3d}
bnd_river_temp = {'value': river_temp_const}
ocean_temp_funcs = {'value': temp_bnd_3d, 'uv': uv_bnd_sum_3d}
west_temp_funcs = {'value': temp_bnd_3d, 'uv': tide_uv_expr_3d}
ocean_uv_funcs = {'uv': uv_bnd_sum_3d, 'baroc_head': bnd_baroc_head_expr}
west_uv_funcs = {'uv': tide_uv_expr_3d, 'baroc_head': bnd_baroc_head_expr}
solver_obj.bnd_functions['shallow_water'] = {
    river_bnd_id: river_swe_funcs,
    south_bnd_id: ocean_tide_funcs,
    north_bnd_id: ocean_tide_funcs,
    west_bnd_id: west_tide_funcs,
}
solver_obj.bnd_functions['momentum'] = {
    river_bnd_id: open_uv_funcs,
    south_bnd_id: ocean_uv_funcs,
    north_bnd_id: ocean_uv_funcs,
    west_bnd_id: west_uv_funcs,
}
solver_obj.bnd_functions['salt'] = {
    river_bnd_id: bnd_river_salt,
    south_bnd_id: ocean_salt_funcs,
    north_bnd_id: ocean_salt_funcs,
    west_bnd_id: west_salt_funcs,
}
solver_obj.bnd_functions['temp'] = {
    river_bnd_id: bnd_river_temp,
    south_bnd_id: ocean_temp_funcs,
    north_bnd_id: ocean_temp_funcs,
    west_bnd_id: west_temp_funcs,
}

# add relaxation terms for T, S, uv
# dT/dt ... - 1/tau*(T_relax - T) = 0
t_bnd_relax = 12.*3600.  # time scale
lx_relax = 30e3  # distance scale from bnd
mask_tmp_2d = solver_obj.function_spaces.P1_2d.get_work_function()
mask_tracer_relax_3d = Function(solver_obj.function_spaces.P1,
                                name='mask_temp_relax_3d')
mask_uv_relax_3d = Function(solver_obj.function_spaces.P1,
                            name='mask_uv_relax_3d')
get_boundary_relaxation_field(mask_tmp_2d,
                              [north_bnd_id, west_bnd_id, south_bnd_id],
                              lx_relax, scalar=1.0/t_bnd_relax)
ExpandFunctionTo3d(mask_tmp_2d, mask_tracer_relax_3d).solve()
get_boundary_relaxation_field(mask_tmp_2d,
                              [north_bnd_id, west_bnd_id],
                              lx_relax, scalar=1.0/t_bnd_relax)
ExpandFunctionTo3d(mask_tmp_2d, mask_uv_relax_3d).solve()
solver_obj.function_spaces.P1_2d.restore_work_function(mask_tmp_2d)
# File('mask.pvd').write(mask_tracer_relax_3d)
# options.temperature_source_3d = mask_tracer_relax_3d*(temp_bnd_3d - solver_obj.fields.temp_3d)
# options.salinity_source_3d = mask_tracer_relax_3d*(salt_bnd_3d - solver_obj.fields.salt_3d)
# options.momentum_source_3d = mask_uv_relax_3d*(uv_bnd_3d - solver_obj.fields.uv_3d)

solver_obj.create_equations()

vel_mask_bath_min = 20.0
vel_mask_bath_max = 500.0
ncom_vel_mask_3d.interpolate(0.5*tanh(3*(2*(solver_obj.fields.bathymetry_3d-vel_mask_bath_min)/(vel_mask_bath_max-vel_mask_bath_min)-1)) + 0.5)
interp_ocean_bnd(0.0)

station_list = [
    ('tpoin', ['elev_2d'], 440659., 5117484., None),
    ('dsdma', ['salt_3d', 'temp_3d'], 426349., 5119564., -7.30),
    ('red26', ['salt_3d', 'temp_3d'], 426607., 5117537., -7.50),
    ('sandi', ['salt_3d', 'temp_3d'], 424296., 5122980., -7.90),
    ('tansy', ['salt_3d', 'temp_3d'], 429120., 5115500., -8.40),
    ('rino', ['salt_3d', 'temp_3d'], 386598., 5208089., 'prof'),
    ('rice', ['salt_3d', 'temp_3d'], 407708., 5113274., 'prof'),
    ('riso', ['salt_3d', 'temp_3d'], 413834., 5039088., 'prof'),
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


# set initial conditions in the estuary
xyz = solver_obj.mesh.coordinates
salt_bnd_3d.interpolate(conditional(ge(xyz[0], 427500.), salt_river, salt_bnd_3d))
temp_bnd_3d.interpolate(conditional(ge(xyz[0], 427500.), river_temp_const, temp_bnd_3d))
uvel_bnd_3d.interpolate(conditional(ge(xyz[0], 427500.), 0.0, uvel_bnd_3d))
vvel_bnd_3d.interpolate(conditional(ge(xyz[0], 427500.), 0.0, vvel_bnd_3d))

# construct bnd velocity splitter
uv_bnd_averager = VerticalIntegrator(uv_bnd_3d,
                                     uv_bnd_dav_3d,
                                     bottom_to_top=True,
                                     bnd_value=Constant((0.0, 0.0, 0.0)),
                                     average=True,
                                     bathymetry=solver_obj.fields.bathymetry_3d,
                                     elevation=solver_obj.fields.elev_cg_2d.view_3d)
extract_uv_bnd = SubFunctionExtractor(uv_bnd_dav_3d, uv_bnd_2d)
copy_uv_bnd_dav_to_3d = ExpandFunctionTo3d(uv_bnd_2d, uv_bnd_dav_3d)
copy_uv_tide_to_3d = ExpandFunctionTo3d(UV_tide_2d, UV_tide_3d)


def split_3d_bnd_velocity():
    uv_bnd_3d.dat.data_with_halos[:, 0] = uvel_bnd_3d.dat.data_with_halos[:]
    uv_bnd_3d.dat.data_with_halos[:, 1] = vvel_bnd_3d.dat.data_with_halos[:]
    uv_bnd_averager.solve()  # uv_bnd_3d -> uv_bnd_dav_3d
    extract_uv_bnd.solve()  # uv_bnd_dav_3d -> uv_bnd_2d
    copy_uv_bnd_dav_to_3d.solve()  # uv_bnd_2d -> uv_bnd_dav_3d
    uv_bnd_3d.assign(uv_bnd_3d - uv_bnd_dav_3d)  # rm depth av


# compute density and baroclinic head at boundary
bnd_density_solver = DensitySolver(salt_bnd_3d, temp_bnd_3d, density_bnd_3d,
                                   solver_obj.equation_of_state)
bnd_rho_integrator = VerticalIntegrator(density_bnd_3d,
                                        baroc_head_bnd_3d,
                                        bottom_to_top=False,
                                        average=False,
                                        bathymetry=solver_obj.fields.bathymetry_3d,
                                        elevation=solver_obj.fields.elev_cg_2d.view_3d)


def compute_bnd_baroclinicity():
    bnd_density_solver.solve()
    bnd_rho_integrator.solve()
    baroc_head_bnd_3d.assign(-physical_constants['rho0_inv']*baroc_head_bnd_3d)


# add custom exporters
# extract and export surface salinity
surf_salt_2d = Function(solver_obj.function_spaces.H_2d, name='surf salinity')
extract_surf_salt = SubFunctionExtractor(solver_obj.fields.salt_3d, surf_salt_2d)
surf_temp_2d = Function(solver_obj.function_spaces.H_2d, name='surf temperature')
extract_surf_temp = SubFunctionExtractor(solver_obj.fields.temp_3d, surf_temp_2d)
surf_uv_2d = Function(solver_obj.function_spaces.U_2d, name='surf velocity')
extract_surf_uv = SubFunctionExtractor(solver_obj.fields.uv_3d, surf_uv_2d)
surf_w_2d = Function(solver_obj.function_spaces.P1DG_2d, name='surf vertical velocity')
extract_surf_w = SubFunctionExtractor(solver_obj.fields.w_3d, surf_w_2d)


def prepare_surf_salt():
    extract_surf_salt.solve()


def prepare_surf_temp():
    extract_surf_temp.solve()


def prepare_surf_uv():
    extract_surf_uv.solve()


def prepare_surf_w():
    extract_surf_w.solve()


solver_obj.exporters['vtk'].add_export(
    'surf_salt_2d', surf_salt_2d, export_type='vtk',
    shortname='Salinity', filename='SurfSalinity2d',
    preproc_func=prepare_surf_salt)
solver_obj.exporters['vtk'].add_export(
    'surf_temp_2d', surf_temp_2d, export_type='vtk',
    shortname='Temperature', filename='SurfTemperature2d',
    preproc_func=prepare_surf_temp)
solver_obj.exporters['vtk'].add_export(
    'surf_uv_2d', surf_uv_2d, export_type='vtk',
    shortname='Velocity', filename='SurfVelocity2d',
    preproc_func=prepare_surf_uv)
solver_obj.exporters['vtk'].add_export(
    'surf_w_2d', surf_w_2d, export_type='vtk',
    shortname='Vertical velocity', filename='SurfVertVelo2d',
    preproc_func=prepare_surf_w)
solver_obj.exporters['vtk'].add_export(
    'atm_pressure_2d', atm_pressure_2d, export_type='vtk',
    shortname='Atm pressure', filename='AtmPressure2d')
solver_obj.exporters['vtk'].add_export(
    'wind_stress_2d', wind_stress_2d, export_type='vtk',
    shortname='Wind stress', filename='WindStress2d')

split_3d_bnd_velocity()
solver_obj.assign_initial_conditions(salt=salt_bnd_3d, temp=temp_bnd_3d,
                                     uv_2d=uv_bnd_2d, uv_3d=uv_bnd_3d)


def update_forcings(t):
    bnd_time.assign(t)
    tide_bnd_interp.set_tidal_field(t)
    copy_uv_tide_to_3d.solve()
    river_flux_const.assign(river_flux_interp(t)[0])
    river_temp_const.assign(river_temp_interp(t)[0])
    oce_bnd_interp.set_fields(t)
    split_3d_bnd_velocity()
    compute_bnd_baroclinicity()
    atm_interp.set_fields(t)
    copy_wind_stress_to_3d.solve()


solver_obj.iterate(update_forcings=update_forcings)
