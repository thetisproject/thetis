from thetis import *
import numpy
import uptide
import datetime


def latlon_from_xyz(xyz):
    r = numpy.linalg.norm(xyz, axis=1)
    lat = numpy.arcsin(xyz[:, 2]/r)
    lon = numpy.arctan2(xyz[:, 1], xyz[:, 0])
    return lat, lon

def get_bathymetry_from_netcdf(bathymetry_file, mesh2d, minimum_depth=2.0, abscissa_name='lon', ordinate_name='lat', variable_name='z', negative_netcdf_depth=True):
  from netCDF4 import Dataset as NetCDFFile
  import scipy.interpolate
  #Read data from NetCDF file.
  print_output('Reading bathymetry/topography data from '+bathymetry_file)
  nc = NetCDFFile(bathymetry_file)
  lat = nc.variables[ordinate_name][:]
  lon = nc.variables[abscissa_name][:]
  values = nc.variables[variable_name][:,:]
  print_output('Structured NetCDF grid abscissa extents: '+str(min(lon))+' to '+str(max(lon)))
  print_output('Structured NetCDF grid ordinate extents: '+str(min(lat))+' to '+str(max(lat)))
  #Construct a bathymetry function in the appropriate function space.
  P1_2d = FunctionSpace(mesh2d, 'CG', 1)
  bathymetry_2d = Function(P1_2d, name="bathymetry")
  #Extract numpy arrays containign the bathymetry function coordinates and
  # data. Will be used to interpollate bewteen the stuctured data.
  xvector = mesh2d.coordinates.dat.data
  bvector = bathymetry_2d.dat.data
  assert xvector.shape[0]==bvector.shape[0]
  print_output('Interpollating bathymetry/topography data onto simulation mesh')
  #Construct a scipy regular grid interpolator obect, defined on the
  # structured-mesh data from the NetCDF file.
  interpolator = scipy.interpolate.RegularGridInterpolator((lat, lon), values)
  #At each point of the mesh, convert the coordinates to lat-lon and interpolate
  # between the structured bathymetry data.
  lat, lon = numpy.degrees(latlon_from_xyz(xvector))
  print_output('Unstructured simulation grid abscissa extents: '+str(min(lon))+' to '+str(max(lon)))
  print_output('Unstructured simulation grid ordinate extents: '+str(min(lat))+' to '+str(max(lat)))
  #In Thetis depth is a positive number, but frequently in NetCDF files it is given
  # as a negative number. Use the negative_netcdf_depth to get the right sign
  if negative_netcdf_depth:
      netcfd_bathy_sign = -1
  else:
      netcfd_bathy_sign = 1
  for i,(lati,loni) in enumerate(zip(lat, lon)):
      bvector[i] = max(netcfd_bathy_sign*interpolator((lati, loni)), minimum_depth)
  print_output('Bathymetry/topography min:'+str(min(bvector)))
  print_output('Bathymetry/topography max:'+str(max(bvector)))
  return bathymetry_2d

class TidalForcing:
    def __init__(self, dt0, lat, lon):
        constituents = ['Q1', 'O1', 'P1', 'K1', 'N2', 'M2', 'S2', 'K2']
        self.tide = uptide.Tides(constituents)
        self.tide.set_initial_time(dt0)
        self.tnci = uptide.tidal_netcdf.OTPSncTidalInterpolator(self.tide,
          'gridMed.nc', 'hf.Med2011.nc')
        self.lat, self.lon = numpy.rad2deg(lat), numpy.rad2deg(lon)

    def set_tidal_field(self, elev, t):
      self.tnci.set_time(t)
      evector = elev.dat.data
      for i, (lat, lon) in enumerate(zip(self.lat, self.lon)):
        try:
          evector[i] = self.tnci.get_val((lon, lat))
        except uptide.netcdf_reader.CoordinateError:
          evector[i] = 0.

def smoothen_bathymetry(bathymetry_2d):
  v = TestFunction(bathymetry_2d.function_space())
  massb = assemble(v * bathymetry_2d *dx)
  massl = assemble(v*dx)
  with massl.dat.vec as ml, massb.dat.vec as mb, bathymetry_2d.dat.vec as sb:
      ml.reciprocal()
      sb.pointwiseMult(ml, mb)

def test_med_tides(do_export=False):

    dt = 600. #Timestep-size
    day = 86400. #One day in seconds
    Omega = 2*pi/day
    nonlin = True
    coriolis = True
    family = 'rt-dg'
    refinement_level = 3
    degree = 2

    outputdir = 'outputs_{nonlin}_{ref}'.format(**{
        'nonlin': 'nonlin' if nonlin else 'lin',
        'coriolis': 'coriolis' if coriolis else 'nocor',
        'family': family,
        'ref': refinement_level,
        'deg': degree})

    mesh2d = Mesh('mesh/med.node', dim=3)
    x = mesh2d.coordinates
    mesh2d.init_cell_orientations(x)

    f = coriolis*2*Omega

    # bathymetry
    bathymetry_2d = get_bathymetry_from_netcdf('mesh/med_bathymetry_filtered.grd', mesh2d, minimum_depth=10.0, abscissa_name='lon', ordinate_name='lat', variable_name='z')

    # --- create solver ---
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    solver_obj.options.element_family = family
    solver_obj.options.use_nonlinear_equations = nonlin
    solver_obj.options.simulation_export_time = dt
    solver_obj.options.output_directory = outputdir
    solver_obj.options.simulation_end_time = 4*day
    solver_obj.options.no_exports = not do_export
    solver_obj.options.fields_to_export = ['uv_2d', 'elev_2d', 'equilibrium_tide']
    solver_obj.options.horizontal_viscosity = Constant(100.0)
    solver_obj.options.timestepper_type = 'CrankNicolson'
    solver_obj.options.timestepper_options.implicitness_theta = 1.0
    solver_obj.options.timestepper_options.solver_parameters = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_package': 'mumps',
        'snes_monitor': False,
        'snes_type': 'newtonls',
    }
    solver_obj.options.coriolis_frequency = Constant(f)
    solver_obj.options.timestep = dt

    # boundary conditions
    solver_obj.bnd_functions['shallow_water'] = {}
    parameters['quadrature_degree'] = 5
    solver_obj.create_function_spaces()
    H_2d = solver_obj.function_spaces.H_2d

    # by defining this function, the equilibrium tide will be applied
    equilibrium_tide = Function(H_2d)
    solver_obj.options.equilibrium_tide = equilibrium_tide
    # Love numbers:
    solver_obj.options.equilibrium_tide_alpha = Constant(0.693)
    solver_obj.options.equilibrium_tide_beta = Constant(0.953)

    # h2dxyz and lat and lon are coordinate functions on the pressure space
    h2dxyz = Function(H_2d*H_2d*H_2d)
    for i, h2dxyzi in enumerate(h2dxyz.split()):
        h2dxyzi.interpolate(x[i])
    lat, lon = latlon_from_xyz(numpy.vstack(h2dxyz.vector()[:]).T)

    dt0 = datetime.datetime(2013, 1, 1)
    tide = uptide.Tides(uptide.ALL_EQUILIBRIUM_TIDAL_CONSTITUENTS)
    tide.set_initial_time(dt0)

    tidal_elev = Function(H_2d)
    tidal_forcing = TidalForcing(dt0, lat, lon)
    solver_obj.bnd_functions['shallow_water'] = {
          384: {'elev': tidal_elev},  # use TPXO solution at open boundary
          388: {'un': 0.0},  # closed boundary
    }


    # a function called every timestep that updates the equilibrium tide and the boundary forcing
    def update_forcings(t):
        print_output('Updating equilibrium tide and forcing at t={}'.format(t))
        equilibrium_tide.vector()[:] = uptide.equilibrium_tide(tide, lat, lon, t)
        tidal_forcing.set_tidal_field(tidal_elev, t)

    update_forcings(0)

    # we start with the TPXO solution as initial condition for elevation
    solver_obj.assign_initial_conditions(elev=tidal_elev)

    solver_obj.iterate(update_forcings=update_forcings)

    uv, eta = solver_obj.fields.solution_2d.split()

    assert True
    print_output("PASSED")


if __name__ == '__main__':
    test_med_tides(do_export=True)
