r"""
Wind stress from WRF/NAM atmospheric model

"""
from firedrake import *
import numpy
import scipy.interpolate
import thetis.timezone as timezone
import thetis.coordsys as coordsys
from thetis.log import *
import datetime
import netCDF4
from thetis.forcing import *
from thetis.utility import get_functionspace

# define model coordinate system
coord_system = coordsys.UTMCoordinateSystem(utm_zone=10)


def test():
    """
    Tests atmospheric model data interpolation.

    .. note::
        The following files must be present
        forcings/atm/wrf/wrf_air.2015_05_16.nc
        forcings/atm/wrf/wrf_air.2015_05_17.nc

        forcings/atm/nam/nam_air.local.2006_05_01.nc
        forcings/atm/nam/nam_air.local.2006_05_02.nc
    """
    mesh2d = Mesh('mesh_cre-plume_03_normal.msh')
    comm = mesh2d.comm
    p1 = get_functionspace(mesh2d, 'CG', 1)
    p1v = get_functionspace(mesh2d, 'CG', 1, vector=True)
    windstress_2d = Function(p1v, name='wind stress')
    atmpressure_2d = Function(p1, name='atm pressure')

    sim_tz = timezone.FixedTimeZone(-8, 'PST')

    # WRF
    # init_date = datetime.datetime(2015, 5, 16, tzinfo=sim_tz)
    # pattern = 'forcings/atm/wrf/wrf_air.2015_*_*.nc'
    # atm_time_step = 3600.  # for verification only
    # test_atm_file = 'forcings/atm/wrf/wrf_air.2015_05_16.nc'

    # NAM
    init_date = datetime.datetime(2006, 5, 1, tzinfo=sim_tz)
    pattern = 'forcings/atm/nam/nam_air.local.2006_*_*.nc'
    atm_time_step = 3*3600.
    test_atm_file = 'forcings/atm/nam/nam_air.local.2006_05_01.nc'

    atm_interp = ATMInterpolator(p1, windstress_2d, atmpressure_2d,
                                 coord_system, pattern, init_date,
                                 verbose=True)

    # create a naive interpolation for first file
    xy = SpatialCoordinate(p1.mesh())
    fsx = Function(p1).interpolate(xy[0]).dat.data_with_halos
    fsy = Function(p1).interpolate(xy[1]).dat.data_with_halos

    mesh_lonlat = []
    for node in range(len(fsx)):
        lon, lat = coord_system.to_lonlat(fsx[node], fsy[node])
        mesh_lonlat.append((lon, lat))
    mesh_lonlat = numpy.array(mesh_lonlat)

    ncfile = netCDF4.Dataset(test_atm_file)
    itime = 6
    grid_lat = ncfile['lat'][:].ravel()
    grid_lon = ncfile['lon'][:].ravel()
    grid_lonlat = numpy.array((grid_lon, grid_lat)).T
    grid_pres = ncfile['prmsl'][itime, :, :].ravel()
    pres = scipy.interpolate.griddata(grid_lonlat, grid_pres, mesh_lonlat, method='linear')
    grid_uwind = ncfile['uwind'][itime, :, :].ravel()
    uwind = scipy.interpolate.griddata(grid_lonlat, grid_uwind, mesh_lonlat, method='linear')
    grid_vwind = ncfile['vwind'][itime, :, :].ravel()
    vwind = scipy.interpolate.griddata(grid_lonlat, grid_vwind, mesh_lonlat, method='linear')
    vrot = coord_system.get_vector_rotator(mesh_lonlat[:, 0], mesh_lonlat[:, 1])
    uwind, vwind = vrot(uwind, vwind)
    u_stress, v_stress = compute_wind_stress(uwind, vwind)

    # compare
    atm_interp.set_fields(itime*atm_time_step - 8*3600.)  # NOTE timezone offset
    assert numpy.allclose(pres, atmpressure_2d.dat.data_with_halos)
    assert numpy.allclose(u_stress, windstress_2d.dat.data_with_halos[:, 0])

    # write fields to disk for visualization
    pres_fn = 'tmp/AtmPressure2d.pvd'
    wind_fn = 'tmp/WindStress2d.pvd'
    print('Saving output to {:} {:}'.format(pres_fn, wind_fn))
    out_pres = File(pres_fn)
    out_wind = File(wind_fn)
    hours = 24*1.5
    granule = 4
    simtime = numpy.arange(granule*hours)*3600./granule
    i = 0
    for t in simtime:
        atm_interp.set_fields(t)
        norm_atm = norm(atmpressure_2d)
        norm_wind = norm(windstress_2d)
        if comm.rank == 0:
            print('{:} {:} {:} {:}'.format(i, t, norm_atm, norm_wind))
        out_pres.write(atmpressure_2d)
        out_wind.write(windstress_2d)
        i += 1


if __name__ == '__main__':
    test()
