"""
Methods for reading ROMS model outputs
"""
from thetis import *
import thetis.coordsys as coordsys
from thetis.timezone import *
from thetis.forcing import *

# define model coordinate system
COORDSYS = coordsys.UTM_ZONE10


def test_time_search():
    """
    Tests time search object.

    Time stamps are deduced from ROMS output files.

    .. note::
        The following ROMS output files must be present:
        ./forcings/liveocean/f2015.05.16/ocean_his_0009.nc
        ./forcings/liveocean/f2015.05.16/ocean_his_0010.nc
        ./forcings/liveocean/f2015.05.16/ocean_his_0011.nc
    """

    # test time parser
    tp = interpolation.NetCDFTimeParser(
        'forcings/liveocean/f2015.05.16/ocean_his_0009.nc',
        time_variable_name='ocean_time')
    nc_start = datetime.datetime(2015, 5, 16, 8, tzinfo=pytz.utc)
    assert tp.start_time == nc_start
    assert tp.end_time == nc_start
    assert numpy.allclose(tp.time_array, numpy.array([datetime_to_epoch(nc_start)]))

    # test time search
    sim_tz = timezone.FixedTimeZone(-8, 'PST')
    init_date = datetime.datetime(2015, 5, 16, tzinfo=sim_tz)
    ncpattern = 'forcings/liveocean/f2015.*/ocean_his_*.nc'
    timesearch_obj = interpolation.NetCDFTimeSearch(
        ncpattern, init_date, interpolation.NetCDFTimeParser,
        time_variable_name='ocean_time', verbose=True)

    sim_time = 100.0
    fn, itime, time = timesearch_obj.find(sim_time, previous=True)
    assert fn == 'forcings/liveocean/f2015.05.16/ocean_his_0009.nc'
    assert itime == 0
    assert time == 0.0
    fn, itime, time = timesearch_obj.find(sim_time)
    assert fn == 'forcings/liveocean/f2015.05.16/ocean_his_0010.nc'
    assert itime == 0
    assert time == 3600.0

    dt = 900
    for i in range(8):
        d = init_date + datetime.timedelta(seconds=i*dt)
        print('Time step {:}, {:}'.format(i, d))
        fn, itime, time = timesearch_obj.find(i*dt, previous=True)
        print('  prev: {:} {:}'.format(fn, itime))
        fn, itime, time = timesearch_obj.find(i*dt, previous=False)
        print('  next: {:} {:}'.format(fn, itime))


def test_interpolator():
    """
    Test ROMS 3d interpolator.

    .. note::
        The following ROMS output files must be present:
        ./forcings/liveocean/f2015.05.16/ocean_his_0009.nc
        ./forcings/liveocean/f2015.05.16/ocean_his_0010.nc
        ./forcings/liveocean/f2015.05.16/ocean_his_0011.nc
    """

    # load and extrude mesh
    from bathymetry import get_bathymetry, smooth_bathymetry, smooth_bathymetry_at_bnd
    nlayers, surf_elem_height, max_z_stretch = (9, 5.0, 4.0)
    mesh2d = Mesh('mesh_cre-plume_03_normal.msh')

    # interpolate bathymetry and smooth it
    bathymetry_2d = get_bathymetry('bathymetry_utm_large.nc', mesh2d, project=False)
    bathymetry_2d = smooth_bathymetry(
        bathymetry_2d, delta_sigma=1.0, bg_diff=0,
        alpha=1e2, exponent=2.5,
        minimum_depth=3.5, niter=30)
    bathymetry_2d = smooth_bathymetry_at_bnd(bathymetry_2d, [2, 7])

    # 3d mesh vertical stretch factor
    z_stretch_fact_2d = Function(bathymetry_2d.function_space(), name='z_stretch')
    # 1.0 (sigma mesh) in shallow areas, 4.0 in deep ocean
    z_stretch_fact_2d.project(-ln(surf_elem_height/bathymetry_2d)/ln(nlayers))
    z_stretch_fact_2d.dat.data[z_stretch_fact_2d.dat.data < 1.0] = 1.0
    z_stretch_fact_2d.dat.data[z_stretch_fact_2d.dat.data > max_z_stretch] = max_z_stretch

    extrude_options = {
        'z_stretch_fact': z_stretch_fact_2d,
    }
    mesh = extrude_mesh_sigma(mesh2d, nlayers, bathymetry_2d,
                              **extrude_options)
    p1 = get_functionspace(mesh, 'CG', 1)

    # make functions
    salt = Function(p1, name='salinity')
    temp = Function(p1, name='temperature')

    sim_tz = timezone.FixedTimeZone(-8, 'PST')
    init_date = datetime.datetime(2015, 5, 16, tzinfo=sim_tz)
    interp = LiveOceanInterpolator(p1,
                                   [salt, temp],
                                   ['salt', 'temp'],
                                   'forcings/liveocean/f2015.*/ocean_his_*.nc',
                                   init_date, COORDSYS)
    interp.set_fields(0.0)
    salt_fn = 'tmp/salt_roms.pvd'
    temp_fn = 'tmp/temp_roms.pvd'
    print('Saving output to {:} {:}'.format(salt_fn, temp_fn))
    out_salt = File(salt_fn)
    out_temp = File(temp_fn)

    out_salt.write(salt)
    out_temp.write(temp)

    dt = 900.
    for i in range(8):
        print('Time step {:}'.format(i))
        interp.set_fields(i*dt)
        out_salt.write(salt)
        out_temp.write(temp)


if __name__ == '__main__':
    test_time_search()
    test_interpolator()
