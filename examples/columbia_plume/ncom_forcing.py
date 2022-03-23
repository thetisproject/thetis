"""
Methods for reading NCOM ocean model outputs
"""
from thetis import *
from thetis.timezone import *
from thetis.forcing import *
from thetis.utility import get_functionspace

# define model coordinate system
COORDSYS = coordsys.UTM_ZONE10


def test_time_search():
    """
    Tests time search object.

    Time stamps are deduced from NCOM output files.

    .. note::
        The following NCOM output files must be present:
        ./forcings/ncom/2006/s3d/s3d.glb8_2f_2006050100.nc
        ./forcings/ncom/2006/s3d/s3d.glb8_2f_2006050200.nc
    """
    # test time search
    sim_tz = timezone.FixedTimeZone(-8, 'PST')
    init_date = datetime.datetime(2006, 5, 1, tzinfo=sim_tz)
    pattern = 'forcings/ncom/{year:04d}/s3d/s3d.glb8_2f_{year:04d}{month:02d}{day:02d}00.nc'
    timesearch_obj = interpolation.DailyFileTimeSearch(pattern, init_date, verbose=True)

    sim_time = 100.0
    fn, itime, time = timesearch_obj.find(sim_time, previous=True)
    assert fn == 'forcings/ncom/2006/s3d/s3d.glb8_2f_2006043000.nc'
    assert itime == 0
    assert time == -72000.0
    fn, itime, time = timesearch_obj.find(sim_time)
    assert fn == 'forcings/ncom/2006/s3d/s3d.glb8_2f_2006050100.nc'
    assert itime == 0
    assert time == 14400.0

    dt = 3*3600.
    for i in range(8):
        d = init_date + datetime.timedelta(seconds=i*dt)
        print('Time step {:}, {:}'.format(i, d))
        fn, itime, time = timesearch_obj.find(i*dt, previous=True)
        print('  prev: {:} {:}'.format(fn, itime))
        fn, itime, time = timesearch_obj.find(i*dt, previous=False)
        print('  next: {:} {:}'.format(fn, itime))


def test_interpolator():
    """
    Test NCOM 3d interpolator.

    .. note::
        The following NCOM output files must be present:
        ./forcings/ncom/model_h.nc
        ./forcings/ncom/model_lat.nc
        ./forcings/ncom/model_ang.nc
        ./forcings/ncom/model_lon.nc
        ./forcings/ncom/model_zm.nc
        ./forcings/ncom/2006/s3d/s3d.glb8_2f_2006050100.nc
        ./forcings/ncom/2006/s3d/s3d.glb8_2f_2006050200.nc
        ./forcings/ncom/2006/t3d/t3d.glb8_2f_2006050100.nc
        ./forcings/ncom/2006/t3d/t3d.glb8_2f_2006050200.nc
        ./forcings/ncom/2006/u3d/u3d.glb8_2f_2006050100.nc
        ./forcings/ncom/2006/u3d/u3d.glb8_2f_2006050200.nc
        ./forcings/ncom/2006/v3d/v3d.glb8_2f_2006050100.nc
        ./forcings/ncom/2006/v3d/v3d.glb8_2f_2006050200.nc
        ./forcings/ncom/2006/ssh/ssh.glb8_2f_2006050100.nc
        ./forcings/ncom/2006/ssh/ssh.glb8_2f_2006050200.nc
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
    p1_2d = get_functionspace(mesh2d, 'CG', 1)
    p1 = get_functionspace(mesh, 'CG', 1)

    # make functions
    salt = Function(p1, name='salinity')
    temp = Function(p1, name='temperature')
    uvel = Function(p1, name='u-velocity')
    vvel = Function(p1, name='v-velocity')
    elev = Function(p1_2d, name='elevation')

    sim_tz = timezone.FixedTimeZone(-8, 'PST')
    init_date = datetime.datetime(2006, 5, 1, tzinfo=sim_tz)
    interp = NCOMInterpolator(
        p1_2d, p1, [salt, temp, uvel, vvel, elev],
        ['Salinity', 'Temperature', 'U_Velocity', 'V_Velocity', 'Surface_Elevation'],
        ['s3d', 't3d', 'u3d', 'v3d', 'ssh'],
        COORDSYS, 'forcings/ncom',
        '{year:04d}/{fieldstr:}/{fieldstr:}.glb8_2f_{year:04d}{month:02d}{day:02d}00.nc',
        init_date, verbose=True
    )
    interp.set_fields(0.0)
    salt_fn = 'tmp/salt.pvd'
    temp_fn = 'tmp/temp.pvd'
    uvel_fn = 'tmp/uvel.pvd'
    vvel_fn = 'tmp/vvel.pvd'
    elev_fn = 'tmp/elev.pvd'
    print('Saving output to {:} {:} {:} {:} {:}'.format(salt_fn, temp_fn, uvel_fn, vvel_fn, elev_fn))
    out_salt = File(salt_fn)
    out_temp = File(temp_fn)
    out_uvel = File(uvel_fn)
    out_vvel = File(vvel_fn)
    out_elev = File(elev_fn)

    out_salt.write(salt)
    out_temp.write(temp)
    out_uvel.write(uvel)
    out_vvel.write(vvel)
    out_elev.write(elev)

    dt = 3*3600.
    for i in range(8):
        print('Time step {:}'.format(i))
        interp.set_fields(i*dt)
        out_salt.write(salt)
        out_temp.write(temp)
        out_uvel.write(uvel)
        out_vvel.write(vvel)
        out_elev.write(elev)


if __name__ == '__main__':
    test_time_search()
    test_interpolator()
