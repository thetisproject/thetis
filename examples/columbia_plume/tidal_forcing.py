from thetis import *
import thetis.coordsys as coordsys
from thetis.timezone import *
from thetis.forcing import *
from thetis.utility import get_functionspace

# define model coordinate system
COORDSYS = coordsys.UTM_ZONE10


def test():
    mesh2d = Mesh('mesh_cre-plume_03_normal.msh')
    p1 = get_functionspace(mesh2d, 'CG', 1)
    p1v = get_functionspace(mesh2d, 'CG', 1, vector=True)
    elev_field = Function(p1, name='elevation')
    uv_field = Function(p1v, name='transport')

    sim_tz = timezone.FixedTimeZone(-8, 'PST')
    init_date = datetime.datetime(2006, 5, 15, tzinfo=sim_tz)

    # tide_cls = FES2004TidalBoundaryForcing
    tide_cls = TPXOTidalBoundaryForcing
    tbnd = tide_cls(
        elev_field, init_date, COORDSYS,
        uv_field=uv_field,
        data_dir='forcings',
        constituents=['M2', 'K1'],
        boundary_ids=[2, 5, 7])
    tbnd.set_tidal_field(0.0)

    elev_outfn = 'tmp/tidal_elev.pvd'
    uv_outfn = 'tmp/tidal_uv.pvd'
    print('Saving to {:} {:}'.format(elev_outfn, uv_outfn))
    elev_out = File(elev_outfn)
    uv_out = File(uv_outfn)
    for t in numpy.linspace(0, 12*3600., 49):
        tbnd.set_tidal_field(t)
        if elev_field.function_space().mesh().comm.rank == 0:
            print('t={:7.1f} elev: {:7.1f} uv: {:7.1f}'.format(t, norm(elev_field), norm(uv_field)))
        elev_out.write(elev_field)
        uv_out.write(uv_field)


if __name__ == '__main__':
    test()
