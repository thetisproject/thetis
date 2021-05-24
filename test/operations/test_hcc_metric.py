import pytest
from firedrake import *
import thetis.utility as utility
import thetis.utility3d as utility3d
from mpi4py import MPI

comm = MPI.COMM_WORLD


def compute_hcc_metric(nelem, nlayers, slope, deform='uniform'):
    mesh2d = UnitSquareMesh(nelem, nelem)
    mesh = ExtrudedMesh(mesh2d, nlayers, 1.0/nlayers)

    mesh.coordinates.dat.data[:, 2] += -1.0

    xyz = SpatialCoordinate(mesh)
    if deform == 'uniform':
        mesh.coordinates.dat.data[:, 2] = (
            mesh.coordinates.dat.data[:, 2]
            + slope*mesh.coordinates.dat.data[:, 0]
        )
    else:
        mesh.coordinates.dat.data[:, 2] = (
            mesh.coordinates.dat.data[:, 2]
            + slope*mesh.coordinates.dat.data[:, 0]*mesh.coordinates.dat.data[:, 2]
        )

    P1DG = utility.get_functionspace(mesh, 'DG', 1)
    f_hcc = Function(P1DG, name='hcc_metric_3d')

    # emulate solver object
    solver_obj = utility.AttrDict()
    solver_obj.mesh = mesh
    solver_obj.comm = comm
    solver_obj.fields = utility.AttrDict()
    solver_obj.function_spaces = utility.AttrDict()

    solver_obj.fields.hcc_metric_3d = f_hcc
    solver_obj.fields.z_coord_3d = Function(P1DG, name='z_coord_3d')
    solver_obj.fields.z_coord_3d.interpolate(xyz[2])

    solver_obj.function_spaces.P1DG = P1DG

    utility3d.Mesh3DConsistencyCalculator(solver_obj).solve()

    hcc_min = f_hcc.dat.data.min()
    hcc_max = f_hcc.dat.data.max()

    return hcc_min, hcc_max


def max_hcc_value(nelem, nlayers, slope):
    """Computes the correct hcc value"""
    dx = 1.0/nelem
    dz = 1.0/nlayers
    hcc = slope*dx/dz
    return hcc


@pytest.mark.parametrize('nelem', [2, 4])
@pytest.mark.parametrize('nlayers', [2, 4])
@pytest.mark.parametrize('slope', [0.0, 0.5, 1.0])
@pytest.mark.parametrize('deform_type', ['uniform', 'bottom_only'])
def test_hcc(nelem, nlayers, slope, deform_type):
    hcc_max = max_hcc_value(nelem, nlayers, slope)
    hcc_min = 0.0 if deform_type == 'bottom_only' else hcc_max
    target = (hcc_min, hcc_max)
    assert compute_hcc_metric(nelem, nlayers, slope, deform=deform_type) == target
