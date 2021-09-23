"""
Columbia river plume simulation
===============================
"""
from thetis import *
from bathymetry import get_bathymetry, smooth_bathymetry, smooth_bathymetry_at_bnd
comm = COMM_WORLD

# set physical constants
physical_constants['rho0'].assign(1000.0)

nlayers = 15
mesh2d = Mesh('mesh_cre-plume_03_normal.msh')
print_output('Loaded mesh ' + mesh2d.name)

dt = 7.0
t_end = 10*24*3600.
t_export = 900.

# bathymetry
bathymetry_2d = get_bathymetry('bathymetry_utm_large.nc', mesh2d, project=False)
print('bath min: {:} max: {:}'.format(bathymetry_2d.dat.data.min(), bathymetry_2d.dat.data.max()))

new_bathymetry_2d = smooth_bathymetry(
    bathymetry_2d, delta_sigma=1.0, bg_diff=0,
    alpha=1e2, exponent=2.5,
    minimum_depth=3.5, niter=30)

out = File('bath.pvd')
out.write(bathymetry_2d)
out.write(new_bathymetry_2d)

new_bathymetry_2d = smooth_bathymetry_at_bnd(new_bathymetry_2d, [2, 7])
out.write(new_bathymetry_2d)


def compute_hcc(bathymetry_2d, nlayers):
    mesh = extrude_mesh_sigma(mesh2d, nlayers, bathymetry_2d)

    P1DG = get_functionspace(mesh, 'DG', 1)
    f_hcc = Function(P1DG, name='hcc_metric_3d')
    xyz = SpatialCoordinate(mesh)

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

    utility.Mesh3DConsistencyCalculator(solver_obj).solve()
    return f_hcc


out = File('hcc.pvd')
hcc = compute_hcc(bathymetry_2d, nlayers)
out.write(hcc)
hcc = compute_hcc(new_bathymetry_2d, nlayers)
out.write(hcc)


def write_npz(outfile, func):
    fs = bathymetry_2d.function_space()
    mesh = fs.mesh()
    connectivity = fs.cell_node_map().values
    x, y = SpatialCoordinate(mesh)

    f = Function(fs, name='test')

    f.interpolate(x)
    x_arr = f.dat.data.copy()
    f.interpolate(y)
    y_arr = f.dat.data.copy()
    data = bathymetry_2d.dat.data

    numpy.savez('bath.npz', x=x_arr, y=y_arr, data=data, connectivity=connectivity)


# write final bathymetry out as numpy array
write_npz('bath.npz', bathymetry_2d)
