import os
import numpy
import scipy.interpolate
from netCDF4 import Dataset
from firedrake import *
from thetis.utility import get_functionspace


def interpolate_onto(interp_func, output_func, coords, min_val):
    bvector = output_func.dat.data
    mesh_xy = coords.dat.data

    assert mesh_xy.shape[0] == bvector.shape[0]
    for i, (node_x, node_y) in enumerate(mesh_xy):
        bvector[i] = interp_func((node_x, node_y))
    shallow_ix = bvector < min_val
    bvector[shallow_ix] = min_val
    bad_ix = ~numpy.isfinite(bvector)
    bvector[bad_ix] = min_val


def retrieve_bath_file(bathfile):
    """Download bathymetry raster if it does not exists."""
    if not os.path.isfile(bathfile):
        import urllib.request
        bath_url = 'http://www.stccmop.org/~karnat/thetis/columbia_plume/'
        print('Downloading bathymetry from {:}'.format(bath_url + bathfile))
        urllib.request.urlretrieve(bath_url + bathfile, bathfile)


def get_bathymetry(bathymetry_file, mesh2d, minimum_depth=5.0, project=False):
    """Interpolates/projects bathymetry from a raster to P1 field."""
    retrieve_bath_file(bathymetry_file)
    d = Dataset(bathymetry_file)
    x = d['x'][:]
    y = d['y'][:]
    bath = -d['bathymetry'][:]
    if numpy.ma.isMaskedArray(bath):
        bath = bath.filled(minimum_depth)
    bath[~numpy.isfinite(bath)] = minimum_depth
    interpolator = scipy.interpolate.RegularGridInterpolator((x, y), bath.T)

    P1_2d = get_functionspace(mesh2d, 'CG', 1)
    bathymetry2d = Function(P1_2d, name='bathymetry')

    if project:
        # interpolate on a high order mesh
        P3_2d = get_functionspace(mesh2d, 'CG', 3)
        P3_2d_v = get_functionspace(mesh2d, 'CG', 3, vector=True)
        bathymetry2d_ho = Function(P3_2d, name='bathymetry')
        coords_ho = Function(P3_2d_v).interpolate(SpatialCoordinate(mesh2d))
        interpolate_onto(interpolator, bathymetry2d_ho, coords_ho, minimum_depth)

        # project on P1
        bathymetry2d.project(bathymetry2d_ho)
        shallow_ix = bathymetry2d.dat.data < minimum_depth
        bathymetry2d.dat.data[shallow_ix] = minimum_depth
    else:
        interpolate_onto(interpolator, bathymetry2d, mesh2d.coordinates, minimum_depth)

    return bathymetry2d


def smooth_bathymetry(bathymetry, delta_sigma=1.0, r_max=0.0, bg_diff=0.0,
                      alpha=1000.0, exponent=1, minimum_depth=None, solution=None,
                      niter=10):
    """
    Smooth bathymetry by minimizing mesh HCC metric r.

    Minimizes HCC metric r while maintaining original bathymetry as much as
    possible.
    """
    fs = bathymetry.function_space()
    mesh = fs.mesh()

    solution = Function(fs, name='bathymetry')
    tmp_bath = Function(fs, name='bathymetry').assign(bathymetry)

    test = TestFunction(fs)

    delta_x = sqrt(CellVolume(mesh))
    bath_grad = grad(tmp_bath)
    grad_h = sqrt(bath_grad[0]**2 + bath_grad[1]**2)
    hcc = (grad_h * delta_x)**exponent / (tmp_bath**0.5 * delta_sigma)

    cost = bg_diff + alpha*hcc
    f = inner(solution - tmp_bath, test)*dx
    f += cost*inner(grad(solution), grad(test))*dx

    prob = NonlinearVariationalProblem(f, solution)
    solver = NonlinearVariationalSolver(prob)

    for i in range(niter):
        # fixed point iteration
        solver.solve()
        if minimum_depth is not None:
            shallow_ix = solution.dat.data < minimum_depth
            solution.dat.data[shallow_ix] = minimum_depth
        tmp_bath.assign(solution)

    return solution


def smooth_bathymetry_at_bnd(bathymetry, bnd_id, strength=8000.):
    """Smooths bathymetry near open boundaries"""
    fs = bathymetry.function_space()
    mesh = fs.mesh()

    solution = Function(fs, name='bathymetry')
    diffusivity = Function(fs, name='diff')

    # step 1: create diffusivity field
    delta_x = sqrt(CellVolume(mesh))
    distance = 2*delta_x
    get_boundary_relaxation_field(diffusivity, bnd_id, distance)

    # step 2: solve diffusion eq
    test = TestFunction(fs)
    f = inner(solution - bathymetry, test)*dx
    f += strength**2*diffusivity*inner(grad(solution), grad(test))*dx

    prob = NonlinearVariationalProblem(f, solution)
    solver = NonlinearVariationalSolver(prob)

    solver.solve()

    return solution


def get_boundary_relaxation_field(mask_func, bnd_id, dist_scale,
                                  scalar=None):
    """
    Generate a smooth relaxation coefficient field near boundaries

    Solution is a linear function ranging from 1.0 at the boundary to 0.0 at
    approximately dist_scale from the boundary. Use `scalar` argument to scale the
    field.
    """
    fs = mask_func.function_space()
    test = TestFunction(fs)
    f = inner(mask_func, test)*dx
    f += (dist_scale**2 * inner(grad(mask_func), grad(test))*dx)
    bc = DirichletBC(fs, 1.0, bnd_id)
    prob = NonlinearVariationalProblem(f, mask_func, bcs=[bc])
    solver = NonlinearVariationalSolver(prob)
    solver.solve()
    # solution is e^(-x), convert to -x
    buff = 1e-7
    mask_func.assign(ln(mask_func + buff) - buff + 1.0)
    # remove negative values
    mask_func.dat.data[mask_func.dat.data < 0.0] = 0.0
    if scalar is not None:
        mask_func.interpolate(mask_func*scalar)

    return mask_func
