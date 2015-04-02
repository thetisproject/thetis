"""
Utility functions and classes for 3D hydrostatic ocean model

Tuomas Karna 2015-02-21
"""
from firedrake import *
import os
import numpy as np
import sys
from cofs.physical_constants import physical_constants
import colorama

comm = op2.MPI.comm
commrank = op2.MPI.comm.rank

colorama.init()

def colorify_text(color):
  def painter(func):
    def func_wrapper(text):
      return color + func(text) + colorama.Style.RESET_ALL
    return func_wrapper
  return painter

@colorify_text(colorama.Back.RED + colorama.Fore.WHITE)
def red(text):
  """Returns given string in uppercase, white on red background."""
  return text.upper()

@colorify_text(colorama.Back.GREEN + colorama.Fore.BLACK)
def green(text):
    """Returns given string in uppercase, black on green background."""
    return text.upper()

@colorify_text(colorama.Style.BRIGHT + colorama.Fore.WHITE)
def bold(text):
  """Returns given string in uppercase, white on green background."""
  return text.upper()

def createDirectory(path):
    if commrank == 0:
        if os.path.exists(path):
            if not os.path.isdir(path):
                raise Exception('file with same name exists', path)
        else:
            os.makedirs(path)
    return path


def extrudeMeshLinear(mesh2d, n_layers, xmin, xmax, dmin, dmax):
    """Extrudes 2d surface mesh with defined by two endpoints in x axis."""
    layer_height = 1.0/n_layers
    base_coords = mesh2d.coordinates
    extrusion_kernel = op2.Kernel("""
            void uniform_extrusion_kernel(double **base_coords,
                        double **ext_coords,
                        int **layer,
                        double *layer_height) {
                for ( int d = 0; d < %(base_map_arity)d; d++ ) {
                    for ( int c = 0; c < %(base_coord_dim)d; c++ ) {
                        ext_coords[2*d][c] = base_coords[d][c];
                        ext_coords[2*d+1][c] = base_coords[d][c];
                    }
                    double s = fmin(fmax((ext_coords[2*d][0]-%(xmin)f)/(%(xmax)f-%(xmin)f), 0.0), 1.0);
                    double depth = s*%(dmax)f + (1.0-s)*%(dmin)f;
                    ext_coords[2*d][%(base_coord_dim)d] = -depth*layer_height[0]*layer[0][0];
                    ext_coords[2*d+1][%(base_coord_dim)d] = -depth*layer_height[0]*(layer[0][0]+1);
                }
            }""" % {'base_map_arity': base_coords.cell_node_map().arity,
                    'base_coord_dim': base_coords.function_space().cdim,
                    'xmin': xmin, 'xmax': xmax, 'dmin': dmin, 'dmax': dmax},
                    'uniform_extrusion_kernel')
    mesh = ExtrudedMesh(mesh2d, layers=n_layers,
                        layer_height=layer_height,
                        extrusion_type='custom',
                        kernel=extrusion_kernel, gdim=3)
    return mesh


def extrudeMeshSigma(mesh2d, n_layers, bathymetry2d):
    """Extrudes 2d surface mesh with bathymetry data defined in a 2d field."""
    mesh = ExtrudedMesh(mesh2d, layers=n_layers, layer_height=1.0/n_layers)

    xyz = mesh.coordinates
    nNodes = xyz.dat.data.shape[0]
    x = xyz.dat.data[:, 0]
    y = xyz.dat.data[:, 0]

    nNodes2d = bathymetry2d.dat.data.shape[0]
    NVert = xyz.dat.data.shape[0]/nNodes2d
    iSource = 0
    # TODO can the loop be circumvented?
    for iNode in range(nNodes2d):
        xyz.dat.data[iNode*NVert:iNode*NVert+NVert, 2] *= -bathymetry2d.dat.data[iNode]
    return mesh


def compVolume2d(eta, dx):
    val = assemble(eta*dx)
    return val


def compVolume3d(dx):
    one = Constant(1.0)
    val = assemble(one*dx)
    return val


def compTracerMass3d(scalarFunc, dx):
    val = assemble(scalarFunc*dx)
    return val


def getZCoordFromMesh(zcoord, solver_parameters={}):
    """Evaluates z coordinates from the 3D mesh"""
    solver_parameters.setdefault('ksp_atol', 1e-12)
    solver_parameters.setdefault('ksp_rtol', 1e-16)
    fs = zcoord.function_space()
    tri = TrialFunction(fs)
    test = TestFunction(fs)
    a = tri*test*dx
    L = fs.mesh().coordinates[2]*test*dx
    solve(a == L, zcoord, solver_parameters=solver_parameters)
    return zcoord


def computeVertVelocity(solution, uv, bathymetry, solver_parameters={}):
    """Computes vertical velocity from 3d continuity equation."""
    # continuity equation must be solved in the space of w (and tracers)
    solver_parameters.setdefault('ksp_atol', 1e-12)
    solver_parameters.setdefault('ksp_rtol', 1e-16)
    H = solution.function_space()
    mesh = H.mesh()
    phi = TestFunction(H)
    tri = TrialFunction(H)
    normal = FacetNormal(mesh)

    w_bottom = -(uv[0]*Dx(bathymetry, 0) + uv[1]*Dx(bathymetry, 1))
    a_w = tri*phi*normal[2]*ds_b - Dx(phi, 2)*tri*dx
    #L_w = (-(Dx(uv[0], 0) + Dx(uv[1], 1))*phi*dx -
           #w_bottom*phi*normal[2]*ds_t)
    # NOTE this is better for DG uv
    L_w = ((uv[0]*Dx(phi, 0) + uv[1]*Dx(phi, 1))*dx -
           (uv[0]*normal[0] + uv[1]*normal[1])*phi*(ds_v + ds_t + ds_b) -
           w_bottom*phi*normal[2]*ds_t)
    solve(a_w == L_w, solution, solver_parameters=solver_parameters)

    return solution


class equation(object):
    """Base class for all equations"""
    # TODO move to equation.py
    def mass_matrix(self, *args, **kwargs):
        """Returns weak form for left hand side."""
        raise NotImplementedError(('This method must be implemented '
                                   'in the derived class'))

    def RHS(self, *args, **kwargs):
        """Returns weak form for the right hand side."""
        raise NotImplementedError(('This method must be implemented '
                                   'in the derived class'))

    def Source(self, *args, **kwargs):
        """Returns weak for for terms that do not depend on the solution."""
        raise NotImplementedError(('This method must be implemented '
                                   'in the derived class'))


def computeVerticalIntegral(input, output, space, bottomToTop=True,
                            bndValue=Constant(0.0), average=False,
                            bathymetry=None,
                            solver_parameters={}):
    """
    Computes vertical integral of the input scalar field in the given
    function space.
    """
    solver_parameters.setdefault('ksp_atol', 1e-12)
    solver_parameters.setdefault('ksp_rtol', 1e-16)
    tri = TrialFunction(space)
    phi = TestFunction(space)

    if bottomToTop:
        bnd_term = inner(bndValue, phi)*ds_t
        mass_bnd_term = inner(tri, phi)*ds_t
    else:
        bnd_term = inner(bndValue, phi)*ds_b
        mass_bnd_term = inner(tri, phi)*ds_b

    a_w = inner(Dx(tri, 2), phi)*dx + mass_bnd_term
    source = input
    if average:
        source = input/bathymetry
    L_w = inner(source, phi)*dx + bnd_term
    solve(a_w == L_w, output, solver_parameters=solver_parameters)

    return output


def copyLayerValueOverVertical(input, output, useBottomValue=True):
    """
    Assings the top/bottom value of the input field to the entire vertical
    dimension of the output field.
    """
    H = input.function_space()
    NVert = H.dofs_per_column[0]
    if NVert == 0:
        raise NotImplementedError('this method doesn\'t support the given function space')
    NNodes = input.dat.data.shape[0]/NVert
    #iNode = 0
    #for i in range(4*NVert):
        #print i, iNode*NVert+i, X.dat.data[iNode*NVert+i], Y.dat.data[iNode*NVert+i], R.dat.data[iNode*NVert+i]
    #iNode = 55
    #for i in range(NVert):
        #print i, iNode*NVert+i, X.dat.data[iNode*NVert+i], Y.dat.data[iNode*NVert+i], R.dat.data[iNode*NVert+i]
    if useBottomValue:
        iSource = NVert-1
    else:
        iSource = 0
    # TODO can the loop be circumvented?
    if len(input.dat.data.shape) > 1:
        for iNode in range(NNodes):
            for p in range(input.dat.data.shape[1]):
                output.dat.data[iNode*NVert:iNode*NVert+NVert, p] = input.dat.data[iNode*NVert+iSource, p]
    else:
        for iNode in range(NNodes):
            output.dat.data[iNode*NVert:iNode*NVert+NVert] = input.dat.data[iNode*NVert+iSource]
    return output


def copy3dFieldTo2d(input3d, output2d, useBottomValue=True, level=None):
    """
    Assings the top/bottom value of the input field to 2d output field.
    """
    H = input3d.function_space()
    parentIsCG = H.dofs_per_column[0] != 0
    if parentIsCG:
        # extruded nodes are laid out for each vertical line
        NVert = H.dofs_per_column[0]
        NNodes = output2d.dat.data.shape[0]
        if useBottomValue:
            iSource = NVert-1
        else:
            iSource = 0
        if level is not None:
            # map positive values to nodes from surface
            # negative values as nodes from bottom
            if level == 0:
                raise Exception('level must be between 1 and NVert')
            if level > 0:
                iSource = level-1
            else:
                iSource = NVert + level
        # TODO can the loop be circumvented?
        if len(input3d.dat.data.shape) > 1:
            for iNode in range(NNodes):
                output2d.dat.data[iNode, 0] = input3d.dat.data[iNode*NVert+iSource, 0]
                output2d.dat.data[iNode, 1] = input3d.dat.data[iNode*NVert+iSource, 1]
        else:
            for iNode in range(NNodes):
                output2d.dat.data[iNode] = input3d.dat.data[iNode*NVert+iSource]
    else:
        # extruded nodes are laid out by elements
        NVert = H.dofs_per_column[2]
        if NVert == 0:
            raise Exception('Unsupported function space, NVert is zero')
        NElem = input3d.dat.data.shape[0]/NVert
        # for P1DGxL1CG
        if useBottomValue:
            iSource = NVert - 3
        else:
            iSource = 0
        if level is not None:
            # map positive values to nodes from surface
            # negative values as nodes from bottom
            if level == 0:
                raise Exception('level must be between 1 and NVert')
            if level > 0:
                iSource = 3*(level-1)
            else:
                iSource = NVert + 3*level
        faceNodes = np.array([0, 1, 2]) + iSource
        ix = (np.tile((NVert*np.arange(NElem)), (3, 1)).T + faceNodes).ravel()
        if len(input3d.dat.data.shape) > 1:
            output2d.dat.data[:, 0] = input3d.dat.data[ix, 0]
            output2d.dat.data[:, 1] = input3d.dat.data[ix, 1]
        else:
            output2d.dat.data[:] = input3d.dat.data[ix]
    return output2d


def copy2dFieldTo3d(input2d, output3d):
    """
    Copies a field from 2d mesh to 3d mesh, assigning the same value over the
    vertical dimension. Horizontal function space must be the same.
    """
    H = output3d.function_space()
    parentIsCG = H.dofs_per_column[0] != 0
    if parentIsCG:
        # extruded nodes are laid out for each vertical line
        NVert = output3d.dat.data.shape[0]/input2d.dat.data.shape[0]
        NNodes = output3d.dat.data.shape[0]/NVert
        # TODO can the loop be circumvented?
        if len(input2d.dat.data.shape) > 1:
            for iNode in range(NNodes):
                output3d.dat.data[iNode*NVert:iNode*NVert+NVert, 0] = input2d.dat.data[iNode, 0]
                output3d.dat.data[iNode*NVert:iNode*NVert+NVert, 1] = input2d.dat.data[iNode, 1]
        else:
            for iNode in range(NNodes):
                output3d.dat.data[iNode*NVert:iNode*NVert+NVert] = input2d.dat.data[iNode]
    else:
        # extruded nodes are laid out by elements
        NVert = H.dofs_per_column[2]
        if NVert == 0:
            raise Exception('Unsupported function space, NVert is zero')
        NElem = output3d.dat.data.shape[0]/NVert
        # for P1DGxL1CG
        faceNodes = np.array([0, 1, 2])
        ix = (np.tile((NVert*np.arange(NElem)), (3, 1)).T + faceNodes).ravel()
        if len(output3d.dat.data.shape) > 1:
            for i in range(NVert-len(faceNodes)+1):
                output3d.dat.data[ix+i, 0] = input2d.dat.data[:, 0]
                output3d.dat.data[ix+i, 1] = input2d.dat.data[:, 1]
        else:
            for i in range(NVert-len(faceNodes)+1):
                output3d.dat.data[ix+i] = input2d.dat.data[:]
    return output3d


def correct3dVelocity(UV2d, uv3d, uv3d_dav, bathymetry):
    """Corrects 3d Horizontal velocity field so that it's depth average
    matches the 2d velocity field."""
    H = uv3d.function_space()
    H2d = UV2d.function_space()
    # compute depth averaged velocity
    bndValue = Constant((0.0, 0.0, 0.0))
    computeVerticalIntegral(uv3d, uv3d_dav, H,
                            bottomToTop=True, bndValue=bndValue,
                            average=True, bathymetry=bathymetry)
    # copy on 2d mesh
    diff = Function(H2d)
    copy3dFieldTo2d(uv3d_dav, diff, useBottomValue=False)
    print 'uv3d_dav', diff.dat.data.min(), diff.dat.data.max()
    # compute difference = UV2d - uv3d_dav
    diff.dat.data[:] *= -1
    diff.dat.data[:] += UV2d.dat.data[:]
    copy2dFieldTo3d(diff, uv3d_dav)
    # correct 3d field
    uv3d.dat.data[:] += uv3d_dav.dat.data


def computeBottomDrag(uv_bottom, z_bottom, bathymetry, drag):
    """Computes bottom drag coefficient (Cd) from boundary log layer."""
    von_karman = physical_constants['von_karman']
    z0_friction = physical_constants['z0_friction']
    drag.assign((von_karman / ln((z_bottom)/z0_friction))**2)
    return drag


def computeBottomFriction(uv3d, uv_bottom2d, uv_bottom3d, z_coord3d,
                          z_bottom2d, z_bottom3d, bathymetry2d,
                          bottom_drag2d, bottom_drag3d):
    copy3dFieldTo2d(uv3d, uv_bottom2d, level=-2)
    copy2dFieldTo3d(uv_bottom2d, uv_bottom3d)
    copy3dFieldTo2d(z_coord3d, z_bottom2d, level=-2)
    copy2dFieldTo3d(z_bottom2d, z_bottom3d)
    z_bottom2d.dat.data[:] += bathymetry2d.dat.data[:]
    computeBottomDrag(uv_bottom2d, z_bottom2d, bathymetry2d, bottom_drag2d)
    copy2dFieldTo3d(bottom_drag2d, bottom_drag3d)


def getHorzontalElemSize(P1_2d, P1_3d=None):
    """
    Computes horizontal element size from the 2D mesh, the copies it over a 3D
    field.
    """
    cellsize = CellSize(P1_2d.mesh())
    test = TestFunction(P1_2d)
    tri = TrialFunction(P1_2d)
    sol2d = Function(P1_2d)
    dx_2d = Measure('dx', domain=P1_2d.mesh(), subdomain_id='everywhere')
    a = test * tri * dx_2d
    L = test * cellsize * dx_2d
    solve(a == L, sol2d)
    if P1_3d is None:
        return sol2d
    sol3d = Function(P1_3d)
    copy2dFieldTo3d(sol2d, sol3d)
    return sol3d


def getVerticalElemSize(P1_2d, P1_3d):
    """
    Computes horizontal element size from the 2D mesh, the copies it over a 3D
    field.
    """
    # compute total depth
    depth2d = Function(P1_2d)
    zbot2d = Function(P1_2d)
    zcoord3d = Function(P1_3d)
    project(Expression('x[2]'), zcoord3d)
    copy3dFieldTo2d(zcoord3d, depth2d, useBottomValue=False)
    copy3dFieldTo2d(zcoord3d, zbot2d, useBottomValue=True)
    depth2d += - zbot2d
    # divide by number of element layers
    n_layers = P1_3d.mesh().layers - 1
    depth2d /= n_layers
    copy2dFieldTo3d(depth2d, zcoord3d)
    return zcoord3d


def updateCoordinates(mesh, eta, bathymetry, z_coord, z_coord_ref,
                      solver_parameters={}):
    """Updates extrusion so that free surface mathces eta3d value"""
    solver_parameters.setdefault('ksp_atol', 1e-12)
    solver_parameters.setdefault('ksp_rtol', 1e-16)
    coords = mesh.coordinates
    fs = z_coord.function_space()
    # sigma stretch function
    new_z = eta*(z_coord_ref + bathymetry)/bathymetry + z_coord_ref
    # update z_coord
    tri = TrialFunction(fs)
    test = TestFunction(fs)
    a = tri*test*dx
    L = new_z*test*dx
    solve(a == L, z_coord, solver_parameters=solver_parameters)
    # assign to mesh
    coords.dat.data[:, 2] = z_coord.dat.data[:]


def computeMeshVelocity(eta, uv, w, w_mesh, w_mesh_surf, dw_mesh_dz_3d,
                        bathymetry, z_coord_ref,
                        solver_parameters={}):
    solver_parameters.setdefault('ksp_atol', 1e-12)
    solver_parameters.setdefault('ksp_rtol', 1e-16)
    fs = w.function_space()
    z = fs.mesh().coordinates[2]
    tri = TrialFunction(fs)
    test = TestFunction(fs)
    # compute w_mesh at the free surface (constant over vertical!)
    # w_mesh_surf = w - eta_grad[0]*uv[0] + eta_grad[1]*uv[1]
    a = tri*test*dx
    eta_grad = nabla_grad(eta)
    L = (w - eta_grad[0]*uv[0] - eta_grad[1]*uv[1])*test*dx
    solve(a == L, w_mesh_surf, solver_parameters=solver_parameters)
    copyLayerValueOverVertical(w_mesh_surf, w_mesh_surf, useBottomValue=False)
    # compute w in the whole water column (0 at bed)
    # w_mesh = w_mesh_surf * (z+h)/(eta+h)
    H = eta + bathymetry
    L = (w_mesh_surf*(z+bathymetry)/H)*test*dx
    solve(a == L, w_mesh, solver_parameters=solver_parameters)
    # compute dw_mesh/dz in the whole water column
    L = (w_mesh_surf/H)*test*dx
    solve(a == L, dw_mesh_dz_3d, solver_parameters=solver_parameters)


def computeParabolicViscosity(uv_bottom, bottom_drag, bathymetry, nu,
                              solver_parameters={}):
    """Computes parabolic eddy viscosity profile assuming log layer flow
    nu = kappa * u_bf * (-z) * (bath + z0 + z) / (bath + z0)
    with
    u_bf = sqrt(Cd)*|uv_bottom|
    """
    solver_parameters.setdefault('ksp_atol', 1e-12)
    solver_parameters.setdefault('ksp_rtol', 1e-16)
    kappa = physical_constants['von_karman']
    z0 = physical_constants['z0_friction']
    H = nu.function_space()
    x = H.mesh().coordinates
    test = TestFunction(H)
    tri = TrialFunction(H)
    a = tri*test*dx
    uv_mag = sqrt(uv_bottom[0]**2 + uv_bottom[1]**2)
    parabola = -x[2]*(bathymetry + z0 + x[2])/(bathymetry + z0)
    L = kappa*sqrt(bottom_drag)*uv_mag*parabola*test*dx
    solve(a == L, nu, solver_parameters=solver_parameters)
    # remove negative values
    neg_ix = nu.dat.data[:] < 1e-10
    nu.dat.data[neg_ix] = 1e-10
    return nu


class exporter(object):
    """Class that handles Paraview file exports."""
    def __init__(self, fs_visu, func_name, outputDir, filename):
        self.fs_visu = fs_visu
        self.outfunc = Function(self.fs_visu, name=func_name)
        self.outfile = File(os.path.join(outputDir, filename))

    def export(self, function):
        self.outfunc.project(function)
        self.outfile << self.outfunc
