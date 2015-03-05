"""
Utility functions and classes for 3D hydrostatic ocean model

Tuomas Karna 2015-02-21
"""
from firedrake import *
import os
import numpy as np
import sys


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


def computeVertVelocity(solution, uv3d, bathymetry):
    """Computes vertical velocity from 3d continuity equation."""
    # continuity equation must be solved in the space of w (and tracers)
    H = solution.function_space()
    mesh = H.mesh()
    phi = TestFunction(H)
    tri = TrialFunction(H)
    normal = FacetNormal(mesh)

    w_bottom = -(uv3d[0]*Dx(bathymetry, 0) + uv3d[1]*Dx(bathymetry, 1))
    a_w = tri*phi*normal[2]*ds_b - Dx(phi, 2)*tri*dx
    #L_w = (-(Dx(uv3d[0], 0) + Dx(uv3d[1], 1))*phi*dx -
           #w_bottom*phi*normal[2]*ds_t)
    # NOTE this is better for DG uv3d
    L_w = ((uv3d[0]*Dx(phi, 0) + uv3d[1]*Dx(phi, 1))*dx -
           (uv3d[0]*normal[0] + uv3d[1]*normal[1])*phi*(ds_v + ds_t + ds_b) -
           w_bottom*phi*normal[2]*ds_t)
    solve(a_w == L_w, solution)

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


class timeIntegrator(object):
    """Base class for all time integrator objects."""
    # TODO move to timeIntegrator.py
    def __init__(self, equation):
        self.equation = equation

    def advance(self):
        raise NotImplementedError(('This method must be implemented '
                                   'in the derived class'))


def computeVerticalIntegral(input, output, space, bottomToTop=True,
                            bndValue=Constant(0.0), average=False,
                            bathymetry=None):
    """
    Computes vertical integral of the input scalar field in the given
    function space.
    """
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
    solve(a_w == L_w, output)

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


def copy3dFieldTo2d(input3d, output2d, useBottomValue=True):
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
            faceNodes = np.array([2, 1, 0]) + NVert - 3
        else:
            faceNodes = np.array([0, 1, 2])
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


class exporter(object):
    """Class that handles Paraview file exports."""
    def __init__(self, fs_visu, func_name, outputDir, filename):
        self.fs_visu = fs_visu
        self.outfunc = Function(self.fs_visu, name=func_name)
        self.outfile = File(os.path.join(outputDir, filename))

    def export(self, function):
        self.outfunc.assign(project(function, self.fs_visu))
        self.outfile << self.outfunc
