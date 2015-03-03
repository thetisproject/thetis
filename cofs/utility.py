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


def extrudeMeshSigmaOld(mesh2d, n_layers, bathymetry):
    """Extrudes 2d surface mesh with bathymetry data defined in a field."""
    layer_height = 1.0/n_layers
    base_coords = mesh2d.coordinates
    extrusion_kernel = op2.Kernel("""
            void uniform_extrusion_kernel(double **base_coords,
                        double **ext_coords,
                        double **aux_data,
                        int **layer,
                        double *layer_height) {
                for ( int d = 0; d < %(base_map_arity)d; d++ ) {
                    for ( int c = 0; c < %(base_coord_dim)d; c++ ) {
                        ext_coords[2*d][c] = base_coords[d][c];
                        ext_coords[2*d+1][c] = base_coords[d][c];
                    }
                    double depth = aux_data[0][0];
                    ext_coords[2*d][%(base_coord_dim)d] = -depth*layer_height[0]*layer[0][0];
                    ext_coords[2*d+1][%(base_coord_dim)d] = -depth*layer_height[0]*(layer[0][0]+1);
                }
            }""" % {'base_map_arity': base_coords.cell_node_map().arity,
                    'base_coord_dim': base_coords.function_space().cdim},
                                'uniform_extrusion_kernel')
    mesh = ExtrudedMesh(mesh2d, layers=n_layers,
                        layer_height=layer_height,
                        extrusion_type='custom',
                        kernel=extrusion_kernel, gdim=3, aux_data=bathymetry)
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
                            bndValue=Constant(0.0)):
    """
    Computes vertical integral of the input scalar field in the given
    function space.
    """
    tri = TrialFunction(space)
    phi = TestFunction(space)

    if bottomToTop:
        bnd_term = bndValue*phi*ds_b
        mass_bnd_term = tri*phi*ds_b
    else:
        bnd_term = bndValue*phi*ds_t
        mass_bnd_term = tri*phi*ds_t

    a_w = Dx(tri, 2)*phi*dx + mass_bnd_term
    L_w = input*phi*dx + bnd_term
    solve(a_w == L_w, output)

    return output


def copyLayerValueOverVertical(input, output, useBottomValue=True):
    """
    Assings the top/bottom value of the input field to the entire vertical
    dimension of the output field.
    """
    H = input.function_space()
    NVert = H.dofs_per_column[0]
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
    for iNode in range(NNodes):
        output.dat.data[iNode*NVert:iNode*NVert+NVert] = input.dat.data[iNode*NVert+iSource]
    return output


def copy2dFieldTo3d(input2d, output3d):
    """
    Copies a field from 2d mesh to 3d mesh, assigning the same value over the
    vertical dimension. Horizontal function space must be the same.
    """
    H = output3d.function_space()
    NVert = H.dofs_per_column[0]
    NNodes = output3d.dat.data.shape[0]/NVert
    iSource = 0
    # TODO can the loop be circumvented?
    for iNode in range(NNodes):
        output3d.dat.data[iNode*NVert:iNode*NVert+NVert] = input2d.dat.data[iNode]
    return output3d


class exporter(object):
    """Class that handles Paraview file exports."""
    def __init__(self, fs_visu, func_name, outputDir, filename):
        self.fs_visu = fs_visu
        self.outfunc = Function(self.fs_visu, name=func_name)
        self.outfile = File(os.path.join(outputDir, filename))

    def export(self, function):
        self.outfunc.assign(project(function, self.fs_visu))
        self.outfile << self.outfunc
