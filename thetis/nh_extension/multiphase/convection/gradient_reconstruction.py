# Copyright (C) 2015-2019 Tormod Landet
# SPDX-License-Identifier: Apache-2.0

import dolfin
import numpy
from ocellaris.utils import cell_dofmap, get_local, set_local
from ocellaris.cpp import load_module


class GradientReconstructor(object):
    def __init__(self, simulation, alpha_func, alpha_name, use_vertex_neighbours=True):
        """
        Reconstructor for the gradient in each cell.

        See for example "An Introduction to Computational Fluid Dynamics -
        The Finite Volume Method" by Versteeg & Malalasekera (2007),
        specifically equation 11.36 on page 322 for details on the method
        """
        assert alpha_func.ufl_element().degree() == 0
        self.simulation = simulation
        self.alpha_function = alpha_func
        self.mesh = alpha_func.function_space().mesh()
        self.use_vertex_neighbours = use_vertex_neighbours
        self.reconstruction_initialized = False
        cpp_key = 'convection/%s/use_cpp_gradient' % alpha_name
        self.use_cpp = simulation.input.get_value(cpp_key, True, 'bool')
        self.initialize()

    def initialize(self):
        """
        Initialize the gradient function and lstsq matrices
        """
        V = self.alpha_function.function_space()
        ndim = V.ufl_cell().topological_dimension()
        mesh = V.mesh()
        tdim = mesh.topology().dim()
        ncells = mesh.topology().ghost_offset(tdim)  # number of owned cells

        # To be used by others accessing this class
        self.gradient = [dolfin.Function(V) for _ in range(ndim)]

        # Connectivity info needed in calculations
        cell_info = self.simulation.data['cell_info']
        conFC = self.simulation.data['connectivity_FC']
        conCF = self.simulation.data['connectivity_CF']
        conVC = self.simulation.data['connectivity_VC']
        conCV = self.simulation.data['connectivity_CV']

        if self.use_vertex_neighbours:
            # Find cells sharing one or more vertices
            con1 = conCV
            con2 = conVC
        else:
            # Find cells sharing one or more facets
            con1 = conCF
            con2 = conFC

        # Precompute connectivity and geometry matrices
        everyones_neighbours = [None] * ncells
        lstsq_matrices = [None] * ncells
        self.lstsq_inv_matrices = numpy.zeros((ncells, ndim, ndim), float, order='C')

        for idx in range(ncells):
            # Find neighbours
            neighbours = []
            facets_or_vertices = con1(idx)
            for ifv in facets_or_vertices:
                cell_neighbours = con2(ifv)
                new_nbs = [
                    ci for ci in cell_neighbours if ci != idx and ci not in neighbours
                ]
                neighbours.extend(new_nbs)

            # Get the centroid of the cell neighbours
            nneigh = len(neighbours)
            A = numpy.zeros((nneigh, ndim), float)
            mp0 = cell_info[idx].midpoint
            for j, ni in enumerate(neighbours):
                mpJ = cell_info[ni].midpoint
                A[j] = mpJ - mp0

            # Calculate the matrices needed for least squares gradient reconstruction
            AT = A.T
            ATA = numpy.dot(AT, A)
            everyones_neighbours[idx] = neighbours
            lstsq_matrices[idx] = AT
            self.lstsq_inv_matrices[idx] = numpy.linalg.inv(ATA)

        # Turn the lists into numpy arrays for ease of communication with C++
        self.num_neighbours = numpy.array(
            [len(nbs) for nbs in everyones_neighbours], dtype='i', order='C'
        )
        NBmax = self.num_neighbours.max()
        self.neighbours = numpy.zeros((ncells, NBmax), dtype='i', order='C')
        self.lstsq_matrices = numpy.zeros((ncells, ndim, NBmax), float, order='C')
        for i in range(ncells):
            Nnb = self.num_neighbours[i]
            self.neighbours[i, :Nnb] = everyones_neighbours[i]
            self.lstsq_matrices[i, :, :Nnb] = lstsq_matrices[i]

        # Eigen does not support 3D arrays
        self.lstsq_matrices = self.lstsq_matrices.reshape(-1, order='C')
        self.lstsq_inv_matrices = self.lstsq_inv_matrices.reshape(-1, order='C')

        self.reconstruction_initialized = True

    def reconstruct(self):
        """
        Reconstruct the gradient in each cell center

        TODO: handle boundary conditions for boundary cells,
              right now the boundary cell gradients are only
              influenced by the cell neighbours
        """
        # Initialize the least squares gradient reconstruction matrices
        # needed to calculate the gradient of a DG0 field
        if not self.reconstruction_initialized:
            self.initialize()

        assert self.alpha_function.ufl_element().degree() == 0

        if not self.use_cpp:
            # Pure Python version
            reconstructor = _reconstruct_gradient
        else:
            # Faster C++ version
            cpp_mod = load_module('linear_convection')
            reconstructor = cpp_mod.reconstruct_gradient

        # Run the gradient reconstruction
        reconstructor(
            self.alpha_function._cpp_object,
            self.num_neighbours,
            self.neighbours,
            self.lstsq_matrices,
            self.lstsq_inv_matrices,
            [gi._cpp_object for gi in self.gradient],
        )


def _reconstruct_gradient(
    alpha_function,
    num_neighbours,
    neighbours,
    lstsq_matrices,
    lstsq_inv_matrices,
    gradient,
):
    """
    Reconstruct the gradient, Python version of the code

    This function used to have a more Pythonic implementation
    that was most likely also faster. See old commits for that
    code. This code is here to verify the C++ version that is
    much faster than this (and the old Pythonic version)
    """
    a_cell_vec = get_local(alpha_function)
    mesh = alpha_function.function_space().mesh()

    V = alpha_function.function_space()
    assert V == gradient[0].function_space()

    cell_dofs = cell_dofmap(V)
    np_gradient = [gi.vector().get_local() for gi in gradient]

    # Reshape arrays. The C++ version needs flatt arrays
    # (limitation in Instant/Dolfin) and we have the same
    # interface for both versions of the code
    ncells = len(num_neighbours)
    ndim = mesh.topology().dim()
    num_cells_owned, num_neighbours_max = neighbours.shape
    assert ncells == num_cells_owned
    lstsq_matrices = lstsq_matrices.reshape((ncells, ndim, num_neighbours_max))
    lstsq_inv_matrices = lstsq_inv_matrices.reshape((ncells, ndim, ndim))

    for icell in range(num_cells_owned):
        cdof = cell_dofs[icell]
        Nnbs = num_neighbours[icell]
        nbs = neighbours[icell, :Nnbs]

        # Get the matrices
        AT = lstsq_matrices[icell, :, :Nnbs]
        ATAI = lstsq_inv_matrices[icell]
        a0 = a_cell_vec[cdof]
        b = [(a_cell_vec[cell_dofs[ni]] - a0) for ni in nbs]
        b = numpy.array(b, float)

        # Calculate the and store the gradient
        g = numpy.dot(ATAI, numpy.dot(AT, b))
        for d in range(ndim):
            np_gradient[d][cdof] = g[d]

    for i, np_grad in enumerate(np_gradient):
        set_local(gradient[i], np_grad, apply='insert')
