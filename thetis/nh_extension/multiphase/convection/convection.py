# Copyright (C) 2014-2019 Tormod Landet
# SPDX-License-Identifier: Apache-2.0

import numpy
import dolfin
from ocellaris.utils import ocellaris_error, cell_dofmap, facet_dofmap
from ocellaris.cpp import load_module
from .gradient_reconstruction import GradientReconstructor


class ConvectionScheme(object):
    """
    A generic convection scheme, to be subclassed
    for actual usage
    """

    description = 'No description available'

    def __init__(self, simulation, func_name):
        """
        The given function space is for the function you
        will be convected. The convection scheme itself
        uses a discontinuous Trace zeroth order element
        to represent the blending function

        The blending function is a downstream blending
        factor (0=upstream, 1=downstream)

        The alpha function is the scalar function to be
        advected
        """
        self.simulation = simulation
        self.func_name = func_name
        self.alpha_function = simulation.data[func_name]
        Va = self.alpha_function.function_space()
        self.alpha_dofmap = Va.dofmap().dofs()

        # Blending function
        self.mesh = Va.mesh()
        Vb = dolfin.FunctionSpace(self.mesh, 'DGT', 0)
        self.blending_function = dolfin.Function(Vb)

        # Mesh size
        self.ncells = self.mesh.num_cells()
        self.nfacets = self.mesh.num_facets()

        # Input for C++ code
        self.cpp_mod = load_module('linear_convection')
        self.cpp_inp = initialize_cpp_input(simulation, self.cpp_mod, Va, Vb)

    def initialize_gradient(self):
        """
        Setup gradient reconstruction code
        """
        self.gradient_reconstructor = GradientReconstructor(
            self.simulation, self.alpha_function, self.func_name
        )

    def update(self, t, dt, velocity):
        raise NotImplementedError()


# Static scheme for testing
class StaticScheme(ConvectionScheme):
    description = 'A scheme that does not move the initial colour function in time'
    need_alpha_gradient = False


def initialize_cpp_input(simulation, cpp_mod, Vcell, Vfacet):
    degree_c = Vcell.ufl_element().degree()
    degree_f = Vfacet.ufl_element().degree()

    cdofs = fdofs = []
    if degree_c == 0:
        cdofs = cell_dofmap(Vcell)
    if degree_f == 0:
        fdofs = facet_dofmap(Vfacet)

    # Precompute dofmaps on first run
    cdofs = numpy.array(cdofs, numpy.intc)
    fdofs = numpy.array(fdofs, numpy.intc)
    cpp_inp = cpp_mod.ConvectionBlendingInput()
    cpp_inp.set_dofmap(cdofs, fdofs)

    # Pass facet info to C++
    fi = simulation.data['facet_info']
    area = numpy.zeros(len(fi), float)
    normals = numpy.zeros((len(fi), simulation.ndim), float)
    midpnts = numpy.zeros((len(fi), simulation.ndim), float)
    for i, finfo in enumerate(fi):
        area[i] = finfo.area
        normals[i] = finfo.normal
        midpnts[i] = finfo.midpoint
    cpp_inp.set_facet_info(area, normals, midpnts)

    # Pass cell info to C++
    ci = simulation.data['cell_info']
    volume = numpy.zeros(len(ci), float)
    midpnts = numpy.zeros((len(ci), simulation.ndim), float)
    for i, cinfo in enumerate(ci):
        volume[i] = cinfo.volume
        midpnts[i] = cinfo.midpoint
    cpp_inp.set_cell_info(volume, midpnts)

    return cpp_inp


class VelocityDGT0Projector(object):
    def __init__(self, simulation, u_conv):
        """
        Given a velocity in DG, e.g DG2, produce a velocity in DGT0,
        i.e. a constant on each facet
        """
        V = u_conv[0].function_space()
        V_dgt0 = dolfin.FunctionSpace(V.mesh(), 'DGT', 0)
        u = dolfin.TrialFunction(V_dgt0)
        v = dolfin.TestFunction(V_dgt0)

        ndim = simulation.ndim
        w = u_conv
        w_new = dolfin.as_vector([dolfin.Function(V_dgt0) for _ in range(ndim)])

        dot, avg, dS, ds = dolfin.dot, dolfin.avg, dolfin.dS, dolfin.ds
        a = dot(avg(u), avg(v)) * dS + dot(u, v) * ds

        L = []
        for d in range(ndim):
            L.append(avg(w[d]) * avg(v) * dS + w[d] * v * ds)

        self.lhs = [dolfin.Form(Li) for Li in L]
        self.A = dolfin.assemble(a)
        self.solver = dolfin.PETScKrylovSolver('cg')
        self.velocity = simulation.data['u_conv_dgt0'] = w_new

    def update(self):
        with dolfin.Timer('Ocellaris produce u_conv_dgt0'):
            for d, L in enumerate(self.lhs):
                b = dolfin.assemble(L)
                self.solver.solve(self.A, self.velocity[d].vector(), b)


from . import upwind
from . import cicsam
from . import hric
