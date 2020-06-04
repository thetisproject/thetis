r"""
2D conservation of mass equation describing bed evolution due to sediment transport

The equation reads

.. math::
    \frac{\partial z_b}{\partial t} + (morfac/(1-p)) \nabla_h \cdot (Q_b)
    = (morfac/(1-p)) H ((Sink S) - Source)
    :label: exner_eq

where :math:'z_b' is the bedlevel, :math:'S' is :math:'q=HT' for conservative
and :math:'T' for non-conservative, :math:`\nabla_h` denotes horizontal gradient,
:math:'morfac' is the morphological scale factor, :math:'p' is the porosity and
:math:'Q_b' is the bedload transport vector

"""

from __future__ import absolute_import
from .equation import Term, Equation
from .utility import *

__all__ = [
    'ExnerEquation',
    'ExnerTerm',
    'ExnerSourceTerm',
    'ExnerBedlevelTerm'
]


class ExnerTerm(Term):
    """
    Generic term that provides commonly used members and mapping for
    boundary functions.
    """
    def __init__(self, function_space, depth, conservative = False):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        """
        super(ExnerTerm, self).__init__(function_space)
        self.n = FacetNormal(self.mesh)
        self.depth = depth

        # define measures with a reasonable quadrature degree
        p = self.function_space.ufl_element().degree()
        self.quad_degree = 2*p + 1
        self.dx = dx() #degree=self.quad_degree)
        self.dS = dS() #degree=self.quad_degree)
        self.ds = ds() #degree=self.quad_degree)
        self.conservative = conservative


class ExnerSourceTerm(ExnerTerm):
    r"""
    Source term accounting for suspended sediment transport

    The weak form reads

    .. math::
        F_s = \int_\Omega \sigma * depth \phi dx

    where :math:`\sigma` is a user defined source scalar field :class:`Function`.

    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):

        f = 0
        tracer = fields.get('tracer')
        source = fields.get('source')
        depth_int_source = fields.get('depth_integrated_source')
        sink = fields.get('sink')
        depth_int_sink = fields.get('depth_integrated_sink')
        morfac = fields.get('morfac')
        porosity = fields.get('porosity')

        if morfac.dat.data[:] <= 0:
            raise ValueError("Morphological acceleration factor must be strictly positive")
        fac = Constant(morfac/(1.0-porosity))
        H = self.depth.get_total_depth(fields_old['elev_2d'])

        if depth_int_source is not None:
            if not self.conservative:
                raise NotImplementedError("Depth-integrated source term not implemented for non-conservative case")
            else:
                if source is not None:
                    raise AttributeError("Assigned both a source term and a depth-integrated source term\
                                 but only one can be implemented. Choose the most appropriate for your case")
                else:
                    source_dep = depth_int_source
        elif source is not None:
            source_dep = source*H

        if depth_int_sink is not None:
            if not self.conservative:
                raise NotImplementedError("Depth-integrated sink term not implemented for non-conservative case")
            else:
                if sink is not None:
                    raise AttributeError("Assigned both a sink term and a depth-integrated sink term\
                                 but only one can be implemented. Choose the most appropriate for your case")
                else:
                    sink_dep = depth_int_sink
        elif sink is not None:
            if self.conservative:
                sink_dep = sink
            else:
                sink_dep = sink*H

        if source_dep is not None and sink_dep is not None:
            f += -inner(fac*(source_dep-tracer*sink_dep), self.test)*self.dx
        elif source_dep is not None and sink_dep is None:
            f += -inner((fac*source_dep), self.test)*self.dx
        elif source_dep is None and sink_dep is not None:
            f += -inner(-fac*tracer*sink_dep, self.test)*self.dx

        return -f

class ExnerBedlevelTerm(ExnerTerm):
    r"""
    some maths
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0
        qbx = fields.get('bedload_x')
        qby = fields.get('bedload_y')

        morfac = fields.get('morfac')
        porosity = fields.get('porosity')

        fac = Constant(morfac/(1.0-porosity))

        f += -(self.test*((qbx*self.n[0]) + (qby*self.n[1])))*ds(1) - (self.test*((qbx*self.n[0]) + (qby*self.n[1])))*ds(2) + (qbx*(self.test.dx(0)) + self.qby*(self.test.dx(1)))*dx

        return 0#-f

class ExnerEquation(Equation):
    """
    """
    def __init__(self, function_space, depth, conservative):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        """
        super().__init__(function_space)

        args = (function_space, depth, conservative)
        self.add_term(ExnerSourceTerm(*args), 'source')
        #self.add_term(ExnerBedlevelTerm(*args), 'implicit')
