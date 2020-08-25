r"""
Exner equation

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
    'ExnerBedloadTerm'
]


class ExnerTerm(Term):
    """
    Generic term that provides commonly used members and mapping for
    boundary functions.
    """
    def __init__(self, function_space, depth, sed_model, conservative=False):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        :arg sed_model: :class: `SedimentModel` containing sediment info
        :kwarg bool conservative: whether to use conservative tracer
        """
        super(ExnerTerm, self).__init__(function_space)
        self.n = FacetNormal(self.mesh)
        self.depth = depth
        self.sed_model = sed_model

        # define measures with a reasonable quadrature degree
        p = self.function_space.ufl_element().degree()
        self.quad_degree = 2*p + 1
        self.dx = dx(degree=self.quad_degree)
        self.dS = dS(degree=self.quad_degree)
        self.ds = ds(degree=self.quad_degree)
        self.conservative = conservative


class ExnerSourceTerm(ExnerTerm):
    r"""
    Source term accounting for suspended sediment transport

    The weak form reads

    .. math::
        F_s = \int_\Omega (\sigma - sediment * \phi) * depth \psi dx

    where :math:`\sigma` is a user defined source scalar field :class:`Function`
    and :math:`\phi` is a user defined source scalar field :class:`Function`.

    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0
        sediment = fields.get('sediment')
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
        else:
            source_dep = None

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
        else:
            sink_dep = None

        if source_dep is not None and sink_dep is not None:
            f += -inner(fac*(source_dep-sediment*sink_dep), self.test)*self.dx
        elif source_dep is not None and sink_dep is None:
            f += -inner((fac*source_dep), self.test)*self.dx
        elif source_dep is None and sink_dep is not None:
            f += -inner(-fac*sediment*sink_dep, self.test)*self.dx

        return -f


class ExnerBedloadTerm(ExnerTerm):
    r"""
    Bedload transport term, \nabla_h \cdot \textbf{qb}

    The weak form is

    .. math::
        \int_\Omega  \nabla_h \cdot \textbf{qb} \psi  dx
        = - \int_\Omega (\textbf{qb} \cdot \nabla) \psi dx
        + \int_\Gamma \psi \textbf{qb} \cdot \textbf{n} dS

    where :math:`\textbf{n}` is the unit normal of the element interfaces.

    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0

        qbx, qby = self.sed_model.get_bedload_term(solution)

        morfac = fields.get('morfac')
        porosity = fields.get('porosity')

        fac = Constant(morfac/(1.0-porosity))

        f += -(self.test*((fac*qbx*self.n[0]) + (fac*qby*self.n[1])))*self.ds(1) - (self.test*((fac*qbx*self.n[0]) + (fac*qby*self.n[1])))*self.ds(2) + (fac*qbx*(self.test.dx(0)) + fac*qby*(self.test.dx(1)))*self.dx

        return -f


class ExnerEquation(Equation):
    """
    Exner equation

    2D conservation of mass equation describing bed evolution due to sediment transport
    """
    def __init__(self, function_space, depth, sed_model, conservative):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        :arg sed_model: :class: `SedimentModel` containing sediment info
        :kwarg bool conservative: whether to use conservative tracer
        """
        super().__init__(function_space)

        if sed_model is None:
            raise ValueError('To use the exner equation must define a sediment model')
        self.depth = depth
        args = (function_space, depth, sed_model, conservative)
        if sed_model.suspendedload:
            self.add_term(ExnerSourceTerm(*args), 'source')
        if sed_model.bedload:
            self.add_term(ExnerBedloadTerm(*args), 'implicit')
