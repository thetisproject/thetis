r"""
Exner equation

2D conservation of mass equation describing bed evolution due to sediment transport

The equation reads

.. math::
    \frac{\partial z_b}{\partial t} + (morfac/(1-p)) \nabla_h \cdot (Q_b)
    = (morfac/(1-p)) H ((Sink S) - Source)
    :label: exner_eq

where :math:'z_b' is the bedlevel, :math:'S' is :math:'q=HT' for conservative (where H is depth
and T is the sediment field) and :math:'T' for non-conservative (where T is the sediment field),
:math:`\nabla_h` denotes horizontal gradient, :math:'morfac' is the morphological scale factor,
:math:'p' is the porosity and :math:'Q_b' is the bedload transport vector

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
    def __init__(self, function_space, depth, sediment_model, depth_integrated_sediment=False):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        :arg sediment_model: :class: `SedimentModel` containing sediment info
        :kwarg bool depth_integrated_sediment: whether the sediment field is depth-integrated
        """
        super(ExnerTerm, self).__init__(function_space)
        self.n = FacetNormal(self.mesh)
        self.depth = depth

        self.sediment_model = sediment_model

        # define measures with a reasonable quadrature degree
        p = self.function_space.ufl_element().degree()
        self.quad_degree = 2*p + 1
        self.dx = dx(degree=self.quad_degree)
        self.dS = dS(degree=self.quad_degree)
        self.ds = ds(degree=self.quad_degree)
        self.depth_integrated_sediment = depth_integrated_sediment


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
        morfac = fields.get('morfac')
        porosity = fields.get('porosity')

        if morfac.dat.data[:] <= 0:
            raise ValueError("Morphological acceleration factor must be strictly positive")
        fac = Constant(morfac/(1.0-porosity))
        H = self.depth.get_total_depth(fields_old['elev_2d'])

        erosion = self.sediment_model.get_erosion_term()
        deposition = self.sediment_model.get_deposition_coefficient() * sediment
        if self.depth_integrated_sediment:
            deposition = deposition/H
        f = self.test*fac*(erosion - deposition)*self.dx

        return f


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

        qbx, qby = self.sediment_model.get_bedload_term(solution)

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
    def __init__(self, function_space, depth, sediment_model, depth_integrated_sediment):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        :arg sediment_model: :class: `SedimentModel` containing sediment info
        :kwarg bool depth_integrated_sediment: whether to use conservative tracer
        """
        super().__init__(function_space)

        if sediment_model is None:
            raise ValueError('To use the exner equation must define a sediment model')

        args = (function_space, depth, sediment_model, depth_integrated_sediment)
        if sediment_model.solve_suspended_sediment:
            self.add_term(ExnerSourceTerm(*args), 'source')
        if sediment_model.use_bedload:
            self.add_term(ExnerBedloadTerm(*args), 'implicit')
