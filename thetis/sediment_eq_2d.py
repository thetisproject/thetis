r"""
2D advection diffusion equation for sediment transport.

This can be either conservative :math:`q=HT` or non-conservative :math:`T` and allows
for a separate source and sink term. The equation reads

.. math::
    \frac{\partial S}{\partial t}
    + \nabla_h \cdot (\textbf{u} S)
    = \nabla_h \cdot (\mu_h \nabla_h S) + Source - (Sink S)
    :label: sediment_eq_2d

where :math:'S' is :math:'q' for conservative and :math:'T' for non-conservative,
:math:`\nabla_h` denotes horizontal gradient, :math:`\textbf{u}` are the horizontal
velocities, and :math:`\mu_h` denotes horizontal diffusivity.
"""
from __future__ import absolute_import
from .utility import *
from .equation import Equation
from .tracer_eq_2d import HorizontalDiffusionTerm, HorizontalAdvectionTerm, TracerTerm
from .conservative_tracer_eq_2d import ConservativeHorizontalAdvectionTerm, ConservativeHorizontalDiffusionTerm

__all__ = [
    'SedimentEquation2D',
    'SedimentTerm',
    'SedimentSourceTerm',
    'SedimentSinkTerm',
]


class SedimentTerm(TracerTerm):
    """
    Generic sediment term that provides commonly used members.
    """
    def __init__(self, function_space, depth,
                 use_lax_friedrichs=True, sipg_parameter=Constant(10.0), conservative = False):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        :kwarg bool use_lax_friedrichs: whether to use Lax Friedrichs stabilisation
        :kwarg sipg_parameter: :class: `Constant` or :class: `Function` penalty parameter for SIPG
        :kwarg bool conservative: whether to use conservative tracer
        """
        super(SedimentTerm, self).__init__(function_space, depth)
        self.conservative = conservative

class SedimentSourceTerm(SedimentTerm):
    r"""
    Generic source term

    The weak form reads

    .. math::
        F_s = \int_\Omega \sigma \phi dx

    where :math:`\sigma` is a user defined scalar :class:`Function`.

    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0
        source = fields.get('source')
        depth_int_source = fields.get('depth_integrated_source')
        if depth_int_source is not None:
            if self.conservative:
                f += -inner(depth_int_source, self.test) * self.dx
            else:
                raise NotImplementedError("Depth-integrated source term not implemented for non-conservative case")
        elif source is not None:
            if self.conservative:
                H = self.depth.get_total_depth(fields['elev_2d'])
                f += -inner(H*source, self.test)*self.dx
            else:
                f += -inner(source, self.test)*self.dx
        else:
            warning("no source term implemented")

        if source is not None and depth_int_source is not None:
            raise AttributeError("Assigned both a source term and a depth-integrated source term\
                                 but only one can be implemented. Choose the most appropriate for your case")

        return -f


class SedimentSinkTerm(SedimentTerm):
    r"""
    Liner sink term

    The weak form reads

    .. math::
        F_s = - \int_\Omega \sigma solution \phi dx

    where :math:`\sigma` is a user defined scalar :class:`Function`.

    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0
        sink = fields.get('sink')
        depth_int_sink = fields.get('depth_integrated_sink')
        if depth_int_sink is not None:
            if self.conservative:
                f += -inner(-depth_int_sink*solution, self.test) * self.dx
            else:
                raise NotImplementedError("Depth-integrated sink term not implemented for non-conservative case")
        elif sink is not None:
            f += -inner(-sink*solution, self.test)*self.dx
        else:
            warning("no sink term implemented")
        if sink is not None and depth_int_sink is not None:
            raise AttributeError("Assigned both a sink term and a depth-integrated sink term\
                                 but only one can be implemented. Choose the most appropriate for your case")
        return -f


class SedimentEquation2D(Equation):
    """
    2D sediment advection-diffusion equation: eq:`tracer_eq` or `conservative_tracer_eq`
    with sediment source and sink term
    """
    def __init__(self, function_space, depth,
                 use_lax_friedrichs=False,
                 sipg_parameter=Constant(10.0),
                 conservative = False):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        :kwarg bool use_lax_friedrichs: whether to use Lax Friedrichs stabilisation
        :kwarg sipg_parameter: :class: `Constant` or :class: `Function` penalty parameter for SIPG
        :kwarg bool conservative: whether to use conservative tracer
        """
        super(SedimentEquation2D, self).__init__(function_space)
        args = (function_space, depth, use_lax_friedrichs, sipg_parameter)
        args_sediment = (function_space, depth, use_lax_friedrichs, sipg_parameter, conservative)
        if conservative:
            self.add_term(ConservativeHorizontalAdvectionTerm(*args), 'explicit')
            self.add_term(ConservativeHorizontalDiffusionTerm(*args), 'explicit')
        else:
            self.add_term(HorizontalAdvectionTerm(*args), 'explicit')
            self.add_term(HorizontalDiffusionTerm(*args), 'explicit')
        self.add_term(SedimentSourceTerm(*args_sediment), 'source')
        self.add_term(SedimentSinkTerm(*args_sediment), 'source')
