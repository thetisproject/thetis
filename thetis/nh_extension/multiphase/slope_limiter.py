# Copyright (C) 2016-2019 Tormod Landet
# SPDX-License-Identifier: Apache-2.0

import numpy
import dolfin
from ocellaris.utils import ocellaris_error
from ocellaris.solver_parts import get_dof_region_marks, mark_cell_layers
from .limiter_bcs import SlopeLimiterBoundaryConditions


DEFAULT_LIMITER = 'None'
DEFAULT_FILTER = 'nofilter'
_SLOPE_LIMITERS = {}


class SlopeLimiterBase(object):
    description = 'No description available'

    active = True
    phi_old = None
    global_bounds = -1e100, 1e100
    has_global_bounds = False
    enforce_global_bounds = False

    def set_phi_old(self, phi_old):
        """
        Some limiters use an old (allready limited) field as an extra
        information about in the limiting process
        """
        self.phi_old = phi_old

    def set_global_bounds(self, phi=None, lo=None, hi=None):
        """
        Set the initial field which we use this to compute the global
        bounds which must be observed everywhere for all time
        """
        if self.enforce_global_bounds:
            if phi is None:
                assert isinstance(lo, (int, float))
                assert isinstance(hi, (int, float))
            else:
                lo = phi.vector().min()
                hi = phi.vector().max()
            self.global_bounds = (lo, hi)
            self.has_global_bounds = True
        else:
            self.global_bounds = SlopeLimiterBase.global_bounds
            self.has_global_bounds = False
        return self.global_bounds


class OnlyBoundSlopeLimiter(SlopeLimiterBase):
    description = 'Bounding limiter'
    active = False

    def __init__(self, simulation, phi_name, phi, skip_cells, boundary_conditions, limiter_input):
        self.phi_name = phi_name
        self.phi = phi
        self.enforce_global_bounds = limiter_input.get_value('enforce_bounds', False, 'bool')
        self.additional_plot_funcs = []
        simulation.log.info('        Enforce global bounds: %r' % self.enforce_global_bounds)

    def run(self):
        if self.has_global_bounds:
            lo, hi = self.global_bounds
            vals = self.phi.vector().get_local()
            numpy.clip(vals, lo, hi, vals)
            self.phi.vector().set_local(vals)
            self.phi.vector().apply('insert')


def SlopeLimiter(simulation, phi_name, phi):
    """
    Return a slope limiter based on the user provided input or the default
    values if no input is provided by the user
    """
    # Get user provided input (or default values)
    inp = simulation.input.get_value('slope_limiter/%s' % phi_name, {}, 'Input')
    method = inp.get_value('method', DEFAULT_LIMITER, 'string')
    skip_boundaries = inp.get_value('skip_boundaries', [], 'list(string)')
    plot_exceedance = inp.get_value('plot', False, 'bool')
    simulation.log.info('    Using slope limiter %s for field %s' % (method, phi_name))
    simulation.log.info('        Skip boundaries: %r' % skip_boundaries)

    # Find degree
    V = phi.function_space()
    degree = V.ufl_element().degree()

    # Skip limiting if degree is 0
    if degree == 0:
        if 'method' in inp and method != 'OnlyBound':
            simulation.log.info(
                '    Switching to slope limiter OnlyBound for field '
                '%s (due to degree == 0)' % phi_name
            )
        return OnlyBoundSlopeLimiter(
            simulation,
            phi_name=phi_name,
            phi=phi,
            skip_cells=None,
            boundary_conditions=None,
            limiter_input=inp,
        )

    # Get boundary region marks and get the helper class used to limit along the boundaries
    drm = get_dof_region_marks(simulation, V)
    bcs = SlopeLimiterBoundaryConditions(simulation, phi_name, drm, V)

    # Mark boundary cells for skipping (cells containing dofs with region marks)
    skip_cells = mark_cell_layers(
        simulation, V, layers=0, dof_region_marks=drm, named_boundaries=skip_boundaries
    )
    if 'all' not in skip_boundaries:
        bcs.activate()

    # Construct the limiter
    limiter_class = get_slope_limiter(method)
    limiter = limiter_class(
        simulation,
        phi_name=phi_name,
        phi=phi,
        skip_cells=skip_cells,
        boundary_conditions=bcs,
        limiter_input=inp,
    )

    if plot_exceedance:
        for func in limiter.additional_plot_funcs:
            simulation.io.add_extra_output_function(func)

    return limiter


