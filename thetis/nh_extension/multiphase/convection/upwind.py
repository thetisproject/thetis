# Copyright (C) 2015-2019 Tormod Landet
# SPDX-License-Identifier: Apache-2.0
"""
The HRIC upwind/downwind blending sheme
"""
from . import ConvectionScheme, register_convection_scheme


@register_convection_scheme('Upwind')
class ConvectionSchemeUpwind(ConvectionScheme):
    description = 'First order upwind'
    need_alpha_gradient = False

    def __init__(self, simulation, func_name):
        """
        Implementation of the upwind convection scheme
        """
        super().__init__(simulation, func_name)

        # Set downwind factor to 0.0
        self.blending_function.vector().zero()

    def update(self, dt, velocity):
        """
        Update the values of the blending function beta at the facets
        """
        pass
