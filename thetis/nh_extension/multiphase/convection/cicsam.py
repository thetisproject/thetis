# Copyright (C) 2014-2019 Tormod Landet
# SPDX-License-Identifier: Apache-2.0
"""
The HRIC upwind/downwind blending sheme
"""
import numpy
import dolfin
import math
from ocellaris.utils import ocellaris_error
from . import ConvectionScheme, register_convection_scheme


@register_convection_scheme('CICSAM')
class ConvectionSchemeHric2D(ConvectionScheme): # is this the right name, or a copy/paste error?
    description = 'Compressive Interface Capturing Scheme for Arbitrary Meshes'
    need_alpha_gradient = True

    def __init__(self, simulation, func_name):
        """
        Implementation of the CICSAM VOF convection scheme

        From:
          "Numerical prediction of two fluid systems with sharp interfaces"
          Imperial College, London, 1997
          Onno Ubbink
        """
        super().__init__(simulation, func_name)

    def update(self, dt, velocity):
        """
        Update the values of the blending function beta at the facets
        according to the HRIC algorithm. Several versions of HRIC
        are implemented
        """
        timer = dolfin.Timer('Ocellaris update CICSAM')

        degree_b = self.blending_function.ufl_element().degree()
        degree_u = velocity[0].ufl_element().degree()
        assert degree_b == 0, (
            'Only facetwise constant blending factors are supported! Got order %d'
            % degree_b
        )
        assert degree_u == 0, (
            'VelocityDGT0Projector must be enabled! Got order %d' % degree_u
        )

        # Check that the input is supported by the C++ code
        degree_a = self.alpha_function.ufl_element().degree()
        if degree_a != 0:
            ocellaris_error(
                'CICSAM scalar field order must be 0',
                'CICSAM implementation does not support order %d fields' % degree_a,
            )

        alpha_arr = self.alpha_function.vector().get_local()
        beta_arr = self.blending_function.vector().get_local()

        cell_dofs = self.cpp_inp.cell_dofmap
        facet_dofs = self.cpp_inp.facet_dofmap

        conFC = self.simulation.data['connectivity_FC']
        facet_info = self.simulation.data['facet_info']
        cell_info = self.simulation.data['cell_info']

        # Get the numpy arrays of the input functions
        gradient = self.gradient_reconstructor.gradient
        gradient_arrs = [gi.vector().get_local() for gi in gradient]
        velocity_arrs = [vi.vector().get_local() for vi in velocity]

        EPS = 1e-6
        Co_max = 0
        for facet in dolfin.facets(self.mesh):
            fidx = facet.index()
            fdof = facet_dofs[fidx]
            finfo = facet_info[fidx]

            # Find the local cells (the two cells sharing this face)
            connected_cells = conFC(fidx)

            if len(connected_cells) != 2:
                # This should be an exterior facet (on ds)
                assert facet.exterior()
                beta_arr[fdof] = 0.0
                continue

            # Indices of the two local cells
            ic0, ic1 = connected_cells

            # Velocity at the facet
            ump = [vi[fdof] for vi in velocity_arrs]

            # Midpoint of local cells
            cell0_mp = cell_info[ic0].midpoint
            cell1_mp = cell_info[ic1].midpoint
            mp_dist = cell1_mp - cell0_mp

            # Normal pointing out of cell 0
            normal = finfo.normal

            # Find indices of downstream ("D") cell and central ("C") cell
            uf = numpy.dot(normal, ump)
            if uf > 0:
                iaC = ic0
                iaD = ic1
                vec_to_downstream = mp_dist
            else:
                iaC = ic1
                iaD = ic0
                vec_to_downstream = -mp_dist

            # Find alpha in D and C cells
            aD = alpha_arr[self.alpha_dofmap[iaD]]
            aC = alpha_arr[self.alpha_dofmap[iaC]]

            # Gradient of alpha in the central cell
            gC = [gi[cell_dofs[iaC]] for gi in gradient_arrs]

            # Upstream value
            # See Ubbink's PhD (1997) equations 4.21 and 4.22
            aU = aD - 2 * numpy.dot(gC, vec_to_downstream)
            aU = min(max(aU, 0.0), 1.0)

            # Calculate the facet Courant number
            Co = abs(uf) * dt * finfo.area / cell_info[iaC].volume
            Co_max = max(Co_max, Co)

            if abs(aC - aD) < EPS or abs(aU - aD) < EPS:
                # No change in this area, use upstream value
                beta_arr[fdof] = 0.0
                continue

            # Introduce normalized variables
            tilde_aC = (aC - aU) / (aD - aU)

            if tilde_aC <= 0 or tilde_aC >= 1:
                # Only upwind is stable
                beta_arr[fdof] = 0.0
                continue

            # Compressive scheme, Hyper-C
            tilde_aF_HC = min(tilde_aC / Co, 1)

            # Less compressive scheme, Ultimate-Quickest
            tilde_aF_UC = min(
                (8 * Co * tilde_aC + (1 - Co) * (6 * tilde_aC + 3)) / 8, tilde_aF_HC
            )

            # Correct tilde_aF to avoid aligning with interfaces
            d = vec_to_downstream
            theta = math.acos(
                abs(numpy.dot(gC, d) / (numpy.dot(gC, gC) * numpy.dot(d, d)) ** 0.5)
            )
            ky = 1.0
            y = min(ky * (math.cos(2 * theta) + 1) / 2, 1)
            tilde_aF_final = tilde_aF_HC * y + tilde_aF_UC * (1 - y)

            # Avoid tilde_aF being slightly lower that tilde_aC due to
            # floating point errors, it must be greater or equal
            if tilde_aC - EPS < tilde_aF_final < tilde_aC:
                tilde_aF_final = tilde_aC

            # Calculate the downstream blending factor (0=upstream, 1=downstream)
            tilde_beta = (tilde_aF_final - tilde_aC) / (1 - tilde_aC)

            if not (0.0 <= tilde_beta <= 1.0):
                print('ERROR, tilde_beta %r is out of range [0, 1]' % tilde_beta)
                print(' face normal: %r' % normal)
                print(' surface gradient: %r' % gC)
                print(' theta: %r' % theta)
                print(' tilde_aF_final %r' % tilde_aF_final)
                print(' tilde_aC %r' % tilde_aC)
                print(' aU %r, aC %r, aD %r' % (aU, aC, aD))

            assert 0.0 <= tilde_beta <= 1.0
            beta_arr[fdof] = tilde_beta

        self.blending_function.vector().set_local(beta_arr)
        self.blending_function.vector().apply('insert')

        Co_max = dolfin.MPI.max(self.mesh.mpi_comm(), Co_max)
        self.simulation.reporting.report_timestep_value('Cof_max', Co_max)
        timer.stop()
