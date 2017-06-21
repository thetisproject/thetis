Depth averaged 2D model formulation
===================================

.. highlight:: python

Governing equations
-------------------

The two dimensional model solves the depth averaged shallow water equations
:eq:`swe_freesurf`\-:eq:`swe_momentum`.
The prognostic variables are the water elevation :math:`\eta` and depth
averaged velocity :math:`\bar{\mathbf{u}}`.
The corresponding fields in Thetis are called ``'elev_2d'`` and  ``'uv_2d'``.

Wetting and drying
------------------

Wetting and drying is included through the modified bathymetry formulation of Karna et al. (2011). The modified equations are given by :eq:`swe_freesurf_wd`\-:eq:`swe_momentum_wd`.

Spatial discretization
----------------------

Thetis supports different finite element discretizations, summarised in the
table below.

.. |uu| replace:: :math:`\bar{\mathbf{u}}`
.. |eta| replace:: :math:`\eta`

================== ============ =========== ========== ===========
Element Family     Name         Degree *n*  |uu| space |eta| space
================== ============ =========== ========== ===========
Equal order DG     ``'dg-dg'``  1, 2        P(n)DG     P(n)DG
Raviart-Thomas DG  ``'rt-dg'``  1, 2        RT(n+1)    P(n)DG
P1DG-P2            ``'dg-cg'``  1           P(n)DG     P(n+1)
================== ============ =========== ========== ===========

Table 1. *Finite element families for polynomial degree n.*

The element family and polynomial degree are set by the :py:attr:`.ModelOptions.element_family` and :py:attr:`.ModelOrder.polynomial_degree` options.

Temporal discretization
-----------------------

Thetis supports different time integration methods, set by the
:py:attr:`.ModelOptions.timestepper_type` option.

=============================== ====================================== ====================== ============
Time integrator                 Thetis class                           Unconditionally stable Description
=============================== ====================================== ====================== ============
``'forwardeuler'``              :py:class:`~.ForwardEuler`             No                     Forward Euler method
``'backwardeuler'``             :py:class:`~.BackwardEuler`            Yes                    Backward Euler method
``'cranknicolson'``             :py:class:`~.CrankNicolson`            Yes                    Crank-Nicolson method
``'dirk33'``                    :py:class:`~.DIRK33`                   Yes                    DIRK(3,4,3) method
``'ssprk33'``                   :py:class:`~.SSPRK33`                  No                     SSPRK(3,3) method
``'sspimex'``                   :py:class:`~.IMEXLPUM2`                No                     LPUM2 SSP IMEX scheme
``'pressureprojectionpicard'``  :py:class:`~.PressureProjectionPicard` No                     Efficient pressure projection solver
``'steadystate'``               :py:class:`~.SteadyState`              --                     Solves equations in steady state
=============================== ====================================== ====================== ============

Table 2. *Time integration methods for 2D model.*

Model time step is defined by the :py:attr:`.ModelOptions.timestep` option.

For explicit solvers, Thetis can also estimate the maximum stable time step
based on the mesh resolution, used element family and time integration scheme.
To use this feature, the user should provide the maximal horizontal velocity
scale with :py:attr:`.ModelOptions.horizontal_velocity_scale` option and leave
:py:attr:`.ModelOptions.timestep` undefined (or set it to ``None``).
