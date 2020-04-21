Baroclinic model formulation
============================

.. highlight:: python

Governing equations
-------------------

The three dimensional model solves the Navier-Stokes equations with Boussinesq
and hydrostatic assumptions.

The model uses mode-splitting, i.e. the three dimensional horizontal
velocity :math:`\mathbf{u}` is split into a depth average
:math:`\bar{\mathbf{u}}` and a deviation
:math:`\mathbf{u}' = \mathbf{u} - \bar{\mathbf{u}}`.
We use the 2D shallow water equations :eq:`swe_freesurf_modesplit`\-
:eq:`swe_momentum_modesplit` to solve :math:`\bar{\mathbf{u}}`, and the
3D momentum equation :eq:`mom_eq_split` to solve :math:`\mathbf{u}'`.
See modules
:py:mod:`~.shallowwater_eq` and :py:mod:`~.momentum_eq` for more information.

Since the model is hydrostatic, the vertical velocity is solved diagnostically
from the continuity equation :eq:`continuity_eq_3d`.
The solver is implemented in :py:class:`~.VerticalVelocitySolver`.

Water temperature and salinity are modeled as tracers by the means of the
tracer advection-diffusion equation :eq:`tracer_eq`.
Options :ref:`ModelOptions3d<model_options_3d>`.\ :py:attr:`.solve_temperature`, :ref:`ModelOptions3d<model_options_3d>`.\ :py:attr:`.solve_salinity`
determine whether the dynamic equations are solved at run time.
If not, we treat these variables as constants whose value is set with
:ref:`ModelOptions3d<model_options_3d>`.\ :py:attr:`.constant_temperature` and
:ref:`ModelOptions3d<model_options_3d>`.\ :py:attr:`.constant_salinity` options.

In baroclinic simulations the water density, :math:`\rho`,  depends on the
temperature and salinity via the equation of state:
:math:`\rho = \rho'(T, S, p) + \rho_0`, where :math:`\rho_0`
is a constant reference density.
The equation of state is set by the option
:ref:`ModelOptions3d<model_options_3d>`.\ :py:attr:`.equation_of_state_type`, and
:math:`\rho_0` is defined in the :py:mod:`~.physical_constants` module.
Water density affects the internal pressure gradient through the baroclinic
head, :math:`r`, which we can solve diagnostically from :eq:`baroc_head`.
The internal pressure gradient, :math:`\mathbf{F}_{pg} = g\nabla_h r`, is
computed weakly as separate field.
The solver is implemented in :py:class:`~.InternalPressureGradientCalculator`.

Setting option :ref:`ModelOptions3d<model_options_3d>`.\ :py:attr:`.use_baroclinic_formulation`\ ``=True`` activates
baroclinicity, i.e. the computation of water density, baroclinic head and
(both 2D and 3D) internal pressure gradients.
If :py:attr:`.use_baroclinic_formulation`\ ``=False``, water density is not computed. Temperature and
salinity may still be simulated, but they are treated as passive tracers.

The following tables summarize the prognostic and diagnostic variables.

================== ======================== ============================= ======================
Variable           Symbol                   Dynamic equation              Thetis field name
================== ======================== ============================= ======================
Water elevation    :math:`\eta`             :eq:`swe_freesurf_modesplit`  ``elev_2d``, ``elev_3d``
Depth av. velocity :math:`\bar{\mathbf{u}}` :eq:`swe_momentum_modesplit`  ``uv_2d``
3D velocity        :math:`\mathbf{u}'`      :eq:`mom_eq_split`            ``uv_3d``
Water temperature  :math:`T`                :eq:`tracer_eq`               ``temp_3d``
Water salinity     :math:`S`                :eq:`tracer_eq`               ``salt_3d``
================== ======================== ============================= ======================

Table 1. *Prognostic variables in the 3D model.*

================== ======================== ============================= ======================
Variable           Symbol                   Equation                      Thetis field name
================== ======================== ============================= ======================
Vertical velocity  :math:`w`                :eq:`continuity_eq_3d`        ``w_3d``
Water density      :math:`\rho`             :eq:`equation_of_state`       ``rho_3d``
Baroclinic head    :math:`r`                :eq:`baroc_head`              ``baroc_head_3d``
Pressure gradient  :math:`\mathbf{F}_{pg}`  :eq:`int_pg_eq`               ``int_pg_3d``
================== ======================== ============================= ======================

Table 2. *Diagnostic variables in the 3D model.*


Spatial discretization
----------------------

Currently Thetis supports two finite element families:
Equal order Discontinuous Galerkin (DG)
(option ``'dg-dg'``), and mimetic Raviart-Thomas-DG family (``'rt-dg'``).
The element family is set by the :ref:`ModelOptions3d<model_options_3d>`.\ :py:attr:`.element_family` option.
Currently only linear elements are supported, i.e.
:ref:`ModelOptions3d<model_options_3d>`.\ :py:attr:`.polynomial_degree` must be 1.

The function spaces for both element families are summarized in the following
tables.

================== ======================== =============================
Variable           Symbol                   Function space
================== ======================== =============================
Water elevation    :math:`\eta`             P1DG
Depth av. velocity :math:`\bar{\mathbf{u}}` P1DG
3D velocity        :math:`\mathbf{u}'`      P1DG x P1DG
Water temperature  :math:`T`                P1DG x P1DG
Water salinity     :math:`S`                P1DG x P1DG
Vertical velocity  :math:`w`                P1DG x P1DG
Water density      :math:`\rho`             P1DG x P1DG
Baroclinic head    :math:`r`                P1DG x P2
Pressure gradient  :math:`\mathbf{F}_{pg}`  P1DG x P1DG
================== ======================== =============================

Table 3. *Equal order Discontinuous Galerkin function spaces (degree=1).*

================== ======================== =============================
Variable           Symbol                   Function space
================== ======================== =============================
Water elevation    :math:`\eta`             P1DG
Depth av. velocity :math:`\bar{\mathbf{u}}` RT2
3D velocity        :math:`\mathbf{u}'`      HDiv(RT2 x P1DG)
Water temperature  :math:`T`                P1DG x P1DG
Water salinity     :math:`S`                P1DG x P1DG
Vertical velocity  :math:`w`                HDiv(P1DG x P2)
Water density      :math:`\rho`             P1DG x P1DG
Baroclinic head    :math:`r`                P1DG x P2
Pressure gradient  :math:`\mathbf{F}_{pg}`  HDiv(RT2 x P1DG)
================== ======================== =============================

Table 4. *Raviart-Thomas Discontinuous Galerkin function spaces (degree=1).*

In both cases the tracers belong to fully discontinuous P1DG x P1DG function
space. Tracer advection is solved with upwinding method and slope limiters
(see :py:class:`~.VertexBasedP1DGLimiter`).

Temporal discretization
-----------------------

The system of coupled equations is marched in time with a
:py:class:`~.CoupledTimeIntegrator`.
The time integration method is set by :ref:`ModelOptions3d<model_options_3d>`.\ :py:attr:`.timestepper_type`
option. Currently supported time integrators are listed below.

======================== ====================================== ======== ================ ============
Time integrator          Thetis class                           2D mode  ALE mesh support Description
======================== ====================================== ======== ================ ============
``'SSPRK22'``            :py:class:`~.CoupledTwoStageRK`        implicit yes              Coupled method based on SSPRK(2,2) scheme
``'LeapFrog'``           :py:class:`~.CoupledLeapFrogAM3`       implicit yes              Leapfrog Adams-Moulton 3 method
======================== ====================================== ======== ================ ============

Table 5. *Supported 3D time integrators.*


The 2D and 3D time steps can be set via :ref:`ModelOptions3d<model_options_3d>`.\ :py:attr:`.timestep` and
:ref:`ModelOptions3d<model_options_3d>`.\ :py:attr:`.timestep_2d` options.
The 2D mode can be treated either implicitly or explicitly.
In case of an implicit 2D mode, the 2D time step is equal to the 3D time step
and :py:attr:`.timestep_2d` option is ignored.

Thetis can also estimate the maximum stable time step based on the mesh
resolution, used element family and time integration scheme.
To use this feature, the user should provide the following estimates:

- :ref:`ModelOptions3d<model_options_3d>`.\ :py:attr:`.horizontal_velocity_scale`: Maximal horizontal velocity scale
- :ref:`ModelOptions3d<model_options_3d>`.\ :py:attr:`.vertical_velocity_scale`: Maximal vertical velocity scale
- :ref:`ModelOptions3d<model_options_3d>`.\ :py:attr:`.horizontal_viscosity_scale`: Maximal horizontal viscosity scale

When the simulation initializes, Thetis will compute the maximal feasible time
step:

.. code-block:: none

    Coupled time integrator: CoupledTwoStageRK
    2D time integrator: TwoStageTrapezoid
    3D time integrator: SSPRK22ALE
    3D implicit time integrator: BackwardEuler
    - dt 2d swe: 7.34794172415
    - dt h. advection: 213.200697179
    - dt v. advection: 729.166666667
    - dt viscosity: 45454.5372777
    - CFL adjusted dt: 2D: inf 3D: 213.200697179
    - chosen dt: 2D: 213.0 3D: 213.0
    - adjusted dt: 2D: 180.0 3D: 180.0
