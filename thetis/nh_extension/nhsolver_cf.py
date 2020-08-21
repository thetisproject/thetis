"""
Module for 2D depth averaged solver in conservative form
"""
from __future__ import absolute_import
from .utility_nh import *
from . import shallowwater_nh
from . import shallowwater_cf
from . import momentum_cf
from . import granular_cf
from . import tracer_sigma
from .. import timeintegrator
from .. import rungekutta
from . import limiter_nh
import weakref
import time as time_mod
from mpi4py import MPI
from .. import exporter
from ..field_defs import field_metadata
from ..options import ModelOptions3d
from .. import callback
from ..log import *
from collections import OrderedDict

g_grav = physical_constants['g_grav']
rho_0 = physical_constants['rho0']

class FlowSolver(FrozenClass):
    """
    Main object for 3D solver in conservative form

    **Example**

    Create 2D mesh

    .. code-block:: python

        from thetis import *
        mesh2d = RectangleMesh(20, 20, 10e3, 10e3)

    Create bathymetry function, set a constant value

    .. code-block:: python

        fs_p1 = FunctionSpace(mesh2d, 'CG', 1)
        bathymetry_2d = Function(fs_p1, name='Bathymetry').assign(10.0)

    Create a 3D model with 6 uniform levels, and set some options
    (see :class:`.ModelOptions3d`)

    .. code-block:: python

        solver_obj = solver_cf.FlowSolver(mesh2d, bathymetry_2d, n_layers=6)
        options = solver_obj.options
        options.element_family = 'dg-dg'
        options.polynomial_degree = 1
        options.timestepper_type = 'SSPRK22'
        options.timestepper_options.use_automatic_timestep = False
        options.solve_salinity = False
        options.solve_temperature = False
        options.simulation_export_time = 50.0
        options.simulation_end_time = 3600.
        options.timestep = 25.0

    Assign initial condition for water elevation

    .. code-block:: python

        solver_obj.create_function_spaces()
        init_elev = Function(solver_obj.function_spaces.H_2d)
        coords = SpatialCoordinate(mesh2d)
        init_elev.project(2.0*exp(-((coords[0] - 4e3)**2 + (coords[1] - 4.5e3)**2)/2.2e3**2))
        solver_obj.assign_initial_conditions(elev=init_elev)

    Run simulation

    .. code-block:: python

        solver_obj.iterate()

    See the manual for more complex examples.
    """
    def __init__(self, mesh2d, bathymetry_2d, n_layers,
                 options=None, extrude_options=None, mesh_ls=None):
        """
        :arg mesh2d: :class:`Mesh` object of the 2D mesh
        :arg bathymetry_2d: Bathymetry of the domain. Bathymetry stands for
            the mean water depth (positive downwards).
        :type bathymetry_2d: :class:`Function`
        :kwarg options: Model options (optional). Model options can also be
            changed directly via the :attr:`.options` class property.
        :type options: :class:`.ModelOptions2d` instance
        """
        self._initialized = False

        self.bathymetry_cg_2d = bathymetry_2d

        self.mesh2d = mesh2d
        # independent landslide mesh for granular flow
        self.mesh_ls = self.mesh2d
        if mesh_ls is not None:
            self.mesh_ls = mesh_ls
        """2D :class`Mesh`"""
        if extrude_options is None:
            extrude_options = {}
        self.mesh = ExtrudedMesh(mesh2d, layers=n_layers, layer_height=1.0/n_layers)
        """3D :class`Mesh`"""
        self.comm = mesh2d.comm

        # add boundary length info
        bnd_len = compute_boundary_length(self.mesh2d)
        self.mesh2d.boundary_len = bnd_len
        self.mesh_ls.boundary_len = bnd_len
        self.mesh.boundary_len = bnd_len
        self.boundary_markers = self.mesh2d.exterior_facets.unique_markers

        self.normal = FacetNormal(self.mesh)

        self.dt = None
        """Time step"""

        # override default options
        self.options = ModelOptions3d()
        """
        Dictionary of all options. A :class:`.ModelOptions3d` object.
        """
        if options is not None:
            self.options.update(options)

        # simulation time step bookkeeping
        self.simulation_time = 0
        self.iteration = 0
        self.i_export = 0
        self.next_export_t = self.simulation_time + self.options.simulation_export_time

        self.bnd_functions = {'shallow_water': {},
                              'momentum': {},
                              'landslide_motion': {},
                              'salt': {},
                              'temp': {},
                              }

        self.callbacks = callback.CallbackManager()
        """
        :class:`.CallbackManager` object that stores all callbacks
        """

        self.fields = FieldDict()
        """
        :class:`.FieldDict` that holds all functions needed by the solver
        object
        """

        self.function_spaces = AttrDict()
        """
        :class:`.AttrDict` that holds all function spaces needed by the
        solver object
        """

        self.export_initial_state = True
        """Do export initial state. False if continuing a simulation"""

        self._simulation_continued = False
        self._isfrozen = True

    def compute_time_step(self, u_scale=Constant(0.0)):
        r"""
        Computes maximum explicit time step from CFL condition.

        .. math :: \Delta t = \frac{\Delta x}{U}

        Assumes velocity scale :math:`U = \sqrt{g H} + U_{scale}` where
        :math:`U_{scale}` is estimated advective velocity.

        :kwarg u_scale: User provided maximum advective velocity scale
        :type u_scale: float or :class:`Constant`
        """
        csize = self.fields.h_elem_size_2d
        bath = self.fields.bathymetry_2d
        fs = bath.function_space()
        bath_pos = Function(fs, name='bathymetry')
        bath_pos.assign(bath)
        min_depth = 0.05
        bath_pos.dat.data[bath_pos.dat.data < min_depth] = min_depth
        test = TestFunction(fs)
        trial = TrialFunction(fs)
        solution = Function(fs)
        g = physical_constants['g_grav']
        u = (sqrt(g * bath_pos) + u_scale)
        a = inner(test, trial) * dx
        l = inner(test, csize / u) * dx
        solve(a == l, solution)
        return solution

    def set_time_step(self, alpha=0.05):
        """
        Sets the model the model time step

        If the time integrator supports automatic time step, and
        :attr:`ModelOptions2d.timestepper_options.use_automatic_timestep` is
        `True`, we compute the maximum time step allowed by the CFL condition.
        Otherwise uses :attr:`ModelOptions2d.timestep`.

        :kwarg float alpha: CFL number scaling factor
        """
        automatic_timestep = (hasattr(self.options.timestepper_options, 'use_automatic_timestep') and
                              self.options.timestepper_options.use_automatic_timestep)
        # TODO revisit math alpha is OBSOLETE
        if automatic_timestep:
            mesh2d_dt = self.compute_time_step(u_scale=self.options.horizontal_velocity_scale)
            dt = self.options.cfl_2d*alpha*float(mesh2d_dt.dat.data.min())
            dt = self.comm.allreduce(dt, op=MPI.MIN)
            self.dt = dt
        else:
            assert self.options.timestep is not None
            assert self.options.timestep > 0.0
            self.dt = self.options.timestep
        if self.comm.rank == 0:
            print_output('dt = {:}'.format(self.dt))
            sys.stdout.flush()

    def create_function_spaces(self):
        """
        Creates function spaces

        Function spaces are accessible via :attr:`.function_spaces`
        object.
        """
        self._isfrozen = False

        # 3D function spaces
        self.function_spaces.P0 = get_functionspace(self.mesh, 'DG', 0, 'DG', 0, name='P0')
        self.function_spaces.P1 = get_functionspace(self.mesh, 'CG', 1, 'CG', 1, name='P1')
        self.function_spaces.P2 = get_functionspace(self.mesh, 'CG', 2, 'CG', 2, name='P2')
        self.function_spaces.P1v = get_functionspace(self.mesh, 'CG', 1, 'CG', 1, name='P1v', vector=True)
        self.function_spaces.P1DG = get_functionspace(self.mesh, 'DG', 1, 'DG', 1, name='P1DG')
        self.function_spaces.P1DGv = get_functionspace(self.mesh, 'DG', 1, 'DG', 1, name='P1DGv', vector=True)
        # function spaces for hu, hv and hw
        if self.options.element_family == 'dg-dg':
            self.function_spaces.H = get_functionspace(self.mesh, 'DG', self.options.polynomial_degree, 'DG', self.options.polynomial_degree, name='H')
            self.function_spaces.U = get_functionspace(self.mesh, 'DG', self.options.polynomial_degree, 'DG', self.options.polynomial_degree, name='U', vector=True)
            if self.options.use_vert_dg0:
                self.function_spaces.H = get_functionspace(self.mesh, 'DG', self.options.polynomial_degree, 'DG', 0, name='H')
                self.function_spaces.U = get_functionspace(self.mesh, 'DG', self.options.polynomial_degree, 'DG', 0, name='U', vector=True)
        else:
            raise Exception('Unsupported finite element family {:}'.format(self.options.element_family))
        self.function_spaces.V = MixedFunctionSpace([self.function_spaces.H, self.function_spaces.H, self.function_spaces.H])

        # 2D function spaces
        self.function_spaces.P0_2d = get_functionspace(self.mesh2d, 'DG', 0, name='P0_2d')
        self.function_spaces.P1_2d = get_functionspace(self.mesh2d, 'CG', 1, name='P1_2d')
        self.function_spaces.P2_2d = get_functionspace(self.mesh2d, 'CG', 2, name='P2_2d')
        self.function_spaces.P1DG_2d = get_functionspace(self.mesh2d, 'DG', 1, name='P1DG_2d')
        # function space w.r.t element family
        if self.options.element_family == 'dg-dg':
            self.function_spaces.H_2d = get_functionspace(self.mesh2d, 'DG', self.options.polynomial_degree, name='H_2d')
            self.function_spaces.U_2d = get_functionspace(self.mesh2d, 'DG', self.options.polynomial_degree, name='U_2d', vector=True)
        else:
            raise Exception('Unsupported finite element family {:}'.format(self.options.element_family))
        self.function_spaces.V_2d = MixedFunctionSpace([self.function_spaces.H_2d, self.function_spaces.U_2d])

        # function spaces for granular landslide
        self.function_spaces.H_ls = get_functionspace(self.mesh_ls, 'DG', self.options.polynomial_degree)
        self.function_spaces.U_ls = get_functionspace(self.mesh_ls, 'DG', self.options.polynomial_degree, vector=True)
        self.function_spaces.V_ls = MixedFunctionSpace([self.function_spaces.H_ls, self.function_spaces.H_ls, self.function_spaces.H_ls])
        self.function_spaces.P1_ls = get_functionspace(self.mesh_ls, 'CG', 1)

        self.function_spaces.turb_space = self.function_spaces.P0

        # define function spaces for baroclinic head and internal pressure gradient
        if self.options.use_quadratic_pressure:
            self.function_spaces.P2DGxP2 = get_functionspace(self.mesh, 'DG', 2, 'CG', 2, name='P2DGxP2')
            self.function_spaces.P2DG_2d = get_functionspace(self.mesh2d, 'DG', 2, name='P2DG_2d')
            if self.options.element_family == 'dg-dg':
                self.function_spaces.P2DGxP1DGv = get_functionspace(self.mesh, 'DG', 2, 'DG', 1, name='P2DGxP1DGv', vector=True, dim=2)
                self.function_spaces.H_bhead = self.function_spaces.P2DGxP2
                self.function_spaces.U_int_pg = self.function_spaces.P2DGxP1DGv
        else:
            self.function_spaces.P1DGxP2 = get_functionspace(self.mesh, 'DG', 1, 'CG', 2, name='P1DGxP2')
            self.function_spaces.H_bhead = self.function_spaces.P1DGxP2
            self.function_spaces.U_int_pg = self.function_spaces.U

        self._isfrozen = True

    def set_sipg_parameter(self):
        r"""
        Compute a penalty parameter which ensures stability of the Interior Penalty method
        used for viscosity and diffusivity terms, from Epshteyn et al. 2007
        (http://dx.doi.org/10.1016/j.cam.2006.08.029).
        The scheme is stable if
        ..math::
            \alpha|_K > 3*X*p*(p+1)*\cot(\theta_K),
        for all elements :math:`K`, where
        ..math::
            X = \frac{\max_{x\in K}(\nu(x))}{\min_{x\in K}(\nu(x))},
        :math:`p` the degree, and :math:`\theta_K` is the minimum angle in the element.
        """
        degree_h, degree_v = self.function_spaces.U.ufl_element().degree()
        alpha_h = Constant(5.0*degree_h*(degree_h+1) if degree_h != 0 else 1.5)
        alpha_v = Constant(5.0*degree_v*(degree_v+1) if degree_v != 0 else 1.0)
        degree_h_tracer, degree_v_tracer = self.function_spaces.H.ufl_element().degree()
        alpha_h_tracer = Constant(5.0*degree_h_tracer*(degree_h_tracer+1) if degree_h_tracer != 0 else 1.5)
        alpha_v_tracer = Constant(5.0*degree_v_tracer*(degree_v_tracer+1) if degree_v_tracer != 0 else 1.0)
        degree_h_turb, degree_v_turb = self.function_spaces.turb_space.ufl_element().degree()
        alpha_h_turb = Constant(5.0*degree_h_turb*(degree_h_turb+1) if degree_h_turb != 0 else 1.5)
        alpha_v_turb = Constant(5.0*degree_v_turb*(degree_v_turb+1) if degree_v_turb != 0 else 1.0)

        if self.options.use_automatic_sipg_parameter:

            # Compute minimum angle in 2d mesh
            theta2d = get_minimum_angles_2d(self.mesh2d)
            min_angle = theta2d.vector().gather().min()
            print_output("Minimum angle in 2D mesh: {:.2f} degrees".format(np.rad2deg(min_angle)))

            # Expand minimum angle field to extruded mesh
            P0 = self.function_spaces.P0
            theta = Function(P0)
            ExpandFunctionTo3d(theta2d, theta).solve()
            cot_theta = 1.0/tan(theta)

            # Horizontal component
            nu = self.options.horizontal_viscosity
            if nu is not None:
                self.options.sipg_parameter = Function(P0)
                self.options.sipg_parameter.interpolate(alpha_h*get_sipg_ratio(nu)*cot_theta)
                max_sipg = self.options.sipg_parameter.vector().gather().max()
                print_output("Maximum SIPG value in horizontal: {:.2f}".format(max_sipg))
            else:
                print_output("SIPG parameter in horizontal: {:.2f}".format(alpha_h.values()[0]))

            # Vertical component
            print_output("SIPG parameter in vertical: {:.2f}".format(alpha_v.values()[0]))

            # Penalty parameter for tracers / turbulence model
            if self.options.solve_salinity or self.options.solve_temperature or self.options.use_turbulence:

                # Horizontal component
                nu = self.options.horizontal_diffusivity
                if nu is not None:
                    scaling = get_sipg_ratio(nu)*cot_theta
                    if self.options.solve_salinity or self.options.solve_temperature:
                        self.options.sipg_parameter_tracer = Function(P0)
                        self.options.sipg_parameter_tracer.interpolate(alpha_h_tracer*scaling)
                        max_sipg = self.options.sipg_parameter_tracer.vector().gather().max()
                        print_output("Maximum tracer SIPG value in horizontal: {:.2f}".format(max_sipg))
                    if self.options.use_turbulence:
                        self.options.sipg_parameter_turb = Function(P0)
                        self.options.sipg_parameter_turb.interpolate(alpha_h_turb*scaling)
                        max_sipg = self.options.sipg_parameter_turb.vector().gather().max()
                        print_output("Maximum turbulence SIPG value in horizontal: {:.2f}".format(max_sipg))
                else:
                    if self.options.solve_salinity or self.options.solve_temperature:
                        print_output("Tracer SIPG parameter in horizontal: {:.2f}".format(alpha_h_tracer.values()[0]))
                    if self.options.use_turbulence:
                        print_output("Turbulence SIPG parameter in horizontal: {:.2f}".format(alpha_h_turb.values()[0]))

                # Vertical component
                if self.options.solve_salinity or self.options.solve_temperature:
                    print_output("Tracer SIPG parameter in vertical: {:.2f}".format(alpha_v_tracer.values()[0]))
                if self.options.use_turbulence:
                    print_output("Turbulence SIPG parameter in vertical: {:.2f}".format(alpha_v_turb.values()[0]))
        else:
            print_output("Using default SIPG parameters")
            self.options.sipg_parameter.assign(alpha_h)
            self.options.sipg_parameter_tracer.assign(alpha_h_tracer)
            self.options.sipg_parameter_turb.assign(alpha_h_turb)
        self.options.sipg_parameter_vertical.assign(alpha_v)
        self.options.sipg_parameter_vertical_tracer.assign(alpha_v_tracer)
        self.options.sipg_parameter_vertical_turb.assign(alpha_v_turb)

    def create_fields(self):
        """
        Creates all fields and extra functions
        """
        if not hasattr(self, 'U_2d'):
            self.create_function_spaces()
        self._isfrozen = False

        if self.options.log_output and not self.options.no_exports:
            logfile = os.path.join(create_directory(self.options.output_directory), 'log')
            filehandler = logging.logging.FileHandler(logfile, mode='w')
            filehandler.setFormatter(logging.logging.Formatter('%(message)s'))
            output_logger.addHandler(filehandler)

        coord_is_dg = element_continuity(self.mesh2d.coordinates.function_space().ufl_element()).horizontal == 'dg'
        if coord_is_dg:
            coord_fs = FunctionSpace(self.mesh, 'DG', 1, vfamily='CG', vdegree=1)
            coord_fs_2d = self.function_spaces.P1DG_2d
        else:
            coord_fs = self.function_spaces.P1
            coord_fs_2d = self.function_spaces.P1_2d

        # bathymetry
        self.fields.bathymetry_3d = Function(self.function_spaces.H)
        self.fields.bathymetry_2d = Function(self.function_spaces.H_2d)
        self.bathymetry_init = Function(self.function_spaces.H_2d)

        # momentum and velocity
        self.fields.mom_3d = Function(self.function_spaces.U, name='Momentum')
        self.mom_3d_old = Function(self.function_spaces.U)
        self.mom_3d_mid = Function(self.function_spaces.U)
        self.fields.elev_3d = Function(self.function_spaces.H)
        self.elev_3d_old = Function(self.function_spaces.H)
        self.fields.uv_3d = Function(self.function_spaces.U, name='Velocity')
        self.uv_3d_old = Function(self.function_spaces.U)
        self.uv_3d_mid = Function(self.function_spaces.U)

        # depth averaged uv
        self.fields.uv_dav_3d = Function(self.function_spaces.U)
        self.uv_dav_3d_mid = Function(self.function_spaces.U)
        self.fields.uv_dav_2d = Function(self.function_spaces.U_2d)

        # element size
        self.fields.v_elem_size_3d = Function(self.function_spaces.P1DG)
        self.fields.v_elem_size_2d = Function(self.function_spaces.P1DG_2d)
        self.fields.h_elem_size_3d = Function(self.function_spaces.P1)
        self.fields.h_elem_size_2d = Function(self.function_spaces.P1_2d)
        get_horizontal_elem_size_3d(self.fields.h_elem_size_2d, self.fields.h_elem_size_3d)
        self.fields.z_coord_3d = Function(coord_fs).project(self.mesh.coordinates[2])
        compute_elem_height(self.fields.z_coord_3d, self.fields.v_elem_size_3d)
        
        # limiter
        if (self.options.use_limiter_for_velocity
                and self.options.polynomial_degree > 0
                and self.options.element_family == 'dg-dg'):
            self.uv_limiter = limiter_nh.VertexBasedP1DGLimiter(self.function_spaces.U)
        else:
            self.uv_limiter = None
        if self.options.use_limiter_for_tracers and self.options.polynomial_degree > 0:
            self.tracer_limiter = limiter_nh.VertexBasedP1DGLimiter(self.function_spaces.H)
        else:
            self.tracer_limiter = None

        self.source_mom = Function(self.function_spaces.U)
        self.fields.q_3d = Function(FunctionSpace(self.mesh, 'CG', self.options.polynomial_degree+1))
        self.fields.q_2d = Function(FunctionSpace(self.mesh2d, 'CG', self.options.polynomial_degree+1))
        if self.options.use_vert_dg0:
            self.fields.q_3d = Function(FunctionSpace(self.mesh, 'CG', 1, 'CG', 1))
            self.fields.q_2d = Function(FunctionSpace(self.mesh2d, 'CG', self.options.polynomial_degree))

        # sigma transformation
        coord_fs = self.function_spaces.P1
        self.sigma_coord = Function(coord_fs).project(self.mesh.coordinates[2])
        self.z_in_sigma = Function(coord_fs)
        self.z_in_sigma_old = Function(coord_fs)
        self.bathymetry_3d_old = Function(self.function_spaces.H)
        self.fields.sigma_dt = Function(coord_fs)
        self.fields.sigma_dx = Function(coord_fs)
        self.fields.sigma_dy = Function(coord_fs)
        self.fields.omega = Function(coord_fs)
        # self.fields.sigma_dx**2 + self.fields.sigma_dy**2 + self.fields.sigma_dz**2
        self.sigma_dxyz = Function(coord_fs)

        self.fields.solution_2d = Function(self.function_spaces.V_2d, name='solution_2d')
        self.fields.elev_2d, self.fields.uv_2d = self.fields.solution_2d.split()
        self.solution_2d_old = Function(self.function_spaces.V_2d)
        self.solution_2d_tmp = Function(self.function_spaces.V_2d)
        self.source_sw = Function(self.function_spaces.V_2d)
        self.elev_2d_old = Function(self.function_spaces.H_2d)
        self.elev_2d_fs = Function(self.function_spaces.H_2d)
        self.elev_2d_init = Function(self.function_spaces.H_2d)
        self.fields.mom_2d = Function(self.function_spaces.U_2d)

        # rigid slide
        if self.options.slide_is_rigid:
            self.fields.h_ls = Function(self.function_spaces.H_ls)
            self.h_ls_old = Function(self.function_spaces.H_ls)
        # granular flow
        if self.options.flow_is_granular:
            self.fields.solution_ls = Function(self.function_spaces.V_ls)
            self.solution_ls_old = Function(self.function_spaces.V_ls)
            self.solution_ls_mid = Function(self.function_spaces.V_ls)
            self.solution_ls_tmp = Function(self.function_spaces.V_ls)
            self.fields.h_ls, self.fields.hu_ls, self.fields.hv_ls = self.fields.solution_ls.split()
            self.h_ls_old, self.hu_ls_old, self.hv_ls_old = self.solution_ls_old.split()
            self.bathymetry_ls = Function(self.function_spaces.H_ls)
            self.phi_i = Function(self.function_spaces.P1_ls).assign(self.options.phi_i)
            self.phi_b = Function(self.function_spaces.P1_ls).assign(self.options.phi_b)
            self.kap = Function(self.function_spaces.P1_ls)
            self.uv_div_ls = Function(self.function_spaces.P1_ls)
            self.strain_rate_ls = Function(self.function_spaces.P1_ls)
            self.grad_p_ls = Function(self.function_spaces.U_ls)
            self.grad_p = Function(self.function_spaces.U_2d)
            self.h_2d_ls = Function(self.function_spaces.P1_ls)
            self.h_2d_cg = Function(self.function_spaces.P1_2d)

        self.landslide = self.options.slide_is_rigid or self.options.slide_is_viscous_fluid or self.options.flow_is_granular
        if self.landslide:
            self.slope = Function(self.function_spaces.H_ls).interpolate(self.options.bed_slope[2])
            self.fields.slide_source_2d = Function(self.function_spaces.H_2d)
            self.fields.slide_source_3d = Function(self.function_spaces.H)

        # compute total viscosity/diffusivity
        self.tot_h_visc = SumFunction()
        self.tot_h_visc.add(self.options.horizontal_viscosity)
        self.tot_h_visc.add(self.fields.get('smag_visc_3d'))
        self.tot_v_visc = SumFunction()
        self.tot_v_visc.add(self.options.vertical_viscosity)
        self.tot_v_visc.add(self.fields.get('eddy_visc_3d'))
        self.tot_v_visc.add(self.fields.get('parab_visc_3d'))
        self.tot_h_diff = SumFunction()
        self.tot_h_diff.add(self.options.horizontal_diffusivity)
        self.tot_v_diff = SumFunction()
        self.tot_v_diff.add(self.options.vertical_diffusivity)
        self.tot_v_diff.add(self.fields.get('eddy_diff_3d'))

        # tracer
        if self.options.solve_salinity:
            self.fields.salt_3d = Function(self.function_spaces.H, name='Salinity')
            self.salt_3d_old = Function(self.function_spaces.H)
            self.salt_3d_mid = Function(self.function_spaces.H)
        if self.options.solve_temperature:
            self.fields.temp_3d = Function(self.function_spaces.H, name='Temperature')
            self.temp_3d_old = Function(self.function_spaces.H)
            self.temp_3d_mid = Function(self.function_spaces.H)
        if self.options.use_baroclinic_formulation:
            if self.options.use_quadratic_density:
                self.fields.density_3d = Function(self.function_spaces.P2DGxP2, name='Density')
            else:
                self.fields.density_3d = Function(self.function_spaces.H, name='Density')
            self.fields.baroc_head_3d = Function(self.function_spaces.H_bhead)
            self.fields.int_pg_3d = Function(self.function_spaces.U_int_pg, name='int_pg_3d')

        self._isfrozen = True

    def create_equations(self):
        """
        Creates shallow water equations
        """
        if 'uv_3d' not in self.fields:
            self.create_fields()
        self._isfrozen = False

        # set a penalty parameter
        self.set_sipg_parameter()

        # ----- Equations
        self.eq_sw = shallowwater_cf.ShallowWaterEquations(
            self.fields.solution_2d.function_space(),
            self.fields.bathymetry_2d,
            self.options)
        self.eq_sw.bnd_functions = self.bnd_functions['shallow_water']

        self.eq_momentum = momentum_cf.MomentumEquation(
            self.fields.mom_3d.function_space(),
            self.fields.bathymetry_3d,
            self.options,
            v_elem_size=self.fields.v_elem_size_3d,
            h_elem_size=self.fields.h_elem_size_3d,
            sipg_parameter=self.options.sipg_parameter,
            sipg_parameter_vertical=self.options.sipg_parameter_vertical)
        self.eq_momentum.bnd_functions = self.bnd_functions['momentum']

        if self.options.flow_is_granular:
            self.eq_ls = granular_cf.GranularEquations(
                self.fields.solution_ls.function_space(),
                self.bathymetry_ls,
                self.options
            )
            self.eq_ls.bnd_functions = self.bnd_functions['landslide_motion']

        self.eq_free_surface = shallowwater_cf.FreeSurfaceEquation(
            TestFunction(self.function_spaces.H_2d),
            self.function_spaces.H_2d,
            self.function_spaces.U_2d,
            self.fields.bathymetry_2d,
            self.options)
        self.eq_free_surface.bnd_functions = self.bnd_functions['shallow_water']

        if not self.options.solve_conservative_momentum:
            self.eq_operator_split = shallowwater_cf.OperatorSplitEquations(
                self.fields.solution_2d.function_space(),
                self.fields.bathymetry_2d,
                self.options)
            self.eq_operator_split.bnd_functions = self.bnd_functions['shallow_water']

        if self.options.use_wetting_and_drying:
            self.wd_modification = wetting_and_drying_modification(self.function_spaces.H_2d)
            self.wd_modification_ls = wetting_and_drying_modification(self.function_spaces.H_ls)

        # initialise limiter
        if self.options.polynomial_degree == 1 and self.options.use_limiter_for_elevation:
            self.limiter_h = limiter_nh.VertexBasedP1DGLimiter(self.function_spaces.H_2d)
            self.limiter_u = limiter_nh.VertexBasedP1DGLimiter(self.function_spaces.U_2d)
        else:
            self.limiter_h = None
            self.limiter_u = None

        # --- operators
        self.copy_elev_to_3d = ExpandFunctionTo3d(self.fields.elev_2d, self.fields.elev_3d)
        if self.options.solve_conservative_momentum:
            solution_3d = self.fields.mom_3d
            solution_2d = self.fields.mom_2d
        else:
            solution_3d = self.fields.uv_3d
            solution_2d = self.fields.uv_2d
        self.uv_averager = VerticalIntegrator(solution_3d,
                                              self.fields.uv_dav_3d,
                                              bottom_to_top=True,
                                              bnd_value=Constant((0.0, 0.0, 0.0)),
                                              average=False,
                                              bathymetry=self.fields.bathymetry_3d,
                                              elevation=self.fields.elev_3d)
        self.extract_surf_dav_uv = SubFunctionExtractor(self.fields.uv_dav_3d,
                                                        solution_2d,#self.fields.uv_dav_2d,
                                                        boundary='top', elem_facet='top',
                                                        elem_height=self.fields.v_elem_size_2d)
       # self.copy_uv_dav_to_uv_dav_3d = ExpandFunctionTo3d(self.fields.uv_dav_2d, self.fields.uv_dav_3d,
       #                                                    elem_height=self.fields.v_elem_size_3d)
        self.copy_uv_to_uv_dav_3d = ExpandFunctionTo3d(self.fields.uv_2d, self.fields.uv_dav_3d,
                                                       elem_height=self.fields.v_elem_size_3d)
        # landslide
        self.fields.bathymetry_2d.project(self.bathymetry_cg_2d)
        self.bathymetry_init.assign(self.fields.bathymetry_2d)
        self.copy_bath_to_3d = ExpandFunctionTo3d(self.fields.bathymetry_2d, self.fields.bathymetry_3d)
        # set initial values
        self.copy_bath_to_3d.solve()
        if self.landslide:
            self.copy_slide_source_to_3d = ExpandFunctionTo3d(self.fields.slide_source_2d, self.fields.slide_source_3d)

        if self.options.flow_is_granular:
            self.extract_bot_q = SubFunctionExtractor(self.fields.q_3d,
                                                      self.fields.q_2d,
                                                      boundary='bottom', elem_facet='bottom',
                                                      elem_height=self.fields.v_elem_size_2d)

        # tracer
        if self.options.solve_salinity:
            self.eq_salt = tracer_sigma.TracerEquation(self.fields.salt_3d.function_space(),
                                                    bathymetry=self.fields.bathymetry_3d,
                                                    v_elem_size=self.fields.v_elem_size_3d,
                                                    h_elem_size=self.fields.h_elem_size_3d,
                                                    use_symmetric_surf_bnd=self.options.element_family == 'dg-dg',
                                                    use_lax_friedrichs=self.options.use_lax_friedrichs_tracer,
                                                    sipg_parameter=self.options.sipg_parameter_tracer,
                                                    sipg_parameter_vertical=self.options.sipg_parameter_vertical_tracer)
            self.eq_salt.bnd_functions = self.bnd_functions['salt']

        assert not self.options.use_implicit_vertical_diffusion, 'this will be implemented'

        if self.options.solve_temperature:
            self.eq_temp = tracer_sigma.TracerEquation(self.fields.temp_3d.function_space(),
                                                    bathymetry=self.fields.bathymetry_3d,
                                                    v_elem_size=self.fields.v_elem_size_3d,
                                                    h_elem_size=self.fields.h_elem_size_3d,
                                                    use_symmetric_surf_bnd=self.options.element_family == 'dg-dg',
                                                    use_lax_friedrichs=self.options.use_lax_friedrichs_tracer,
                                                    sipg_parameter=self.options.sipg_parameter_tracer,
                                                    sipg_parameter_vertical=self.options.sipg_parameter_vertical_tracer)
            self.eq_temp.bnd_functions = self.bnd_functions['temp']

        if self.options.use_baroclinic_formulation:
            if self.options.solve_salinity:
                s = self.fields.salt_3d
            else:
                s = self.options.constant_salinity
            if self.options.solve_temperature:
                t = self.fields.temp_3d
            else:
                t = self.options.constant_temperature
            if self.options.equation_of_state_type == 'linear':
                eos_options = self.options.equation_of_state_options
                self.equation_of_state = LinearEquationOfState(eos_options.rho_ref,
                                                               eos_options.alpha,
                                                               eos_options.beta,
                                                               eos_options.th_ref,
                                                               eos_options.s_ref)
            else:
                self.equation_of_state = JackettEquationOfState()
            if self.options.use_quadratic_density:
                self.density_solver = DensitySolverWeak(s, t, self.fields.density_3d,
                                                        self.equation_of_state)
            else:
                self.density_solver = DensitySolver(s, t, self.fields.density_3d,
                                                    self.equation_of_state)
            self.rho_integrator = VerticalIntegrator(self.fields.density_3d,
                                                     self.fields.baroc_head_3d,
                                                     bottom_to_top=False,
                                                     average=False,
                                                     bathymetry=self.fields.bathymetry_3d,
                                                     elevation=self.fields.elev_3d, # TODO check, original: 'self.fields.elev_cg_3d'
                                                     use_in_sigma=True,
                                                     )
            self.int_pg_calculator = momentum_cf.InternalPressureGradientCalculator(
                self.fields, self.options, self.bnd_functions['momentum'],
                solver_parameters=self.options.timestepper_options.solver_parameters_momentum_explicit)

        self._isfrozen = True  # disallow creating new attributes

    def create_timestepper(self):
        """
        Creates time stepper instance
        """
        if not hasattr(self, 'eq_sw'):
            self.create_equations()

        self._isfrozen = False

        if self.options.log_output and not self.options.no_exports:
            logfile = os.path.join(create_directory(self.options.output_directory), 'log')
            filehandler = logging.logging.FileHandler(logfile, mode='w')
            filehandler.setFormatter(logging.logging.Formatter('%(message)s'))
            output_logger.addHandler(filehandler)

        # ----- Time integrators
        # fields dic for 2d timestepper
        self.fields_sw = {
            'mom_2d': self.fields.mom_2d,
            'source_sw': self.source_sw, 
            'uv_2d': self.fields.uv_2d,
            'sponge_damping_2d': self.set_sponge_damping(self.options.sponge_layer_length, self.options.sponge_layer_start, alpha=10., sponge_is_2d=True),
            }
        if self.landslide:
            self.fields_sw.update({'slide_source': self.fields.slide_source_2d,})
        if self.options.flow_is_granular:
            self.fields_ls = {
                'phi_i': self.phi_i,
                'phi_b': self.phi_b,
                #'kap': self.kap,
                'uv_div': self.uv_div_ls,
                'strain_rate': self.strain_rate_ls,
                'fluid_pressure_gradient': self.grad_p_ls,
                'h_2d': self.h_2d_ls,
                }
        # fields dic for 3d timestepper
        self.fields_3d = {
            'sigma_dx': self.fields.sigma_dx,
            'sigma_dy': self.fields.sigma_dy,
            'sigma_dxyz': self.sigma_dxyz,
            'omega': self.fields.omega,
            'int_pg': self.fields.get('int_pg_3d'),
            'uv_3d': self.fields.uv_3d,
            'elev_3d': self.fields.elev_3d, 
            'source_mom': self.source_mom, 
            'sponge_damping_3d': self.set_sponge_damping(
                self.options.sponge_layer_length, self.options.sponge_layer_start, alpha=10., sponge_is_2d=False),
            'viscosity_v': self.tot_v_visc.get_sum(), # for explicit vertical diffusion
            'viscosity_h': self.tot_h_visc.get_sum(),
            # for tracer
            'diffusivity_h': self.tot_h_diff.get_sum(),
            'diffusivity_v': self.tot_v_diff.get_sum(), # for not self.options.use_implicit_vertical_diffusion
            'source_tracer': self.options.salinity_source_3d,
            'lax_friedrichs_tracer_scaling_factor': self.options.lax_friedrichs_tracer_scaling_factor,
            }
        self.set_time_step()

        if not self.options.solve_conservative_momentum:
            self.timestepper_operator_split = timeintegrator.CrankNicolson(self.eq_operator_split, self.fields.solution_2d,
                                                              self.fields_sw, self.dt, bnd_conditions=self.bnd_functions['shallow_water'],
                                                              solver_parameters=self.options.timestepper_options.solver_parameters_2d_swe)
            self.timestepper_free_surface_implicit = timeintegrator.CrankNicolson(self.eq_free_surface, self.elev_2d_fs,
                                                              self.fields_sw, self.dt,
                                                              bnd_conditions=self.bnd_functions['shallow_water'],
                                                              semi_implicit=False,
                                                              theta=0.5)
            self.timestepper_free_surface_explicit = timeintegrator.CrankNicolson(self.eq_free_surface, self.elev_2d_fs,
                                                              self.fields_sw, self.dt,
                                                              bnd_conditions=self.bnd_functions['shallow_water'],
                                                              semi_implicit=False,
                                                              theta=1.)
            self.timestepper_sw = timeintegrator.CrankNicolson(self.eq_sw, self.fields.solution_2d,
                                                               self.fields_sw, self.dt, bnd_conditions=self.bnd_functions['shallow_water'],
                                                               solver_parameters=self.options.timestepper_options.solver_parameters_2d_swe)

        if self.options.timestepper_type == 'SSPRK33': # TODO delete
            self.timestepper = rungekutta.SSPRK33(self.eq_sw, self.fields.solution_2d,
                                                  self.fields_sw, self.dt,
                                                  bnd_conditions=self.bnd_functions['shallow_water'],
                                                  solver_parameters=self.options.timestepper_options.solver_parameters)
        elif self.options.timestepper_type == 'CrankNicolson':
            self.timestepper = timeintegrator.CrankNicolson(self.eq_sw, self.fields.solution_2d,
                                                            self.fields_sw, self.dt,
                                                            bnd_conditions=self.bnd_functions['shallow_water'],
                                                            solver_parameters=self.options.timestepper_options.solver_parameters,
                                                            semi_implicit=self.options.timestepper_options.use_semi_implicit_linearization,
                                                            theta=self.options.timestepper_options.implicitness_theta)

       # print_output('Using time integrator: {:}'.format(self.timestepper.__class__.__name__))
        self._isfrozen = True  # disallow creating new attributes

    def create_exporters(self):
        """
        Creates file exporters
        """
        if not hasattr(self, 'timestepper'):
            self.create_timestepper()
        self._isfrozen = False
        self.exporters = OrderedDict()
        if not self.options.no_exports:
            e = exporter.ExportManager(self.options.output_directory,
                                       self.options.fields_to_export,
                                       self.fields,
                                       field_metadata,
                                       export_type='vtk',
                                       verbose=self.options.verbose > 0)
            self.exporters['vtk'] = e
            hdf5_dir = os.path.join(self.options.output_directory, 'hdf5')
            e = exporter.ExportManager(hdf5_dir,
                                       self.options.fields_to_export_hdf5,
                                       self.fields,
                                       field_metadata,
                                       export_type='hdf5',
                                       verbose=self.options.verbose > 0)
            self.exporters['hdf5'] = e

        self._isfrozen = True  # disallow creating new attributes

    def initialize(self):
        """
        Creates function spaces, equations, time stepper and exporters
        """
        if not hasattr(self, 'U_2d'):
            self.create_function_spaces()
        if not hasattr(self, 'eq_sw'):
            self.create_equations()
        if not hasattr(self, 'timestepper'):
            self.create_timestepper()
        if not hasattr(self, 'exporters'):
            self.create_exporters()
        self._initialized = True

    def assign_initial_conditions(self, elev_2d=None, uv_2d=None, h_ls=None, uv_ls=None,
                                  salt=None, temp=None):
        """
        Assigns initial conditions

        :kwarg elev: Initial condition for water elevation
        :type elev: scalar :class:`Function`, :class:`Constant`, or an expression
        :kwarg uv: Initial condition for depth averaged velocity
        :type uv: vector valued :class:`Function`, :class:`Constant`, or an expression
        """
        if not self._initialized:
            self.initialize()

        if elev_2d is not None:
            self.fields.elev_2d.project(elev_2d)
        # prevent negative initial water depth
        if self.options.use_hllc_flux:
            h_2d_array = self.fields.elev_2d.dat.data + self.fields.bathymetry_2d.dat.data
            ind_2d = np.where(h_2d_array[:] <= 0)[0]
            self.fields.elev_2d.dat.data[ind_2d] = -self.fields.bathymetry_2d.dat.data[ind_2d]
            self.elev_2d_init.assign(self.fields.elev_2d)

        if uv_2d is not None:
            self.fields.uv_2d.project(uv_2d)
            self.fields.mom_2d.project((self.fields.elev_2d + self.fields.bathymetry_2d)*uv_2d)

        if self.landslide:
            if h_ls is not None:
                self.fields.h_ls.project(h_ls)
            if uv_ls is not None:
                self.fields.solution_ls.sub(1).project(self.fields.h_ls*uv_ls[0])
                self.fields.solution_ls.sub(2).project(self.fields.h_ls*uv_ls[1])

        if salt is not None and self.options.solve_salinity:
            self.fields.salt_3d.project(salt)
        if temp is not None and self.options.solve_temperature:
            self.fields.temp_3d.project(temp)
        # TODO add more

       # self.timestepper.initialize(self.fields.solution_2d)

    def add_callback(self, callback, eval_interval='export'):
        """
        Adds callback to solver object

        :arg callback: :class:`.DiagnosticCallback` instance
        :kwarg string eval_interval: Determines when callback will be evaluated,
            either 'export' or 'timestep' for evaluating after each export or
            time step.
        """
        self.callbacks.add(callback, eval_interval)

    def export(self):
        """
        Export all fields to disk

        Also evaluates all callbacks set to 'export' interval.
        """
        self.callbacks.evaluate(mode='export')
        for e in self.exporters.values():
            e.export()

    def load_state(self, i_export, outputdir=None, t=None, iteration=None):
        """
        Loads simulation state from hdf5 outputs.

        This replaces :meth:`.assign_initial_conditions` in model initilization.

        This assumes that model setup is kept the same (e.g. time step) and
        all pronostic state variables are exported in hdf5 format. The required
        state variables are: h_2d, hu_2d, hv_2d

        Currently hdf5 field import only works for the same number of MPI
        processes.

        :arg int i_export: export index to load
        :kwarg string outputdir: (optional) directory where files are read from.
            By default ``options.output_directory``.
        :kwarg float t: simulation time. Overrides the time stamp stored in the
            hdf5 files.
        :kwarg int iteration: Overrides the iteration count in the hdf5 files.
        """
        if not self._initialized:
            self.initialize()
        if outputdir is None:
            outputdir = self.options.output_directory
        # create new ExportManager with desired outputdir
        state_fields = ['uv_2d', 'elev_2d', 'uv_3d']
        hdf5_dir = os.path.join(outputdir, 'hdf5')
        e = exporter.ExportManager(hdf5_dir,
                                   state_fields,
                                   self.fields,
                                   field_metadata,
                                   export_type='hdf5',
                                   verbose=self.options.verbose > 0)
        e.exporters['uv_2d'].load(i_export, self.fields.uv_2d)
        e.exporters['elev_2d'].load(i_export, self.fields.elev_2d)
        e.exporters['uv_3d'].load(i_export, self.fields.uv_3d)
        self.assign_initial_conditions()

        # time stepper bookkeeping for export time step
        self.i_export = i_export
        self.next_export_t = self.i_export*self.options.simulation_export_time
        if iteration is None:
            iteration = int(np.ceil(self.next_export_t/self.dt))
        if t is None:
            t = iteration*self.dt
        self.iteration = iteration
        self.simulation_time = t

        # for next export
        self.export_initial_state = outputdir != self.options.output_directory
        if self.export_initial_state:
            offset = 0
        else:
            offset = 1
        self.next_export_t += self.options.simulation_export_time
        for e in self.exporters.values():
            e.set_next_export_ix(self.i_export + offset)

    def print_state(self, cputime):
        """
        Print a summary of the model state on stdout

        :arg float cputime: Measured CPU time
        """
        norm_h = norm(self.fields.elev_2d)
        norm_u = norm(self.fields.uv_3d)

        line = ('{iexp:5d} {i:5d} T={t:10.2f} '
                'eta norm: {e:10.4f} u norm: {u:10.4f} {cpu:5.2f}')
        print_output(line.format(iexp=self.i_export, i=self.iteration,
                                 t=self.simulation_time, e=norm_h,
                                 u=norm_u, cpu=cputime))
        sys.stdout.flush()

    def set_sponge_damping(self, length, sponge_start_point, alpha=10., sponge_is_2d=True):
        """
        Set damping terms to reduce the reflection on solid boundaries.
        """
        pi = 4*np.arctan(1.)
        if length == [0., 0.]:
            return None
        if sponge_is_2d:
            damping_coeff = Function(self.function_spaces.P1_2d)
        else:
            damping_coeff = Function(self.function_spaces.P1)
        damp_vector = damping_coeff.dat.data[:]
        mesh = damping_coeff.ufl_domain()
        xvector = mesh.coordinates.dat.data[:, 0]
        yvector = mesh.coordinates.dat.data[:, 1]
        assert xvector.shape[0] == damp_vector.shape[0]
        assert yvector.shape[0] == damp_vector.shape[0]
        if xvector.max() <= sponge_start_point[0] + length[0]:
            length[0] = xvector.max() - sponge_start_point[0]
        if yvector.max() <= sponge_start_point[1] + length[1]:
            length[1] = yvector.max() - sponge_start_point[1]

        if length[0] > 0.:
            for i, x in enumerate(xvector):
                x = (x - sponge_start_point[0])/length[0]
                if x > 0 and x < 0.5:
                    damp_vector[i] = alpha*0.25*(np.tanh(np.sin(pi*(2.*x - 0.5))/(1. - (4.*x - 1.)**2)) + 1.)
                elif x > 0.5 and x < 1.:
                    damp_vector[i] = alpha*0.25*(np.tanh(np.sin(pi*(1.5 - 2*x))/(1. - (3. - 4.*x)**2)) + 1.)
                else:
                    damp_vector[i] = 0.
        if length[1] > 0.:
            for i, y in enumerate(yvector):
                x = (y - sponge_start_point[1])/length[1]
                if x > 0 and x < 0.5:
                    damp_vector[i] = alpha*0.25*(np.tanh(np.sin(pi*(2.*x - 0.5))/(1. - (4.*x - 1.)**2)) + 1.)
                elif x > 0.5 and x < 1.:
                    damp_vector[i] = alpha*0.25*(np.tanh(np.sin(pi*(1.5 - 2*x))/(1. - (3. - 4.*x)**2)) + 1.)
                else:
                    damp_vector[i] = 0.

        return damping_coeff

    def iterate(self, update_forcings=None,
                export_func=None):
        """
        Runs the simulation

        Iterates over the time loop until time ``options.simulation_end_time`` is reached.
        Exports fields to disk on ``options.simulation_export_time`` intervals.

        :kwarg update_forcings: User-defined function that takes simulation
            time as an argument and updates time-dependent boundary conditions
            (if any).
        :kwarg export_func: User-defined function (with no arguments) that will
            be called on every export.
        """
        # TODO I think export function is obsolete as callbacks are in place
        if not self._initialized:
            self.initialize()

        self.options.check_salinity_conservation &= self.options.solve_salinity
        self.options.check_salinity_overshoot &= self.options.solve_salinity
        self.options.check_temperature_conservation &= self.options.solve_temperature
        self.options.check_temperature_overshoot &= self.options.solve_temperature
        self.options.check_volume_conservation_3d &= self.options.use_ale_moving_mesh
        self.options.use_limiter_for_tracers &= self.options.polynomial_degree > 0
        self.options.use_limiter_for_velocity &= self.options.polynomial_degree > 0
        self.options.use_limiter_for_velocity &= self.options.element_family == 'dg-dg'

        t_epsilon = 1.0e-5
        cputimestamp = time_mod.clock()
        next_export_t = self.simulation_time + self.options.simulation_export_time

        dump_hdf5 = self.options.export_diagnostics and not self.options.no_exports
        if self.options.check_volume_conservation_2d:
            c = callback.VolumeConservation2DCallback(self,
                                                      export_to_hdf5=dump_hdf5,
                                                      append_to_log=True)
            self.add_callback(c, eval_interval='export')
        if self.options.check_volume_conservation_3d:
            c = callback.VolumeConservation3DCallback(self,
                                                      export_to_hdf5=dump_hdf5,
                                                      append_to_log=True)
            self.add_callback(c, eval_interval='export')
        if self.options.check_salinity_conservation:
            c = callback.TracerMassConservationCallback('salt_3d',
                                                        self,
                                                        export_to_hdf5=dump_hdf5,
                                                        append_to_log=True)
            self.add_callback(c, eval_interval='export')
        if self.options.check_salinity_overshoot:
            c = callback.TracerOvershootCallBack('salt_3d',
                                                 self,
                                                 export_to_hdf5=dump_hdf5,
                                                 append_to_log=True)
            self.add_callback(c, eval_interval='export')
        if self.options.check_temperature_conservation:
            c = callback.TracerMassConservationCallback('temp_3d',
                                                        self,
                                                        export_to_hdf5=dump_hdf5,
                                                        append_to_log=True)
            self.add_callback(c, eval_interval='export')
        if self.options.check_temperature_overshoot:
            c = callback.TracerOvershootCallBack('temp_3d',
                                                 self,
                                                 export_to_hdf5=dump_hdf5,
                                                 append_to_log=True)
            self.add_callback(c, eval_interval='export')

        if self._simulation_continued:
            # set all callbacks to append mode
            for m in self.callbacks:
                for k in self.callbacks[m]:
                    self.callbacks[m][k].set_write_mode('append')

        # initial export
        self.print_state(0.0)
        if self.export_initial_state:
            self.export()
            if export_func is not None:
                export_func()
            if 'vtk' in self.exporters and isinstance(self.fields.bathymetry_2d, Function):
                self.exporters['vtk'].export_bathymetry(self.fields.bathymetry_2d)

        initial_simulation_time = self.simulation_time
        internal_iteration = 0

        # solver for advancing 3d momentum equations
        solution_3d = self.fields.uv_3d
        solution_mid = self.uv_3d_mid
        if self.options.solve_conservative_momentum:
            solution_3d = self.fields.mom_3d
            solution_mid = self.mom_3d_mid
        a_mom = self.eq_momentum.mass_term(self.eq_momentum.trial)
        l_mom = (self.eq_momentum.mass_term(solution_3d) + Constant(self.dt)*
                self.eq_momentum.residual('all', solution_3d, solution_3d, 
                                          self.fields_3d, self.fields_3d, self.bnd_functions['momentum'])
                )
        prob_mom = LinearVariationalProblem(a_mom, l_mom, solution_mid)
        solver_mom = LinearVariationalSolver(prob_mom, solver_parameters=self.options.timestepper_options.solver_parameters_momentum_explicit)

        # solver for advancing 3d tracer equation
        if self.options.solve_salinity:
            a_salt = self.eq_salt.mass_term(self.eq_salt.trial)
            l_salt = self.eq_salt.mass_term(self.fields.salt_3d) + \
                     self.dt*self.eq_salt.residual('all', self.fields.salt_3d, self.fields.salt_3d, self.fields_3d, self.fields_3d, self.bnd_functions['salt'])
            prob_salt = LinearVariationalProblem(a_salt, l_salt, self.salt_3d_mid)
            solver_salt = LinearVariationalSolver(prob_salt, solver_parameters=self.options.timestepper_options.solver_parameters_tracer_explicit)
        if self.options.solve_temperature:
            a_temp = self.eq_temp.mass_term(self.eq_temp.trial)
            l_temp = self.eq_temp.mass_term(self.fields.temp_3d) + \
                     self.dt*self.eq_temp.residual('all', self.fields.temp_3d, self.fields.temp_3d, self.fields_3d, self.fields_3d, self.bnd_functions['temp'])
            prob_temp = LinearVariationalProblem(a_temp, l_temp, self.temp_3d_mid)
            solver_temp = LinearVariationalSolver(prob_temp, solver_parameters=self.options.timestepper_options.solver_parameters_tracer_explicit)

        if self.options.flow_is_granular:
            dt_ls = self.dt / self.options.n_dt
            # solver for granular landslide motion
            a_ls = self.eq_ls.mass_term(self.eq_ls.trial)
            l_ls = (self.eq_ls.mass_term(self.fields.solution_ls) + Constant(dt_ls)*
                    self.eq_ls.residual('all', self.fields.solution_ls, self.fields.solution_ls,
                                        self.fields_ls, self.fields_ls, self.bnd_functions['landslide_motion'])
                   )
            prob_ls = LinearVariationalProblem(a_ls, l_ls, self.solution_ls_tmp)
            solver_ls = LinearVariationalSolver(prob_ls, solver_parameters=self.options.timestepper_options.solver_parameters_granular_explicit)
            # solver for div(velocity)
            h_ls, hu_ls, hv_ls = self.fields.solution_ls.split()
            u_ls = conditional(h_ls <= 0, zero(hu_ls.ufl_shape), hu_ls/h_ls)
            v_ls = conditional(h_ls <= 0, zero(hv_ls.ufl_shape), hv_ls/h_ls)
            tri_div = TrialFunction(self.uv_div_ls.function_space())
            test_div = TestFunction(self.uv_div_ls.function_space())
            a_div = tri_div*test_div*dx
            l_div = (Dx(u_ls, 0) + Dx(v_ls, 1))*test_div*dx
            prob_div = LinearVariationalProblem(a_div, l_div, self.uv_div_ls)
            solver_div = LinearVariationalSolver(prob_div)
            # solver for strain rate
            l_sr = 0.5*(Dx(u_ls, 1) + Dx(v_ls, 0))*test_div*dx
            prob_sr = LinearVariationalProblem(a_div, l_sr, self.strain_rate_ls)
            solver_sr = LinearVariationalSolver(prob_sr)
            # solver for fluid pressure at slide surface
            h_2d = self.fields.bathymetry_2d + self.fields.elev_2d
            tri_pf = TrialFunction(self.grad_p.function_space())
            test_pf = TestFunction(self.grad_p.function_space())
            a_pf = dot(tri_pf, test_pf)*dx
            l_pf = dot(conditional(h_2d <= 0, zero(self.grad_p.ufl_shape), 
                       grad(self.options.rho_fluid*physical_constants['g_grav']*h_2d + self.fields.q_2d)), test_pf)*dx
            prob_pf = LinearVariationalProblem(a_pf, l_pf, self.grad_p)
            solver_pf = LinearVariationalSolver(prob_pf)

        if True:
            h_3d = self.fields.elev_3d + self.fields.bathymetry_3d
            h_3d_old = self.elev_3d_old + self.bathymetry_3d_old
            alpha = self.options.depth_wd_interface

            if self.options.use_hllc_flux and self.options.use_wetting_and_drying:
                h_total = conditional(h_3d <= alpha, alpha, h_3d)
                h_total_old = conditional(h_3d_old <= alpha, alpha, h_3d_old)
            elif self.options.use_wetting_and_drying:
                h_total = 2 * alpha**2 / (2 * alpha + abs(h_3d)) + 0.5 * (abs(h_3d) + h_3d)
                h_total_old = 2 * alpha**2 / (2 * alpha + abs(h_3d_old)) + 0.5 * (abs(h_3d_old) + h_3d_old)
            else:
                h_total = h_3d
                h_total_old = h_3d_old

            uv_3d = self.fields.uv_3d#conditional(h_3d <= alpha, zero(self.fields.mom_3d.ufl_shape), self.fields.uv_3d/h_3d)
            sigma_coord = self.sigma_coord
            z_in_sigma = sigma_coord*h_total + ( - 1.)*self.fields.bathymetry_3d
            z_in_sigma_old = sigma_coord*h_total_old + ( - 1.)*self.bathymetry_3d_old
            # solvers for terms in omega
            tri_omega = TrialFunction(self.z_in_sigma.function_space())
            test_omega = TestFunction(self.z_in_sigma.function_space())
            sigma_dz = 1./h_total#conditional(h_3d <= alpha, zero(h_total.ufl_shape), 1./h_total)
            sigma_dx = -sigma_dz*Dx(z_in_sigma, 0)
            sigma_dy = -sigma_dz*Dx(z_in_sigma, 1)
            a_omega = tri_omega*test_omega*dx
            l_sigma_dt = -sigma_dz*(z_in_sigma - z_in_sigma_old)/self.dt*test_omega*dx
            l_sigma_dx = sigma_dx*test_omega*dx
            l_sigma_dy = sigma_dy*test_omega*dx
            l_omega = (self.fields.sigma_dt + uv_3d[0]*self.fields.sigma_dx + uv_3d[1]*self.fields.sigma_dy + uv_3d[2]*sigma_dz)*test_omega*dx
            l_sigma_dxyz = (sigma_dx**2 + sigma_dy**2 + sigma_dz**2)*test_omega*dx
            prob_sigma_dt = LinearVariationalProblem(a_omega, l_sigma_dt, self.fields.sigma_dt)
            prob_sigma_dx = LinearVariationalProblem(a_omega, l_sigma_dx, self.fields.sigma_dx)
            prob_sigma_dy = LinearVariationalProblem(a_omega, l_sigma_dy, self.fields.sigma_dy)
            prob_omega = LinearVariationalProblem(a_omega, l_omega, self.fields.omega)
            prob_sigma_dxyz = LinearVariationalProblem(a_omega, l_sigma_dxyz, self.sigma_dxyz)
            solver_sigma_dt = LinearVariationalSolver(prob_sigma_dt)
            solver_sigma_dx = LinearVariationalSolver(prob_sigma_dx)
            solver_sigma_dy = LinearVariationalSolver(prob_sigma_dy)
            solver_omega = LinearVariationalSolver(prob_omega)
            solver_sigma_dxyz = LinearVariationalSolver(prob_sigma_dxyz)

            # solver for the Poisson equation
            q_3d = self.fields.q_3d
            test_q = TestFunction(q_3d.function_space())
            lhs = -Dx(test_q, 0)*(Dx(q_3d, 0) + Dx(q_3d, 2)*sigma_dx)*dx - Dx(test_q, 1)*(Dx(q_3d, 1) + Dx(q_3d, 2)*sigma_dy)*dx - \
                   (sigma_dx**2 + sigma_dy**2 + sigma_dz**2)*Dx(test_q, 2)*Dx(q_3d, 2)*dx - \
                   Dx(sigma_dx*test_q, 2)*Dx(q_3d, 0)*dx - Dx(sigma_dy*test_q, 2)*Dx(q_3d, 1)*dx - \
                   sigma_dz*(Dx(h_total, 0)*sigma_dx + Dx(h_total, 1)*sigma_dy)*Dx(q_3d, 2)*test_q*dx
            rhs = -rho_0/self.dt*(Dx(test_q, 0)*uv_3d[0] + Dx(sigma_dx*test_q, 2)*uv_3d[0] +
                                  Dx(test_q, 1)*uv_3d[1] + Dx(sigma_dy*test_q, 2)*uv_3d[1] + Dx(sigma_dz*test_q, 2)*uv_3d[2])*dx
            if self.landslide:
                rhs += rho_0/self.dt*sigma_dz*self.fields.slide_source_3d*self.normal[2]*test_q*ds_bottom
            F = lhs - rhs
            # boundary conditions: to refer to the top and bottom use "top" and "bottom"
            # for other boundaries use the normal numbers (ids) from the horizontal mesh
            # (UnitSquareMesh automatically defines 1,2,3, and 4)
            bc_top = DirichletBC(q_3d.function_space(), 0., "top")
            bcs = [bc_top]
            if not self.options.update_free_surface:
                bcs = []
            for bnd_marker in self.boundary_markers:
                func = self.bnd_functions['shallow_water'].get(bnd_marker)
                if func is not None: #TODO set more general and accurate conditional statement
                    bc = DirichletBC(q_3d.function_space(), 0., int(bnd_marker))
                    bcs.append(bc)
            prob = NonlinearVariationalProblem(F, q_3d, bcs=bcs)
            solver_q = NonlinearVariationalSolver(prob,
                                                  solver_parameters={'snes_type': 'ksponly',#'newtonls''ksponly', final: 'ksponly'
                                                                     'ksp_type': 'gmres',#'gmres''preonly',              'gmres'
                                                                     'pc_type': 'gamg'},#'ilu''gamg',                     'ilu'
                                                  bcs=bcs,
                                                  options_prefix='poisson_solver')

            # solver for updating uv_3d
            q_3d_mod = self.fields.q_3d#conditional(h_3d <= 0, 0, self.fields.q_3d)
            tri_uv_3d = TrialFunction(self.function_spaces.U)
            test_uv_3d = TestFunction(self.function_spaces.U)
            a_u = dot(tri_uv_3d, test_uv_3d)*dx
            l_u = dot(self.fields.uv_3d, test_uv_3d)*dx - self.dt/rho_0*((Dx(q_3d_mod, 0) + Dx(q_3d_mod, 2)*sigma_dx)*test_uv_3d[0] +
                                                                         (Dx(q_3d_mod, 1) + Dx(q_3d_mod, 2)*sigma_dy)*test_uv_3d[1] +
                                                                         (Dx(q_3d_mod, 2)*sigma_dz)*test_uv_3d[2])*dx
            prob_u = LinearVariationalProblem(a_u, l_u, self.fields.uv_3d)
            solver_u = LinearVariationalSolver(prob_u)

        # solver for advancing 2d shallow water equations
     #   a_sw = self.eq_sw.mass_term(self.eq_sw.trial)
      #  l_sw = (self.eq_sw.mass_term(self.fields.solution_2d) + Constant(self.dt)*
       #         self.eq_sw.residual('all', self.fields.solution_2d, self.fields.solution_2d, 
        #                            self.fields_sw, self.fields_sw, self.bnd_functions['shallow_water'])
         #      )
       # prob_sw = LinearVariationalProblem(a_sw, l_sw, self.solution_2d_tmp)
        #solver_sw = LinearVariationalSolver(prob_sw, solver_parameters=self.options.timestepper_options.solver_parameters)

        # solver to update free surface
        if self.options.use_hllc_flux:
            a_fs = self.eq_free_surface.mass_term(self.eq_free_surface.trial)
            l_fs = (self.eq_free_surface.mass_term(self.fields.elev_2d) + Constant(self.dt)*
                        self.eq_free_surface.residual('all', self.fields.elev_2d, self.fields.elev_2d, 
                                                      self.fields_sw, self.fields_sw, self.bnd_functions['shallow_water'])
                   )
            prob_fs = LinearVariationalProblem(a_fs, l_fs, self.elev_2d_fs)
            solver_fs = LinearVariationalSolver(prob_fs, solver_parameters=self.options.timestepper_options.solver_parameters_granular_explicit)

        timestepper_free_surface_crank = timeintegrator.CrankNicolson(self.eq_free_surface, self.elev_2d_fs,
                                                              self.fields_sw, self.dt,
                                                              bnd_conditions=self.bnd_functions['shallow_water'],
                                                              # solver_parameters=solver_parameters,
                                                              semi_implicit=False,
                                                              theta=1.0)
       # timestepper_free_surface_ssprk = rungekutta.SSPRK33(self.eq_free_surface, self.elev_2d_fs,
        #                                                      self.fields_sw, self.dt,
         #                                                     bnd_conditions=self.bnd_functions['shallow_water'],
          #                                                    solver_parameters=self.options.timestepper_options.solver_parameters_granular_explicit)

        while self.simulation_time <= self.options.simulation_end_time - t_epsilon:

            if self.options.timestepper_type == 'SSPRK22':
                n_stages = 2
                coeff = [[0., 1.], [1./2., 1./2.]]

            self.uv_3d_old.assign(self.fields.uv_3d)
            self.mom_3d_old.assign(self.fields.mom_3d)
            self.elev_3d_old.assign(self.fields.elev_3d)
            self.bathymetry_3d_old.assign(self.fields.bathymetry_3d)
            self.elev_2d_fs.assign(self.fields.elev_2d)
            self.elev_2d_old.assign(self.fields.elev_2d)

            if self.options.solve_salinity:
                self.salt_3d_old.assign(self.fields.salt_3d)
            if self.options.solve_temperature:
                self.temp_3d_old.assign(self.fields.temp_3d)

            if self.options.slide_is_rigid:
                self.h_ls_old.assign(self.fields.h_ls)
            if self.options.flow_is_granular:
                self.solution_ls_old.assign(self.fields.solution_ls)
                self.solution_ls_mid.assign(self.fields.solution_ls)

            h_2d_array = self.fields.elev_2d.dat.data + self.fields.bathymetry_2d.dat.data
            h_3d_array = self.fields.elev_3d.dat.data + self.fields.bathymetry_3d.dat.data

            couple_granular_and_wave_in_ssprk = not True
            solve_hydrostatic_eq = not True
            if (not couple_granular_and_wave_in_ssprk):
                if self.options.flow_is_granular:
                    if not self.options.lamda == 0.:
                        self.h_2d_cg.project(self.fields.bathymetry_2d + self.fields.elev_2d)
                        self.h_2d_ls.dat.data[:] = self.h_2d_cg.dat.data[:]
                    for i in range(self.options.n_dt):
                        # solve fluid pressure on slide
                       # self.extract_bot_q.solve()
                        self.fields.bathymetry_2d.dat.data[:] = self.bathymetry_init.dat.data[:] - self.fields.h_ls.dat.data[:]/self.slope.dat.data.min()
                       # solver_pf.solve()
                       # self.grad_p_ls.dat.data[:] = self.grad_p.dat.data[:]

                        self.solution_ls_mid.assign(self.fields.solution_ls)
                        for i_stage in range(n_stages):
                            #self.timestepper.solve_stage(i_stage, self.simulation_time, update_forcings)
                            solver_ls.solve()
                            self.fields.solution_ls.assign(coeff[i_stage][0]*self.solution_ls_mid + coeff[i_stage][1]*self.solution_ls_tmp)
                            if self.options.use_wetting_and_drying:
                                limiter_start_time = 0.
                                limiter_end_time = self.options.simulation_end_time - t_epsilon
                                use_limiter = self.options.use_limiter_for_granular and self.simulation_time >= limiter_start_time and self.simulation_time <= limiter_end_time
                                self.wd_modification_ls.apply(self.fields.solution_ls, self.options.wetting_and_drying_threshold, use_limiter)
                            solver_div.solve()
                           # solver_sr.solve()

                if self.landslide:
                    # update landslide motion source
                    if update_forcings is not None:
                        update_forcings(self.simulation_time + self.dt)

                    ind_wet_2d = np.where(h_2d_array[:] > 0)[0]
                    if self.simulation_time >= 0.:
                        self.fields.slide_source_2d.assign(0.)
                        self.fields.slide_source_2d.dat.data[ind_wet_2d] = (self.fields.h_ls.dat.data[ind_wet_2d] 
                                                                            - self.h_ls_old.dat.data[ind_wet_2d])/self.dt/self.slope.dat.data.min()
                    # copy slide source to 3d
                    self.copy_slide_source_to_3d.solve()

                    # update bathymetry
                    self.fields.bathymetry_2d.dat.data[:] = self.bathymetry_init.dat.data[:] - self.fields.h_ls.dat.data[:]/self.slope.dat.data.min()
                    self.copy_bath_to_3d.solve()

                    if self.options.use_hllc_flux:
                        # restore water depth without landslide
                        ind_ls_2d = np.where(h_2d_array[:] <= 0)[0]
                        self.fields.elev_2d.dat.data[ind_ls_2d] = self.elev_2d_init.dat.data[ind_ls_2d]
                        # update elevation to kepp positive water depth
                        ind_dry_2d = np.where(h_2d_array[:] <= 0)[0]
                        self.fields.elev_2d.dat.data[ind_dry_2d] = -self.fields.bathymetry_2d.dat.data[ind_dry_2d]

                        self.elev_2d_old.assign(self.fields.elev_2d)
                        self.elev_2d_fs.assign(self.fields.elev_2d)
                        self.copy_elev_to_3d.solve()

            if self.options.solve_conservative_momentum and (not solve_hydrostatic_eq) and (not self.options.no_wave_flow):
                assert (not couple_granular_and_wave_in_ssprk)
               # solver_sigma_dt.solve()
                solver_sigma_dx.solve()
                solver_sigma_dy.solve()
                solver_omega.solve()
               # solver_sigma_dxyz.solve()

                for i_stage in range(n_stages):
                    #self.timestepper.solve_stage(i_stage, self.simulation_time, update_forcings)
                    solver_mom.solve()
                    self.fields.mom_3d.assign(coeff[i_stage][0]*self.mom_3d_old + coeff[i_stage][1]*self.mom_3d_mid)
                    if self.options.use_limiter_for_velocity:
                        self.uv_limiter.apply(self.fields.mom_3d)

                    # update velocity
                    ind_wet_3d = np.where(h_3d_array[:] > alpha)[0]
                    self.fields.uv_3d.assign(0.)
                    for i in range(3):
                        self.fields.uv_3d.dat.data[ind_wet_3d, i] = self.fields.mom_3d.dat.data[ind_wet_3d, i] / h_3d_array[ind_wet_3d]

                   # solver_sigma_dt.solve()
                    solver_sigma_dx.solve()
                    solver_sigma_dy.solve()
                    solver_omega.solve()
                   # solver_sigma_dxyz.solve()

              #  if True:
                    # solve nh pressure
                    solver_q.solve()
                    # update velocity
                    solver_u.solve()

                    # update momentum
                    self.fields.mom_3d.assign(0.)
                    for i in range(3):
                        self.fields.mom_3d.dat.data[ind_wet_3d, i] = self.fields.uv_3d.dat.data[ind_wet_3d, i] * h_3d_array[ind_wet_3d]

                    # update free surface elevation
                    if self.options.update_free_surface and i_stage == 1:
                        self.uv_averager.solve() # mom_3d -> uv_dav_3d
                        self.extract_surf_dav_uv.solve() # uv_dav_3d -> mom_2d

                        if self.options.use_hllc_flux:
                            self.fields.elev_2d.assign(self.elev_2d_old)
                            for i_stage in range(n_stages):
                                solver_fs.solve()
                                self.fields.elev_2d.assign(coeff[i_stage][0]*self.elev_2d_old + coeff[i_stage][1]*self.elev_2d_fs)
                        else:
                            self.fields.uv_2d.assign(0.)
                            ind_wet_2d = np.where(h_2d_array[:] > alpha)[0]
                            for i in range(2):
                                self.fields.uv_2d.dat.data[ind_wet_2d, i] = self.fields.mom_2d.dat.data[ind_wet_2d, i] / h_2d_array[ind_wet_2d]
                            self.elev_2d_fs.assign(self.elev_2d_old)
                            timestepper_free_surface_crank.advance(self.simulation_time, update_forcings)
                            self.fields.elev_2d.assign(self.elev_2d_fs)

                        if self.options.use_limiter_for_elevation:
                            if self.limiter_h is not None:
                                self.limiter_h.apply(self.fields.elev_2d)

                        if self.options.use_hllc_flux and self.options.use_wetting_and_drying:
                            self.wd_modification.apply(self.fields.solution_2d, alpha, 
                                                       use_limiter=False, use_eta_solution=True, bathymetry=self.fields.bathymetry_2d)
                        self.copy_elev_to_3d.solve()
                        solver_sigma_dt.solve()

                    # prevent negative water depth
                   # ind_dry_3d = np.where(h_3d_array[:] <= 0)[0]
                   # ind_dry_2d = np.where(h_2d_array[:] <= 0)[0]
                   # self.fields.elev_2d.dat.data[ind_dry_2d] = - self.fields.bathymetry_2d.dat.data[ind_dry_2d]
                   # self.fields.elev_3d.dat.data[ind_dry_3d] = - self.fields.bathymetry_3d.dat.data[ind_dry_3d]
                   # self.fields.mom_3d.dat.data[ind_dry_3d] = [0., 0., 0.]

                    solver_sigma_dx.solve()
                    solver_sigma_dy.solve()
                    solver_omega.solve()
                   # solver_sigma_dxyz.solve()

                if self.options.solve_salinity:
                    for i_stage in range(n_stages):
                        solver_salt.solve()
                        self.fields.salt_3d.assign(coeff[i_stage][0]*self.salt_3d_old + coeff[i_stage][1]*self.salt_3d_mid)
                        if self.options.use_limiter_for_tracers:
                            self.tracer_limiter.apply(self.fields.salt_3d)
                if self.options.solve_temperature:
                    for i_stage in range(n_stages):
                        solver_temp.solve()
                        self.fields.temp_3d.assign(coeff[i_stage][0]*self.temp_3d_old + coeff[i_stage][1]*self.temp_3d_mid)
                        if self.options.use_limiter_for_tracers:
                            self.tracer_limiter.apply(self.fields.temp_3d)

                # update baroclinicity
                if self.options.use_baroclinic_formulation:
                    compute_baroclinic_head(weakref.proxy(self))

            elif (not self.options.no_wave_flow) and (not solve_hydrostatic_eq):

                for i_stage in range(n_stages):
                    # advance granular landslide motion
                    if couple_granular_and_wave_in_ssprk and self.options.flow_is_granular:
                        if not self.options.lamda == 0.:
                            self.h_2d_cg.project(self.fields.bathymetry_2d + self.fields.elev_2d)
                            self.h_2d_ls.dat.data[:] = self.h_2d_cg.dat.data[:]

                        # solve fluid pressure on slide
                       # self.extract_bot_q.solve()
                        self.fields.bathymetry_2d.dat.data[:] = self.bathymetry_init.dat.data[:] - self.fields.h_ls.dat.data[:]/self.slope.dat.data.min()
                        solver_pf.solve()
                        self.grad_p_ls.dat.data[:] = self.grad_p.dat.data[:]

                        solver_ls.solve()
                        self.fields.solution_ls.assign(coeff[i_stage][0]*self.solution_ls_mid + coeff[i_stage][1]*self.solution_ls_tmp)

                        if self.options.use_wetting_and_drying:
                            limiter_start_time = 0.
                            limiter_end_time = self.options.simulation_end_time - t_epsilon
                            use_limiter = self.options.use_limiter_for_granular and self.simulation_time >= limiter_start_time and self.simulation_time <= limiter_end_time
                            self.wd_modification_ls.apply(self.fields.solution_ls, self.options.wetting_and_drying_threshold, use_limiter)
                        solver_div.solve()

                        ind_wet_2d = np.where(h_2d_array[:] > 0)[0]
                        if self.simulation_time >= 0.:
                            self.fields.slide_source_2d.assign(0.)
                            self.fields.slide_source_2d.dat.data[ind_wet_2d] = (self.fields.h_ls.dat.data[ind_wet_2d] 
                                                                                 - self.h_ls_old.dat.data[ind_wet_2d])/self.dt/self.slope.dat.data.min()
                        # copy slide source to 3d
                        self.copy_slide_source_to_3d.solve()

                        # update bathymetry
                        self.fields.bathymetry_2d.dat.data[:] = self.bathymetry_init.dat.data[:] - self.fields.h_ls.dat.data[:]/self.slope.dat.data.min()
                        self.copy_bath_to_3d.solve()

                    # 2d advance
                    if self.options.solve_separate_elevation_gradient and self.options.update_free_surface:
                        if i_stage == 1:
                            self.uv_averager.solve() # uv_3d -> uv_dav_3d
                            self.extract_surf_dav_uv.solve() # uv_dav_3d -> uv_2d
                            self.fields.elev_2d.assign(self.elev_2d_old)
                            self.copy_uv_to_uv_dav_3d.solve()
                            self.uv_dav_3d_mid.assign(self.fields.uv_dav_3d)
                            self.timestepper_operator_split.advance(self.simulation_time, update_forcings)
                            if self.options.use_limiter_for_elevation:
                                if self.limiter_h is not None:
                                    self.limiter_h.apply(self.fields.elev_2d)
                                if self.limiter_u is not None:
                                    self.limiter_u.apply(self.fields.uv_2d)
                            self.copy_elev_to_3d.solve()

                    # 3d advance
                    solver_sigma_dx.solve()
                    solver_sigma_dy.solve()
                    solver_omega.solve()

                    solver_mom.solve()
                    self.fields.uv_3d.assign(self.uv_3d_mid)
                    if self.options.use_limiter_for_velocity:
                        self.uv_limiter.apply(self.fields.uv_3d)

                    # wetting and drying treatment
                    if self.options.use_wetting_and_drying:
                        ind_dry_3d = np.where(h_3d_array[:] <= 0)[0]
                        self.fields.uv_3d.dat.data[ind_dry_3d] = [0, 0, 0]

                    if i_stage == 1 and self.options.solve_separate_elevation_gradient:
                        self.fields.uv_3d.assign(0.5*(self.uv_3d_old + self.fields.uv_3d))
                        # update 2d coupling, i.e. including the elevation gradient contribution
                        if self.options.update_free_surface:
                            self.copy_uv_to_uv_dav_3d.solve()
                            self.fields.uv_3d.assign(self.fields.uv_3d - (self.uv_dav_3d_mid - self.fields.uv_dav_3d))

                        # solve nh pressure
                        solver_q.solve()
                        # update velocity
                        solver_u.solve()
                        # wetting and drying treatment
                        if self.options.use_wetting_and_drying:
                            ind_dry_3d = np.where(h_3d_array[:] <= 0)[0]
                            self.fields.uv_3d.dat.data[ind_dry_3d] = [0, 0, 0]
                        # update free surface elevation
                        if self.options.update_free_surface:
                            self.uv_averager.solve()
                            self.extract_surf_dav_uv.solve()
                            self.elev_2d_fs.assign(self.elev_2d_old)
                            self.timestepper_free_surface_implicit.advance(self.simulation_time, update_forcings)
                            self.fields.elev_2d.assign(self.elev_2d_fs)
                            if self.options.use_limiter_for_elevation:
                                if self.limiter_h is not None:
                                    self.limiter_h.apply(self.fields.elev_2d)
                            self.copy_elev_to_3d.solve()
                            solver_sigma_dt.solve()

                    if (not self.options.solve_separate_elevation_gradient):
                        # solve nh pressure
                        solver_q.solve()
                        # update velocity
                        solver_u.solve()
                        # wetting and drying treatment
                        if self.options.use_wetting_and_drying:
                            ind_dry_3d = np.where(h_3d_array[:] <= 0)[0]
                            self.fields.uv_3d.dat.data[ind_dry_3d] = [0, 0, 0]
                        if i_stage == 1:
                            self.fields.uv_3d.assign(0.5*(self.uv_3d_old + self.fields.uv_3d))
                        # update free surface elevation
                        if self.options.update_free_surface:
                            self.uv_averager.solve()
                            self.extract_surf_dav_uv.solve()
                            self.elev_2d_fs.assign(self.elev_2d_old)
                            self.timestepper_free_surface_explicit.advance(self.simulation_time, update_forcings)
                            self.fields.elev_2d.assign(self.elev_2d_fs)
                            if self.options.use_limiter_for_elevation:
                                if self.limiter_h is not None:
                                    self.limiter_h.apply(self.fields.elev_2d)
                            self.copy_elev_to_3d.solve()
                            solver_sigma_dt.solve()

            elif (not self.options.no_wave_flow):
                self.timestepper_sw.advance(self.simulation_time, update_forcings)
                if self.options.use_limiter_for_elevation:
                    if self.limiter_h is not None:
                        self.limiter_h.apply(self.fields.elev_2d)
                    if self.limiter_u is not None:
                        self.limiter_u.apply(self.fields.uv_2d)

            # Move to next time step
            self.iteration += 1
            internal_iteration += 1
            self.simulation_time = initial_simulation_time + internal_iteration*self.dt

            self.callbacks.evaluate(mode='timestep')

            # Write the solution to file
            if self.simulation_time >= next_export_t - t_epsilon:
                self.i_export += 1
                next_export_t += self.options.simulation_export_time

                cputime = time_mod.clock() - cputimestamp
                cputimestamp = time_mod.clock()
                self.print_state(cputime)

                # exporter with wetting-drying handle
                if self.options.use_wetting_and_drying:
                    self.solution_2d_tmp.assign(self.fields.solution_2d)
                    ind = np.where(h_2d_array[:] <= 1E-6)[0]
                    self.fields.elev_2d.dat.data[ind] = 1E-6 - self.fields.bathymetry_2d.dat.data[ind]
                if self.options.flow_is_granular:
                    self.solution_ls_tmp.assign(self.fields.solution_ls)
                    ind = np.where(self.fields.h_ls.dat.data[:] > 1E-6)[0]
                    self.fields.hu_ls.dat.data[ind] = self.fields.hu_ls.dat.data[ind] / self.fields.h_ls.dat.data[ind] # TODO note here
                self.export()
                if self.options.use_wetting_and_drying:
                    self.fields.solution_2d.assign(self.solution_2d_tmp)
                if self.options.flow_is_granular:
                    self.fields.solution_ls.assign(self.solution_ls_tmp)

                if export_func is not None:
                    export_func()
