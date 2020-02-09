"""
Module for vertical two dimensional solver in sigma mesh
"""
from __future__ import absolute_import
from . import shallowwater_nh
from . import landslide_motion
from . import momentum_sigma
from . import tracer_sigma
from . import coupled_timeintegrator_nh
from .utility_nh import *
from .. import turbulence
from .. import timeintegrator
from .. import rungekutta
import thetis.limiter as limiter
import time as time_mod
from mpi4py import MPI
from .. import exporter
import weakref
from ..field_defs import field_metadata
from ..options import ModelOptions3d
from .. import callback
from ..log import *
from collections import OrderedDict

g_grav = physical_constants['g_grav']
rho_0 = physical_constants['rho0']

class FlowSolver(FrozenClass):
    """
    Main object for 3D solver

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

        solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, n_layers=6)
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
                 options=None, extrude_options=None):
        """
        :arg mesh2d: :class:`Mesh` object of the 2D mesh
        :arg bathymetry_2d: Bathymetry of the domain. Bathymetry stands for
            the mean water depth (positive downwards).
        :type bathymetry_2d: 2D :class:`Function`
        :arg int n_layers: Number of layers in the vertical direction.
            Elements are distributed uniformly over the vertical.
        :kwarg options: Model options (optional). Model options can also be
            changed directly via the :attr:`.options` class property.
        :type options: :class:`.ModelOptions3d` instance
        """
        self._initialized = False

        self.bathymetry_cg_2d = bathymetry_2d

        self.mesh2d = mesh2d
        """2D :class`Mesh`"""
        if extrude_options is None:
            extrude_options = {}
       # self.mesh = extrude_mesh_sigma(mesh2d, n_layers, bathymetry_2d, **extrude_options)
        self.mesh = ExtrudedMesh(mesh2d, layers=n_layers, layer_height=1.0/n_layers) # for sigma mesh formulation
        self.horizontal_domain_is_2d = self.mesh2d.geometric_dimension() == 2

        self.normal_2d = FacetNormal(self.mesh2d)
        self.normal = FacetNormal(self.mesh)
        self.boundary_markers = self.mesh.exterior_facets.unique_markers
        self.n_layers = n_layers
        """3D :class`Mesh`"""
        self.comm = mesh2d.comm

        # add boundary length info
        bnd_len = compute_boundary_length(self.mesh2d)
        self.mesh2d.boundary_len = bnd_len
        self.mesh.boundary_len = bnd_len

        # override default options
        self.options = ModelOptions3d()
        """
        Dictionary of all options. A :class:`.ModelOptions3d` object.
        """
        if options is not None:
            self.options.update(options)

        self.dt = self.options.timestep
        """Time step"""
        self.dt_2d = self.options.timestep_2d
        """Time of the 2D solver"""
        self.M_modesplit = None
        """Mode split ratio (int)"""

        # simulation time step bookkeeping
        self.simulation_time = 0
        self.iteration = 0
        self.i_export = 0
        self.next_export_t = self.simulation_time + self.options.simulation_export_time

        self.bnd_functions = {'shallow_water': {},
                              'landslide_motion': {},
                              'momentum': {},
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

    def compute_dx_factor(self):
        """
        Computes normalized distance between nodes in the horizontal direction

        The factor depends on the finite element space and its polynomial
        degree. It is used to compute maximal stable time steps.
        """
        p = self.options.polynomial_degree
        if self.options.element_family == 'rt-dg':
            # velocity space is essentially p+1
            p = self.options.polynomial_degree + 1
        # assuming DG basis functions on triangles
        l_r = p**2/3.0 + 7.0/6.0*p + 1.0
        factor = 0.5*0.25/l_r
        return factor

    def compute_dz_factor(self):
        """
        Computes a normalized distance between nodes in the vertical direction

        The factor depends on the finite element space and its polynomial
        degree. It is used to compute maximal stable time steps.
        """
        p = self.options.polynomial_degree
        # assuming DG basis functions in an interval
        l_r = 1.0/max(p, 1)
        factor = 0.5*0.25/l_r
        return factor

    def compute_dt_2d(self, u_scale):
        r"""
        Computes maximum explicit time step from CFL condition.

        .. math :: \Delta t = \frac{\Delta x}{U}

        Assumes velocity scale :math:`U = \sqrt{g H} + U_{scale}` where
        :math:`U_{scale}` is estimated advective velocity.

        :arg u_scale: User provided maximum advective velocity scale
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
        dt = float(solution.dat.data.min())
        dt = self.comm.allreduce(dt, op=MPI.MIN)
        dt *= self.compute_dx_factor()
        return dt

    def compute_dt_h_advection(self, u_scale):
        r"""
        Computes maximum explicit time step for horizontal advection

        .. math :: \Delta t = \frac{\Delta x}{U_{scale}}

        where :math:`U_{scale}` is estimated horizontal advective velocity.

        :arg u_scale: User provided maximum horizontal velocity scale
        :type u_scale: float or :class:`Constant`
        """
        u = u_scale
        if isinstance(u_scale, FiredrakeConstant):
            u = u_scale.dat.data[0]
        min_dx = self.fields.h_elem_size_2d.dat.data.min()
        # alpha = 0.5 if self.options.element_family == 'rt-dg' else 1.0
        # dt = alpha*1.0/10.0/(self.options.polynomial_degree + 1)*min_dx/u
        min_dx *= self.compute_dx_factor()
        dt = min_dx/u
        dt = self.comm.allreduce(dt, op=MPI.MIN)
        return dt

    def compute_dt_v_advection(self, w_scale):
        r"""
        Computes maximum explicit time step for vertical advection

        .. math :: \Delta t = \frac{\Delta z}{W_{scale}}

        where :math:`W_{scale}` is estimated vertical advective velocity.

        :arg w_scale: User provided maximum vertical velocity scale
        :type w_scale: float or :class:`Constant`
        """
        w = w_scale
        if isinstance(w_scale, FiredrakeConstant):
            w = w_scale.dat.data[0]
        min_dz = self.fields.v_elem_size_2d.dat.data.min()
        # alpha = 0.5 if self.options.element_family == 'rt-dg' else 1.0
        # dt = alpha*1.0/1.5/(self.options.polynomial_degree + 1)*min_dz/w
        min_dz *= self.compute_dz_factor()
        dt = min_dz/w
        dt = self.comm.allreduce(dt, op=MPI.MIN)
        return dt

    def compute_dt_diffusion(self, nu_scale):
        r"""
        Computes maximum explicit time step for horizontal diffusion.

        .. math :: \Delta t = \alpha \frac{(\Delta x)^2}{\nu_{scale}}

        where :math:`\nu_{scale}` is estimated diffusivity scale.
        """
        nu = nu_scale
        if isinstance(nu_scale, FiredrakeConstant):
            nu = nu_scale.dat.data[0]
        min_dx = self.fields.h_elem_size_2d.dat.data.min()
        factor = 2.0
        if self.options.timestepper_type == 'LeapFrog':
            factor = 1.2
        min_dx *= factor*self.compute_dx_factor()
        dt = (min_dx)**2/nu
        dt = self.comm.allreduce(dt, op=MPI.MIN)
        return dt

    def compute_mesh_stats(self):
        """
        Computes number of elements, nodes etc and prints to sdtout
        """
        nnodes = self.function_spaces.P1_2d.dim()
        ntriangles = int(self.function_spaces.P1DG_2d.dim()/3)
        nlayers = self.mesh.topology.layers - 1
        nprisms = ntriangles*nlayers
        dofs_per_elem = len(self.function_spaces.H.finat_element.entity_dofs())
        ntracer_dofs = dofs_per_elem*nprisms
        min_h_size = self.comm.allreduce(self.fields.h_elem_size_2d.dat.data.min(), MPI.MIN)
        max_h_size = self.comm.allreduce(self.fields.h_elem_size_2d.dat.data.max(), MPI.MAX)
        min_v_size = self.comm.allreduce(self.fields.v_elem_size_3d.dat.data.min(), MPI.MIN)
        max_v_size = self.comm.allreduce(self.fields.v_elem_size_3d.dat.data.max(), MPI.MAX)

        print_output('2D mesh: {:} nodes, {:} triangles'.format(nnodes, ntriangles))
        print_output('3D mesh: {:} layers, {:} prisms'.format(nlayers, nprisms))
        print_output('Horizontal element size: {:.2f} ... {:.2f} m'.format(min_h_size, max_h_size))
        print_output('Vertical element size: {:.3f} ... {:.3f} m'.format(min_v_size, max_v_size))
        print_output('Element family: {:}, degree: {:}'.format(self.options.element_family, self.options.polynomial_degree))
        print_output('Number of tracer DOFs: {:}'.format(ntracer_dofs))
        print_output('Number of cores: {:}'.format(self.comm.size))
        print_output('Tracer DOFs per core: ~{:.1f}'.format(float(ntracer_dofs)/self.comm.size))

    def set_time_step(self):
        """
        Sets the model the model time step

        If the time integrator supports automatic time step, and
        :attr:`ModelOptions3d.timestepper_options.use_automatic_timestep` is
        `True`, we compute the maximum time step allowed by the CFL condition.
        Otherwise uses :attr:`ModelOptions3d.timestep`.

        Once the time step is determined, will adjust it to be an integer
        fraction of export interval ``options.simulation_export_time``.
        """
        automatic_timestep = (hasattr(self.options.timestepper_options, 'use_automatic_timestep') and
                              self.options.timestepper_options.use_automatic_timestep)

        cfl2d = self.timestepper.cfl_coeff_2d
        cfl3d = self.timestepper.cfl_coeff_3d
        max_dt_swe = self.compute_dt_2d(self.options.horizontal_velocity_scale)
        max_dt_hadv = self.compute_dt_h_advection(self.options.horizontal_velocity_scale)
        max_dt_vadv = self.compute_dt_v_advection(self.options.vertical_velocity_scale)
        max_dt_diff = self.compute_dt_diffusion(self.options.horizontal_viscosity_scale)
        print_output('  - dt 2d swe: {:}'.format(max_dt_swe))
        print_output('  - dt h. advection: {:}'.format(max_dt_hadv))
        print_output('  - dt v. advection: {:}'.format(max_dt_vadv))
        print_output('  - dt viscosity: {:}'.format(max_dt_diff))
        max_dt_2d = cfl2d*max_dt_swe
        max_dt_3d = cfl3d*min(max_dt_hadv, max_dt_vadv, max_dt_diff)
        print_output('  - CFL adjusted dt: 2D: {:} 3D: {:}'.format(max_dt_2d, max_dt_3d))
        if not automatic_timestep:
            print_output('  - User defined dt: 2D: {:} 3D: {:}'.format(self.options.timestep_2d, self.options.timestep))
        self.dt = self.options.timestep
        self.dt_2d = self.options.timestep_2d
        if automatic_timestep:
            assert self.options.timestep is not None
            assert self.options.timestep > 0.0
            assert self.options.timestep_2d is not None
            assert self.options.timestep_2d > 0.0

        if self.dt_mode == 'split':
            if automatic_timestep:
                self.dt = max_dt_3d
                self.dt_2d = max_dt_2d
            # compute mode split ratio and force it to be integer
            self.M_modesplit = int(np.ceil(self.dt/self.dt_2d))
            self.dt_2d = self.dt/self.M_modesplit
        elif self.dt_mode == '2d':
            if automatic_timestep:
                self.dt = min(max_dt_2d, max_dt_3d)
            self.dt_2d = self.dt
            self.M_modesplit = 1
        elif self.dt_mode == '3d':
            if automatic_timestep:
                self.dt = max_dt_3d
            self.dt_2d = self.dt
            self.M_modesplit = 1

        print_output('  - chosen dt: 2D: {:} 3D: {:}'.format(self.dt_2d, self.dt))

        # fit dt to export time
        m_exp = int(np.ceil(self.options.simulation_export_time/self.dt))
        self.dt = float(self.options.simulation_export_time)/m_exp
        if self.dt_mode == 'split':
            self.M_modesplit = int(np.ceil(self.dt/self.dt_2d))
            self.dt_2d = self.dt/self.M_modesplit
        else:
            self.dt_2d = self.dt

        print_output('  - adjusted dt: 2D: {:} 3D: {:}'.format(self.dt_2d, self.dt))

        print_output('dt = {0:f}'.format(self.dt))
        if self.dt_mode == 'split':
            print_output('2D dt = {0:f} {1:d}'.format(self.dt_2d, self.M_modesplit))
        sys.stdout.flush()

    def create_function_spaces(self):
        """
        Creates function spaces

        Function spaces are accessible via :attr:`.function_spaces`
        object.
        """
        self._isfrozen = False
        # ----- function spaces: elev in H, uv in U, mixed is W
        self.function_spaces.P0 = get_functionspace(self.mesh, 'DG', 0, 'DG', 0, name='P0')
        self.function_spaces.P1 = get_functionspace(self.mesh, 'CG', 1, 'CG', 1, name='P1')
        self.function_spaces.P2 = get_functionspace(self.mesh, 'CG', 2, 'CG', 2, name='P2')
        self.function_spaces.P1v = get_functionspace(self.mesh, 'CG', 1, 'CG', 1, name='P1v', vector=True)
        self.function_spaces.P1DG = get_functionspace(self.mesh, 'DG', 1, 'DG', 1, name='P1DG')
        self.function_spaces.P1DGv = get_functionspace(self.mesh, 'DG', 1, 'DG', 1, name='P1DGv', vector=True)

        # function spaces for (u,v) and w
        if self.options.element_family == 'rt-dg':
            self.function_spaces.U = get_functionspace(self.mesh, 'RT', self.options.polynomial_degree+1, 'DG', self.options.polynomial_degree, name='U', hdiv=True)
            self.function_spaces.W = get_functionspace(self.mesh, 'DG', self.options.polynomial_degree, 'CG', self.options.polynomial_degree+1, name='W', hdiv=True)
        elif self.options.element_family == 'dg-dg':
            self.function_spaces.U = get_functionspace(self.mesh, 'DG', self.options.polynomial_degree, 'DG', self.options.polynomial_degree, name='U', vector=True)
            self.function_spaces.W = get_functionspace(self.mesh, 'DG', self.options.polynomial_degree, 'DG', self.options.polynomial_degree, name='W', vector=True)
        else:
            raise Exception('Unsupported finite element family {:}'.format(self.options.element_family))

        self.function_spaces.Uint = self.function_spaces.U  # vertical integral of uv
        # tracers
        self.function_spaces.H = get_functionspace(self.mesh, 'DG', self.options.polynomial_degree, 'DG', self.options.polynomial_degree, name='H')
       # self.function_spaces.H = get_functionspace(self.mesh, 'DG', self.options.polynomial_degree, 'DG', 0, name='H')
        self.function_spaces.turb_space = self.function_spaces.P0

        # 2D spaces
        self.function_spaces.P1_2d = get_functionspace(self.mesh2d, 'CG', 1, name='P1_2d')
        self.function_spaces.P2_2d = get_functionspace(self.mesh2d, 'CG', 1, name='P2_2d')
        self.function_spaces.P1v_2d = get_functionspace(self.mesh2d, 'CG', 1, name='P1v_2d', vector=True)
        self.function_spaces.P1DG_2d = get_functionspace(self.mesh2d, 'DG', 1, name='P1DG_2d')
        self.function_spaces.P1DGv_2d = get_functionspace(self.mesh2d, 'DG', 1, name='P1DGv_2d', vector=True)
        # 2D velocity space
        if self.options.element_family == 'rt-dg':
            self.function_spaces.U_2d = get_functionspace(self.mesh2d, 'RT', self.options.polynomial_degree+1, name='U_2d')
        elif self.options.element_family == 'dg-dg':
            if self.horizontal_domain_is_2d:
                self.function_spaces.U_2d = get_functionspace(self.mesh2d, 'DG', self.options.polynomial_degree, name='U_2d', vector=True)
            else:
                self.function_spaces.U_2d = get_functionspace(self.mesh2d, 'DG', self.options.polynomial_degree, name='U_2d')
        self.function_spaces.H_2d = get_functionspace(self.mesh2d, 'DG', self.options.polynomial_degree, name='H_2d')
        self.function_spaces.V_2d = MixedFunctionSpace([self.function_spaces.U_2d, self.function_spaces.H_2d], name='V_2d')

        # define function spaces for baroclinic head and internal pressure gradient
        if self.options.use_quadratic_pressure:
            self.function_spaces.P2DGxP2 = get_functionspace(self.mesh, 'DG', 2, 'CG', 2, name='P2DGxP2')
            self.function_spaces.P2DG_2d = get_functionspace(self.mesh2d, 'DG', 2, name='P2DG_2d')
            if self.options.element_family == 'dg-dg':
                self.function_spaces.P2DGxP1DGv = get_functionspace(self.mesh, 'DG', 2, 'DG', 1, name='P2DGxP1DGv', vector=True, dim=2)
                self.function_spaces.H_bhead = self.function_spaces.P2DGxP2
                self.function_spaces.H_bhead_2d = self.function_spaces.P2DG_2d
                self.function_spaces.U_int_pg = self.function_spaces.P2DGxP1DGv
            elif self.options.element_family == 'rt-dg':
                self.function_spaces.H_bhead = self.function_spaces.P2DGxP2
                self.function_spaces.H_bhead_2d = self.function_spaces.P2DG_2d
                self.function_spaces.U_int_pg = self.function_spaces.U
        else:
            self.function_spaces.P1DGxP2 = get_functionspace(self.mesh, 'DG', 1, 'CG', 2, name='P1DGxP2')
            self.function_spaces.H_bhead = self.function_spaces.P1DGxP2
            self.function_spaces.H_bhead_2d = self.function_spaces.P1DG_2d
            self.function_spaces.U_int_pg = self.function_spaces.U

        self._isfrozen = True

    def create_fields(self):
        """
        Creates all fields
        """
        if not hasattr(self, 'U_2d'):
            self.create_function_spaces()
        self._isfrozen = False

        if self.options.log_output and not self.options.no_exports:
            logfile = os.path.join(create_directory(self.options.output_directory), 'log')
            filehandler = logging.logging.FileHandler(logfile, mode='w')
            filehandler.setFormatter(logging.logging.Formatter('%(message)s'))
            output_logger.addHandler(filehandler)

        # mesh velocity etc fields must be in the same space as 3D coordinates
        coord_is_dg = element_continuity(self.mesh2d.coordinates.function_space().ufl_element()).horizontal == 'dg'
        if coord_is_dg:
            coord_fs = FunctionSpace(self.mesh, 'DG', 1, vfamily='CG', vdegree=1)
            coord_fs_2d = self.function_spaces.P1DG_2d
        else:
            coord_fs = self.function_spaces.P1
            coord_fs_2d = self.function_spaces.P1_2d

        # ----- fields
        self.fields.solution_2d = Function(self.function_spaces.V_2d)
        # correct treatment of the split 2d functions
        uv_2d, eta2d = self.fields.solution_2d.split()
        self.fields.uv_2d = uv_2d
        self.fields.elev_2d = eta2d
        if self.options.use_bottom_friction:
            self.fields.uv_bottom_2d = Function(self.function_spaces.P1v_2d)
            self.fields.z_bottom_2d = Function(coord_fs_2d)
            self.fields.bottom_drag_2d = Function(coord_fs_2d)

        self.fields.elev_3d = Function(self.function_spaces.H)
        self.fields.elev_cg_3d = Function(coord_fs)
        self.fields.elev_cg_2d = Function(coord_fs_2d)
        self.fields.uv_3d = Function(self.function_spaces.U)
        if self.options.use_bottom_friction:
            self.fields.uv_bottom_3d = Function(self.function_spaces.P1v)
            self.fields.bottom_drag_3d = Function(coord_fs)
        self.fields.bathymetry_2d = Function(coord_fs_2d)
        self.fields.bathymetry_3d = Function(coord_fs)
        # z coordinate in the strecthed mesh
        self.fields.z_coord_3d = Function(coord_fs)
        # z coordinate in the reference mesh (eta=0)
        self.fields.z_coord_ref_3d = Function(coord_fs)
        self.fields.uv_dav_3d = Function(self.function_spaces.U)
        self.fields.uv_dav_2d = Function(self.function_spaces.U_2d)
        self.fields.split_residual_2d = Function(self.function_spaces.U_2d)
        self.fields.uv_mag_3d = Function(self.function_spaces.P0)
        self.fields.uv_p1_3d = Function(self.function_spaces.P1v)
        self.fields.w_3d = Function(self.function_spaces.W)
        self.fields.hcc_metric_3d = Function(self.function_spaces.P1DG, name='mesh consistency')
        if self.options.use_ale_moving_mesh:
            self.fields.w_mesh_3d = Function(coord_fs)
            self.fields.w_mesh_surf_3d = Function(coord_fs)
            self.fields.w_mesh_surf_2d = Function(coord_fs_2d)
        if self.options.solve_salinity:
            self.fields.salt_3d = Function(self.function_spaces.H, name='Salinity')
        if self.options.solve_temperature:
            self.fields.temp_3d = Function(self.function_spaces.H, name='Temperature')
        if self.options.use_implicit_vertical_diffusion and self.options.use_parabolic_viscosity:
            self.fields.parab_visc_3d = Function(self.function_spaces.P1)
        if self.options.use_baroclinic_formulation:
            if self.options.use_quadratic_density:
                self.fields.density_3d = Function(self.function_spaces.P2DGxP2, name='Density')
            else:
                self.fields.density_3d = Function(self.function_spaces.H, name='Density')
            self.fields.baroc_head_3d = Function(self.function_spaces.H_bhead)
            self.fields.int_pg_3d = Function(self.function_spaces.U_int_pg, name='int_pg_3d')
        else:
            self.fields.density_3d = Function(self.function_spaces.H, name='Density').assign(self.options.rho_water) # WPan added.
        if self.options.coriolis_frequency is not None:
            if isinstance(self.options.coriolis_frequency, FiredrakeConstant):
                self.fields.coriolis_3d = self.options.coriolis_frequency
            else:
                self.fields.coriolis_3d = Function(self.function_spaces.P1)
                ExpandFunctionTo3d(self.options.coriolis_frequency, self.fields.coriolis_3d).solve()
        if self.options.wind_stress is not None:
            if isinstance(self.options.wind_stress, FiredrakeFunction):
                assert self.options.wind_stress.function_space().mesh().geometric_dimension() == 3, \
                    'wind stress field must be a 3D function'
                self.fields.wind_stress_3d = self.options.wind_stress
            elif isinstance(self.options.wind_stress, FiredrakeConstant):
                self.fields.wind_stress_3d = self.options.wind_stress
            else:
                raise Exception('Unsupported wind stress type: {:}'.format(type(self.options.wind_stress)))
        self.fields.v_elem_size_3d = Function(self.function_spaces.P1DG)
        self.fields.v_elem_size_2d = Function(self.function_spaces.P1DG_2d)
        self.fields.h_elem_size_3d = Function(self.function_spaces.P1)
        self.fields.h_elem_size_2d = Function(self.function_spaces.P1_2d)
        get_horizontal_elem_size_3d(self.fields.h_elem_size_2d, self.fields.h_elem_size_3d)
        self.fields.max_h_diff = Function(self.function_spaces.P1)
        if self.options.use_smagorinsky_viscosity:
            self.fields.smag_visc_3d = Function(self.function_spaces.P1)
        if self.options.use_limiter_for_tracers and self.options.polynomial_degree > 0:
            self.tracer_limiter = limiter.VertexBasedP1DGLimiter(self.function_spaces.H)
        else:
            self.tracer_limiter = None
        if (self.options.use_limiter_for_velocity
                and self.options.polynomial_degree > 0
                and self.options.element_family == 'dg-dg'):
            self.uv_limiter = limiter.VertexBasedP1DGLimiter(self.function_spaces.U)
        else:
            self.uv_limiter = None
        if self.options.use_turbulence:
            if self.options.turbulence_model_type == 'gls':
                # NOTE tke and psi should be in H as tracers ??
                self.fields.tke_3d = Function(self.function_spaces.turb_space)
                self.fields.psi_3d = Function(self.function_spaces.turb_space)
                # NOTE other turb. quantities should share the same nodes ??
                self.fields.eps_3d = Function(self.function_spaces.turb_space)
                self.fields.len_3d = Function(self.function_spaces.turb_space)
                self.fields.eddy_visc_3d = Function(self.function_spaces.turb_space)
                self.fields.eddy_diff_3d = Function(self.function_spaces.turb_space)
                # NOTE M2 and N2 depend on d(.)/dz -> use CG in vertical ?
                self.fields.shear_freq_3d = Function(self.function_spaces.turb_space)
                self.fields.buoy_freq_3d = Function(self.function_spaces.turb_space)
                self.turbulence_model = turbulence_nh.GenericLengthScaleModel(
                    weakref.proxy(self),
                    self.fields.tke_3d,
                    self.fields.psi_3d,
                    self.fields.uv_3d,
                    self.fields.get('density_3d'),
                    self.fields.len_3d,
                    self.fields.eps_3d,
                    self.fields.eddy_diff_3d,
                    self.fields.eddy_visc_3d,
                    self.fields.buoy_freq_3d,
                    self.fields.shear_freq_3d,
                    options=self.options.turbulence_model_options)
            elif self.options.turbulence_model_type == 'pacanowski':
                self.fields.eddy_visc_3d = Function(self.function_spaces.turb_space)
                self.fields.eddy_diff_3d = Function(self.function_spaces.turb_space)
                self.fields.shear_freq_3d = Function(self.function_spaces.turb_space)
                self.fields.buoy_freq_3d = Function(self.function_spaces.turb_space)
                self.turbulence_model = turbulence_nh.PacanowskiPhilanderModel(
                    weakref.proxy(self),
                    self.fields.uv_3d,
                    self.fields.get('density_3d'),
                    self.fields.eddy_diff_3d,
                    self.fields.eddy_visc_3d,
                    self.fields.buoy_freq_3d,
                    self.fields.shear_freq_3d,
                    options=self.options.turbulence_model_options)
            else:
                raise Exception('Unsupported turbulence model: {:}'.format(self.options.turbulence_model))
        else:
            self.turbulence_model = None
        # copute total viscosity/diffusivity
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

        self.create_functions() # WPan added.

        self._isfrozen = True

    def create_functions(self):
        """
        Creates extra functions, including fields
        """
        self.bathymetry_dg_old = Function(self.function_spaces.H_2d)
        self.bathymetry_dg = Function(self.function_spaces.H_2d).project(self.bathymetry_cg_2d)
        self.bathymetry_ls = Function(self.function_spaces.H_2d).project(self.bathymetry_cg_2d)
        self.bathymetry_wd = Function(self.function_spaces.P1_2d).project(self.bathymetry_cg_2d) # for wetting-drying use of 3D, temporarily
        self.bathymetry_3d_old = Function(self.fields.bathymetry_3d.function_space())
        self.elev_2d_old = Function(self.function_spaces.H_2d)
        self.elev_2d_mid = Function(self.function_spaces.H_2d)
        self.elev_3d_old = Function(self.function_spaces.H)
        self.elev_3d_mid = Function(self.function_spaces.H)
        self.elev_3d_star = Function(self.function_spaces.H)

        self.uv_2d_dg = Function(self.function_spaces.P1DGv_2d)
        self.uv_2d_old = Function(self.function_spaces.U_2d)
        self.uv_2d_mid = Function(self.function_spaces.U_2d)
        self.uv_3d_old = Function(self.function_spaces.U)
        self.uv_3d_mid = Function(self.function_spaces.U)
        self.uv_3d_tmp = Function(self.function_spaces.U)
        self.uv_dav_3d_mid = Function(self.function_spaces.U)
        self.uv_dav_2d_mid = Function(self.function_spaces.U_2d)
        self.w_3d_old = Function(self.function_spaces.U)

        self.w_surface = Function(self.function_spaces.H_2d)
        self.w_interface = Function(self.function_spaces.H_2d)
        self.fields.w_nh = Function(self.function_spaces.H_2d)
        self.fields.q_3d = Function(FunctionSpace(self.mesh, 'CG', self.options.polynomial_degree+1))
        self.q_3d_mid = Function(self.fields.q_3d.function_space())
        self.q_3d_dq = Function(self.fields.q_3d.function_space())
        self.fields.q_2d = Function(self.function_spaces.P2_2d)
        self.q_2d_mid = Function(self.fields.q_2d.function_space())
        self.q_2d_old = Function(self.function_spaces.P2_2d)
        self.fields.ext_pg_3d = Function(self.function_spaces.U_int_pg)

        self.fields.uv_delta = Function(self.function_spaces.U_2d)
        self.fields.uv_delta_2 = Function(self.function_spaces.U_2d)
        self.fields.uv_delta_3 = Function(self.function_spaces.U_2d)
        self.fields.uv_nh = Function(self.function_spaces.U_2d)
        self.uv_av_1 = Function(self.function_spaces.U_2d)
        self.uv_av_2 = Function(self.function_spaces.U_2d)
        self.uv_av_3 = Function(self.function_spaces.U_2d)
        self.w_01 = Function(self.function_spaces.H_2d)
        self.w_12 = Function(self.function_spaces.H_2d)
        self.w_23 = Function(self.function_spaces.H_2d)
        self.function_spaces.q_mixed_two_layers = MixedFunctionSpace([self.function_spaces.P2_2d, self.function_spaces.P2_2d])
        self.function_spaces.q_mixed_three_layers = MixedFunctionSpace([self.function_spaces.P2_2d, self.function_spaces.P2_2d, self.function_spaces.P2_2d])
        self.q_mixed_two_layers = Function(self.function_spaces.q_mixed_two_layers)
        self.q_mixed_three_layers = Function(self.function_spaces.q_mixed_three_layers)
        self.q_0 = Function(self.function_spaces.P2_2d)
        self.q_1 = Function(self.function_spaces.P2_2d)
        self.q_2 = Function(self.function_spaces.P2_2d)
        self.fields.elev_nh = Function(self.function_spaces.H_bhead)
        self.solution_2d_old = Function(self.function_spaces.V_2d)
        self.solution_2d_tmp = Function(self.function_spaces.V_2d)

        self.fields.solution_ls = Function(self.function_spaces.V_2d)
        self.solution_ls_old = Function(self.function_spaces.V_2d)
        self.solution_ls_tmp = Function(self.function_spaces.V_2d)
        # functions for landslide modelling
        uv_ls, elev_ls = self.fields.solution_ls.split()
        self.fields.uv_ls = uv_ls
        self.fields.elev_ls = elev_ls
        self.elev_ls_real = Function(self.function_spaces.H_2d)
        self.fields.slide_source = Function(self.function_spaces.H_2d)
        self.fields.bed_slope = Function(self.function_spaces.H_2d)
        self.dudy = Function(self.function_spaces.H_2d)
        self.dvdx = Function(self.function_spaces.H_2d)

        for k in range(self.n_layers):
            setattr(self, 'uv_av_' + str(k+1), Function(self.function_spaces.U_2d))
            #self.__dict__['uv_av_' + str(k+1)] = Function(self.function_spaces.U_2d)
            setattr(self, 'w_' + str(k+1), Function(self.function_spaces.H_2d))
            setattr(self, 'w_av_' + str(k+1), Function(self.function_spaces.H_2d))
            setattr(self, 'q_' + str(k+1), Function(self.function_spaces.P2_2d))
            if k == 0:
                list_fs = [self.function_spaces.P2_2d]
                setattr(self, 'w_' + str(k), Function(self.function_spaces.H_2d))
                setattr(self, 'q_' + str(k), Function(self.function_spaces.P2_2d))
                setattr(self, 'q_' + str(self.n_layers+1), Function(self.function_spaces.P2_2d))
            else:
                list_fs.append(self.function_spaces.P2_2d)
        self.function_spaces.q_mixed_n_layers = MixedFunctionSpace(list_fs)
        self.q_mixed_n_layers = Function(self.function_spaces.q_mixed_n_layers)

        coord_fs = FunctionSpace(self.mesh, 'DG', 1, vfamily='CG', vdegree=1)
        if self.horizontal_domain_is_2d:
            self.sigma_coord = Function(coord_fs).project(self.mesh.coordinates[2])
        else:
            self.sigma_coord = Function(coord_fs).project(self.mesh.coordinates[1])

        self.z_in_sigma = Function(coord_fs)
        self.z_in_sigma_old = Function(coord_fs)
        self.fields.sigma_dt = Function(coord_fs)
        self.fields.sigma_dx = Function(coord_fs)
        self.fields.sigma_dy = Function(coord_fs)
       # self.fields.sigma_dz = Function(coord_fs)
        self.fields.omega = Function(coord_fs)

        # p1dg 3d functions
        self.uv_3d_p1dg = Function(self.function_spaces.P1DGv)
        self.tracer_3d_p1dg = Function(self.function_spaces.P1DG)

        if self.options.solve_salinity:
            self.salt_3d_old = Function(self.function_spaces.H)
            self.salt_3d_mid = Function(self.function_spaces.H)
        if self.options.solve_temperature:
            self.temp_3d_old = Function(self.function_spaces.H)
            self.temp_3d_mid = Function(self.function_spaces.H)

    def create_equations(self):
        """
        Creates all dynamic equations and time integrators
        """
        if 'uv_3d' not in self.fields:
            self.create_fields()
        self._isfrozen = False

        if self.options.log_output and not self.options.no_exports:
            logfile = os.path.join(create_directory(self.options.output_directory), 'log')
            filehandler = logging.logging.FileHandler(logfile, mode='w')
            filehandler.setFormatter(logging.logging.Formatter('%(message)s'))
            output_logger.addHandler(filehandler)

        self.eq_sw = shallowwater_nh.ModeSplit2DEquations(
            self.fields.solution_2d.function_space(),
            self.bathymetry_dg,
            self.options)

        self.eq_sw_nh = shallowwater_nh.ShallowWaterEquations(
            self.fields.solution_2d.function_space(),
            self.bathymetry_dg,
            self.options)

        self.eq_sw_mom = shallowwater_nh.ShallowWaterMomentumEquation(
            TestFunction(self.function_spaces.U_2d),
            self.function_spaces.U_2d,
            self.function_spaces.H_2d,
            self.bathymetry_dg,
            self.options)

        self.eq_free_surface = shallowwater_nh.FreeSurfaceEquation(
            TestFunction(self.function_spaces.H_2d),
            self.function_spaces.H_2d,
            self.function_spaces.U_2d,
            self.bathymetry_dg,
            self.options)

        if not self.options.slide_is_granular:
            self.eq_ls = landslide_motion.LiquidSlideEquations(
            self.fields.solution_ls.function_space(),
            self.bathymetry_ls,
            self.options)
        else:
            self.eq_ls = landslide_motion.GranularSlideEquations(
            self.fields.solution_ls.function_space(),
            self.bathymetry_ls,
            self.options)
        self.eq_ls.bnd_functions = self.bnd_functions['landslide_motion']

        expl_bottom_friction = self.options.use_bottom_friction and not self.options.use_implicit_vertical_diffusion
        self.eq_momentum = momentum_sigma.MomentumEquation(self.fields.uv_3d.function_space(),
                                                        bathymetry=self.fields.bathymetry_3d,
                                                        v_elem_size=self.fields.v_elem_size_3d,
                                                        h_elem_size=self.fields.h_elem_size_3d,
                                                        use_nonlinear_equations=self.options.use_nonlinear_equations,
                                                        use_lax_friedrichs=self.options.use_lax_friedrichs_velocity,
                                                        use_bottom_friction=expl_bottom_friction)

        if self.options.use_implicit_vertical_diffusion:
            self.eq_vertmomentum = momentum_sigma.MomentumEquation(self.fields.uv_3d.function_space(),
                                                                bathymetry=self.fields.bathymetry_3d,
                                                                v_elem_size=self.fields.v_elem_size_3d,
                                                                h_elem_size=self.fields.h_elem_size_3d,
                                                                use_nonlinear_equations=False, # i.e. advection terms neglected
                                                                use_lax_friedrichs=self.options.use_lax_friedrichs_velocity,
                                                                use_bottom_friction=self.options.use_bottom_friction)
        if self.options.solve_salinity:
            self.eq_salt = tracer_sigma.TracerEquation(self.fields.salt_3d.function_space(),
                                                    bathymetry=self.fields.bathymetry_3d,
                                                    v_elem_size=self.fields.v_elem_size_3d,
                                                    h_elem_size=self.fields.h_elem_size_3d,
                                                    use_lax_friedrichs=self.options.use_lax_friedrichs_tracer,
                                                    use_symmetric_surf_bnd=self.options.element_family == 'dg-dg')
            if self.options.use_implicit_vertical_diffusion:
                self.eq_salt_vdff = tracer_sigma.TracerEquation(self.fields.salt_3d.function_space(),
                                                             bathymetry=self.fields.bathymetry_3d,
                                                             v_elem_size=self.fields.v_elem_size_3d,
                                                             h_elem_size=self.fields.h_elem_size_3d,
                                                             use_lax_friedrichs=self.options.use_lax_friedrichs_tracer)

        if self.options.solve_temperature:
            self.eq_temp = tracer_sigma.TracerEquation(self.fields.temp_3d.function_space(),
                                                    bathymetry=self.fields.bathymetry_3d,
                                                    v_elem_size=self.fields.v_elem_size_3d,
                                                    h_elem_size=self.fields.h_elem_size_3d,
                                                    use_lax_friedrichs=self.options.use_lax_friedrichs_tracer,
                                                    use_symmetric_surf_bnd=self.options.element_family == 'dg-dg')
            if self.options.use_implicit_vertical_diffusion:
                self.eq_temp_vdff = tracer_sigma.TracerEquation(self.fields.temp_3d.function_space(),
                                                             bathymetry=self.fields.bathymetry_3d,
                                                             v_elem_size=self.fields.v_elem_size_3d,
                                                             h_elem_size=self.fields.h_elem_size_3d,
                                                             use_lax_friedrichs=self.options.use_lax_friedrichs_tracer)

        self.eq_sw.bnd_functions = self.bnd_functions['shallow_water']
        self.eq_momentum.bnd_functions = self.bnd_functions['momentum']
        if self.options.solve_salinity:
            self.eq_salt.bnd_functions = self.bnd_functions['salt']
        if self.options.solve_temperature:
            self.eq_temp.bnd_functions = self.bnd_functions['temp']
        if self.options.use_turbulence and self.options.turbulence_model_type == 'gls':
            if self.options.use_turbulence_advection:
                # explicit advection equations
                self.eq_tke_adv = tracer_sigma.TracerEquation(self.fields.tke_3d.function_space(),
                                                           bathymetry=self.fields.bathymetry_3d,
                                                           v_elem_size=self.fields.v_elem_size_3d,
                                                           h_elem_size=self.fields.h_elem_size_3d,
                                                           use_lax_friedrichs=self.options.use_lax_friedrichs_tracer)
                self.eq_psi_adv = tracer_sigma.TracerEquation(self.fields.psi_3d.function_space(),
                                                           bathymetry=self.fields.bathymetry_3d,
                                                           v_elem_size=self.fields.v_elem_size_3d,
                                                           h_elem_size=self.fields.h_elem_size_3d,
                                                           use_lax_friedrichs=self.options.use_lax_friedrichs_tracer)
            # implicit vertical diffusion eqn with production terms
            self.eq_tke_diff = turbulence.TKEEquation(self.fields.tke_3d.function_space(),
                                                      self.turbulence_model,
                                                      bathymetry=self.fields.bathymetry_3d,
                                                      v_elem_size=self.fields.v_elem_size_3d,
                                                      h_elem_size=self.fields.h_elem_size_3d)
            self.eq_psi_diff = turbulence.PsiEquation(self.fields.psi_3d.function_space(),
                                                      self.turbulence_model,
                                                      bathymetry=self.fields.bathymetry_3d,
                                                      v_elem_size=self.fields.v_elem_size_3d,
                                                      h_elem_size=self.fields.h_elem_size_3d)

        # ----- Time integrators
        self.dt_mode = '3d'  # 'split'|'2d'|'3d' use constant 2d/3d dt, or split
        if self.options.timestepper_type == 'LeapFrog':
            raise Exception('Not surpport this time integrator: '+str(self.options.timestepper_type))
            self.timestepper = coupled_timeintegrator_nh.CoupledLeapFrogAM3(weakref.proxy(self))
        elif self.options.timestepper_type == 'SSPRK22':
            self.timestepper = coupled_timeintegrator_nh.CoupledTwoStageRK(weakref.proxy(self))
        else:
            raise Exception('Unknown time integrator type: '+str(self.options.timestepper_type))

        # ----- File exporters
        # create export_managers and store in a list
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

        # ----- Operators
        #tot_uv_3d = self.fields.uv_3d + self.fields.uv_dav_3d    <<---------- #################################################
        tot_uv_3d = self.fields.uv_3d # modified for operator-splitting method used in Telemac3D
        self.w_solver = VerticalVelocitySolver(self.fields.w_3d,
                                               tot_uv_3d,
                                               self.fields.bathymetry_3d,
                                               self.eq_momentum.bnd_functions)
        if self.horizontal_domain_is_2d:
            zero_bnd_value = Constant((0.0, 0.0, 0.0))
        else:
            zero_bnd_value = Constant((0.0, 0.0))
        self.uv_averager = VerticalIntegrator(self.fields.uv_3d,
                                              self.fields.uv_dav_3d,
                                              bottom_to_top=True,
                                              bnd_value=zero_bnd_value,
                                              average=False,
                                              bathymetry=self.fields.bathymetry_3d,
                                              elevation=self.fields.elev_cg_3d)
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
                                                     elevation=self.fields.elev_cg_3d)
            self.int_pg_calculator = momentum_sigma.InternalPressureGradientCalculator(
                self.fields, self.options,
                self.bnd_functions['momentum'],
                solver_parameters=self.options.timestepper_options.solver_parameters_momentum_explicit)
        self.extract_surf_dav_uv = SubFunctionExtractor(self.fields.uv_dav_3d,
                                                        self.fields.uv_dav_2d,
                                                        boundary='top', elem_facet='top',
                                                        elem_height=self.fields.v_elem_size_2d)
        self.copy_elev_to_3d = ExpandFunctionTo3d(self.fields.elev_2d, self.fields.elev_3d)
        self.copy_elev_cg_to_3d = ExpandFunctionTo3d(self.fields.elev_cg_2d, self.fields.elev_cg_3d) # seems ok to delete?
        self.copy_uv_dav_to_uv_dav_3d = ExpandFunctionTo3d(self.fields.uv_dav_2d, self.fields.uv_dav_3d,
                                                           elem_height=self.fields.v_elem_size_3d)
        self.copy_uv_to_uv_dav_3d = ExpandFunctionTo3d(self.fields.uv_2d, self.fields.uv_dav_3d,
                                                       elem_height=self.fields.v_elem_size_3d)
        self.uv_mag_solver = VelocityMagnitudeSolver(self.fields.uv_mag_3d, u=self.fields.uv_3d)
        if self.options.use_bottom_friction:
            self.extract_uv_bottom = SubFunctionExtractor(self.fields.uv_p1_3d, self.fields.uv_bottom_2d,
                                                          boundary='bottom', elem_facet='average',
                                                          elem_height=self.fields.v_elem_size_2d)
            self.extract_z_bottom = SubFunctionExtractor(self.fields.z_coord_3d, self.fields.z_bottom_2d,
                                                         boundary='bottom', elem_facet='average',
                                                         elem_height=self.fields.v_elem_size_2d)
            if self.options.use_parabolic_viscosity:
                self.copy_uv_bottom_to_3d = ExpandFunctionTo3d(self.fields.uv_bottom_2d,
                                                               self.fields.uv_bottom_3d,
                                                               elem_height=self.fields.v_elem_size_3d)
                self.copy_bottom_drag_to_3d = ExpandFunctionTo3d(self.fields.bottom_drag_2d,
                                                                 self.fields.bottom_drag_3d,
                                                                 elem_height=self.fields.v_elem_size_3d)
        self.mesh_updater = ALEMeshUpdater(self)

        if self.options.use_smagorinsky_viscosity:
            self.smagorinsky_diff_solver = SmagorinskyViscosity(self.fields.uv_p1_3d, self.fields.smag_visc_3d,
                                                                self.options.smagorinsky_coefficient, self.fields.h_elem_size_3d,
                                                                self.fields.max_h_diff,
                                                                weak_form=self.options.polynomial_degree == 0)
        if self.options.use_parabolic_viscosity:
            self.parabolic_viscosity_solver = ParabolicViscosity(self.fields.uv_bottom_3d,
                                                                 self.fields.bottom_drag_3d,
                                                                 self.fields.bathymetry_3d,
                                                                 self.fields.parab_visc_3d)
        self.uv_p1_projector = Projector(self.fields.uv_3d, self.fields.uv_p1_3d)
        self.elev_3d_to_cg_projector = Projector(self.fields.elev_3d, self.fields.elev_cg_3d)
        self.elev_2d_to_cg_projector = Projector(self.fields.elev_2d, self.fields.elev_cg_2d)

        # ----- set initial values
        self.fields.bathymetry_2d.project(self.bathymetry_cg_2d)
        ExpandFunctionTo3d(self.fields.bathymetry_2d, self.fields.bathymetry_3d).solve()
        self.mesh_updater.initialize()
        self.compute_mesh_stats()
        self.set_time_step()
        self.timestepper.set_dt(self.dt, self.dt_2d)
        # compute maximal diffusivity for explicit schemes
        degree_h, degree_v = self.function_spaces.H.ufl_element().degree()
        max_diff_alpha = 1.0/60.0/max((degree_h*(degree_h + 1)), 1.0)  # FIXME depends on element type and order
        self.fields.max_h_diff.assign(max_diff_alpha/self.dt * self.fields.h_elem_size_3d**2)
        d = self.fields.max_h_diff.dat.data
        print_output('max h diff {:} - {:}'.format(d.min(), d.max()))

        self.next_export_t = self.simulation_time + self.options.simulation_export_time
        self._initialized = True
        self._isfrozen = True

    def assign_initial_conditions(self, elev=None, salt=None, temp=None,
                                  uv_2d=None, uv_3d=None, tke=None, psi=None,
                                  elev_slide=None, uv_slide=None):
        """
        Assigns initial conditions

        :kwarg elev: Initial condition for water elevation
        :type elev: scalar 2D :class:`Function`, :class:`Constant`, or an expression
        :kwarg salt: Initial condition for salinity field
        :type salt: scalar 3D :class:`Function`, :class:`Constant`, or an expression
        :kwarg temp: Initial condition for temperature field
        :type temp: scalar 3D :class:`Function`, :class:`Constant`, or an expression
        :kwarg uv_2d: Initial condition for depth averaged velocity
        :type uv_2d: vector valued 2D :class:`Function`, :class:`Constant`, or an expression
        :kwarg uv_3d: Initial condition for horizontal velocity
        :type uv_3d: vector valued 3D :class:`Function`, :class:`Constant`, or an expression
        :kwarg tke: Initial condition for turbulent kinetic energy field
        :type tke: scalar 3D :class:`Function`, :class:`Constant`, or an expression
        :kwarg psi: Initial condition for turbulence generic lenght scale field
        :type psi: scalar 3D :class:`Function`, :class:`Constant`, or an expression
        """
        if not self._initialized:
            self.create_equations()
        if elev is not None:
            self.fields.elev_2d.project(elev)
        if uv_2d is not None:
            self.fields.uv_2d.project(uv_2d)
            if uv_3d is None:
                ExpandFunctionTo3d(self.fields.uv_2d, self.fields.uv_3d,
                                   elem_height=self.fields.v_elem_size_3d).solve()
        # landslide
        uv_ls, elev_ls = self.fields.solution_ls.split()
        if elev_slide is not None:
            elev_ls.project(elev_slide)
        if uv_slide is not None:
            uv_ls.project(uv_slide)

        if uv_3d is not None:
            self.fields.uv_3d.project(uv_3d)
        if salt is not None and self.options.solve_salinity:
            self.fields.salt_3d.project(salt)
        if temp is not None and self.options.solve_temperature:
            self.fields.temp_3d.project(temp)
        if self.options.use_turbulence and self.options.turbulence_model_type == 'gls':
            if tke is not None:
                self.fields.tke_3d.project(tke)
            if psi is not None:
                self.fields.psi_3d.project(psi)
            self.turbulence_model.initialize()

        if self.options.use_ale_moving_mesh:
            self.timestepper._update_3d_elevation()
            self.timestepper._update_moving_mesh()
        self.timestepper.initialize()
        # update all diagnostic variables
        self.timestepper._update_all_dependencies(self.simulation_time, 
                                                  do_2d_coupling=False,
                                                  do_vert_diffusion=False,
                                                  do_ale_update=True,
                                                  do_stab_params=True,
                                                  do_turbulence=False)
        if self.options.use_turbulence:
            self.turbulence_model.initialize()

    def add_callback(self, callback, eval_interval='export'):
        """
        Adds callback to solver object

        :arg callback: :class:`.DiagnosticCallback` instance
        :kwarg str eval_interval: Determines when callback will be evaluated,
            either 'export' or 'timestep' for evaluating after each export or
            time step.
        """
        self.callbacks.add(callback, eval_interval)

    def export(self):
        """
        Export all fields to disk

        Also evaluates all callbacks set to 'export' interval.
        """
        self.callbacks.evaluate(mode='export', index=self.i_export)
        # set uv to total uv instead of deviation from depth average
        # TODO find a cleaner way of doing this ...
        #self.fields.uv_3d += self.fields.uv_dav_3d
        for e in self.exporters.values():
            e.export()
        # restore uv_3d
        #self.fields.uv_3d -= self.fields.uv_dav_3d

    def load_state(self, i_export, outputdir=None, t=None, iteration=None):
        """
        Loads simulation state from hdf5 outputs.

        This replaces :meth:`.assign_initial_conditions` in model initilization.

        This assumes that model setup is kept the same (e.g. time step) and
        all pronostic state variables are exported in hdf5 format.  The required
        state variables are: elev_2d, uv_2d, uv_3d, salt_3d, temp_3d, tke_3d,
        psi_3d

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
            self.create_equations()
        if outputdir is None:
            outputdir = self.options.output_directory
        # create new ExportManager with desired outputdir
        state_fields = ['uv_2d', 'elev_2d', 'uv_3d',
                        'salt_3d', 'temp_3d', 'tke_3d', 'psi_3d']
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
        # NOTE remove mean from uv_3d
        self.timestepper._remove_depth_average_from_uv_3d()
        salt = temp = tke = psi = None
        if self.options.solve_salinity:
            salt = self.fields.salt_3d
            e.exporters['salt_3d'].load(i_export, salt)
        if self.options.solve_temperature:
            temp = self.fields.temp_3d
            e.exporters['temp_3d'].load(i_export, temp)
        if self.options.use_turbulence:
            if 'tke_3d' in self.fields:
                tke = self.fields.tke_3d
                e.exporters['tke_3d'].load(i_export, tke)
            if 'psi_3d' in self.fields:
                psi = self.fields.psi_3d
                e.exporters['psi_3d'].load(i_export, psi)
        self.assign_initial_conditions(elev=self.fields.elev_2d,
                                       uv_2d=self.fields.uv_2d,
                                       uv_3d=self.fields.uv_3d,
                                       salt=salt, temp=temp,
                                       tke=tke, psi=psi,
                                       )

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

        self._simulation_continued = True

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

    def solve_poisson_eq(self, q, uv_3d, w_3d, A=None, B=None, C=None, multi_layers=True):
        """
        Solve Poisson equation in two modes controlled by parameter `multi_layers'.

        Generic forms:
           2D: `div(grad(q)) + inner(A, grad(q)) + B*q = C`
           3d: `div(grad(q)) = C*(div(uv_3d) + Dx(w_3d, 2))`

        :arg A, B and C: Known functions, constants or expressions
        :type A: vector, B: scalar, C: scalar (3D: div terms). Valued :class:`Function`, `Constant`, or an expression
        :arg q: Non-hydrostatic pressure to be solved and output
        :type q: scalar function 3D or 2D :class:`Function`
        """
        q_test = TestFunction(q.function_space())
        normal = FacetNormal(q.function_space().mesh())
        boundary_markers = q.function_space().mesh().exterior_facets.unique_markers
        horizontal_is_dg = element_continuity(q.function_space().ufl_element()).horizontal in ['dg', 'hdiv']
        vertical_is_dg = element_continuity(q.function_space().ufl_element()).vertical in ['dg', 'hdiv']

        if multi_layers:
            # nabla^2-term is integrated by parts
            laplace_term = -inner(grad(q_test), grad(q)) * dx
            forcing = -(Dx(C*q_test, 0) * uv_3d[0] + Dx(C*q_test, 1) * uv_3d[1] + Dx(C*q_test, 2) * w_3d[2]) * dx 
            #C*dot(q_test, div(uv_3d) + Dx(w_3d[2], 2)) * dx
            F = laplace_term - forcing
            pressure_correction = False#True
            if pressure_correction:
                F += (-inner(Dx(q_test, 2), Dx(self.fields.q_3d, 2))*dx)

            # boundary conditions: to refer to the top and bottom use "top" and "bottom"
            # for other boundaries use the normal numbers (ids) from the horizontal mesh
            # (UnitSquareMesh automatically defines 1,2,3, and 4)
            bc_top = DirichletBC(q.function_space(), 0., "top")
            bcs = [bc_top]
            for bnd_marker in boundary_markers:
                func = self.bnd_functions['shallow_water'].get(bnd_marker)
                if func is not None: #TODO set more general and accurate conditional statement
                    bc = DirichletBC(q.function_space(), 0., int(bnd_marker))
                    bcs.append(bc)

            # you can add Dirichlet bcs to other boundaries if needed
            # any boundary that is not specified gets the natural zero Neumann bc
            #solve(F==0, q, bcs=bcs)

            prob = NonlinearVariationalProblem(F, q, bcs=bcs)
            solver = NonlinearVariationalSolver(prob,
                                            solver_parameters={'snes_type': 'ksponly',#'newtonls''ksponly'
                                                               'ksp_type': 'gmres',#'gmres''preonly'
                                                               'pc_type': 'ilu',},#'ilu''gamg'
                                            bcs=bcs,
                                            options_prefix='poisson_solver')

            solver.solve()

            return q

        else:
            # weak forms
            f = (-dot(grad(q), grad(q_test)) + B*q*q_test)*dx - C*q_test*dx - q*div(A*q_test)*dx
            if horizontal_is_dg:
                f += inner(grad(avg(q)), jump(q_test, normal))*dS + avg(q)*jump(q_test, inner(A, normal))*dS
            # boundary conditions
            for bnd_marker in boundary_markers:
                func = self.bnd_functions['shallow_water'].get(bnd_marker)
                ds_bnd = ds(int(bnd_marker))
                #q_open_bc = self.q_bnd.assign(0.)
                if self.bnd_functions['shallow_water'] == {}:#func is None or 'q' not in func:
                    # Neumann boundary condition => inner(grad(q), normal)=0.
                    f += (q*inner(A, normal))*q_test*ds_bnd

            prob = NonlinearVariationalProblem(f, q)
            solver = NonlinearVariationalSolver(prob,
                                            solver_parameters={'snes_type': 'ksponly', # ksponly, newtonls
                                                               'ksp_type': 'preonly', # gmres, preonly
                                                               'mat_type': 'aij',
                                                               'pc_type': 'lu', #'bjacobi', 'lu'
                                                               },
                                            options_prefix='poisson_solver')
            solver.solve()

            return q

    def calculate_external_pressure_gradient(self, pressure='elevation'):
        """
        Calculate three-dimensional external pressure gradient.
        """
        if pressure == 'elevation':
            eta = self.fields.elev_3d
        else:
            eta = self.fields.q_3d
        trial = TrialFunction(self.function_spaces.U_int_pg)
        test = TestFunction(self.function_spaces.U_int_pg)
        p, q = self.function_spaces.U_int_pg.ufl_element().degree()
        quad_degree = (2*p + 1, 2*q + 1)
        f = 0
        if element_continuity(eta.function_space().ufl_element()).horizontal == 'dg':
            div_test = (Dx(test[0], 0) + Dx(test[1], 1))
            f += -g_grav*eta*div_test*dx
            head_star = avg(eta)
            jump_n_dot_test = (jump(test[0], self.normal[0]) +
                               jump(test[1], self.normal[1]))
            f += g_grav*head_star*jump_n_dot_test*(dS_v + dS_h)
            n_dot_test = (self.normal[0]*test[0] +
                          self.normal[1]*test[1])
            f += g_grav*eta*n_dot_test*(ds_bottom + ds_surf)
            for bnd_marker in self.boundary_markers:
                funcs = self.bnd_functions['momentum'].get(bnd_marker)
                ds_bnd = ds_v(int(bnd_marker), degree=quad_degree)
                if eta is not None:
                    if funcs is not None and 'elev3dxx' in funcs:
                        r_ext = funcs['elev3d']
                        head_ext = r_ext
                        head_in = eta
                        head_star = 0.5*(head_ext + head_in)
                    else:
                        head_star = eta
                    f += g_grav*head_star*n_dot_test*ds_bnd
        else:
            grad_head_dot_test = (Dx(eta, 0)*test[0] +
                                  Dx(eta, 1)*test[1])
            f += g_grav * grad_head_dot_test * dx

        a = inner(trial, test) * dx
        if pressure == 'elevation':
            l = f
        else:
            if element_continuity(eta.function_space().ufl_element()).horizontal == 'dg':
                f += -g_grav*eta*Dx(test[2], 2)*dx + avg(eta)*jump(test[2], self.normal[2])*(dS_h)
                f += g_grav*eta*test[2]*self.normal[2]*(ds_bottom + ds_surf)
                for bnd_marker in self.boundary_markers:
                    funcs = self.bnd_functions['shallow_water'].get(bnd_marker)
                    ds_bnd = ds_v(int(bnd_marker), degree=quad_degree)
                    f += g_grav*eta*test[2]*self.normal[2]*ds_bnd
            else:
                f += g_grav*Dx(eta, 2)*test[2]*dx
            a = g_grav*inner(trial, test) * dx
            l = f
        #####
        a = inner(trial, test) * dx
        l = g_grav * (eta + self.fields.bathymetry_3d) * (Dx(eta, 0)*test[0]) * dx
        l = g_grav*dot(grad(eta), test)*dx
        solve(a == l, self.fields.ext_pg_3d)
        #####
       # prob = LinearVariationalProblem(a, l, self.fields.ext_pg_3d)
       # lin_solver = LinearVariationalSolver(prob, solver_parameters= \
       #              self.options.timestepper_options.solver_parameters_momentum_explicit)
#
       # lin_solver.solve()

    def set_sponge_damping(self, length, x_start, y_start =None, alpha=10., sponge_is_2d=True):
        """
        Set damping terms to reduce the reflection on solid boundaries.
        """
        if length == [0., 0.]:
            return None
        if sponge_is_2d is True:
            damping_coeff = Function(self.function_spaces.P1_2d)
        else:
            damping_coeff = Function(self.function_spaces.P1)
        damp_vector = damping_coeff.dat.data[:]
        mesh = damping_coeff.ufl_domain()
        if (not self.horizontal_domain_is_2d) and sponge_is_2d:
            xvector = mesh.coordinates.dat.data[:]
        else:
            xvector = mesh.coordinates.dat.data[:, 0]

        if mesh.coordinates.sub(0).dat.data.max() <= x_start[0] + length[0]:
            length[0] = xvector.max() - x_start[0]
            #if length[0] < 0:
                #print('Start point of the first sponge layer is out of computational domain!')
                #raise ValueError('Start point of the first sponge layer is out of computational domain!')
        if mesh.coordinates.sub(0).dat.data.max() <= x_start[1] + length[1]:
            length[1] = xvector.max() - x_start[1]
            #if length[1] < 0:
                #print('Start point of the second sponge layer is out of computational domain!')
                #raise ValueError('Start point of the second sponge layer is out of computational domain!')

        assert xvector.shape[0] == damp_vector.shape[0]
        for i, xy in enumerate(xvector):
            pi = 4*np.arctan(1.)
            x = (xy - x_start[0])/length[0]
            if y_start is not None:
                x = (xy[1] - y_start)/length[0]
            if x > 0 and x < 0.5:
                damp_vector[i] = alpha*0.25*(np.tanh(np.sin(pi*(2.*x - 0.5))/(1. - (4.*x - 1.)**2)) + 1.)
            elif x > 0.5 and x < 1.:
                damp_vector[i] = alpha*0.25*(np.tanh(np.sin(pi*(1.5 - 2*x))/(1. - (3. - 4.*x)**2)) + 1.)
            else:
                damp_vector[i] = 0.
        if length[1] == 0.:
            return damping_coeff
        for i, xy in enumerate(xvector):
            pi = 4*np.arctan(1.)
            x = (xy - x_start[1])/length[1]
            if y_start is not None:
                x = (xy[1] - y_start)/length[1]
            if x > 0 and x < 0.5:
                damp_vector[i] = alpha*0.25*(np.tanh(np.sin(pi*(2.*x - 0.5))/(1. - (4.*x - 1.)**2)) + 1.)
            elif x > 0.5 and x < 1.:
                damp_vector[i] = alpha*0.25*(np.tanh(np.sin(pi*(1.5 - 2*x))/(1. - (3. - 4.*x)**2)) + 1.)
            else:
                damp_vector[i] = 0.
        return damping_coeff

    def get_alpha(self, H0):
        """
        An alternative to try alpha, finding minimum alpha to let all depths below the threshold wd_mindep.

        :arg H0: Minimum water depth
        """     
        if H0 > 1.0E-5:
            return 0.
        elif not self.options.constant_mindep:
            return np.sqrt(0.25*self.options.wd_mindep**2 - 0.5*self.options.wd_mindep*H0) + 0.5*self.options.wd_mindep # new formulated function, WPan
            #return np.sqrt(self.options.wd_mindep**2 - self.options.wd_mindep*H0) + self.options.wd_mindep # artificial porosity method
            #return np.sqrt(4*self.options.wd_mindep*(self.options.wd_mindep-H0)) # original bathymetry changed method
        else:
            return self.options.wd_mindep

    def slide_shape(self, simulation_time):
        """
        Specific slide shape function for rigid landslide generated tsunami modelling.
        """
        L = 215.E3
        B = self.mesh2d.coordinates.sub(1).dat.data.max()
        S = 7.5E3
        hmax = 144.
        Umax = 35.
        T = self.options.t_landslide
        Ta = 0.5*T
        Tc = 0.
        Td = T - Ta - Tc
        R = 150.E3
        Ra = 75.E3
        Rc = Umax*Tc
        Rd = R - Ra - Rc
        phi = 0.
        x0 = 1112.5E3
        y0 = 0.5*B
        if simulation_time < Ta:
            s = Ra*(1. - cos(Umax/Ra*simulation_time))
        elif simulation_time >= Ta and simulation_time < (Ta + Tc):
            s = Ra + Umax*(simulation_time - Ta)
        elif simulation_time >= (Ta + Tc) and simulation_time < T:
            s = Ra + Rc + Rd*(sin(Umax/Rd*(simulation_time - Ta - Tc)))
        else:
            s = R
        xs = x0 + s*cos(phi)
        ys = y0 + s*sin(phi)
        # calculate slide shape below
        hs = Function(self.function_spaces.P1_2d)
        xy_vector = self.mesh2d.coordinates.dat.data
        hs_vector = hs.dat.data
        assert xy_vector.shape[0] == hs_vector.shape[0]
        for i, xy in enumerate(xy_vector):
            x = (xy[0] - xs)*cos(phi) + (xy[1] - ys)*sin(phi)
            y = -(xy[0] - xs)*sin(phi) + (xy[1] - ys)*cos(phi)
            if x < -(L+S) and x > -(L+2.*S):
                hs_vector[i] = hmax*exp(-(2.*(x+S+L)/S)**4 - (2.*y/B)**4)
            elif x < -S and x >= -(L+S):
                hs_vector[i] = hmax*exp(-(2.*y/B)**4)
            elif x < 0. and x >= -S:
                hs_vector[i] = hmax*exp(-(2.*(x+S)/S)**4 - (2.*y/B)**4)
            else:
                hs_vector[i] = 0.

            is_block = True # i.e. block slide
            if is_block:
                if x < -(L+S) and x > -(L+2.*S):
                   hs_vector[i] = hmax*exp(-(2.*(x+S+L)/S)**4)
                elif x < -S and x >= -(L+S):
                   hs_vector[i] = hmax
                elif x < 0. and x >= -S:
                   hs_vector[i] = hmax*exp(-(2.*(x+S)/S)**4)
                else:
                   hs_vector[i] = 0.

        return hs

    def iterate(self, update_forcings=None, update_forcings3d=None,
                export_func=None):
        """
        Runs the simulation

        Iterates over the time loop until time ``options.simulation_end_time`` is reached.

        Exports fields to disk on ``options.simulation_export_time`` intervals.

        :kwarg update_forcings: User-defined function that takes simulation
            time as an argument and updates time-dependent boundary conditions
            of the 2D system (if any).
        :kwarg update_forcings_3d: User-defined function that takes simulation
            time as an argument and updates time-dependent boundary conditions
            of the 3D equations (if any).
        :kwarg export_func: User-defined function (with no arguments) that will
            be called on every export.
        """
        if not self._initialized:
            self.create_equations()

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

        # split solution to facilitate the following
        uv_2d_old, elev_2d_old = self.solution_2d_old.split()
        uv_2d, elev_2d = self.fields.solution_2d.split()
        uta_2d_old, eta_2d_old = split(self.solution_2d_old) # note: not '.split()'
        uta_2d, eta_2d = split(self.fields.solution_2d)
        # functions associated with landslide
        uv_ls_old, elev_ls_old = self.solution_ls_old.split()
        uv_ls, elev_ls = self.fields.solution_ls.split()
        uta_ls_old, eta_ls_old = split(self.solution_ls_old) # note: not '.split()'
        uta_ls, eta_ls = split(self.fields.solution_ls)
        # trial and test functions used to update
        tri_uv_2d = TrialFunction(self.function_spaces.U_2d)
        test_uv_2d = TestFunction(self.function_spaces.U_2d)
        tri_elev_2d = TrialFunction(self.function_spaces.H_2d)
        test_elev_2d = TestFunction(self.function_spaces.H_2d)
        test_uta_2d, test_eta_2d = TestFunctions(self.fields.solution_2d.function_space())
        # for 3d velocities
        tri_uv_3d = TrialFunction(self.function_spaces.U)
        test_uv_3d = TestFunction(self.function_spaces.U)
        tri_h_3d = TrialFunction(self.function_spaces.H)
        test_h_3d = TestFunction(self.function_spaces.H)

        # initial export
        self.print_state(0.0)
        if self.export_initial_state:
            self.export()
            if export_func is not None:
                export_func()
            if 'vtk' in self.exporters:
                self.exporters['vtk'].export_bathymetry(self.fields.bathymetry_2d)

        while self.simulation_time <= self.options.simulation_end_time - t_epsilon:

            hydrostatic_solver_2d = False #TODO use more common set control
            hydrostatic_solver_3d = False
            conventional_3d_NH_solver = True

            # Original mode-splitting method
            #self.timestepper.advance(self.simulation_time,
            #                         update_forcings, update_forcings3d)
            self.uv_3d_old.assign(self.fields.uv_3d)
            self.uv_2d_old.assign(self.fields.uv_2d)
            self.elev_3d_old.assign(self.fields.elev_3d)
            self.elev_2d_old.assign(self.fields.elev_2d)
            self.elev_2d_mid.assign(self.fields.elev_2d)

            self.solution_2d_old.assign(self.fields.solution_2d)
            self.solution_ls_old.assign(self.fields.solution_ls)
            self.bathymetry_dg_old.assign(self.bathymetry_dg)
            self.bathymetry_3d_old.assign(self.fields.bathymetry_3d)

            if self.options.solve_salinity:
                self.salt_3d_old.assign(self.fields.salt_3d)
            if self.options.solve_temperature:
                self.temp_3d_old.assign(self.fields.temp_3d)

            # ----- Self-defined time integrator for layer-integrated NH solver
            fields_dav = {
                    'linear_drag_coefficient': self.options.linear_drag_coefficient,
                    'quadratic_drag_coefficient': self.options.quadratic_drag_coefficient,
                    'manning_drag_coefficient': self.options.manning_drag_coefficient,
                    'viscosity_h': self.options.horizontal_viscosity,
                    'lax_friedrichs_velocity_scaling_factor': self.options.lax_friedrichs_velocity_scaling_factor,
                    'coriolis': self.options.coriolis_frequency,
                    'wind_stress': self.options.wind_stress,
                    'atmospheric_pressure': self.options.atmospheric_pressure,
                    'momentum_source': self.fields.split_residual_2d,#self.options.momentum_source_2d,
                    'volume_source': self.options.volume_source_2d,
                    'w_nh': self.fields.w_nh,
                    'uv_nh': self.fields.uv_nh,
                    'eta': self.fields.elev_2d,
                    'uv': self.fields.uv_2d,
                    'bathymetry_init': self.fields.bathymetry_2d,
                    'slide_viscosity': self.options.slide_viscosity,
                    'slide_source': self.fields.slide_source,
                    'bed_slope': self.fields.bed_slope,
                    'ext_pressure': self.fields.q_2d + self.options.rho_water*g_grav*(self.bathymetry_dg + self.fields.elev_2d),
                    'dudy': self.dudy,
                    'dvdx': self.dvdx,
                    'sponge_damping_2d': self.set_sponge_damping(self.options.sponge_layer_length, self.options.sponge_layer_xstart, alpha=10., sponge_is_2d=True),
                         }

            solver_parameters = {'snes_type': 'newtonls', # ksponly, newtonls
                                 'ksp_type': 'gmres', # gmres, preonly
                                 'pc_type': 'fieldsplit'}

            if self.simulation_time <= t_epsilon:
                # timestepper for operator splitting in 3D NH solver
                if hydrostatic_solver_3d or self.options.use_operator_splitting:
                    theta_os = 0.5
                else:
                    theta_os = 1.0
                timestepper_operator_splitting = timeintegrator.CrankNicolson(self.eq_sw, self.fields.solution_2d,
                                                              fields_dav, self.dt, bnd_conditions=self.bnd_functions['shallow_water'],
                                                              solver_parameters=solver_parameters,
                                                              semi_implicit=False,
                                                              theta=theta_os)
                timestepper_operator_splitting_explicit = timeintegrator.CrankNicolson(self.eq_sw, self.fields.solution_2d,
                                                              fields_dav, self.dt, bnd_conditions=self.bnd_functions['shallow_water'],
                                                              solver_parameters=solver_parameters,
                                                              semi_implicit=False,
                                                              theta=0.)
                # timestepper for depth-integrated NH solver
                timestepper_depth_integrated = timeintegrator.CrankNicolson(self.eq_sw_nh, self.fields.solution_2d,
                                                              fields_dav, self.dt,
                                                              bnd_conditions=self.bnd_functions['shallow_water'],
                                                              solver_parameters=solver_parameters,
                                                              semi_implicit=False,
                                                              theta=0.5)

                # timestepper for free surface equation
                if self.options.use_operator_splitting:
                    theta_fs = 0.5
                else:
                    theta_fs = 1.0
                timestepper_free_surface = timeintegrator.CrankNicolson(self.eq_free_surface, self.elev_2d_mid,
                                                              fields_dav, self.dt,
                                                              bnd_conditions=self.bnd_functions['shallow_water'],
                                                              # solver_parameters=solver_parameters,
                                                              semi_implicit=False,
                                                              theta=theta_fs)

                # timestepper for only elevation gradient term
                timestepper_mom_2d = timeintegrator.CrankNicolson(self.eq_sw_mom, self.uv_2d_mid,
                                                              fields_dav, self.dt,
                                                              bnd_conditions=self.bnd_functions['shallow_water'],
                                                              # solver_parameters=solver_parameters,
                                                              semi_implicit=False,
                                                              theta=0.)

            # --- Hydrostatic solver ---
            if hydrostatic_solver_2d:
                if self.options.landslide and (not self.options.slide_is_rigid) and (not self.options.slide_is_granular):
                    if self.simulation_time <= t_epsilon:
                        timestepper_depth_integrated.F += -self.dt*self.eq_sw_nh.add_landslide_term(uv_ls, elev_ls, fields, self.bathymetry_ls, self.bnd_functions['landslide_motion'])
                        timestepper_depth_integrated.update_solver()
                    if self.simulation_time == self.options.t_landslide:
                        timestepper_depth_integrated.F += self.dt*self.eq_sw_nh.add_landslide_term(uv_ls, elev_ls, fields, self.bathymetry_ls, self.bnd_functions['landslide_motion'])
                        timestepper_depth_integrated.update_solver()

                if self.options.landslide and self.options.slide_is_rigid:
                    self.bathymetry_dg.project(self.fields.bathymetry_2d - self.slide_shape(self.simulation_time))
                    elev_ls.project(-self.bathymetry_dg)
                    self.fields.slide_source.project((self.slide_shape(self.simulation_time + self.dt) - self.slide_shape(self.simulation_time))/self.dt)

                timestepper_depth_integrated.advance(self.simulation_time, update_forcings)

            elif hydrostatic_solver_3d:
                #self.timestepper.advance(self.simulation_time,
                #                         update_forcings, update_forcings3d)
                self.bathymetry_cg_2d.project(self.bathymetry_dg)
                ExpandFunctionTo3d(self.bathymetry_cg_2d, self.fields.bathymetry_3d).solve()
                n_stages = 2
                for i_stage in range(n_stages):
                    # 2D advance
                    self.uv_averager.solve()
                    self.extract_surf_dav_uv.solve()
                    self.fields.uv_2d.assign(self.fields.uv_dav_2d)
                    self.copy_uv_dav_to_uv_dav_3d.solve()
                    self.uv_dav_3d_mid.assign(self.fields.uv_dav_3d)
                    self.timestepper.store_elevation(i_stage)
                    if i_stage == 1:
                        timestepper_operator_splitting.advance(self.simulation_time, update_forcings)
                    #self.timestepper.timesteppers.swe2d.solve_stage(i_stage, self.simulation_time, update_forcings)
                    # compute mesh velocity
                    self.timestepper.compute_mesh_velocity(i_stage)
                    # 3D advance in old mesh
                    self.timestepper.timesteppers.mom_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                    # update mesh
                    self.copy_elev_to_3d.solve()
                    if self.options.use_ale_moving_mesh:
                        self.mesh_updater.update_mesh_coordinates()
                    # solve 3D
                    self.timestepper.timesteppers.mom_expl.solve_stage(i_stage)
                    if self.options.use_limiter_for_velocity:
                        self.uv_limiter.apply(self.fields.uv_3d)
                    # correct uv_3d
                    self.copy_uv_to_uv_dav_3d.solve()
                    self.fields.uv_3d.project(self.fields.uv_3d - (self.uv_dav_3d_mid - self.fields.uv_dav_3d))
                    # update w
                    self.w_solver.solve()

            # based on operator-splitting method used in Telemac3D
            # Jankowski, J.A., 1999. A non-hydrostatic model for free surface flows.
            # Two-stage second-order nonlinear Strong Stability-Preserving (SSP) Runge\A8CKutta scheme
            # Gottlieb et al., 2001. doi: https://doi.org/10.1137/S003614450036757X
            elif conventional_3d_NH_solver:

                if self.options.landslide:
                    self.bathymetry_cg_2d.project(self.bathymetry_dg)
                    ExpandFunctionTo3d(self.bathymetry_cg_2d, self.fields.bathymetry_3d).solve() # for landslide

                if self.simulation_time <= t_epsilon:
                    uv_3d = self.fields.uv_3d
                    sigma_coord = self.sigma_coord
                    z_in_sigma = sigma_coord*self.fields.elev_3d + (sigma_coord - 1.)*self.fields.bathymetry_3d
                    z_in_sigma_old = sigma_coord*self.elev_3d_old + (sigma_coord - 1.)*self.bathymetry_3d_old
                    h_total = self.fields.elev_3d + self.fields.bathymetry_3d

                    # solvers for terms in omega
                    tri_omega = TrialFunction(self.z_in_sigma.function_space())
                    test_omega = TestFunction(self.z_in_sigma.function_space())
                    sigma_dz = 1./h_total
                    sigma_dx = -sigma_dz*Dx(z_in_sigma, 0)
                    a_omega = tri_omega*test_omega*dx
                    l_sigma_dt = -sigma_dz*(z_in_sigma - z_in_sigma_old)/self.dt*test_omega*dx
                    l_sigma_dx = sigma_dx*test_omega*dx
                    l_omega = (self.fields.sigma_dt + uv_3d[0]*self.fields.sigma_dx + sigma_dz*uv_3d[1])*test_omega*dx
                    if self.horizontal_domain_is_2d:
                        l_omega = (self.fields.sigma_dt + uv_3d[0]*self.fields.sigma_dx + uv_3d[1]*self.fields.sigma_dy + sigma_dz*uv_3d[2])*test_omega*dx
                        sigma_dy = -sigma_dz*Dx(z_in_sigma, 1)
                        l_sigma_dy = sigma_dy*test_omega*dx
                        prob_sigma_dy = LinearVariationalProblem(a_omega, l_sigma_dy, self.fields.sigma_dy)
                        solver_sigma_dy = LinearVariationalSolver(prob_sigma_dy)
                    prob_sigma_dt = LinearVariationalProblem(a_omega, l_sigma_dt, self.fields.sigma_dt)
                    prob_sigma_dx = LinearVariationalProblem(a_omega, l_sigma_dx, self.fields.sigma_dx)
                    prob_omega = LinearVariationalProblem(a_omega, l_omega, self.fields.omega)
                    solver_sigma_dt = LinearVariationalSolver(prob_sigma_dt)
                    solver_sigma_dx = LinearVariationalSolver(prob_sigma_dx)
                    solver_omega = LinearVariationalSolver(prob_omega)

                    # solver for the Poisson equation
                    q_3d = self.fields.q_3d
                    if self.options.use_pressure_correction:
                        q_3d = self.q_3d_dq
                    test_q = TestFunction(q_3d.function_space())
                    if self.horizontal_domain_is_2d:
                        lhs = -Dx(test_q, 0)*(Dx(q_3d, 0) + Dx(q_3d, 2)*sigma_dx)*dx - Dx(test_q, 1)*(Dx(q_3d, 1) + Dx(q_3d, 2)*sigma_dy)*dx - \
                              (sigma_dx**2 + sigma_dy**2 + sigma_dz**2)*Dx(test_q, 2)*Dx(q_3d, 2)*dx - \
                              Dx(sigma_dx*test_q, 2)*Dx(q_3d, 0)*dx - Dx(sigma_dy*test_q, 2)*Dx(q_3d, 1)*dx - \
                              sigma_dz*(Dx(h_total, 0)*sigma_dx + Dx(h_total, 1)*sigma_dy)*Dx(q_3d, 2)*test_q*dx
                       # lhs = -Dx(test_q, 0)*Dx(q_3d, 0)*dx - Dx(test_q, 1)*Dx(q_3d, 1)*dx - (sigma_dx**2 + sigma_dy**2 + sigma_dz**2)*Dx(test_q, 2)*Dx(q_3d, 2)*dx - \
                       #        sigma_dx*(Dx(test_q, 0)*Dx(q_3d, 2) + Dx(test_q, 2)*Dx(q_3d, 0))*dx - \
                       #        sigma_dy*(Dx(test_q, 1)*Dx(q_3d, 2) + Dx(test_q, 2)*Dx(q_3d, 1))*dx - \
                       #        Dx(test_q*(Dx(sigma_dx, 0) + Dx(sigma_dx, 2)*sigma_dx + Dx(sigma_dy, 0) + Dx(sigma_dy, 2)*sigma_dy), 2)*q_3d*dx
                        rhs = -rho_0/self.dt*(Dx(test_q, 0)*uv_3d[0] + Dx(sigma_dx*test_q, 2)*uv_3d[0] +
                                              Dx(test_q, 1)*uv_3d[1] + Dx(sigma_dy*test_q, 2)*uv_3d[1] + 
                                              Dx(sigma_dz*test_q, 2)*uv_3d[2])*dx
                    else:
                        lhs = -Dx(test_q, 0)*(Dx(q_3d, 0) + Dx(q_3d, 1)*sigma_dx)*dx - \
                              (sigma_dx**2 + sigma_dz**2)*Dx(test_q, 1)*Dx(q_3d, 1)*dx - \
                              Dx(sigma_dx*test_q, 1)*Dx(q_3d, 0)*dx - \
                              sigma_dz*(Dx(h_total, 0)*sigma_dx)*Dx(q_3d, 1)*test_q*dx
                        rhs = -rho_0/self.dt*(Dx(test_q, 0)*uv_3d[0] + Dx(sigma_dx*test_q, 1)*uv_3d[0] +
                                              Dx(sigma_dz*test_q, 1)*uv_3d[1])*dx
                       # old version below
                       # lhs = -Dx(test_q, 0)*Dx(q_3d, 0)*dx - (sigma_dx**2 + sigma_dz**2)*Dx(test_q, 1)*Dx(q_3d, 1)*dx - \
                       #        sigma_dx*(Dx(test_q, 0)*Dx(q_3d, 1) + Dx(test_q, 1)*Dx(q_3d, 0))*dx - \
                       #        Dx(test_q*(Dx(sigma_dx, 0) + Dx(sigma_dx, 1)*sigma_dx), 1)*q_3d*dx
                       # rhs = -rho_0/self.dt*(Dx(test_q, 0)*uv_3d[0] + Dx(sigma_dx*test_q, 1)*uv_3d[0] + Dx(sigma_dz*test_q, 1)*uv_3d[1])*dx
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
                    a_u = dot(tri_uv_3d, test_uv_3d)*dx
                    if self.horizontal_domain_is_2d:
                        l_u = dot(uv_3d, test_uv_3d)*dx - self.dt/rho_0*((Dx(q_3d, 0) + Dx(q_3d, 2)*sigma_dx)*test_uv_3d[0] +
                                                                         (Dx(q_3d, 1) + Dx(q_3d, 2)*sigma_dy)*test_uv_3d[1] +
                                                                         (Dx(q_3d, 2)*sigma_dz)*test_uv_3d[2])*dx
                    else:
                        l_u = dot(uv_3d, test_uv_3d)*dx - self.dt/rho_0*((Dx(q_3d, 0) + Dx(q_3d, 1)*sigma_dx)*test_uv_3d[0] +
                                                                         (Dx(q_3d, 1)*sigma_dz)*test_uv_3d[1])*dx
                    prob_u = LinearVariationalProblem(a_u, l_u, uv_3d)
                    solver_u = LinearVariationalSolver(prob_u)

                    # solver for advancing the momentum equation
                    fields_3d = {'elev_3d': self.fields.elev_3d,
                          'int_pg': self.fields.get('int_pg_3d'),
                          'ext_pg': self.fields.get('ext_pg_3d'),
                          'uv_3d': self.fields.uv_3d,
                          'uv_depth_av': self.fields.get('uv_dav_3d'),
                          'w': self.fields.w_3d,
                          'w_mesh': self.fields.get('w_mesh_3d'),
                          'viscosity_h': self.tot_h_visc.get_sum(),
                          'viscosity_v': self.tot_v_visc.get_sum(), # for not self.options.use_implicit_vertical_diffusion
                          'source_mom': self.options.momentum_source_3d,
                          # 'uv_mag': self.fields.uv_mag_3d,
                          'uv_p1': self.fields.get('uv_p1_3d'),
                          'lax_friedrichs_velocity_scaling_factor': self.options.lax_friedrichs_velocity_scaling_factor,
                          'coriolis': self.fields.get('coriolis_3d'),
                          'sigma_dt': self.fields.sigma_dt,
                          'sigma_dx': self.fields.sigma_dx,
                          'sigma_dy': self.fields.sigma_dy,
                          'omega': self.fields.omega,
                          'q_3d': self.fields.q_3d,
                          'use_pressure_correction': self.options.use_pressure_correction,
                          'solve_elevation_gradient_separately': self.options.solve_elevation_gradient_separately,
                          'sponge_damping_3d': self.set_sponge_damping(self.options.sponge_layer_length, self.options.sponge_layer_xstart, alpha=10., sponge_is_2d=False),
                              }
                    a_mom = dot(tri_uv_3d, test_uv_3d)*dx
                    l_mom = dot(uv_3d, test_uv_3d)*dx + self.dt*self.eq_momentum.residual('all', uv_3d, uv_3d, fields_3d, fields_3d, self.bnd_functions['momentum'])
                    prob_mom_ssprk = LinearVariationalProblem(a_mom, l_mom, self.uv_3d_mid)
                    solver_mom_ssprk = LinearVariationalSolver(prob_mom_ssprk, solver_parameters=self.options.timestepper_options.solver_parameters_momentum_explicit)

                    # solver for advancing the tracer equation
                    fields_3d.update({
                          'diffusivity_h': self.tot_h_diff.get_sum(),
                          'diffusivity_v': self.tot_v_diff.get_sum(), # for not self.options.use_implicit_vertical_diffusion
                          'source_tracer': self.options.salinity_source_3d,
                          'lax_friedrichs_tracer_scaling_factor': self.options.lax_friedrichs_tracer_scaling_factor,
                              })
                    a_tracer = dot(tri_h_3d, test_h_3d)*dx
                    if self.options.solve_salinity:
                        l_salt = dot(self.fields.salt_3d, test_h_3d)*dx + \
                                   self.dt*self.eq_salt.residual('all', self.fields.salt_3d, self.fields.salt_3d, fields_3d, fields_3d, self.bnd_functions['salt'])
                        prob_salt_ssprk = LinearVariationalProblem(a_tracer, l_salt, self.salt_3d_mid)
                        solver_salt_ssprk = LinearVariationalSolver(prob_salt_ssprk, solver_parameters=self.options.timestepper_options.solver_parameters_tracer_explicit)
                    if self.options.solve_temperature:
                        l_temp = dot(self.fields.temp_3d, test_h_3d)*dx + \
                                   self.dt*self.eq_temp.residual('all', self.fields.temp_3d, self.fields.temp_3d, fields_3d, fields_3d, self.bnd_functions['temp'])
                        prob_temp_ssprk = LinearVariationalProblem(a_tracer, l_temp, self.temp_3d_mid)
                        solver_temp_ssprk = LinearVariationalSolver(prob_temp_ssprk, solver_parameters=self.options.timestepper_options.solver_parameters_tracer_explicit)

                    # solver for the pressure gradient
                    a_grad_q = dot(tri_h_3d, test_h_3d)*dx
                    l_grad_q = 1./rho_0*(Dx(q_3d, 0) + Dx(q_3d, 1)*sigma_dx)*test_h_3d*dx
                    prob_grad_q = LinearVariationalProblem(a_grad_q, l_grad_q, self.q_3d_mid)
                    solver_grad_q = LinearVariationalSolver(prob_grad_q)

                    # solver for updating free surface
                    a_fs = dot(tri_elev_2d, test_elev_2d)*dx
                    l_fs = dot(elev_2d, test_elev_2d)*dx + self.dt*self.eq_free_surface.residual('all', elev_2d, elev_2d, fields_dav, fields_dav, self.bnd_functions['shallow_water'])
                    prob_fs_ssprk = LinearVariationalProblem(a_fs, l_fs, self.elev_2d_mid)
                    solver_fs_ssprk = LinearVariationalSolver(prob_fs_ssprk, solver_parameters=self.options.timestepper_options.solver_parameters_momentum_explicit)

                    # solver for calculating 2D external pressure gradient
                    a_pg = dot(tri_uv_2d, test_uv_2d)*dx
                    l_pg = -self.eq_sw_mom.residual('all', uv_2d, uv_2d, fields_dav, fields_dav, self.bnd_functions['shallow_water'])
                    prob_pg_ssprk = LinearVariationalProblem(a_pg, l_pg, self.uv_2d_mid)
                    solver_pg_ssprk = LinearVariationalSolver(prob_pg_ssprk)

                    # integrate q_3d in vertical
                 #   q_integrator = VerticalIntegrator(q_3d,
                  #                                self.q_3d_mid,
                   #                               bottom_to_top=False,
                    #                              average=False,
                     #                             bathymetry=self.fields.bathymetry_3d,
                      #                            elevation=self.fields.elev_cg_3d)
                 #   extract_int_q = SubFunctionExtractor(self.q_3d_mid,
                  #                                   self.q_2d_mid,
                   #                                  boundary='bottom', elem_facet='bottom',
                    #                                 elem_height=self.fields.v_elem_size_2d)
                 #   extract_bottom_q = SubFunctionExtractor(self.fields.q_3d,
                  #                                      self.fields.q_2d,
                   #                                     boundary='bottom', elem_facet='bottom',
                    #                                    elem_height=self.fields.v_elem_size_2d)

                   # h_mid = eta_2d + self.bathymetry_dg
                   # timestepper_depth_integrated.F += (self.dt/rho_0*1./h_mid*inner(Dx(self.q_2d_mid*h_mid, 0) - self.fields.q_2d*Dx(self.bathymetry_dg, 0), test_uta_2d)*dx)
                   # prob_with_int_q = NonlinearVariationalProblem(timestepper_depth_integrated.F, self.fields.solution_2d)
                   # solver_with_int_q = NonlinearVariationalSolver(prob_with_int_q, solver_parameters=solver_parameters)
                   # timestepper_depth_integrated.update_solver()

                    # update solver for correcting eta
                   # timestepper_operator_splitting.F += -self.dt*g_grav*inner(Dx(self.elev_2d_mid, 0), test_uta_2d)*dx
                   # timestepper_operator_splitting.update_solver()

                n_stages = 2
                if self.options.use_operator_splitting:
                    assert self.options.solve_elevation_gradient_separately is True, \
                        'Use operator splitting needs solving elevation gradient separately!'
                    for i_stage in range(n_stages):
                        # 2d advance
                        advancing_elev_once = True
                        if self.options.update_free_surface:
                            if i_stage == 0 and (not advancing_elev_once):
                                self.copy_uv_to_uv_dav_3d.solve()
                                self.uv_dav_3d_mid.assign(self.fields.uv_dav_3d)
                                timestepper_operator_splitting_explicit.advance(self.simulation_time, update_forcings)
                                self.copy_elev_to_3d.solve()
                                solver_sigma_dt.solve()
                                solver_sigma_dx.solve()
                                if self.horizontal_domain_is_2d:
                                    solver_sigma_dy.solve()
                            elif i_stage == 1:
                               # self.uv_3d_mid.assign(self.fields.uv_3d)
                               # self.fields.uv_3d.assign(self.uv_3d_old)
                                self.uv_averager.solve()
                                self.extract_surf_dav_uv.solve()
                                self.fields.uv_2d.assign(self.fields.uv_dav_2d)
                                self.fields.elev_2d.assign(self.elev_2d_old)
                                self.copy_uv_dav_to_uv_dav_3d.solve()
                                self.uv_dav_3d_mid.assign(self.fields.uv_dav_3d)
                               # self.fields.uv_3d.assign(self.uv_3d_mid)
                                timestepper_operator_splitting.advance(self.simulation_time, update_forcings)
                                self.copy_elev_to_3d.solve()
                                solver_sigma_dt.solve()
                                solver_sigma_dx.solve()
                                if self.horizontal_domain_is_2d:
                                    solver_sigma_dy.solve()

                        # 3d advance
                        solver_omega.solve()
                        solver_mom_ssprk.solve()
                        self.fields.uv_3d.assign(self.uv_3d_mid)
                        if self.options.use_limiter_for_velocity:
                            self.uv_limiter.apply(self.fields.uv_3d)

                        if i_stage == 0 and (not advancing_elev_once):
                            # update 2d coupling, i.e. including the elevation gradient contribution
                            if self.options.update_free_surface:
                                self.copy_uv_to_uv_dav_3d.solve()
                                self.fields.uv_3d.project(self.fields.uv_3d - (self.uv_dav_3d_mid - self.fields.uv_dav_3d))
                        elif i_stage == 1:
                            self.fields.uv_3d.assign(0.5*(self.uv_3d_old + self.fields.uv_3d))
                            # update 2d coupling, i.e. including the elevation gradient contribution
                            if self.options.update_free_surface:
                                self.copy_uv_to_uv_dav_3d.solve()
                                self.fields.uv_3d.project(self.fields.uv_3d - (self.uv_dav_3d_mid - self.fields.uv_dav_3d))

                            if self.options.use_implicit_vertical_diffusion:
                                if self.options.solve_salinity:
                                    with timed_stage('impl_salt_vdiff'):
                                        self.timestepper.timesteppers.salt_impl.advance(self.simulation_time)
                                if self.options.solve_temperature:
                                    with timed_stage('impl_temp_vdiff'):
                                        self.timestepper.timesteppers.temp_impl.advance(self.simulation_time)
                                with timed_stage('impl_mom_vvisc'):
                                    self.timestepper.timesteppers.mom_impl.advance(self.simulation_time)

                            solver_q.solve()
                            solver_u.solve()
                            if self.options.use_pressure_correction:
                                self.fields.q_3d.assign(self.fields.q_3d + self.q_3d_dq)
                            self.timestepper._update_turbulence(self.simulation_time)
                            self.timestepper._update_bottom_friction()
                            self.timestepper._update_stabilization_params()

                            # compute final free surface elevation
                            if self.options.update_free_surface:
                                self.uv_averager.solve()
                                self.extract_surf_dav_uv.solve()
                                self.fields.uv_2d.assign(self.fields.uv_dav_2d)
                                timestepper_free_surface.solution.assign(self.elev_2d_old)
                                timestepper_free_surface.advance(self.simulation_time, update_forcings)
                                self.fields.elev_2d.assign(self.elev_2d_mid)
                                self.copy_elev_to_3d.solve()
                               # self.calculate_external_pressure_gradient(pressure='elevation')
                                solver_sigma_dt.solve()
                                solver_sigma_dx.solve()
                                if self.horizontal_domain_is_2d:
                                    solver_sigma_dy.solve()
                                    if self.options.set_vertical_2d:
                                        self.fields.uv_3d.dat.data[:, 1] = 0.

                    # TODO modify to use self-defined timestepper
                    solver_omega.solve()
                    if self.options.solve_salinity:
                        self.timestepper.timesteppers.salt_expl.advance(self.simulation_time, update_forcings)
                        if self.options.use_limiter_for_tracers:
                            self.tracer_limiter.apply(self.fields.salt_3d)
                    # update baroclinicity
                    self.timestepper._update_baroclinicity()
                   
                else: # i.e. ssprk used in NHWAVE
                    for i_stage in range(n_stages):
                        # mom advance
                        solver_mom_ssprk.solve()
                        self.fields.uv_3d.assign(self.uv_3d_mid)
                        if self.options.use_limiter_for_velocity:
                            self.uv_limiter.apply(self.fields.uv_3d)

                        advancing_elev_implicitly = False
                        # 2d advance
                        if self.options.solve_elevation_gradient_separately and self.options.update_free_surface:
                            self.uv_averager.solve()
                            self.extract_surf_dav_uv.solve()
                            self.copy_uv_dav_to_uv_dav_3d.solve()
                            self.uv_dav_3d_mid.assign(self.fields.uv_dav_3d)
                            if i_stage == 0 or (not advancing_elev_implicitly):
                                self.uv_2d_mid.assign(self.fields.uv_dav_2d)
                                timestepper_mom_2d.advance(self.simulation_time, update_forcings)
                                self.fields.uv_2d.assign(self.uv_2d_mid)
                            else:
                                self.fields.uv_2d.assign(self.fields.uv_dav_2d)
                                self.fields.elev_2d.assign(self.elev_2d_old)
                                timestepper_operator_splitting.advance(self.simulation_time, update_forcings)

                            # couple 2d (elevation gradient) into 3d
                            self.copy_uv_to_uv_dav_3d.solve()
                            self.fields.uv_3d.assign(self.fields.uv_3d + (self.fields.uv_dav_3d - self.uv_dav_3d_mid))

                        # update non-hydrostatic pressure
                        solver_q.solve()
                        solver_u.solve()
                        if self.options.use_pressure_correction:
                            self.fields.q_3d.assign(self.fields.q_3d + self.q_3d_dq)

                        if i_stage == 1:
                            self.fields.uv_3d.assign(0.5*(self.uv_3d_old + self.fields.uv_3d))
                            if self.options.use_implicit_vertical_diffusion:
                                if self.options.solve_salinity:
                                    with timed_stage('impl_salt_vdiff'):
                                        self.timestepper.timesteppers.salt_impl.advance(self.simulation_time)
                                if self.options.solve_temperature:
                                    with timed_stage('impl_temp_vdiff'):
                                        self.timestepper.timesteppers.temp_impl.advance(self.simulation_time)
                                with timed_stage('impl_mom_vvisc'):
                                    self.timestepper.timesteppers.mom_impl.advance(self.simulation_time)
                            self.timestepper._update_turbulence(self.simulation_time)
                            self.timestepper._update_bottom_friction()
                            self.timestepper._update_stabilization_params()

                        # update free surface elevation
                        if self.options.update_free_surface:
                            self.uv_averager.solve()
                            self.extract_surf_dav_uv.solve()
                            self.fields.uv_2d.assign(self.fields.uv_dav_2d)
                            self.elev_2d_mid.assign(self.elev_2d_old)
                            timestepper_free_surface.advance(self.simulation_time, update_forcings)
                            self.fields.elev_2d.assign(self.elev_2d_mid)
                            self.copy_elev_to_3d.solve()
                            solver_sigma_dt.solve()
                            solver_sigma_dx.solve()
                            if self.horizontal_domain_is_2d:
                                solver_sigma_dy.solve()
                          #  solver_pg_ssprk.solve()
                          #  self.fields.uv_dav_2d.assign(self.uv_2d_mid)
                          #  self.copy_uv_dav_to_uv_dav_3d.solve()
                          #  self.fields.ext_pg_3d.assign(self.fields.uv_dav_3d)

                        # tracer advance
                        solver_omega.solve()
                        if self.options.solve_salinity:
                            solver_salt_ssprk.solve()
                            self.fields.salt_3d.assign(self.salt_3d_mid)
                            if self.options.use_limiter_for_tracers:
                                self.tracer_limiter.apply(self.fields.salt_3d)
                            if i_stage == 1:
                                self.fields.salt_3d.assign(0.5*(self.salt_3d_old + self.fields.salt_3d))
                        if self.options.solve_temperature:
                            solver_temp_ssprk.solve()
                            self.fields.temp_3d.assign(self.temp_3d_mid)
                            if self.options.use_limiter_for_tracers:
                                self.tracer_limiter.apply(self.fields.temp_3d)
                            if i_stage == 1:
                                self.fields.temp_3d.assign(0.5*(self.temp_3d_old + self.fields.temp_3d))
                        # update baroclinicity
                        self.timestepper._update_baroclinicity()

            # Move to next time step
            self.simulation_time += self.dt
            self.iteration += 1

            self.callbacks.evaluate(mode='timestep')

            # Write the solution to file
            if self.simulation_time >= self.next_export_t - t_epsilon:
                self.i_export += 1
                self.next_export_t += self.options.simulation_export_time

                cputime = time_mod.clock() - cputimestamp
                cputimestamp = time_mod.clock()
                self.print_state(cputime)

                # exporter with wetting-drying handle
                self.solution_2d_tmp.assign(self.fields.solution_2d)
                self.solution_ls_tmp.assign(self.fields.solution_ls)
                H = self.bathymetry_dg.dat.data + elev_2d.dat.data
                h_ls = self.bathymetry_ls.dat.data + elev_ls.dat.data
                ind = np.where(H[:] <= 0.)[0]
                ind_ls = np.where(h_ls[:] <= 0.)[0]
                elev_2d.dat.data[ind] = 1E-6 - self.bathymetry_dg.dat.data[ind]
                elev_ls.dat.data[ind_ls] = 1E-6 - self.bathymetry_ls.dat.data[ind_ls]
                self.export()
                self.fields.solution_2d.assign(self.solution_2d_tmp)
                self.fields.solution_ls.assign(self.solution_ls_tmp)

                if export_func is not None:
                    export_func()

                if hydrostatic_solver_2d:
                    print_output('Adopting 2d hydrostatic solver with P{degree:} {element:} ...'.
                                 format(degree=self.options.polynomial_degree, element=self.options.element_family))
                elif hydrostatic_solver_3d:
                    print_output('Adopting 3d hydrostatic solver with P{degree:} {element:} ...'.
                                 format(degree=self.options.polynomial_degree, element=self.options.element_family))
                elif conventional_3d_NH_solver:
                    print_output('Adopting 3d non-hydrostatic solver with P{degree:} {element:} ...'.
                                 format(degree=self.options.polynomial_degree, element=self.options.element_family))
                elif one_layer_NH_solver:
                    print_output('Adopting one-layer non-hydrostatic solver with P{degree:} {element:} ...'.
                                 format(degree=self.options.polynomial_degree, element=self.options.element_family))
                elif reduced_two_layer_NH_solver:
                    print_output('Adopting reduced two-layer non-hydrostatic solver with P{degree:} {element:} ...'.
                                 format(degree=self.options.polynomial_degree, element=self.options.element_family))
                elif coupled_two_layer_NH_solver:
                    print_output('Adopting coupled two-layer non-hydrostatic solver with P{degree:} {element:} ...'.
                                 format(degree=self.options.polynomial_degree, element=self.options.element_family))
                elif coupled_three_layer_NH_solver:
                    print_output('Adopting coupled three-layer non-hydrostatic solver with P{degree:} {element:} ...'.
                                 format(degree=self.options.polynomial_degree, element=self.options.element_family))
                else:
                    print_output('Adopting {nlayer:}-layer non-hydrostatic solver with P{degree:} {element:} ...'.
                                 format(nlayer=self.n_layers, degree=self.options.polynomial_degree, element=self.options.element_family))

