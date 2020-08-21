"""
Module for 3D barotropic/baroclinic non-hydrostatic solver
"""
from __future__ import absolute_import
from .utility_nh import *
from . import shallowwater_nh
from . import fluid_slide
from . import momentum_ale
from . import tracer_ale
from . import sediment_ale
from . import coupled_timeintegrator_nh
from . import turbulence_ale
from .. import timeintegrator
from .. import rungekutta
from . import limiter_nh as limiter
import time as time_mod
from mpi4py import MPI
from .. import exporter
import weakref
from ..field_defs import field_metadata
from ..options import ModelOptions3d
from .. import callback
from ..log import *
from collections import OrderedDict


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

        solver_obj = solver_nh.FlowSolver(mesh2d, bathymetry_2d, n_layers=6)
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
        self.mesh = extrude_mesh_sigma(mesh2d, n_layers, bathymetry_2d, **extrude_options)
        """3D :class`Mesh`"""
        self.comm = mesh2d.comm

        self.horizontal_domain_is_2d = self.mesh2d.geometric_dimension() == 2
        if self.horizontal_domain_is_2d:
            self.vert_ind = 2
        else:
            self.vert_ind = 1
        self.normal = FacetNormal(self.mesh)
        self.boundary_markers = self.mesh.exterior_facets.unique_markers

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
                              'sediment': {},
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
        self.function_spaces.P2_2d = get_functionspace(self.mesh2d, 'CG', 2, name='P2_2d')
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
        if self.options.use_baroclinic_formulation:
            if self.options.use_quadratic_density:
                self.fields.density_3d = Function(self.function_spaces.P2DGxP2, name='Density')
            else:
                self.fields.density_3d = Function(self.function_spaces.H, name='Density')
            self.fields.baroc_head_3d = Function(self.function_spaces.H_bhead)
            self.fields.int_pg_3d = Function(self.function_spaces.U_int_pg, name='int_pg_3d')
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
                self.turbulence_model = turbulence_ale.GenericLengthScaleModel(
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
                self.turbulence_model = turbulence_ale.PacanowskiPhilanderModel(
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

        # ----- creates extra functions for nh extension
        self.fields.q_3d = Function(FunctionSpace(self.mesh, 'CG', self.options.polynomial_degree+1))
        self.bathymetry_dg = Function(self.function_spaces.H_2d).project(self.bathymetry_cg_2d)
        self.bathymetry_ls = Function(self.function_spaces.H_2d).project(self.bathymetry_cg_2d)
        self.elev_2d_old = Function(self.function_spaces.H_2d)
        self.elev_2d_fs = Function(self.function_spaces.H_2d)

        self.uv_2d_dg = Function(self.function_spaces.P1DGv_2d)
        self.uv_2d_old = Function(self.function_spaces.U_2d)
        self.uv_2d_mid = Function(self.function_spaces.U_2d)
        self.uv_dav_3d_mid = Function(self.function_spaces.U)
        self.solution_2d_tmp = Function(self.function_spaces.V_2d)

        # functions for landslide modelling
        self.fields.solution_ls = Function(self.function_spaces.V_2d)
        self.solution_ls_old = Function(self.function_spaces.V_2d)
        self.solution_ls_tmp = Function(self.function_spaces.V_2d)
        self.fields.uv_ls, self.fields.elev_ls = self.fields.solution_ls.split()
        self.fields.slide_source_2d = Function(self.function_spaces.H_2d)
        self.fields.slide_source_3d = Function(self.function_spaces.H)
        self.fields.h_ls = Function(self.function_spaces.H_2d)
        self.h_ls_old = Function(self.function_spaces.H_2d)

        # for sediment transport
        if self.options.solve_sediment:
            self.fields.c_3d = Function(self.function_spaces.H)

        self._isfrozen = True

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

        self.eq_operator_split = shallowwater_nh.ModeSplit2DEquations(
            self.fields.solution_2d.function_space(),
            self.bathymetry_dg,
            self.options)
        self.eq_operator_split.bnd_functions = self.bnd_functions['shallow_water']

        self.eq_sw_nh = shallowwater_nh.ShallowWaterEquations(
            self.fields.solution_2d.function_space(),
            self.bathymetry_dg,
            self.options)

        self.eq_mom_2d = shallowwater_nh.MomentumEquation2D(
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

        if self.options.slide_is_viscous_fluid:
            self.eq_ls = fluid_slide.LiquidSlideEquations(
            self.fields.solution_ls.function_space(),
            self.bathymetry_ls,
            self.options)
            self.eq_ls.bnd_functions = self.bnd_functions['landslide_motion']

        # sediment transport equation
        if self.options.solve_sediment:
            assert (not self.options.solve_salinity) and (not self.options.solve_temperature), \
                   'Sediment transport equation is being solved... \
                    Temporarily it is not supported to solve other tracers simultaneously.'
            self.eq_sediment = sediment_ale.SedimentEquation(self.fields.c_3d.function_space(),
                                                            bathymetry=self.fields.bathymetry_3d,
                                                            v_elem_size=self.fields.v_elem_size_3d,
                                                            h_elem_size=self.fields.h_elem_size_3d,
                                                            use_lax_friedrichs=self.options.use_lax_friedrichs_velocity,
                                                            use_symmetric_surf_bnd=self.options.element_family == 'dg-dg')
            self.eq_sediment.bnd_functions = self.bnd_functions['sediment']
            if self.options.use_implicit_vertical_diffusion:
                self.eq_sediment_vdff = sediment_ale.SedimentEquation(self.fields.c_3d.function_space(),
                                                             bathymetry=self.fields.bathymetry_3d,
                                                             v_elem_size=self.fields.v_elem_size_3d,
                                                             h_elem_size=self.fields.h_elem_size_3d,
                                                             use_lax_friedrichs=self.options.use_lax_friedrichs_tracer)

        expl_bottom_friction = self.options.use_bottom_friction and not self.options.use_implicit_vertical_diffusion
        self.eq_momentum = momentum_ale.MomentumEquation(self.fields.uv_3d.function_space(),
                                                         bathymetry=self.fields.bathymetry_3d,
                                                         v_elem_size=self.fields.v_elem_size_3d,
                                                         h_elem_size=self.fields.h_elem_size_3d,
                                                         use_nonlinear_equations=self.options.use_nonlinear_equations,
                                                         use_lax_friedrichs=self.options.use_lax_friedrichs_velocity,
                                                         use_bottom_friction=expl_bottom_friction)
        self.eq_momentum.bnd_functions = self.bnd_functions['momentum']

        if self.options.use_implicit_vertical_diffusion:
            self.eq_vertmomentum = momentum_ale.MomentumEquation(self.fields.uv_3d.function_space(),
                                                                 bathymetry=self.fields.bathymetry_3d,
                                                                 v_elem_size=self.fields.v_elem_size_3d,
                                                                 h_elem_size=self.fields.h_elem_size_3d,
                                                                 use_nonlinear_equations=False, # i.e. advection terms neglected
                                                                 use_lax_friedrichs=self.options.use_lax_friedrichs_velocity,
                                                                 use_bottom_friction=self.options.use_bottom_friction)
        if self.options.solve_salinity:
            self.eq_salt = tracer_ale.TracerEquation(self.fields.salt_3d.function_space(),
                                                     bathymetry=self.fields.bathymetry_3d,
                                                     v_elem_size=self.fields.v_elem_size_3d,
                                                     h_elem_size=self.fields.h_elem_size_3d,
                                                     use_lax_friedrichs=self.options.use_lax_friedrichs_tracer,
                                                     use_symmetric_surf_bnd=self.options.element_family == 'dg-dg')
            self.eq_salt.bnd_functions = self.bnd_functions['salt']
            if self.options.use_implicit_vertical_diffusion:
                self.eq_salt_vdff = tracer_ale.TracerEquation(self.fields.salt_3d.function_space(),
                                                              bathymetry=self.fields.bathymetry_3d,
                                                              v_elem_size=self.fields.v_elem_size_3d,
                                                              h_elem_size=self.fields.h_elem_size_3d,
                                                              use_lax_friedrichs=self.options.use_lax_friedrichs_tracer)

        if self.options.solve_temperature:
            self.eq_temp = tracer_ale.TracerEquation(self.fields.temp_3d.function_space(),
                                                     bathymetry=self.fields.bathymetry_3d,
                                                     v_elem_size=self.fields.v_elem_size_3d,
                                                     h_elem_size=self.fields.h_elem_size_3d,
                                                     use_lax_friedrichs=self.options.use_lax_friedrichs_tracer,
                                                     use_symmetric_surf_bnd=self.options.element_family == 'dg-dg')
            self.eq_temp.bnd_functions = self.bnd_functions['temp']
            if self.options.use_implicit_vertical_diffusion:
                self.eq_temp_vdff = tracer_ale.TracerEquation(self.fields.temp_3d.function_space(),
                                                              bathymetry=self.fields.bathymetry_3d,
                                                              v_elem_size=self.fields.v_elem_size_3d,
                                                              h_elem_size=self.fields.h_elem_size_3d,
                                                              use_lax_friedrichs=self.options.use_lax_friedrichs_tracer)

        if self.options.use_turbulence and self.options.turbulence_model_type == 'gls':
            if self.options.use_turbulence_advection:
                # explicit advection equations
                self.eq_tke_adv = tracer_ale.TracerEquation(self.fields.tke_3d.function_space(),
                                                           bathymetry=self.fields.bathymetry_3d,
                                                           v_elem_size=self.fields.v_elem_size_3d,
                                                           h_elem_size=self.fields.h_elem_size_3d,
                                                           use_lax_friedrichs=self.options.use_lax_friedrichs_tracer)
                self.eq_psi_adv = tracer_ale.TracerEquation(self.fields.psi_3d.function_space(),
                                                           bathymetry=self.fields.bathymetry_3d,
                                                           v_elem_size=self.fields.v_elem_size_3d,
                                                           h_elem_size=self.fields.h_elem_size_3d,
                                                           use_lax_friedrichs=self.options.use_lax_friedrichs_tracer)
            # implicit vertical diffusion eqn with production terms
            self.eq_tke_diff = turbulence_ale.TKEEquation(self.fields.tke_3d.function_space(),
                                                      self.turbulence_model,
                                                      bathymetry=self.fields.bathymetry_3d,
                                                      v_elem_size=self.fields.v_elem_size_3d,
                                                      h_elem_size=self.fields.h_elem_size_3d)
            self.eq_psi_diff = turbulence_ale.PsiEquation(self.fields.psi_3d.function_space(),
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
        self.w_solver = VerticalVelocitySolver(self.fields.w_3d,
                                               self.fields.uv_3d,
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
                                              average=True,
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
            if self.options.solve_sediment:
                self.density_solver = DensitySolverSediment(self.fields.c_3d, self.fields.density_3d, 
                                                            self.options.rho_slide, self.options.rho_fluid)
            elif self.options.use_quadratic_density:
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
            self.int_pg_calculator = momentum_ale.InternalPressureGradientCalculator(
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
        if self.options.use_bottom_friction:
            self.extract_uv_bottom = SubFunctionExtractor(self.fields.uv_p1_3d, self.fields.uv_bottom_2d,
                                                          boundary='bottom', elem_facet='average',
                                                          elem_height=self.fields.v_elem_size_2d)
            self.extract_z_bottom = SubFunctionExtractor(self.fields.z_coord_3d, self.fields.z_bottom_2d,
                                                         boundary='bottom', elem_facet='average',
                                                         elem_height=self.fields.v_elem_size_2d)
        self.mesh_updater = ALEMeshUpdater(self)

        if self.options.use_smagorinsky_viscosity:
            self.smagorinsky_diff_solver = SmagorinskyViscosity(self.fields.uv_p1_3d, self.fields.smag_visc_3d,
                                                                self.options.smagorinsky_coefficient, self.fields.h_elem_size_3d,
                                                                self.fields.max_h_diff,
                                                                weak_form=self.options.polynomial_degree == 0)
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
                                  elev_slide=None, uv_slide=None, sedi=None, h_ls=None):
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
        if self.options.landslide:
            uv_ls, elev_ls = self.fields.solution_ls.split()
            if elev_slide is not None:
                elev_ls.project(elev_slide)
            if uv_slide is not None:
                uv_ls.project(uv_slide)
            if h_ls is not None:
                self.fields.h_ls.project(h_ls)

        if sedi is not None and self.options.solve_sediment:
            self.fields.c_3d.project(sedi)

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
        for e in self.exporters.values():
            e.export()

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
        if (not self.horizontal_domain_is_2d) and sponge_is_2d:
            xvector = mesh.coordinates.dat.data[:]
        else:
            xvector = mesh.coordinates.dat.data[:, 0]
            yvector = mesh.coordinates.dat.data[:, 1]
            assert yvector.shape[0] == damp_vector.shape[0]
            if yvector.max() <= sponge_start_point[1] + length[1]:
                length[1] = yvector.max() - sponge_start_point[1]
        assert xvector.shape[0] == damp_vector.shape[0]
        if xvector.max() <= sponge_start_point[0] + length[0]:
            length[0] = xvector.max() - sponge_start_point[0]

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

    def set_vertical_2d(self):
        """
        Set zero in y- direction, forming artificial vertical two dimensional.
        """
        if self.horizontal_domain_is_2d:
            uv_2d, elev_2d = self.fields.solution_2d.split()
            self.uv_2d_dg.project(uv_2d)
            self.uv_2d_dg.sub(1).assign(0.)
            uv_2d.project(self.uv_2d_dg)
            # landslide
            uv_ls, elev_ls = self.fields.solution_ls.split()
            self.uv_2d_dg.project(uv_ls)
            self.uv_2d_dg.sub(1).assign(0.)
            uv_ls.project(self.uv_2d_dg)

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

        # initial export
        self.print_state(0.0)
        if self.export_initial_state:
            self.export()
            if export_func is not None:
                export_func()
            if 'vtk' in self.exporters:
                self.exporters['vtk'].export_bathymetry(self.fields.bathymetry_2d)

        initial_simulation_time = self.simulation_time
        internal_iteration = 0

        if self.options.set_vertical_2d: # True for 1D case
            self.set_vertical_2d()

        # ----- Self-defined time integrators
        fields_2d = {
                    'linear_drag_coefficient': self.options.linear_drag_coefficient,
                    'quadratic_drag_coefficient': self.options.quadratic_drag_coefficient,
                    'manning_drag_coefficient': self.options.manning_drag_coefficient,
                    'viscosity_h': self.options.horizontal_viscosity,
                    'lax_friedrichs_velocity_scaling_factor': self.options.lax_friedrichs_velocity_scaling_factor,
                    'coriolis': self.options.coriolis_frequency,
                    'wind_stress': self.options.wind_stress,
                    'atmospheric_pressure': self.options.atmospheric_pressure,
                    'momentum_source': self.options.momentum_source_2d,
                    'volume_source': self.options.volume_source_2d,
                    'eta': self.fields.elev_2d,
                    'uv': self.fields.uv_2d,
                    'slide_source': self.fields.slide_source_2d,
                    'sponge_damping_2d': self.set_sponge_damping(self.options.sponge_layer_length, self.options.sponge_layer_start, alpha=10., sponge_is_2d=True),}

        solver_parameters = {'snes_type': 'newtonls', # ksponly, newtonls
                             'ksp_type': 'gmres', # gmres, preonly
                             'pc_type': 'fieldsplit'}

        # timestepper for operator splitting in 3D NH solver
        timestepper_operator_split = timeintegrator.CrankNicolson(self.eq_operator_split, self.fields.solution_2d,
                                                              fields_2d, self.dt, bnd_conditions=self.bnd_functions['shallow_water'],
                                                              solver_parameters=solver_parameters,
                                                              semi_implicit=False,
                                                              theta=0.5)
        timestepper_operator_split_explicit = timeintegrator.CrankNicolson(self.eq_operator_split, self.fields.solution_2d,
                                                              fields_2d, self.dt, bnd_conditions=self.bnd_functions['shallow_water'],
                                                              solver_parameters=solver_parameters,
                                                              semi_implicit=False,
                                                              theta=0.)
        # timestepper for depth-integrated NH solver
        timestepper_depth_integrated = timeintegrator.CrankNicolson(self.eq_sw_nh, self.fields.solution_2d,
                                                              fields_2d, self.dt,
                                                              bnd_conditions=self.bnd_functions['shallow_water'],
                                                              solver_parameters=solver_parameters,
                                                              semi_implicit=False,
                                                              theta=0.5)
        # timestepper for free surface equation
        timestepper_free_surface = timeintegrator.CrankNicolson(self.eq_free_surface, self.elev_2d_fs,
                                                              fields_2d, self.dt,
                                                              bnd_conditions=self.bnd_functions['shallow_water'],
                                                              # solver_parameters=solver_parameters,
                                                              semi_implicit=False,
                                                              theta=0.5)
        timestepper_free_surface_implicit = timeintegrator.CrankNicolson(self.eq_free_surface, self.elev_2d_fs,
                                                              fields_2d, self.dt,
                                                              bnd_conditions=self.bnd_functions['shallow_water'],
                                                              # solver_parameters=solver_parameters,
                                                              semi_implicit=False,
                                                              theta=1.0)
        # timestepper for only elevation gradient term
        timestepper_mom_2d = timeintegrator.CrankNicolson(self.eq_mom_2d, self.uv_2d_mid,
                                                              fields_2d, self.dt,
                                                              bnd_conditions=self.bnd_functions['shallow_water'],
                                                              # solver_parameters=solver_parameters,
                                                              semi_implicit=False,
                                                              theta=0.)

        # solver for viscous fluid landslide
        if self.options.landslide and self.options.slide_is_viscous_fluid:
            uv_ls_old, elev_ls_old = self.solution_ls_old.split()
            uv_ls, elev_ls = self.fields.solution_ls.split()
            theta = 0.5
            solution_if_semi = self.fields.solution_ls
            F_liquid_ls = (self.eq_ls.mass_term(self.fields.solution_ls) - self.eq_ls.mass_term(self.solution_ls_old) - self.dt*(
                            theta*self.eq_ls.residual('all', self.fields.solution_ls, solution_if_semi, fields, fields, self.bnd_functions['landslide_motion']) + 
                            (1-theta)*self.eq_ls.residual('all', self.solution_ls_old, self.solution_ls_old, fields, fields, self.bnd_functions['landslide_motion'])) - 
                            self.dt*self.eq_ls.add_external_surface_term(self.uv_2d_old, self.elev_2d_old, self.uv_2d_old, elev_ls_old, fields, self.bnd_functions['shallow_water']) 
                           )
            prob_liquid_ls = NonlinearVariationalProblem(F_liquid_ls, self.fields.solution_ls)
            solver_liquid_ls = NonlinearVariationalSolver(prob_liquid_ls,
                                                          solver_parameters={'snes_type': 'newtonls', # ksponly, newtonls
                                                                             'ksp_type': 'gmres', # gmres, preonly
                                                                             'pc_type': 'fieldsplit'})
        # Poisson solver for the non-hydrostatic pressure
        assert self.options.use_pressure_correction is False, \
               'Pressure correction method is temporarily implemented in only sigma model.'
        q_3d = self.fields.q_3d
        fs_q = q_3d.function_space()
        q_is_dg = element_continuity(fs_q.ufl_element()).horizontal == 'dg'
        uv_3d = self.fields.uv_3d
        Const = physical_constants['rho0']/self.dt
       # if self.fields.density_3d is not None:
       #     Const += self.fields.density_3d/self.dt
        trial_q = TrialFunction(fs_q)
        test_q = TestFunction(fs_q)

        # nabla^2-term is integrated by parts
        a_q = dot(grad(test_q), grad(trial_q)) * dx #+ test_q*inner(grad(q), normal)*ds_surf
        l_q = Const * dot(grad(test_q), uv_3d) * dx
        if self.options.landslide:
            l_q += -Const*self.fields.slide_source_3d*self.normal[self.vert_ind]*test_q*ds_bottom

        if q_is_dg:
            degree_h, degree_v = fs_q.ufl_element().degree()
            if self.horizontal_domain_is_2d:
                elemsize = (self.fields.h_elem_size_3d*(self.normal[0]**2 + self.normal[1]**2) +
                            self.fields.v_elem_size_3d*self.normal[2]**2)
            else:
                elemsize = (self.fields.h_elem_size_3d*self.normal[0]**2 +
                            self.fields.v_elem_size_3d*self.normal[1]**2)
            sigma = 5.0*degree_h*(degree_h + 1)/elemsize
            if degree_h == 0:
                sigma = 1.5/elemsize
            alpha_q = avg(sigma)
            # Nitsche proved that if gamma_q is taken as η/h, where
            # h is the element size and η is a sufficiently large constant, then the discrete solution
            # converges to the exact solution with optimal order in H1 and L2.
            gamma_q = 2*sigma*Const # 1E10

            a_q += - dot(avg(grad(test_q)), jump(trial_q, self.normal))*(dS_v + dS_h) \
                   - dot(jump(test_q, self.normal), avg(grad(trial_q)))*(dS_v + dS_h) \
                   + alpha_q*dot(jump(test_q, self.normal), jump(trial_q, self.normal))*(dS_v + dS_h)

            incompressibility_flux_type = 'central'
            if incompressibility_flux_type == 'central':
                u_flux = avg(uv_3d)
            elif incompressibility_flux_type == 'upwind':
                switch = conditional(
                            gt(abs(dot(uv_3d, self.normal))('+'), 0.0), 1.0, 0.0
                            )
                u_flux = switch * uv_3d('+') + (1 - switch) * uv_3d('-')

            l_q += -Const * dot(u_flux, self.normal('+')) * jump(test_q) * (dS_v + dS_h)
           # l_q += -Const * dot(uv_3d, self.normal) * test_q * ds_surf
           # l_q = -Const * div(uv_3d) * test_q * dx
            # zero Dirichlet top boundary
            q0 = Constant(0.)
            a_q += - dot(grad(test_q), trial_q*self.normal)*ds_surf \
                   - dot(test_q*self.normal, grad(trial_q))*ds_surf \
                   + gamma_q*test_q*trial_q*ds_surf
            l_q += -q0*dot(grad(test_q), self.normal)*ds_surf + gamma_q*q0*test_q*ds_surf

        # boundary conditions: to refer to the top and bottom use "top" and "bottom"
        # for other boundaries use the normal numbers (ids) from the horizontal mesh
        # (UnitSquareMesh automatically defines 1,2,3, and 4)
        bc_top = DirichletBC(fs_q, 0., "top")
        bcs = [bc_top]
        if not self.options.update_free_surface:
            bcs = []
        for bnd_marker in self.boundary_markers:
            func = self.bnd_functions['shallow_water'].get(bnd_marker)
            ds_bnd = ds_v(int(bnd_marker))
            if func is not None: # TODO set more general and accurate conditional statement
                bc = DirichletBC(fs_q, 0., int(bnd_marker))
                bcs.append(bc)
                if q_is_dg:
                    a_q += - dot(grad(test_q), trial_q*self.normal)*ds_bnd \
                           - dot(test_q*self.normal, grad(trial_q))*ds_bnd \
                           + gamma_q*test_q*trial_q*ds_bnd
                    l_q += -q0*dot(grad(test_q), self.normal)*ds_bnd + gamma_q*q0*test_q*ds_bnd
                l_q += -Const * dot(uv_3d, self.normal) * test_q * ds_bnd

        # you can add Dirichlet bcs to other boundaries if needed
        # any boundary that is not specified gets the natural zero Neumann bc
        prob_q = LinearVariationalProblem(a_q, l_q, q_3d, bcs=bcs)
        if q_is_dg:
            prob_q = LinearVariationalProblem(a_q, l_q, q_3d)
        solver_q = LinearVariationalSolver(prob_q,
                                           solver_parameters={'snes_type': 'ksponly',#'newtonls''ksponly', final: 'ksponly'
                                                              'ksp_type': 'gmres',#'gmres''preonly',              'gmres'
                                                              'pc_type': 'gamg'},#'ilu''gamg',                     'ilu'
                                           options_prefix='poisson_solver')

        # solver for updating uv_3d
        tri_uv_3d = TrialFunction(self.function_spaces.U)
        test_uv_3d = TestFunction(self.function_spaces.U)
        a_u = dot(tri_uv_3d, test_uv_3d)*dx
        l_u = dot(uv_3d - self.dt/physical_constants['rho0']*grad(q_3d), test_uv_3d)*dx
        prob_u = LinearVariationalProblem(a_u, l_u, uv_3d)
        solver_u = LinearVariationalSolver(prob_u)

        while self.simulation_time <= self.options.simulation_end_time - t_epsilon:

            # Original mode-splitting method
            #self.timestepper.advance(self.simulation_time,
            #                         update_forcings, update_forcings3d)

            self.uv_2d_old.assign(self.fields.uv_2d)
            self.elev_2d_old.assign(self.fields.elev_2d)
            self.elev_2d_fs.assign(self.fields.elev_2d)
            self.solution_ls_old.assign(self.fields.solution_ls)

            if self.options.landslide and self.options.slide_is_viscous_fluid:
                if update_forcings is not None:
                    update_forcings(self.simulation_time + self.dt)
                if self.simulation_time <= self.options.t_landslide:
                    solver_liquid_ls.solve()
                    # replace bathymetry by landslide position
                    self.bathymetry_dg.project(-(self.eq_ls.water_height_displacement(elev_ls) + elev_ls))
                else:
                    print_output('Landslide motion has been stopped, and waves continue propagating')

            hydrostatic_solver_2d = False # TODO use more common set control
            hydrostatic_solver_3d = False
            conventional_3d_NH_solver = True

            # --- Hydrostatic solver ---
            if hydrostatic_solver_2d:
                if self.options.landslide and self.options.slide_is_viscous_fluid:
                    if self.simulation_time <= t_epsilon:
                        timestepper_depth_integrated.F += -self.dt*self.eq_sw_nh.add_landslide_term(uv_ls, elev_ls, fields, self.bathymetry_ls, self.bnd_functions['landslide_motion'])
                        timestepper_depth_integrated.update_solver()
                    if self.simulation_time == self.options.t_landslide:
                        timestepper_depth_integrated.F += self.dt*self.eq_sw_nh.add_landslide_term(uv_ls, elev_ls, fields, self.bathymetry_ls, self.bnd_functions['landslide_motion'])
                        timestepper_depth_integrated.update_solver()

                timestepper_depth_integrated.advance(self.simulation_time, update_forcings)

                if self.options.set_vertical_2d:
                    self.set_vertical_2d()

            elif hydrostatic_solver_3d:
                #self.timestepper.advance(self.simulation_time,
                #                         update_forcings, update_forcings3d)
                self.bathymetry_cg_2d.project(self.bathymetry_dg)
                ExpandFunctionTo3d(self.bathymetry_cg_2d, self.fields.bathymetry_3d).solve()
                n_stages = 2
                if True:
                    for i_stage in range(n_stages):
                        ## 2D advance
                        if i_stage == 1 and self.options.update_free_surface and self.options.solve_separate_elevation_gradient:
                            self.timestepper.store_elevation(i_stage - 1)
                            self.uv_averager.solve()
                            self.extract_surf_dav_uv.solve()
                            self.fields.uv_2d.assign(self.fields.uv_dav_2d)
                            self.copy_uv_dav_to_uv_dav_3d.solve()
                            self.uv_dav_3d_mid.assign(self.fields.uv_dav_3d)
                            timestepper_operator_split.advance(self.simulation_time, update_forcings)
                            #self.timestepper.timesteppers.swe2d.solve_stage(i_stage, self.simulation_time, update_forcings)
                            # compute mesh velocity
                            self.timestepper.compute_mesh_velocity(i_stage - 1)

                        ## 3D advance in old mesh
                        # salt_eq
                        if self.options.solve_salinity:
                            self.timestepper.timesteppers.salt_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # tmp_eq
                        if self.options.solve_temperature:
                            self.timestepper.timesteppers.temp_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # sediment_eq
                        if self.options.solve_sediment:
                            self.timestepper.timesteppers.sediment_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # turb_advection
                        if 'psi_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.psi_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        if 'tke_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.tke_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # momentum_eq
                        self.timestepper.timesteppers.mom_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)

                        ## update mesh
                        if self.options.update_free_surface:
                            self.copy_elev_to_3d.solve()
                            if self.options.use_ale_moving_mesh:
                                self.mesh_updater.update_mesh_coordinates()

                        ## solve 3D
                        # salt_eq
                        if self.options.solve_salinity:
                            self.timestepper.timesteppers.salt_expl.solve_stage(i_stage)
                            if self.options.use_limiter_for_tracers:
                                self.tracer_limiter.apply(self.fields.salt_3d)
                        # temp_eq
                        if self.options.solve_temperature:
                            self.timestepper.timesteppers.temp_expl.solve_stage(i_stage)
                            if self.options.use_limiter_for_tracers:
                                self.tracer_limiter.apply(self.fields.temp_3d)
                        # sediment_eq
                        if self.options.solve_sediment:
                            self.timestepper.timesteppers.sediment_expl.solve_stage(i_stage)
                            if self.options.use_limiter_for_tracers:
                                self.tracer_limiter.apply(self.fields.c_3d)
                        # turb_advection
                        if 'psi_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.psi_expl.solve_stage(i_stage)
                        if 'tke_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.tke_expl.solve_stage(i_stage)
                        # momentum_eq
                        self.timestepper.timesteppers.mom_expl.solve_stage(i_stage)
                        if self.options.use_limiter_for_velocity:
                            self.uv_limiter.apply(self.fields.uv_3d)

                        last_stage = i_stage == n_stages - 1

                        if last_stage:
                            ## compute final prognostic variables
                            # correct uv_3d
                            if self.options.update_free_surface and self.options.solve_separate_elevation_gradient:
                                self.copy_uv_to_uv_dav_3d.solve()
                                self.fields.uv_3d.project(self.fields.uv_3d - (self.uv_dav_3d_mid - self.fields.uv_dav_3d))
                            if self.options.use_implicit_vertical_diffusion:
                                if self.options.solve_salinity:
                                    with timed_stage('impl_salt_vdiff'):
                                        self.timestepper.timesteppers.salt_impl.advance(self.simulation_time)
                                if self.options.solve_temperature:
                                    with timed_stage('impl_temp_vdiff'):
                                        self.timestepper.timesteppers.temp_impl.advance(self.simulation_time)
                                if self.options.solve_sediment:
                                    with timed_stage('impl_sediment_vdiff'):
                                        self.timestepper.timesteppers.sediment_impl.advance(self.simulation_time)
                                with timed_stage('impl_mom_vvisc'):
                                    self.timestepper.timesteppers.mom_impl.advance(self.simulation_time)
                            ## compute final diagnostic fields
                            # update baroclinicity
                            self.timestepper._update_baroclinicity()
                            # update w
                            self.fields.uv_3d.dat.data[:, self.vert_ind] = 0. # TODO
                            self.w_solver.solve()
                            # update parametrizations
                            self.timestepper._update_turbulence(self.simulation_time)
                            self.timestepper._update_stabilization_params()
                            self.fields.uv_3d.dat.data[:, self.vert_ind] = self.fields.w_3d.dat.data[:, self.vert_ind] # TODO
                        else:
                            ## update variables that explict solvers depend on
                            # correct uv_3d
                          #  self.copy_uv_to_uv_dav_3d.solve()
                          #  self.fields.uv_3d.project(self.fields.uv_3d - (self.uv_dav_3d_mid - self.fields.uv_dav_3d))
                            # update baroclinicity
                            self.timestepper._update_baroclinicity()
                            # update w
                            self.fields.uv_3d.dat.data[:, self.vert_ind] = 0. # TODO
                            self.w_solver.solve()
                            self.fields.uv_3d.dat.data[:, self.vert_ind] = self.fields.w_3d.dat.data[:, self.vert_ind] # TODO

            # --- Non-hydrostatic solver ---
            # Comparing non-hydrostatic extensions to a discontinuous finite element coastal ocean model
            # Pan et al., 2020. doi: https://doi.org/10.1016/j.ocemod.2020.101634
            elif conventional_3d_NH_solver:

                if self.options.landslide: # for rigid landslide motion
                    if update_forcings is not None:
                        update_forcings(self.simulation_time)
                    self.bathymetry_cg_2d.project(self.bathymetry_dg)
                    if self.simulation_time <= t_epsilon:
                        bath_2d_to_3d = ExpandFunctionTo3d(self.bathymetry_cg_2d, self.fields.bathymetry_3d)
                        slide_source_2d_to_3d = ExpandFunctionTo3d(self.fields.slide_source_2d, self.fields.slide_source_3d)
                    bath_2d_to_3d.solve()
                    slide_source_2d_to_3d.solve()

                n_stages = 2
                if self.options.solve_separate_elevation_gradient:
                    for i_stage in range(n_stages):
                        ## 2D advance
                        if i_stage == 1 and self.options.update_free_surface:
                            self.timestepper.store_elevation(i_stage - 1)
                            self.uv_averager.solve()
                            self.extract_surf_dav_uv.solve()
                            self.fields.uv_2d.assign(self.fields.uv_dav_2d)
                            self.copy_uv_dav_to_uv_dav_3d.solve()
                            self.uv_dav_3d_mid.assign(self.fields.uv_dav_3d)
                            timestepper_operator_split.advance(self.simulation_time, update_forcings)
                            #self.timestepper.timesteppers.swe2d.solve_stage(i_stage, self.simulation_time, update_forcings)
                            # compute mesh velocity
                            self.timestepper.compute_mesh_velocity(i_stage - 1)

                        ## 3D advance in old mesh
                        # salt_eq
                        if self.options.solve_salinity:
                            self.timestepper.timesteppers.salt_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # tmp_eq
                        if self.options.solve_temperature:
                            self.timestepper.timesteppers.temp_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # sediment_eq
                        if self.options.solve_sediment:
                            self.timestepper.timesteppers.sediment_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # turb_advection
                        if 'psi_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.psi_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        if 'tke_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.tke_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # momentum_eq
                        self.timestepper.timesteppers.mom_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)

                        ## update mesh
                        if self.options.update_free_surface:
                            self.copy_elev_to_3d.solve()
                            if self.options.use_ale_moving_mesh:
                                self.mesh_updater.update_mesh_coordinates()

                        ## solve 3D
                        # salt_eq
                        if self.options.solve_salinity:
                            self.timestepper.timesteppers.salt_expl.solve_stage(i_stage)
                            if self.options.use_limiter_for_tracers:
                                self.tracer_limiter.apply(self.fields.salt_3d)
                        # temp_eq
                        if self.options.solve_temperature:
                            self.timestepper.timesteppers.temp_expl.solve_stage(i_stage)
                            if self.options.use_limiter_for_tracers:
                                self.tracer_limiter.apply(self.fields.temp_3d)
                        # sediment_eq
                        if self.options.solve_sediment:
                            self.timestepper.timesteppers.sediment_expl.solve_stage(i_stage)
                            if self.options.use_limiter_for_tracers:
                                self.tracer_limiter.apply(self.fields.c_3d)
                        # turb_advection
                        if 'psi_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.psi_expl.solve_stage(i_stage)
                        if 'tke_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.tke_expl.solve_stage(i_stage)
                        # momentum_eq
                        self.timestepper.timesteppers.mom_expl.solve_stage(i_stage)
                        if self.options.use_limiter_for_velocity:
                            self.uv_limiter.apply(self.fields.uv_3d)

                        last_stage = i_stage == n_stages - 1

                        if last_stage:
                            ## compute final prognostic variables
                            # correct uv_3d
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
                                if self.options.solve_sediment:
                                    with timed_stage('impl_sediment_vdiff'):
                                        self.timestepper.timesteppers.sediment_impl.advance(self.simulation_time)
                                with timed_stage('impl_mom_vvisc'):
                                    self.timestepper.timesteppers.mom_impl.advance(self.simulation_time)
                            ## compute final diagnostic fields
                            # update baroclinicity
                            self.timestepper._update_baroclinicity()
                            # update parametrizations
                            self.timestepper._update_turbulence(self.simulation_time)
                            self.timestepper._update_stabilization_params()
                        else:
                            # update baroclinicity
                            self.timestepper._update_baroclinicity()

                        if last_stage:
                            # solve q_3d
                            solver_q.solve()
                            # update uv_3d
                            solver_u.solve()

                            # update water level elev_2d
                            if self.options.update_free_surface:
                                # update final depth-averaged uv_2d
                                self.uv_averager.solve()
                                self.extract_surf_dav_uv.solve()
                                self.fields.uv_2d.assign(self.fields.uv_dav_2d)
                                self.elev_2d_fs.assign(self.elev_2d_old)
                                timestepper_free_surface.advance(self.simulation_time, update_forcings)
                                self.fields.elev_2d.assign(self.elev_2d_fs)
                                ## update mesh
                                self.copy_elev_to_3d.solve()
                                if self.options.use_ale_moving_mesh:
                                    self.mesh_updater.update_mesh_coordinates()

                else: # ssprk in NHWAVE TODO optimise ALE part w.r.t nh pressure updating free surface
                    for i_stage in range(n_stages):
                        ## 3D advance in old mesh
                        # salt_eq
                        if self.options.solve_salinity:
                            self.timestepper.timesteppers.salt_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # tmp_eq
                        if self.options.solve_temperature:
                            self.timestepper.timesteppers.temp_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # sediment_eq
                        if self.options.solve_sediment:
                            self.timestepper.timesteppers.sediment_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # turb_advection
                        if 'psi_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.psi_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        if 'tke_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.tke_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # momentum_eq
                        self.timestepper.timesteppers.mom_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)

                        ## update mesh
                        if self.options.update_free_surface and i_stage == 1:
                            self.copy_elev_to_3d.solve()
                            if self.options.use_ale_moving_mesh:
                                self.mesh_updater.update_mesh_coordinates()

                        ## solve 3D
                        # salt_eq
                        if self.options.solve_salinity:
                            self.timestepper.timesteppers.salt_expl.solve_stage(i_stage)
                            if self.options.use_limiter_for_tracers:
                                self.tracer_limiter.apply(self.fields.salt_3d)
                        # temp_eq
                        if self.options.solve_temperature:
                            self.timestepper.timesteppers.temp_expl.solve_stage(i_stage)
                            if self.options.use_limiter_for_tracers:
                                self.tracer_limiter.apply(self.fields.temp_3d)
                        # sediment_eq
                        if self.options.solve_sediment:
                            self.timestepper.timesteppers.sediment_expl.solve_stage(i_stage)
                            if self.options.use_limiter_for_tracers:
                                self.tracer_limiter.apply(self.fields.c_3d)
                        # turb_advection
                        if 'psi_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.psi_expl.solve_stage(i_stage)
                        if 'tke_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.tke_expl.solve_stage(i_stage)
                        # momentum_eq
                        self.timestepper.timesteppers.mom_expl.solve_stage(i_stage)
                        if self.options.use_limiter_for_velocity:
                            self.uv_limiter.apply(self.fields.uv_3d)

                        # solve q_3d
                        solver_q.solve()
                        # update uv_3d
                        solver_u.solve()

                       # self.timestepper.timesteppers.mom_expl.solve_pg_nh(i_stage)

                        if i_stage == 1:
                            if self.options.use_implicit_vertical_diffusion:
                                if self.options.solve_salinity:
                                    with timed_stage('impl_salt_vdiff'):
                                        self.timestepper.timesteppers.salt_impl.advance(self.simulation_time)
                                if self.options.solve_temperature:
                                    with timed_stage('impl_temp_vdiff'):
                                        self.timestepper.timesteppers.temp_impl.advance(self.simulation_time)
                                if self.options.solve_sediment:
                                    with timed_stage('impl_sediment_vdiff'):
                                        self.timestepper.timesteppers.sediment_impl.advance(self.simulation_time)
                                with timed_stage('impl_mom_vvisc'):
                                    self.timestepper.timesteppers.mom_impl.advance(self.simulation_time)
                            self.timestepper._update_turbulence(self.simulation_time)
                            self.timestepper._update_stabilization_params()

                        # update free surface elevation
                        if self.options.update_free_surface and i_stage == 1: # TODO modify; higher dissipation if solving free surface at each stage
                            self.uv_averager.solve()
                            self.extract_surf_dav_uv.solve()
                            self.fields.uv_2d.assign(self.fields.uv_dav_2d)
                            self.elev_2d_fs.assign(self.elev_2d_old)

                            self.timestepper.store_elevation(i_stage)
                            timestepper_free_surface_implicit.advance(self.simulation_time, update_forcings)
                            self.fields.elev_2d.assign(self.elev_2d_fs)
                           # self.timestepper.compute_mesh_velocity(i_stage)
                            self.copy_elev_to_3d.solve()
                            if i_stage == 1:
                                # compute mesh velocity
                               # self.timestepper.compute_mesh_velocity(0)
                               # self.fields.w_mesh_3d.assign(0.)
                                ## update mesh
                                if self.options.use_ale_moving_mesh:
                                    self.mesh_updater.update_mesh_coordinates()
                            else:
                                self.timestepper.compute_mesh_velocity(0)

            # Move to next time step
            self.iteration += 1
            internal_iteration += 1
            self.simulation_time = initial_simulation_time + internal_iteration*self.dt

            self.callbacks.evaluate(mode='timestep')

            # Write the solution to file
            if self.simulation_time >= self.next_export_t - t_epsilon:
                self.i_export += 1
                self.next_export_t += self.options.simulation_export_time

                cputime = time_mod.clock() - cputimestamp
                cputimestamp = time_mod.clock()
                self.print_state(cputime)

                # exporter with wetting-drying treating
                if self.options.use_wetting_and_drying:
                    self.solution_2d_tmp.assign(self.fields.solution_2d)
                    H = self.bathymetry_dg.dat.data + self.fields.elev_2d.dat.data
                    ind = np.where(H[:] <= 0.)[0]
                    self.fields.elev_2d.dat.data[ind] = 1E-6 - self.bathymetry_dg.dat.data[ind]
                    if self.options.landslide and self.options.slide_is_viscous_fluid:
                        self.solution_ls_tmp.assign(self.fields.solution_ls)
                        h_ls = self.bathymetry_ls.dat.data + elev_ls.dat.data
                        ind_ls = np.where(h_ls[:] <= 0.)[0]
                        elev_ls.dat.data[ind_ls] = 1E-6 - self.bathymetry_ls.dat.data[ind_ls]
                self.export()
                if self.options.use_wetting_and_drying:
                    self.fields.solution_2d.assign(self.solution_2d_tmp)
                    if self.options.landslide and self.options.slide_is_viscous_fluid:
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
                    print_output('Adopting 3d ALE non-hydrostatic solver with P{degree:} {element:} ...'.
                                 format(degree=self.options.polynomial_degree, element=self.options.element_family))
