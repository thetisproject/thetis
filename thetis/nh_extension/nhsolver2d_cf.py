"""
Module for 2D depth averaged solver in conservative form
"""
from __future__ import absolute_import
from .utility_nh import *
from . import shallowwater_nh
from . import shallowwater_cf
from . import granular_cf
from .. import timeintegrator
from .. import rungekutta
from .. import implicitexplicit
from .. import coupled_timeintegrator_2d
from .. import tracer_eq_2d
from . import limiter_nh as limiter
import weakref
import time as time_mod
from mpi4py import MPI
from .. import exporter
from ..field_defs import field_metadata
from ..options import ModelOptions2d
from .. import callback
from ..log import *
from collections import OrderedDict


class FlowSolver(FrozenClass):
    """
    Main object for 2D depth averaged solver in conservative form

    **Example**

    Create mesh

    .. code-block:: python

        from thetis import *
        mesh2d = RectangleMesh(20, 20, 10e3, 10e3)

    Create bathymetry function, set a constant value

    .. code-block:: python

        fs_p1 = FunctionSpace(mesh2d, 'CG', 1)
        bathymetry_2d = Function(fs_p1, name='Bathymetry').assign(10.0)

    Create solver object and set some options

    .. code-block:: python

        solver_obj = solver2d_cf.FlowSolver(mesh2d, bathymetry_2d)
        options = solver_obj.options
        options.element_family = 'dg-dg'
        options.polynomial_degree = 1
        options.timestepper_type = 'CrankNicolson'
        options.simulation_export_time = 50.0
        options.simulation_end_time = 3600.
        options.timestep = 25.0

    Assign initial condition for water elevation

    .. code-block:: python

        solver_obj.create_function_spaces()
        init_elev = Function(solver_obj.function_spaces.H_2d)
        coords = SpatialCoordinate(mesh2d)
        init_elev.project(exp(-((coords[0] - 4e3)**2 + (coords[1] - 4.5e3)**2)/2.2e3**2))
        solver_obj.assign_initial_conditions(elev=init_elev)

    Run simulation

    .. code-block:: python

        solver_obj.iterate()

    See the manual for more complex examples.
    """
    def __init__(self, mesh2d, bathymetry_2d, options=None, mesh_ls=None):
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
        self.mesh2d = mesh2d
        # independent landslide mesh for granular flow
        self.mesh_ls = self.mesh2d
        if mesh_ls is not None:
            self.mesh_ls = mesh_ls
        self.comm = mesh2d.comm

        # add boundary length info
        bnd_len = compute_boundary_length(self.mesh2d)
        self.mesh2d.boundary_len = bnd_len
        self.mesh_ls.boundary_len = bnd_len
        self.normal_2d = FacetNormal(self.mesh2d)
        self.boundary_markers = self.mesh2d.exterior_facets.unique_markers

        self.dt = None
        """Time step"""

        self.options = ModelOptions2d()
        """
        Dictionary of all options. A :class:`.ModelOptions2d` object.
        """
        if options is not None:
            self.options.update(options)

        # simulation time step bookkeeping
        self.simulation_time = 0
        self.iteration = 0
        self.i_export = 0
        self.next_export_t = self.simulation_time + self.options.simulation_export_time

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

        self.fields.bathymetry_2d = bathymetry_2d

        self.export_initial_state = True
        """Do export initial state. False if continuing a simulation"""

        self.bnd_functions = {'shallow_water': {}, 'momentum': {}, 'tracer': {}, 'landslide_motion': {}}

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
        if self.options.use_hllc_flux:
            self.function_spaces.V_2d = MixedFunctionSpace([self.function_spaces.H_2d, self.function_spaces.H_2d, self.function_spaces.H_2d])
        else:
            self.function_spaces.V_2d = MixedFunctionSpace([self.function_spaces.H_2d, self.function_spaces.U_2d])

        # function spaces for granular landslide
        self.function_spaces.H_ls = get_functionspace(self.mesh_ls, 'DG', self.options.polynomial_degree)
        self.function_spaces.U_ls = get_functionspace(self.mesh_ls, 'DG', self.options.polynomial_degree, vector=True)
        self.function_spaces.V_ls = MixedFunctionSpace([self.function_spaces.H_ls, self.function_spaces.H_ls, self.function_spaces.H_ls])
        self.function_spaces.P1_ls = get_functionspace(self.mesh_ls, 'CG', 1)

        self._isfrozen = True

    def create_functions(self):
        """
        Creates extra functions
        """
        self.fields.solution_2d = Function(self.function_spaces.V_2d, name='solution_2d')
        if self.options.use_hllc_flux:
            self.fields.elev_2d, self.fields.hu_2d, self.fields.hv_2d = self.fields.solution_2d.split()
            self.fields.uv_2d = Function(self.function_spaces.U_2d)
        else:
            self.fields.elev_2d, self.fields.uv_2d = self.fields.solution_2d.split()
        self.solution_old = Function(self.function_spaces.V_2d)
        self.solution_tmp = Function(self.function_spaces.V_2d)

        self.uv_2d_old = Function(self.function_spaces.U_2d)
        self.fields.mom_2d = Function(self.function_spaces.U_2d)
        self.fields.h_elem_size_2d = Function(self.function_spaces.P1_2d)
        self.source_sw = Function(self.function_spaces.V_2d)
        self.bathymetry_dg = Function(self.function_spaces.H_2d).project(self.fields.bathymetry_2d)
        self.bathymetry_old = Function(self.function_spaces.H_2d).assign(self.bathymetry_dg)
        self.bathymetry_init = Function(self.function_spaces.H_2d).assign(self.bathymetry_dg)
        self.elev_init = Function(self.function_spaces.H_2d)
        self.elev_fs = Function(self.function_spaces.H_2d)
        self.w_surface = Function(self.function_spaces.H_2d)

        # landslide
        if self.options.landslide or self.options.flow_is_granular:
            self.fields.slide_source_2d = Function(self.function_spaces.H_2d)
            self.fields.solution_ls = Function(self.function_spaces.V_ls, name='solution_ls')
            self.fields.h_ls = self.fields.solution_ls.split()[0]
            self.solution_ls_old = Function(self.function_spaces.V_ls)
            self.solution_ls_mid = Function(self.function_spaces.V_ls)
            self.solution_ls_tmp = Function(self.function_spaces.V_ls)
            self.slope = Function(self.function_spaces.H_ls).interpolate(self.options.bed_slope[2])
        # granular flow
        if self.options.flow_is_granular:
            self.bathymetry_ls = Function(self.function_spaces.H_ls)
            self.phi_i = Function(self.function_spaces.P1_ls).assign(self.options.phi_i)
            self.phi_b = Function(self.function_spaces.P1_ls).assign(self.options.phi_b)
            self.kap = Function(self.function_spaces.P1_ls)
            self.uv_div_ls = Function(self.function_spaces.P1_ls)
            self.strain_rate_ls = Function(self.function_spaces.P1_ls)
            self.grad_p_ls = Function(self.function_spaces.U_ls)
            self.grad_p = Function(self.function_spaces.U_2d)
           # self.slope = Function(self.function_spaces.H_ls).interpolate(self.options.bed_slope[2])
            self.h_2d_ls = Function(self.function_spaces.P1_ls)
            self.h_2d_cg = Function(self.function_spaces.P1_2d)

        # multi-layer approach
        fs_q = get_functionspace(self.mesh2d, 'CG', self.options.polynomial_degree)
        self.fields.q_2d = Function(fs_q)
        q_fs_list = []
        for k in range(self.options.n_layers):
            q_fs_list.append(fs_q)
            setattr(self, 'uv_av_' + str(k+1), Function(self.function_spaces.U_2d))
            setattr(self, 'w_av_' + str(k+1), Function(self.function_spaces.H_2d))
        self.function_spaces.q_mixed = MixedFunctionSpace(q_fs_list)
        self.q_mixed = Function(self.function_spaces.q_mixed)

    def create_equations(self):
        """
        Creates shallow water equations
        """
        if not hasattr(self, 'U_2d'):
            self.create_function_spaces()
        self._isfrozen = False
        # ----- fields
        self.create_functions()
        get_horizontal_elem_size_2d(self.fields.h_elem_size_2d)

        # ----- Equations
        self.eq_sw = shallowwater_cf.ShallowWaterEquations(
            self.fields.solution_2d.function_space(),
            self.bathymetry_dg,
            self.options)
        if self.options.flow_is_granular:
            self.eq_ls = granular_cf.GranularEquations(
                self.fields.solution_ls.function_space(),
                self.bathymetry_ls,
                self.options)
        self.eq_free_surface = shallowwater_cf.FreeSurfaceEquation(
            TestFunction(self.function_spaces.H_2d),
            self.function_spaces.H_2d,
            self.function_spaces.U_2d,
            self.bathymetry_dg,
            self.options)
        self.eq_uv_mom = shallowwater_nh.ShallowWaterMomentumEquation(
            TestFunction(self.function_spaces.U_2d),
            self.function_spaces.U_2d,
            self.function_spaces.H_2d,
            self.bathymetry_dg,
            self.options)

        if self.options.use_wetting_and_drying:
            self.wd_modification = wetting_and_drying_modification(self.function_spaces.H_2d)
            self.wd_modification_ls = wetting_and_drying_modification(self.function_spaces.H_ls)

        # initialise limiter
        if self.options.polynomial_degree == 1:
            self.limiter_h = limiter.VertexBasedP1DGLimiter(self.function_spaces.H_2d)
            self.limiter_u = limiter.VertexBasedP1DGLimiter(self.function_spaces.U_2d)
        else:
            self.limiter_h = None
            self.limiter_u = None

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
        self.fields_sw = {
            'source_sw': self.source_sw,
            'mom_2d': self.fields.mom_2d,
            'eta': self.fields.elev_2d,
            'uv_2d': self.fields.uv_2d,
            'sponge_damping_2d': self.set_sponge_damping(self.options.sponge_layer_length, self.options.sponge_layer_start, alpha=10., sponge_is_2d=True),
            }
        if self.options.landslide:
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
        self.set_time_step()
        if self.options.timestepper_type == 'SSPRK33':
            self.timestepper = rungekutta.SSPRK33(self.eq_sw, self.fields.solution_2d,
                                                  self.fields_sw, self.dt,
                                                  bnd_conditions=self.bnd_functions['shallow_water'],
                                                  solver_parameters=self.options.timestepper_options.solver_parameters)
            self.timestepper_free_surface = rungekutta.SSPRK33(self.eq_free_surface, self.elev_fs,
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
            self.timestepper_free_surface = timeintegrator.CrankNicolson(self.eq_free_surface, self.elev_fs,
                                                  self.fields_sw, self.dt,
                                                  bnd_conditions=self.bnd_functions['shallow_water'],
                                                  solver_parameters=self.options.timestepper_options.solver_parameters_momentum,
                                                  semi_implicit=False,
                                                  theta=1.0)
        else:
            raise Exception('Unknown time integrator type: '+str(self.options.timestepper_type))
        print_output('Using time integrator: {:}'.format(self.timestepper.__class__.__name__))
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

    def assign_initial_conditions(self, elev_2d=None, uv_2d=None, h_ls=None, uv_ls=None):
        """
        Assigns initial conditions

        :kwarg elev_2d: Initial condition for water elevation
        :type elev_2d: scalar :class:`Function`, :class:`Constant`, or an expression
        :kwarg uv_2d: Initial condition for depth averaged velocity
        :type uv_2d: vector valued :class:`Function`, :class:`Constant`, or an expression
        """
        if not self._initialized:
            self.initialize()

        if elev_2d is not None:
            self.fields.elev_2d.project(elev_2d)
        # prevent negative initial water depth
        if self.options.use_hllc_flux:
            h_2d = self.fields.elev_2d.dat.data + self.bathymetry_dg.dat.data
            ind = np.where(h_2d[:] <= 0)[0]
            self.fields.elev_2d.dat.data[ind] = -self.bathymetry_dg.dat.data[ind]
            self.elev_init.assign(self.fields.elev_2d)

        if uv_2d is not None:
            self.fields.uv_2d.project(uv_2d)
            if self.options.use_hllc_flux:
                for i in range(2):
                    self.fields.solution_2d.sub(i+1).dat.data[:] = self.fields.uv_2d.dat.data[:, i] * h_2d[:]

        if self.options.landslide or self.options.flow_is_granular:
            if h_ls is not None:
                self.fields.solution_ls.sub(0).project(h_ls)
            h_ls = self.fields.solution_ls.sub(0).dat.data[:]
            ind_ls = np.where(h_ls[:] <= self.options.wetting_and_drying_threshold)[0]
            h_ls[ind_ls] = 0.
            if uv_ls is not None:
                self.fields.solution_ls.sub(1).project(self.fields.solution_ls.sub(0)*uv_ls[0])
                self.fields.solution_ls.sub(2).project(self.fields.solution_ls.sub(0)*uv_ls[1])

        self.timestepper.initialize(self.fields.solution_2d)

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
        state variables are: elev_2d, hu_2d, hv_2d

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
        state_fields = ['elev_2d']
        hdf5_dir = os.path.join(outputdir, 'hdf5')
        e = exporter.ExportManager(hdf5_dir,
                                   state_fields,
                                   self.fields,
                                   field_metadata,
                                   export_type='hdf5',
                                   verbose=self.options.verbose > 0)
        e.exporters['elev_2d'].load(i_export, self.fields.elev_2d)
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
        if self.options.tracer_only:
            norm_q = norm(self.fields.tracer_2d)

            line = ('{iexp:5d} {i:5d} T={t:10.2f} '
                    'tracer norm: {q:10.4f} {cpu:5.2f}')

            print_output(line.format(iexp=self.i_export, i=self.iteration,
                                     t=self.simulation_time, q=norm_q,
                                     cpu=cputime))
        else:
            lx = self.mesh2d.coordinates.sub(0).dat.data.max() - self.mesh2d.coordinates.sub(0).dat.data.min()
            ly = self.mesh2d.coordinates.sub(1).dat.data.max() - self.mesh2d.coordinates.sub(1).dat.data.min()
            if self.options.use_hllc_flux:
                eta, hu, hv = self.fields.solution_2d.split()
                norm_uv = norm(as_vector((hu, hv))/(self.bathymetry_dg + eta)) / sqrt(lx * ly)
            else:
                eta, uv = self.fields.solution_2d.split()
                norm_uv = norm(uv) / sqrt(lx * ly)
            norm_eta = norm(eta) / sqrt(lx * ly)
            if self.options.flow_is_granular:
                norm_hs = norm(self.fields.h_ls) / sqrt(lx * ly)
            else:
                norm_hs = 0

            line = ('{iexp:5d} {i:5d} T={t:10.2f} '
                    'eta norm: {e:10.4f} uv norm: {u:10.4f} hs norm: {h:10.4f} {cpu:5.2f}')
            print_output(line.format(iexp=self.i_export, i=self.iteration,
                                     t=self.simulation_time, e=norm_eta,
                                     u=norm_uv, h=norm_hs, cpu=cputime))
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

        self.options.use_limiter_for_tracers &= self.options.polynomial_degree > 0

        t_epsilon = 1.0e-5
        cputimestamp = time_mod.clock()
        next_export_t = self.simulation_time + self.options.simulation_export_time

        dump_hdf5 = self.options.export_diagnostics and not self.options.no_exports
        if self.options.check_volume_conservation_2d:
            c = callback.VolumeConservation2DCallback(self,
                                                      export_to_hdf5=dump_hdf5,
                                                      append_to_log=True)
            self.add_callback(c)

        if self.options.check_tracer_conservation:
            c = callback.TracerMassConservation2DCallback('tracer_2d',
                                                          self,
                                                          export_to_hdf5=dump_hdf5,
                                                          append_to_log=True)
            self.add_callback(c, eval_interval='export')

        if self.options.check_tracer_overshoot:
            c = callback.TracerOvershootCallBack('tracer_2d',
                                                 self,
                                                 export_to_hdf5=dump_hdf5,
                                                 append_to_log=True)
            self.add_callback(c, eval_interval='export')

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

        # solver for advancing main equations
        if self.options.timestepper_type == 'SSPRK33':
            a_sw = self.eq_sw.mass_term(self.eq_sw.trial)
            l_sw = (self.eq_sw.mass_term(self.fields.solution_2d) + Constant(self.dt)*
                    self.eq_sw.residual('all', self.fields.solution_2d, self.fields.solution_2d, 
                                        self.fields_sw, self.fields_sw, self.bnd_functions['shallow_water'])
                   )
            prob_sw = LinearVariationalProblem(a_sw, l_sw, self.solution_tmp)
            solver_sw = LinearVariationalSolver(prob_sw, solver_parameters=self.options.timestepper_options.solver_parameters)

        if self.options.flow_is_granular:
            dt_ls = self.dt / self.options.n_dt
            # solver for granular landslide motion
            a_ls = self.eq_ls.mass_term(self.eq_ls.trial)
            l_ls = (self.eq_ls.mass_term(self.fields.solution_ls) + Constant(dt_ls)*
                    self.eq_ls.residual('all', self.fields.solution_ls, self.fields.solution_ls,
                                        self.fields_ls, self.fields_ls, self.bnd_functions['landslide_motion'])
                   )
            prob_ls = LinearVariationalProblem(a_ls, l_ls, self.solution_ls_tmp)
            solver_ls = LinearVariationalSolver(prob_ls, solver_parameters=self.options.timestepper_options.solver_parameters)
            # solver for div(velocity)
            h_ls = self.fields.solution_ls.sub(0)
            hu_ls = self.fields.solution_ls.sub(1)
            hv_ls = self.fields.solution_ls.sub(2)
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
            h_2d = self.bathymetry_dg + self.fields.elev_2d
            tri_pf = TrialFunction(self.grad_p.function_space())
            test_pf = TestFunction(self.grad_p.function_space())
            a_pf = dot(tri_pf, test_pf)*dx
            l_pf = dot(conditional(h_2d <= 0, zero(self.grad_p.ufl_shape), 
                       grad(self.options.rho_fluid*physical_constants['g_grav']*h_2d + self.fields.q_2d)), test_pf)*dx
            prob_pf = LinearVariationalProblem(a_pf, l_pf, self.grad_p)
            solver_pf = LinearVariationalSolver(prob_pf)

        # solvers for non-hydrostatic pressure
        solve_nh_pressure = True # TODO set in `options`
        if solve_nh_pressure:
            # Poisson solver
            theta = 1#0.5
            par = 0.5 # approximation parameter for NH terms
            d_2d = self.bathymetry_dg
            h_2d = d_2d + self.fields.elev_2d
            h_mid = conditional(h_2d <= self.options.depth_wd_interface, self.options.depth_wd_interface, h_2d)#2 * alpha**2 / (2 * alpha + abs(h_2d)) + 0.5 * (abs(h_2d) + h_2d)
            A = theta*grad(self.fields.elev_2d - d_2d)/h_mid# + (1. - theta)*grad(self.solution_old.sub(0) - d_2d)/h_old
            B = div(A) - 2./(par*h_mid*h_mid)
            C = (div(self.fields.uv_2d) + (self.w_surface + inner(2.*self.fields.uv_2d - self.uv_2d_old, grad(d_2d)))/h_mid)/(par*self.dt)
            if self.options.flow_is_granular:
                C = (div(self.fields.uv_2d) + (self.w_surface + inner(2.*self.fields.uv_2d - self.uv_2d_old, grad(d_2d)) - self.fields.slide_source_2d)/h_mid)/(par*self.dt)
            # weak forms
            q_2d = self.fields.q_2d
            q_test = TestFunction(self.fields.q_2d.function_space())
            f_q = (-dot(grad(q_2d), grad(q_test)) + B*q_2d*q_test)*dx - C*q_test*dx - q_2d*div(A*q_test)*dx
            # boundary conditions
            for bnd_marker in self.boundary_markers:
                func = self.bnd_functions['shallow_water'].get(bnd_marker)
                ds_bnd = ds(int(bnd_marker))
                #q_open_bc = self.q_bnd.assign(0.)
                if self.bnd_functions['shallow_water'] == {}:#func is None or 'q' not in func:
                    # Neumann boundary condition => inner(grad(q), normal)=0.
                    f_q += (q_2d*inner(A, self.normal_2d))*q_test*ds_bnd
            prob_q = NonlinearVariationalProblem(f_q, self.fields.q_2d)
            solver_q = NonlinearVariationalSolver(prob_q,
                                            solver_parameters={'snes_type': 'ksponly', # ksponly, newtonls
                                                               'ksp_type': 'preonly', # gmres, preonly
                                                               'mat_type': 'aij',
                                                               'pc_type': 'lu', #'bjacobi', 'lu'
                                                               },
                                            options_prefix='poisson_solver')
            # solver to update velocities
            # update uv_2d
            uv_tri = TrialFunction(self.function_spaces.U_2d)
            uv_test = TestFunction(self.function_spaces.U_2d)
            a_u = inner(uv_tri, uv_test)*dx
            l_u = inner(self.fields.uv_2d - par*self.dt*(grad(q_2d) + A*q_2d), uv_test)*dx
            prob_u = LinearVariationalProblem(a_u, l_u, self.fields.uv_2d)
            solver_u = LinearVariationalSolver(prob_u)
            # update w_surf
            w_tri = TrialFunction(self.function_spaces.H_2d)
            w_test = TestFunction(self.function_spaces.H_2d)
            a_w = w_tri*w_test*dx
            l_w = (self.w_surface + 2.*self.dt*q_2d/h_mid + inner(self.fields.uv_2d - self.uv_2d_old, grad(d_2d)))*w_test*dx
            prob_w = LinearVariationalProblem(a_w, l_w, self.w_surface)
            solver_w = LinearVariationalSolver(prob_w)

        # multi-layer approach
        uv_2d = self.fields.uv_2d
        elev_2d = self.fields.elev_2d
        # layer thickness accounting for total depth
        n_layers = self.options.n_layers
        alpha = self.options.alpha_nh
        if len(alpha) < n_layers:
            n = n_layers - len(alpha)
            sum = 0.
            if len(alpha) >= 1:
                for k in range(len(alpha)):
                    sum = sum + alpha[k]
            if sum >= 1.:
                print_output('Using uniform vertical layers due to improper option set ...')
                alpha = [1./n_layers for i in range(n_layers)]
            else:
                for k in range(n):
                    alpha.append((1. - sum)/n)
        else:
            sum = 0.
            for i in range(n_layers):
                sum += alpha[i]
            if not sum == 1:
                print_output('Using uniform vertical layers due to improper option set ...')
                alpha = [1./n_layers for i in range(n_layers)]
        # location parameter of layer interface
        beta = [0.]
        for k in range(n_layers):
            val = beta[k] + alpha[k]
            beta.append(val)

        # list the layer-averaged velocities
        u_list = []
        w_list = []
        q_list = []
        for k in range(n_layers):
            u_list.append(getattr(self, 'uv_av_'+str(k+1)))
            w_list.append(getattr(self, 'w_av_'+str(k+1)))
            q_list.append(self.q_mixed.split()[k])
            if k == n_layers - 1:
                q_list.append(0.)

        # store fields for convenience
        h_2d = self.bathymetry_dg + self.fields.elev_2d
        if self.options.use_hllc_flux and self.options.use_wetting_and_drying:
            h_tot = conditional(h_2d <= self.options.depth_wd_interface, self.options.depth_wd_interface, h_2d)
        elif self.options.use_wetting_and_drying:
            h_tot = 2 * self.options.depth_wd_interface**2 / (2 * self.options.depth_wd_interface + abs(h_2d)) + 0.5 * (abs(h_2d) + h_2d)
        else:
            h_tot = h_2d
        #h_tot = shallowwater_nh.ShallowWaterTerm(self.fields.solution_2d.function_space(), self.bathymetry_dg, self.options).get_total_depth(self.fields.elev_2d)
        fields = self.fields_sw
        elev_dt = Function(self.function_spaces.H_2d)
        bath_dt = 0#Function(self.function_spaces.H_2d) # TODO note here
        for k in range(n_layers + 1):
            # velocities at layer interface
            if k == 0:
                if n_layers == 1:
                    fields['u_z_'+str(k)] = u_list[k]
                else:
                    fields['u_z_'+str(k)] = 2.*u_list[k] - (alpha[k+1]/(alpha[k]+alpha[k+1])*u_list[k] + 
                                                            alpha[k]/(alpha[k]+alpha[k+1])*u_list[k+1])
                fields['w_z_'+str(k)] = -inner(fields['u_z_'+str(k)], grad(self.bathymetry_dg))
                if self.options.landslide:
                    fields['w_z_'+str(k)] += -self.fields.slide_source_2d # (self.bathymetry_dg - self.bathymetry_dg_old)/self.dt
            elif k > 0 and k < n_layers:
                fields['u_z_'+str(k)] = alpha[k]/(alpha[k-1]+alpha[k])*u_list[k-1] + alpha[k-1]/(alpha[k-1]+alpha[k])*u_list[k]
                fields['w_z_'+str(k)] = 2.*w_list[k-1] - fields['w_z_'+str(k-1)]
            else:
                fields['u_z_'+str(k)] = 2.*u_list[k-1] - fields['u_z_'+str(k-1)]
                fields['w_z_'+str(k)] = 2.*w_list[k-1] - fields['w_z_'+str(k-1)]
            # z-coordinate
            fields['z_'+str(k)] = beta[k]*elev_2d + (beta[k] - 1)*self.bathymetry_dg
            if k > 0:
                fields['z_'+str(k-1)+str(k)] = (beta[k-1] + beta[k])*elev_2d + (beta[k-1] + beta[k] - 2)*self.bathymetry_dg
                fields['omega_'+str(k-1)] = w_list[k-1] - 0.5*(dot(fields['u_z_'+str(k-1)], grad(fields['z_'+str(k-1)])) + beta[k-1]*elev_dt + (beta[k-1] -1)*bath_dt +
                                                               dot(fields['u_z_'+str(k)], grad(fields['z_'+str(k)])) + beta[k]*elev_dt + (beta[k] - 1)*bath_dt)

        if n_layers >= 2:
            timestepper_dic = {}
            for k in range(n_layers - 1):
                if self.options.timestepper_type == 'SSPRK33':
                    solver_parameters = self.options.timestepper_options.solver_parameters
                else:
                    solver_parameters = self.options.timestepper_options.solver_parameters_momentum
                timestepper_dic['uv_layer_'+str(k+1)] = timeintegrator.CrankNicolson(self.eq_uv_mom, u_list[k],
                                                              self.fields_sw, self.dt,
                                                              bnd_conditions=self.bnd_functions['momentum'],
                                                              solver_parameters=solver_parameters,
                                                              semi_implicit=False,
                                                              theta=1.0)
                consider_vertical_mass_flux = False
                if consider_vertical_mass_flux:
                    uv_mass_term = fields['omega_'+str(k)]*(fields['u_z_'+str(k+1)] - fields['u_z_'+str(k)])/(alpha[k]*h_tot)
                    timestepper_dic['uv_layer_'+str(k+1)].F += self.dt*inner(uv_mass_term, uv_test)*dx
                    timestepper_dic['uv_layer_'+str(k+1)].update_solver()

            if True:
                if True:
                    solve_vertical_velocity = False
                    if n_layers >= 2 and solve_vertical_velocity:
                        for k in range(n_layers):
                            timestepper_dic['w_layer_'+str(k+1)] = timeintegrator.CrankNicolson(self.eq_w_mom, w_list[k],
                                                              self.field_dic, self.dt,
                                                              bnd_conditions=self.bnd_functions['momentum'],
                                                              solver_parameters=self.options.timestepper_options.solver_parameters_momentum,
                                                              semi_implicit=self.options.timestepper_options.use_semi_implicit_linearization,
                                                              theta=self.options.timestepper_options.implicitness_theta)
                            if consider_vertical_mass_flux:
                                w_mass_term = fields['omega_'+str(k)]*(fields['w_z_'+str(k+1)] - fields['w_z_'+str(k)])/(alpha[k]*h_tot)
                                timestepper_dic['w_layer_'+str(k+1)].F += self.dt*w_mass_term*w_test*dx
                                timestepper_dic['w_layer_'+str(k+1)].update_solver()

        # solver for the mixed Poisson equation
        if True:
            if True:
                if True:
                    q_test = TestFunctions(self.function_spaces.q_mixed)
                    q_tuple = split(self.q_mixed)
                    if n_layers == 1:
                        q_test = [TestFunction(self.fields.q_2d.function_space())]
                        q_tuple = [self.fields.q_2d]
                    # re-arrange the list of q
                    q = []
                    for k in range(n_layers):
                        q.append(q_tuple[k])
                        if k == n_layers - 1:
                            # free-surface NH pressure
                            q.append(0.)
                    f = 0.
                    for k in range(n_layers):
                        # weak form of div(h_{k+1}*uv_av_{k+1})
                        div_hu_term = div((alpha[k]*h_tot)*u_list[k])*q_test[k]*dx + \
                                      0.5*self.dt*(alpha[k]*h_tot)*dot(grad(q[k]+q[k+1]), grad(q_test[k]))*dx + \
                                      0.5*self.dt*(q[k]-q[k+1])*dot(grad(fields['z_'+str(k)+str(k+1)]), grad(q_test[k]))*dx
                        if k >= 1:
                            for i in range(k):
                                div_hu_term += 2.*(div((alpha[i]*h_tot)*u_list[i])*q_test[k]*dx + \
                                               0.5*self.dt*(alpha[i]*h_tot)*dot(grad(q[i]+q[i+1]), grad(q_test[k]))*dx + \
                                               0.5*self.dt*(q[i]-q[i+1])*dot(grad(fields['z_'+str(i)+str(i+1)]), grad(q_test[k]))*dx)
                        # weak form of w_{k}{k+1}
                        vert_vel_term = 2.*(w_list[k] + self.dt*(q[k] - q[k+1])/(alpha[k]*h_tot))*q_test[k]*dx
                        consider_vert_adv = False#True
                        if consider_vert_adv: # TODO if make sure that considering vertical advection is benefitial, delete this logical variable
                            #vert_vel_term += -2.*self.dt*dot(u_list[k], grad(w_list[k]))*q_test[k]*dx
                            vert_vel_term += 2.*self.dt*(div(u_list[k]*q_test[k])*w_list[k]*dx -
                                                         avg(w_list[k])*jump(q_test[k], inner(u_list[k], self.normal_2d))*dS)
                            if consider_mesh_relative_velocity:
                                vert_vel_term += -2.*self.dt/(alpha[k]*h_tot)*inner(omega_dic['z_'+str(k+1)]*(fields['w_z_'+str(k+1)] - w_list[k]) -
                                                                                omega_dic['z_'+str(k)]*(fields['w_z_'+str(k)] - w_list[k]), q_test[k])*dx
                        # weak form of RHS terms
                        if k == 0: # i.e. the layer adjacent to the bottom
                            if n_layers == 1:
                                grad_1_layer1 = grad(fields['z_'+str(k)+str(k+1)])
                                interface_term = dot(grad_1_layer1, u_list[k])*q_test[k]*dx - \
                                                 0.5*self.dt*(-div(grad_1_layer1*q_test[k])*(q[k]+q[k+1]))*dx - \
                                                 0.5*self.dt*(1./(alpha[k]*h_tot)*dot(grad_1_layer1, grad(fields['z_'+str(k)+str(k+1)]))*(q[k]-q[k+1]))*q_test[k]*dx
                            else:
                                grad_1_layer1 = grad(2.*fields['z_'+str(k)] + alpha[k]*alpha[k+1]/(alpha[k] + alpha[k+1])*h_tot)
                                grad_2_layer1 = grad(alpha[k]*alpha[k]/(alpha[k] + alpha[k+1])*h_tot)
                                interface_term = (dot(grad_1_layer1, u_list[k]) + dot(grad_2_layer1, u_list[k+1]))*q_test[k]*dx - \
                                                 0.5*self.dt*(-div(grad_1_layer1*q_test[k])*(q[k]+q[k+1]) - div(grad_2_layer1*q_test[k])*(q[k+1]+q[k+2]))*dx - \
                                                 0.5*self.dt*(1./(alpha[k]*h_tot)*dot(grad_1_layer1, grad(fields['z_'+str(k)+str(k+1)]))*(q[k]-q[k+1]) + 
                                                          1./(alpha[k+1]*h_tot)*dot(grad_2_layer1, grad(fields['z_'+str(k+1)+str(k+2)]))*(q[k+1]-q[k+2]))*q_test[k]*dx
                        elif k == n_layers - 1: # i.e. the layer adjacent to the free surface
                            grad_1_layern = grad(-alpha[k]*alpha[k]/(alpha[k-1] + alpha[k])*h_tot)
                            grad_2_layern = grad(2.*fields['z_'+str(k+1)] - alpha[k-1]*alpha[k]/(alpha[k-1] + alpha[k])*h_tot)
                            interface_term = (dot(grad_1_layern, u_list[k-1]) + dot(grad_2_layern, u_list[k]))*q_test[k]*dx - \
                                             0.5*self.dt*(-div(grad_1_layern*q_test[k])*(q[k-1]+q[k]) - div(grad_2_layern*q_test[k])*(q[k]+q[k+1]))*dx - \
                                             0.5*self.dt*(1./(alpha[k-1]*h_tot)*dot(grad_1_layern, grad(fields['z_'+str(k-1)+str(k)]))*(q[k-1]-q[k]) + 
                                                      1./(alpha[k]*h_tot)*dot(grad_2_layern, grad(fields['z_'+str(k)+str(k+1)]))*(q[k]-q[k+1]))*q_test[k]*dx
                        else:
                            grad_1_layerk = alpha[k]/(alpha[k-1] + alpha[k])*grad(fields['z_'+str(k)])
                            grad_2_layerk = alpha[k-1]/(alpha[k-1] + alpha[k])*grad(fields['z_'+str(k)]) + \
                                            alpha[k+1]/(alpha[k] + alpha[k+1])*grad(fields['z_'+str(k+1)])
                            grad_3_layerk = alpha[k]/(alpha[k] + alpha[k+1])*grad(fields['z_'+str(k+1)])
                            interface_term = (dot(grad_1_layerk, u_list[k-1]) + 
                                              dot(grad_2_layerk, u_list[k]) + 
                                              dot(grad_3_layerk, u_list[k+1]))*q_test[k]*dx - \
                                             0.5*self.dt*(-div(grad_1_layerk*q_test[k])*(q[k-1]+q[k]) - 
                                                          div(grad_2_layerk*q_test[k])*(q[k]+q[k+1]) - 
                                                          div(grad_3_layerk*q_test[k])*(q[k+1]+q[k+2]))*dx - \
                                             0.5*self.dt*(1./(alpha[k-1]*h_tot)*dot(grad_1_layerk, grad(fields['z_'+str(k-1)+str(k)]))*(q[k-1]-q[k]) + 
                                                          1./(alpha[k]*h_tot)*dot(grad_2_layerk, grad(fields['z_'+str(k)+str(k+1)]))*(q[k]-q[k+1]) + 
                                                          1./(alpha[k+1]*h_tot)*dot(grad_3_layerk, grad(fields['z_'+str(k+1)+str(k+2)]))*(q[k+1]-q[k+2]))*q_test[k]*dx
                        # weak form of slide source term
                        if self.options.landslide:
                            slide_source_term = -2.*self.fields.slide_source_2d*q_test[k]*dx
                            f += slide_source_term
                        f += div_hu_term + vert_vel_term - interface_term

                        for bnd_marker in self.boundary_markers:
                            func = self.bnd_functions['shallow_water'].get(bnd_marker)
                            ds_bnd = ds(int(bnd_marker))
                            if self.bnd_functions['shallow_water'] == {}:#func is None or 'q' not in func:
                                # bnd terms of div(h_{k+1}*uv_av_{k+1})
                                f += -0.5*self.dt*(q[k]-q[k+1])*dot(grad(fields['z_'+str(k)+str(k+1)]), self.normal_2d)*q_test[k]*ds_bnd
                                if k >= 1:
                                    for i in range(k):
                                        f += -self.dt*(q[i]-q[i+1])*dot(grad(fields['z_'+str(i)+str(i+1)]), self.normal_2d)*q_test[k]*ds_bnd
                                # bnd terms of RHS terms about interface connection
                                if k == 0:
                                    if n_layers == 1:
                                        f += 0.5*self.dt*dot(grad_1_layer1, self.normal_2d)*(q[k]+q[k+1])*q_test[k]*ds_bnd
                                    else:
                                        f += 0.5*self.dt*(dot(grad_1_layer1, self.normal_2d)*(q[k]+q[k+1]) + 
                                                          dot(grad_2_layer1, self.normal_2d)*(q[k+1]+q[k+2]))*q_test[k]*ds_bnd
                                elif k == n_layers - 1:
                                    f += 0.5*self.dt*(dot(grad_1_layern, self.normal_2d)*(q[k-1]+q[k]) + 
                                                      dot(grad_2_layern, self.normal_2d)*(q[k]+q[k+1]))*q_test[k]*ds_bnd
                                else:
                                    f += 0.5*self.dt*(dot(grad_1_layerk, self.normal_2d)*(q[k-1]+q[k]) +
                                                      dot(grad_2_layerk, self.normal_2d)*(q[k]+q[k+1]) +
                                                      dot(grad_3_layerk, self.normal_2d)*(q[k+1]+q[k+2]))*q_test[k]*ds_bnd

                    prob_q = NonlinearVariationalProblem(f, self.q_mixed)
                    if n_layers == 1:
                        prob_q = NonlinearVariationalProblem(f, self.fields.q_2d)
                    solver_q_ml = NonlinearVariationalSolver(prob_q,
                                                        solver_parameters={'snes_type': 'ksponly', # ksponly, newtonls
                                                               'ksp_type': 'preonly', # gmres, preonly
                                                               'mat_type': 'aij',
                                                               'pc_type': 'lu'})

        # solvers for updating layer-averaged velocities
        solver_u_ml = []
        solver_w_ml = []
        uv_tri = TrialFunction(self.function_spaces.U_2d)
        uv_test = TestFunction(self.function_spaces.U_2d)
        w_tri = TrialFunction(self.function_spaces.H_2d)
        w_test = TestFunction(self.function_spaces.H_2d)
        a_u = inner(uv_tri, uv_test)*dx
        a_w = inner(w_tri, w_test)*dx
        def q_list_mod(k):
            return conditional(h_2d <= 0, 0, q_list[k])

        for k in range(n_layers):
            l_u = dot(u_list[k] - 0.5*self.dt*(grad(q_list_mod(k) + q_list_mod(k+1)) - 
                                               (q_list_mod(k+1) - q_list_mod(k))/(alpha[k]*h_tot)*grad(fields['z_'+str(k)+str(k+1)])), uv_test)*dx
            l_w = (w_list[k] - self.dt*(q_list_mod(k+1) - q_list_mod(k))/(alpha[k]*h_tot))*w_test*dx
            prob_u = LinearVariationalProblem(a_u, l_u, u_list[k])
            prob_w = LinearVariationalProblem(a_w, l_w, w_list[k])
            solver_u_ml.append(LinearVariationalSolver(prob_u))
            solver_w_ml.append(LinearVariationalSolver(prob_w))

        # solver to update free surface
        if self.options.use_hllc_flux:
            a_fs = self.eq_free_surface.mass_term(self.eq_free_surface.trial)
            l_fs = (self.eq_free_surface.mass_term(self.fields.elev_2d) + Constant(self.dt)*
                    self.eq_free_surface.residual('all', self.fields.elev_2d, self.fields.elev_2d, 
                                        self.fields_sw, self.fields_sw, self.bnd_functions['shallow_water'])
                   )
            prob_fs = LinearVariationalProblem(a_fs, l_fs, self.elev_fs)
            solver_fs = LinearVariationalSolver(prob_fs, solver_parameters=self.options.timestepper_options.solver_parameters)

        while self.simulation_time <= self.options.simulation_end_time - t_epsilon:

            self.bathymetry_old.assign(self.bathymetry_dg)
            self.uv_2d_old.assign(self.fields.uv_2d)
            # original line: self.timestepper.advance(self.simulation_time, update_forcings)

            # facilitate wetting and drying treatment at each stage
            if self.options.timestepper_type == 'SSPRK33':
                n_stages = self.timestepper.n_stages
                coeff = [[0., 1.], [3./4., 1./4.], [1./3., 2./3.]]
            use_ssprk22 = True # i.e. compatible with nh wave model
            if use_ssprk22:
                n_stages = 2
                coeff = [[0., 1.], [1./2., 1./2.]]
            if self.options.landslide:
                self.solution_ls_old.assign(self.fields.solution_ls)
            if self.options.flow_is_granular:
                if not self.options.lamda == 0.:
                    self.h_2d_cg.project(self.bathymetry_dg + self.fields.elev_2d)
                    self.h_2d_ls.dat.data[:] = self.h_2d_cg.dat.data[:]
                for i in range(self.options.n_dt):
                    # solve fluid pressure on slide
                    self.bathymetry_dg.dat.data[:] = self.bathymetry_init.dat.data[:] - self.fields.h_ls.dat.data[:]/self.slope.dat.data.min()
                    solver_pf.solve()
                    self.grad_p_ls.dat.data[:] = self.grad_p.dat.data[:]

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
                        solver_sr.solve()

            if not self.options.no_wave_flow:

                self.solution_old.assign(self.fields.solution_2d)
                self.elev_fs.assign(self.fields.elev_2d)

                h_array = self.fields.elev_2d.dat.data + self.bathymetry_dg.dat.data

                # update landslide motion source
                if self.options.landslide:
                    # update landslide motion source
                    if update_forcings is not None:
                        update_forcings(self.simulation_time + self.dt)
                    ind_wet = np.where(h_array[:] > 0)[0]
                    self.fields.slide_source_2d.assign(0.)
                    if self.simulation_time >= 0.:
                        self.fields.slide_source_2d.dat.data[ind_wet] = (self.fields.solution_ls.sub(0).dat.data[ind_wet] 
                                                                         - self.solution_ls_old.sub(0).dat.data[ind_wet])/self.dt/self.slope.dat.data.min()

                    # NOTE `self.bathymetry_init` initialised does not vary with time
                    self.bathymetry_dg.dat.data[:] = self.bathymetry_init.dat.data[:] - self.fields.h_ls.dat.data[:]/self.slope.dat.data.min()
                    if self.options.use_hllc_flux:
                        # detect before hitting water
                        h_init = self.elev_init.dat.data + self.bathymetry_dg.dat.data
                        ind = np.where(h_init[:] <= 0)[0]
                        self.fields.elev_2d.dat.data[ind] = -self.bathymetry_dg.dat.data[ind]

                if self.options.use_hllc_flux:
                    for i_stage in range(n_stages):
                        #self.timestepper.solve_stage(i_stage, self.simulation_time, update_forcings)
                        solver_sw.solve()
                        self.fields.solution_2d.assign(coeff[i_stage][0]*self.solution_old + coeff[i_stage][1]*self.solution_tmp)
                        if False:#self.options.use_limiter_for_elevation:
                            for i in range(3):
                                self.limiter_h.apply(self.fields.solution_2d.sub(i))
                        if self.options.use_wetting_and_drying:
                            limiter_start_time = 0.
                            limiter_end_time = 1000
                            use_limiter = self.options.use_limiter_for_elevation and self.simulation_time >= limiter_start_time and self.simulation_time <= limiter_end_time
                            self.wd_modification.apply(self.fields.solution_2d, self.options.wetting_and_drying_threshold, 
                                                       use_limiter, use_eta_solution=True, bathymetry=self.bathymetry_dg)

                else:
                    self.timestepper.advance(self.simulation_time, update_forcings)

                if self.options.use_limiter_for_elevation:
                    if self.limiter_h is not None:
                        self.limiter_h.apply(self.fields.elev_2d)
                    if self.limiter_u is not None:
                        self.limiter_u.apply(self.fields.uv_2d)

                ind_dry = np.where(h_array[:] <= 0)[0]
                self.fields.uv_2d.dat.data[ind_dry] = [0, 0]

                if True:
                    if self.options.use_hllc_flux:
                        # calculate velocity
                        ind_wet = np.where(h_array[:] > self.options.depth_wd_interface)[0]
                        ind_dry = np.where(h_array[:] <= self.options.depth_wd_interface)[0]
                        self.fields.uv_2d.assign(0.)
                        for i in range(2):
                            self.fields.uv_2d.dat.data[ind_wet, i] = self.fields.solution_2d.sub(i+1).dat.data[ind_wet] / h_array[ind_wet]

                    if True:

                       # solver_q.solve()
                       # solver_u.solve()
                       # solver_w.solve()

                        if n_layers >= 2:
                            # solve layer-averaged horizontal velocity in the hydrostatic step
                            sum_uv_av = 0. 
                            # except the layer adjacent to the free surface
                            for k in range(n_layers - 1):
                                if self.options.landslide: # save time
                                    timestepper_dic['uv_layer_'+str(k+1)].advance(self.simulation_time) #TODO note here remove update_forcings, save computational cost
                                else:
                                    timestepper_dic['uv_layer_'+str(k+1)].advance(self.simulation_time, update_forcings)
                                if self.options.use_limiter_for_multi_layer:
                                    if self.limiter_u is not None:
                                        self.limiter_u.apply(u_list[k])
                                u_list[k].dat.data[ind_dry] = [0., 0.]
                                #sum_uv_av += u_list[k] # cannot sum by this way
                                sum_uv_av = sum_uv_av + alpha[k]*u_list[k]
                            u_list[n_layers-1].project((uv_2d - sum_uv_av)/alpha[n_layers-1])
                            # solve layer-averaged vertical velocity in the hydrostatic step
                            #for k in range(n_layers):
                            #    self.fields.uv_nh.assign(u_list[k])
                            #    timestepper_dic['w_layer_'+str(k+1)].advance(self.simulation_time, update_forcings)
                        else:
                            self.uv_av_1.assign(uv_2d)

                        # solve non-hydrostatic pressure q
                        solver_q_ml.solve()
                        if n_layers == 1:
                            q_list[0].assign(self.fields.q_2d)
                        else:
                            self.fields.q_2d.assign(q_list[0])

                        # update layer-averaged velocities
                        sum_uv_av = 0.
                        for k in range(n_layers):
                            solver_u_ml[k].solve()
                            solver_w_ml[k].solve()
                            sum_uv_av = sum_uv_av + alpha[k]*u_list[k]
                        self.fields.uv_2d.project(sum_uv_av)

                    if self.options.use_hllc_flux:
                        # update momentum
                        for i in range(2):
                            self.fields.solution_2d.sub(i+1).dat.data[:] = 0
                            self.fields.solution_2d.sub(i+1).dat.data[ind_wet] = self.fields.uv_2d.dat.data[ind_wet, i] * h_array[ind_wet]
                            self.fields.mom_2d.dat.data[:, i] = self.fields.solution_2d.sub(i+1).dat.data[:]

                    if True:

                        if self.options.use_limiter_for_elevation:
                            if self.limiter_u is not None:
                                self.limiter_u.apply(self.fields.uv_2d)
                        self.fields.uv_2d.dat.data[ind_dry] = [0, 0]

                        # update free surface elevation
                        if self.options.update_free_surface:
                            if self.options.use_hllc_flux:
                                self.fields.elev_2d.assign(self.elev_fs)
                                for i_stage in range(n_stages):
                                    solver_fs.solve()
                                    self.fields.elev_2d.assign(coeff[i_stage][0]*self.solution_old.sub(0) + coeff[i_stage][1]*self.elev_fs)
                            else:
                                self.elev_fs.assign(self.solution_old.sub(0))
                                self.timestepper_free_surface.advance(self.simulation_time, update_forcings)
                                self.fields.elev_2d.assign(self.elev_fs)

                            if self.options.use_limiter_for_elevation:
                                if self.limiter_h is not None:
                                    self.limiter_h.apply(self.fields.elev_2d)

                            ind_dry = np.where(h_array[:] <= 0)[0]
                            self.fields.uv_2d.dat.data[ind_dry] = [0, 0]

                            if self.options.use_hllc_flux and self.options.use_wetting_and_drying:
                                self.wd_modification.apply(self.fields.solution_2d, self.options.depth_wd_interface, 
                                                           use_limiter=False, use_eta_solution=True, bathymetry=self.bathymetry_dg)



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
                if self.options.use_wetting_and_drying and (not self.options.no_wave_flow):
                    self.solution_tmp.assign(self.fields.solution_2d)
                    ind = np.where(h_array[:] <= 1E-6)[0]
                    self.fields.elev_2d.dat.data[ind] = 1E-6 - self.bathymetry_dg.dat.data[ind]
                self.export()
                if self.options.use_wetting_and_drying and (not self.options.no_wave_flow):
                    self.fields.solution_2d.assign(self.solution_tmp)

                if export_func is not None:
                    export_func()
