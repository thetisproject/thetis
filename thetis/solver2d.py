"""
Module for 2D solver class.

Tuomas Karna 2015-10-17
"""
from __future__ import absolute_import
from .utility import *
from . import shallowwater_eq
from . import timeintegrator
import time as time_mod
from mpi4py import MPI
from . import exporter
from .field_defs import field_metadata
from .options import ModelOptions
from . import callback
from .log import *


class FlowSolver2d(FrozenClass):
    """Creates and solves 2D depth averaged equations with RT1-P1DG elements"""
    def __init__(self, mesh2d, bathymetry_2d, order=1, options=None):
        self._initialized = False
        self.mesh2d = mesh2d
        self.comm = mesh2d.comm
        # add boundary length info
        bnd_len = compute_boundary_length(self.mesh2d)
        self.mesh2d.boundary_len = bnd_len

        # Time integrator setup
        self.dt = None

        # 2d model specific default options
        self.options = ModelOptions()
        self.options.setdefault('timestepper_type', 'SSPRK33')
        self.options.setdefault('fields_to_export', ['elev_2d', 'uv_2d'])
        if options is not None:
            self.options.update(options)

        # simulation time step bookkeeping
        self.simulation_time = 0
        self.iteration = 0
        self.i_export = 0
        self.next_export_t = self.simulation_time + self.options.t_export

        self.callbacks = callback.CallbackManager()
        """Callback manager object"""

        self.fields = FieldDict()
        """Holds all functions needed by the solver object."""
        self.function_spaces = AttrDict()
        """Holds all function spaces needed by the solver object."""
        self.fields.bathymetry_2d = bathymetry_2d
        self.export_initial_state = True
        """Do export initial state. False if continuing a simulation"""

        self.bnd_functions = {'shallow_water': {}}
        self._isfrozen = True  # disallow creating new attributes

    def compute_time_step(self, u_mag=Constant(0.0)):
        """
        Computes maximum explicit time step from CFL condition.

        dt = CellSize/U

        Assumes velocity scale U = sqrt(g*H) + u_mag
        where u_mag is estimated advective velocity
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
        u = (sqrt(g * bath_pos) + u_mag)
        a = inner(test, trial) * dx
        l = inner(test, csize / u) * dx
        solve(a == l, solution)
        return solution

    def set_time_step(self, alpha=0.05):
        self.dt = self.options.dt
        if self.dt is None:
            mesh2d_dt = self.compute_time_step(u_mag=self.options.u_advection)
            dt = self.options.cfl_2d*alpha*float(mesh2d_dt.dat.data.min())
            dt = self.comm.allreduce(dt, op=MPI.MIN)
            self.dt = dt
        if self.comm.rank == 0:
            print_output('dt = {:}'.format(self.dt))
            sys.stdout.flush()

    def create_function_spaces(self):
        """Creates function spaces"""
        self._isfrozen = False
        # ----- function spaces: elev in H, uv in U, mixed is W
        self.function_spaces.P0_2d = FunctionSpace(self.mesh2d, 'DG', 0)
        self.function_spaces.P1_2d = FunctionSpace(self.mesh2d, 'CG', 1)
        self.function_spaces.P1v_2d = VectorFunctionSpace(self.mesh2d, 'CG', 1)
        self.function_spaces.P1DG_2d = FunctionSpace(self.mesh2d, 'DG', 1)
        self.function_spaces.P1DGv_2d = VectorFunctionSpace(self.mesh2d, 'DG', 1)
        # 2D velocity space
        if self.options.element_family == 'rt-dg':
            self.function_spaces.U_2d = FunctionSpace(self.mesh2d, 'RT', self.options.order+1)
            self.function_spaces.H_2d = FunctionSpace(self.mesh2d, 'DG', self.options.order)
        elif self.options.element_family == 'dg-cg':
            self.function_spaces.U_2d = VectorFunctionSpace(self.mesh2d, 'DG', self.options.order, name='U_2d')
            self.function_spaces.H_2d = FunctionSpace(self.mesh2d, 'CG', self.options.order+1)
        elif self.options.element_family == 'dg-dg':
            self.function_spaces.U_2d = VectorFunctionSpace(self.mesh2d, 'DG', self.options.order, name='U_2d')
            self.function_spaces.H_2d = FunctionSpace(self.mesh2d, 'DG', self.options.order)
        else:
            raise Exception('Unsupported finite element family {:}'.format(self.options.element_family))
        self.function_spaces.V_2d = MixedFunctionSpace([self.function_spaces.U_2d, self.function_spaces.H_2d])

        self._isfrozen = True

    def create_equations(self):
        """Creates functions, equations and time steppers."""
        if not hasattr(self, 'U_2d'):
            self.create_function_spaces()
        self._isfrozen = False
        # ----- fields
        self.fields.solution_2d = Function(self.function_spaces.V_2d, name='solution_2d')
        self.fields.h_elem_size_2d = Function(self.function_spaces.P1_2d)
        get_horizontal_elem_size_2d(self.fields.h_elem_size_2d)

        # ----- Equations
        self.eq_sw = shallowwater_eq.ShallowWaterEquations(
            self.fields.solution_2d.function_space(),
            self.fields.bathymetry_2d,
            nonlin=self.options.nonlin,
            include_grad_div_viscosity_term=self.options.include_grad_div_viscosity_term,
            include_grad_depth_viscosity_term=self.options.include_grad_depth_viscosity_term
        )
        self.eq_sw.bnd_functions = self.bnd_functions['shallow_water']
        self._isfrozen = True  # disallow creating new attributes

    def create_timestepper(self):
        self._isfrozen = False
        # ----- Time integrators
        fields = {
            'linear_drag': self.options.linear_drag,
            'quadratic_drag': self.options.quadratic_drag,
            'mu_manning': self.options.mu_manning,
            'viscosity_h': self.options.h_viscosity,
            'uv_lax_friedrichs': self.options.uv_lax_friedrichs,
            'coriolis': self.options.coriolis,
            'wind_stress': self.options.wind_stress,
            'uv_source': self.options.uv_source_2d,
            'elev_source': self.options.elev_source_2d, }
        self.set_time_step()
        if self.options.timestepper_type.lower() == 'ssprk33':
            self.timestepper = timeintegrator.SSPRK33Stage(self.eq_sw, self.fields.solution_2d,
                                                           fields, self.dt,
                                                           bnd_conditions=self.bnd_functions['shallow_water'],
                                                           solver_parameters=self.options.solver_parameters_sw)
        elif self.options.timestepper_type.lower() == 'ssprk33semi':
            self.timestepper = timeintegrator.SSPRK33StageSemiImplicit(self.eq_sw, self.fields.solution_2d,
                                                                       fields, self.dt,
                                                                       bnd_conditions=self.bnd_functions['shallow_water'],
                                                                       solver_parameters=self.options.solver_parameters_sw,
                                                                       semi_implicit=self.options.use_linearized_semi_implicit_2d,
                                                                       theta=self.options.shallow_water_theta)

        elif self.options.timestepper_type.lower() == 'forwardeuler':
            self.timestepper = timeintegrator.ForwardEuler(self.eq_sw, self.fields.solution_2d,
                                                           fields, self.dt,
                                                           bnd_conditions=self.bnd_functions['shallow_water'],
                                                           solver_parameters=self.options.solver_parameters_sw)
        elif self.options.timestepper_type.lower() == 'backwardeuler':
            self.timestepper = timeintegrator.BackwardEuler(self.eq_sw, self.fields.solution_2d,
                                                            fields, self.dt,
                                                            bnd_conditions=self.bnd_functions['shallow_water'],
                                                            solver_parameters=self.options.solver_parameters_sw)
        elif self.options.timestepper_type.lower() == 'cranknicolson':
            self.timestepper = timeintegrator.CrankNicolson(self.eq_sw, self.fields.solution_2d,
                                                            fields, self.dt,
                                                            bnd_conditions=self.bnd_functions['shallow_water'],
                                                            solver_parameters=self.options.solver_parameters_sw,
                                                            semi_implicit=self.options.use_linearized_semi_implicit_2d,
                                                            theta=self.options.shallow_water_theta)
        elif self.options.timestepper_type.lower() == 'steadystate':
            self.timestepper = timeintegrator.SteadyState(self.eq_sw, self.fields.solution_2d,
                                                          fields, self.dt,
                                                          bnd_conditions=self.bnd_functions['shallow_water'],
                                                          solver_parameters=self.options.solver_parameters_sw)
        elif self.options.timestepper_type.lower() == 'sspimex':
            # TODO meaningful solver params
            sp_impl = {
                'ksp_type': 'gmres',
                'pc_type': 'fieldsplit',
                'pc_fieldsplit_type': 'multiplicative',
            }
            sp_expl = {
                'ksp_type': 'gmres',
                'pc_type': 'fieldsplit',
                'pc_fieldsplit_type': 'multiplicative',
            }
            self.timestepper = timeintegrator.SSPIMEX(self.eq_sw, self.fields.solution_2d, fields, self.dt,
                                                      bnd_conditions=self.bnd_functions['shallow_water'],
                                                      solver_parameters=sp_expl,
                                                      solver_parameters_dirk=sp_impl)
        else:
            raise Exception('Unknown time integrator type: '+str(self.options.timestepper_type))
        self._isfrozen = True  # disallow creating new attributes

    def create_exporters(self):
        self._isfrozen = False
        # correct treatment of the split 2d functions
        uv_2d, elev_2d = self.fields.solution_2d.split()
        self.fields.uv_2d = uv_2d
        self.fields.elev_2d = elev_2d
        self.exporters = {}
        if not self.options.no_exports:
            e = exporter.ExportManager(self.options.outputdir,
                                       self.options.fields_to_export,
                                       self.fields,
                                       field_metadata,
                                       export_type='vtk',
                                       verbose=self.options.verbose > 0)
            self.exporters['vtk'] = e
            numpy_dir = os.path.join(self.options.outputdir, 'numpy')
            e = exporter.ExportManager(numpy_dir,
                                       self.options.fields_to_export_numpy,
                                       self.fields,
                                       field_metadata,
                                       export_type='numpy',
                                       verbose=self.options.verbose > 0)
            self.exporters['numpy'] = e
            hdf5_dir = os.path.join(self.options.outputdir, 'hdf5')
            e = exporter.ExportManager(hdf5_dir,
                                       self.options.fields_to_export_hdf5,
                                       self.fields,
                                       field_metadata,
                                       export_type='hdf5',
                                       verbose=self.options.verbose > 0)
            self.exporters['hdf5'] = e

        self._isfrozen = True  # disallow creating new attributes

    def initialize(self):
        if not hasattr(self, 'U_2d'):
            self.create_function_spaces()
        if not hasattr(self, 'eq_sw'):
            self.create_equations()
        if not hasattr(self, 'timestepper'):
            self.create_timestepper()
        if not hasattr(self, 'exporters'):
            self.create_exporters()
        self._initialized = True

    def assign_initial_conditions(self, elev=None, uv_init=None):
        if not self._initialized:
            self.initialize()
        uv_2d, elev_2d = self.fields.solution_2d.split()
        if elev is not None:
            elev_2d.project(elev)
        if uv_init is not None:
            uv_2d.project(uv_init)

        self.timestepper.initialize(self.fields.solution_2d)

    def add_callback(self, callback, eval_interval='export'):
        """Adds callback to solver object

        :arg callback: DiagnosticCallback instance
        "arg eval_interval: 'export'|'timestep' Determines when callback will be evaluated.
        """
        self.callbacks.add(callback, eval_interval)

    def export(self):
        self.callbacks.evaluate(mode='export')
        for key in self.exporters:
            self.exporters[key].export()

    def load_state(self, i_export, outputdir=None, t=None, iteration=None):
        """
        Loads simulation state from hdf5 outputs.

        This replaces assign_initial_conditions in model initilization.

        This assumes that model setup is kept the same (e.g. time step) and
        all pronostic state variables are exported in hdf5 format. The required
        state variables are: elev_2d, uv_2d

        Currently hdf5 field import only works for the same number of MPI
        processes.
        """
        if not self._initialized:
            self.initialize()
        if outputdir is None:
            outputdir = self.options.outputdir
        # create new ExportManager with desired outputdir
        state_fields = ['uv_2d', 'elev_2d']
        hdf5_dir = os.path.join(outputdir, 'hdf5')
        e = exporter.ExportManager(hdf5_dir,
                                   state_fields,
                                   self.fields,
                                   field_metadata,
                                   export_type='hdf5',
                                   verbose=self.options.verbose > 0)
        e.exporters['uv_2d'].load(i_export, self.fields.uv_2d)
        e.exporters['elev_2d'].load(i_export, self.fields.elev_2d)
        self.assign_initial_conditions()

        # time stepper bookkeeping for export time step
        self.i_export = i_export
        self.next_export_t = self.i_export*self.options.t_export
        if iteration is None:
            iteration = int(np.ceil(self.next_export_t/self.dt))
        if t is None:
            t = iteration*self.dt
        self.iteration = iteration
        self.simulation_time = t

        # for next export
        self.export_initial_state = outputdir != self.options.outputdir
        if self.export_initial_state:
            offset = 0
        else:
            offset = 1
        self.next_export_t += self.options.t_export
        for k in self.exporters:
            self.exporters[k].set_next_export_ix(self.i_export + offset)

    def print_state(self, cputime):
        norm_h = norm(self.fields.solution_2d.split()[1])
        norm_u = norm(self.fields.solution_2d.split()[0])

        line = ('{iexp:5d} {i:5d} T={t:10.2f} '
                'eta norm: {e:10.4f} u norm: {u:10.4f} {cpu:5.2f}')
        print_output(line.format(iexp=self.i_export, i=self.iteration,
                                 t=self.simulation_time, e=norm_h,
                                 u=norm_u, cpu=cputime))
        sys.stdout.flush()

    def iterate(self, update_forcings=None,
                export_func=None):
        if not self._initialized:
            self.initialize()

        t_epsilon = 1.0e-5
        cputimestamp = time_mod.clock()
        next_export_t = self.simulation_time + self.options.t_export

        dump_hdf5 = self.options.export_diagnostics and not self.options.no_exports
        if self.options.check_vol_conservation_2d:
            c = callback.VolumeConservation2DCallback(self,
                                                      export_to_hdf5=dump_hdf5,
                                                      append_to_log=True)
            self.add_callback(c)

        # initial export
        self.print_state(0.0)
        if self.export_initial_state:
            self.export()
            if export_func is not None:
                export_func()
            if 'vtk' in self.exporters:
                self.exporters['vtk'].export_bathymetry(self.fields.bathymetry_2d)

        while self.simulation_time <= self.options.t_end + t_epsilon:

            self.timestepper.advance(self.simulation_time, self.dt, self.fields.solution_2d,
                                     update_forcings)

            # Move to next time step
            self.iteration += 1
            self.simulation_time = self.iteration*self.dt

            self.callbacks.evaluate(mode='timestep')

            # Write the solution to file
            if self.simulation_time >= next_export_t - t_epsilon:
                self.i_export += 1
                next_export_t += self.options.t_export

                cputime = time_mod.clock() - cputimestamp
                cputimestamp = time_mod.clock()
                self.print_state(cputime)

                self.export()
                if export_func is not None:
                    export_func()
