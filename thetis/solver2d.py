"""
Module for 2D solver class.

Tuomas Karna 2015-10-17
"""
from utility import *
import shallowwater_eq
import timeintegrator
import time as time_mod
from mpi4py import MPI
import exporter
from thetis.field_defs import field_metadata
from thetis.options import ModelOptions
import thetis.callback as callback


class FlowSolver2d(FrozenClass):
    """Creates and solves 2D depth averaged equations with RT1-P1DG elements"""
    def __init__(self, mesh2d, bathymetry_2d, order=1, options=None):
        self._initialized = False
        self.mesh2d = mesh2d

        # add boundary length info
        bnd_len = compute_boundary_length(self.mesh2d)
        self.mesh2d.boundary_len = bnd_len

        # Time integrator setup
        self.dt = None

        # 2d model specific default options
        self.options = ModelOptions()
        self.options.setdefault('timestepper_type', 'SSPRK33')
        self.options.setdefault('timer_labels', ['mode2d'])
        self.options.setdefault('fields_to_export', ['elev_2d', 'uv_2d'])
        if options is not None:
            self.options.update(options)

        # simulation time step bookkeeping
        self.simulation_time = 0
        self.iteration = 0
        self.i_export = 1

        self.callbacks = OrderedDict()
        """List of callback functions that will be called during exports"""

        self.fields = FieldDict()
        """Holds all functions needed by the solver object."""
        self.function_spaces = AttrDict()
        """Holds all function spaces needed by the solver object."""
        self.fields.bathymetry_2d = bathymetry_2d

        self.bnd_functions = {'shallow_water': {}}
        self._isfrozen = True  # disallow creating new attributes

    def set_time_step(self):
        self.dt = self.options.dt
        if self.dt is None:
            mesh2d_dt = self.eq_sw.get_time_step(u_mag=self.options.u_advection)
            dt = self.options.cfl_2d*float(mesh2d_dt.dat.data.min()/20.0)
            dt = comm.allreduce(dt, op=MPI.MIN)
            self.dt = dt
        if commrank == 0:
            print 'dt =', self.dt
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
        if self.options.mimetic and self.options.continuous_pressure:
            raise ValueError("Cannot combine options mimetic and continuous_pressure")
        # 2D velocity space
        if self.options.mimetic:
            self.function_spaces.U_2d = FunctionSpace(self.mesh2d, 'RT', self.options.order+1)
            self.function_spaces.H_2d = FunctionSpace(self.mesh2d, 'DG', self.options.order)
        elif self.options.continuous_pressure:
            self.function_spaces.U_2d = VectorFunctionSpace(self.mesh2d, 'DG', self.options.order, name='U_2d')
            self.function_spaces.H_2d = FunctionSpace(self.mesh2d, 'CG', self.options.order+1)
        else:
            self.function_spaces.U_2d = VectorFunctionSpace(self.mesh2d, 'DG', self.options.order, name='U_2d')
            self.function_spaces.H_2d = FunctionSpace(self.mesh2d, 'DG', self.options.order)
        # TODO can this be omitted?
        # self.function_spaces.U_scalar_2d = FunctionSpace(self.mesh2d, 'DG', self.options.order)
        self.function_spaces.V_2d = MixedFunctionSpace([self.function_spaces.U_2d, self.function_spaces.H_2d])

        self._isfrozen = True

    def create_equations(self):
        """Creates functions, equations and time steppers."""
        if not hasattr(self, 'U_2d'):
            self.create_function_spaces()
        self._isfrozen = False
        # ----- fields
        self.fields.solution_2d = Function(self.function_spaces.V_2d)

        # ----- Equations
        self.eq_sw = shallowwater_eq.ShallowWaterEquations(
            self.fields.solution_2d.function_space(),
            self.fields.bathymetry_2d,
            nonlin=self.options.nonlin,
            include_grad_div_viscosity_term=self.options.include_grad_div_viscosity_term,
            include_grad_depth_viscosity_term=self.options.include_grad_depth_viscosity_term
        )
        self.eq_sw.bnd_functions = self.bnd_functions['shallow_water']

        # ----- Time integrators
        fields = {
            'lin_drag': self.options.lin_drag,
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
                                                           solver_parameters=self.eq_sw.solver_parameters)
        elif self.options.timestepper_type.lower() == 'ssprk33semi':
            self.timestepper = timeintegrator.SSPRK33StageSemiImplicit(self.eq_sw, self.fields.solution_2d,
                                                                       fields, self.dt,
                                                                       bnd_conditions=self.bnd_functions['shallow_water'],
                                                                       solver_parameters=self.eq_sw.solver_parameters,
                                                                       semi_implicit=options.use_linearized_semi_implicit_2d)

        elif self.options.timestepper_type.lower() == 'forwardeuler':
            self.timestepper = timeintegrator.ForwardEuler(self.eq_sw, self.fields.solution_2d,
                                                           fields, self.dt,
                                                           bnd_conditions=self.bnd_functions['shallow_water'],
                                                           solver_parameters=self.eq_sw.solver_parameters)
        elif self.options.timestepper_type.lower() == 'cranknicolson':
            self.timestepper = timeintegrator.CrankNicolson(self.eq_sw, self.fields.solution_2d,
                                                            fields, self.dt,
                                                            bnd_conditions=self.bnd_functions['shallow_water'],
                                                            solver_parameters=self.eq_sw.solver_parameters,
                                                            semi_implicit=options.use_linearized_semi_implicit_2d)
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
            self.timestepper = timeintegrator.SSPIMEX(self.eq_sw, self.dt,
                                                      solver_parameters=sp_expl,
                                                      solver_parameters_dirk=sp_impl)
        else:
            raise Exception('Unknown time integrator type: '+str(self.options.timestepper_type))

        # ----- File exporters
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

        self._initialized = True
        self._isfrozen = True  # disallow creating new attributes

    def assign_initial_conditions(self, elev=None, uv_init=None):
        if not self._initialized:
            self.create_equations()
        uv_2d, elev_2d = self.fields.solution_2d.split()
        if elev is not None:
            elev_2d.project(elev)
        if uv_init is not None:
            uv_2d.project(uv_init)

        self.timestepper.initialize(self.fields.solution_2d)

    def export(self):
        for key in self.exporters:
            self.exporters[key].export()

    def load_state(self, i_export, t, iteration):
        """Loads simulation state from hdf5 outputs."""
        uv_2d, elev_2d = self.fields.solution_2d.split()
        self.exporters['hdf5'].exporters['uv_2d'].load(i_export, uv_2d)
        self.exporters['hdf5'].exporters['elev_2d'].load(i_export, elev_2d)
        self.assign_initial_conditions(elev=elev_2d, uv_init=uv_2d)
        self.i_export = i_export
        self.simulation_time = t
        self.iteration = iteration
        self.print_state(0.0)
        self.i_export += 1
        for k in self.exporters:
            self.exporters[k].set_next_export_ix(self.i_export)

    def print_state(self, cputime):
        norm_h = norm(self.fields.solution_2d.split()[1])
        norm_u = norm(self.fields.solution_2d.split()[0])

        if commrank == 0:
            line = ('{iexp:5d} {i:5d} T={t:10.2f} '
                    'eta norm: {e:10.4f} u norm: {u:10.4f} {cpu:5.2f}')
            print(bold(line.format(iexp=self.i_export, i=self.iteration, t=self.simulation_time, e=norm_h,
                                   u=norm_u, cpu=cputime)))
            sys.stdout.flush()

    def iterate(self, update_forcings=None,
                export_func=None):
        if not self._initialized:
            self.create_equations()

        t_epsilon = 1.0e-5
        cputimestamp = time_mod.clock()
        next_export_t = self.simulation_time + self.options.t_export

        if self.options.check_vol_conservation_2d:
            self.callbacks['vol2d'] = callback.VolumeConservation2DCallback()

        for key in self.callbacks:
            self.callbacks[key].initialize(self)

        # initial export
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

            # Write the solution to file
            if self.simulation_time >= next_export_t - t_epsilon:
                cputime = time_mod.clock() - cputimestamp
                cputimestamp = time_mod.clock()
                self.print_state(cputime)

                for key in self.callbacks:
                    self.callbacks[key].update(self)
                    self.callbacks[key].report()

                self.export()
                if export_func is not None:
                    export_func()

                next_export_t += self.options.t_export
                self.i_export += 1

                if commrank == 0 and len(self.options.timer_labels) > 0:
                    cost = {}
                    relcost = {}
                    totcost = 0
                    for label in self.options.timer_labels:
                        value = timing(label, reset=True)
                        cost[label] = value
                        totcost += value
                    for label in self.options.timer_labels:
                        c = cost[label]
                        relcost = c/max(totcost, 1e-6)
                        print '{0:25s} : {1:11.6f} {2:11.2f}'.format(
                            label, c, relcost)
                        sys.stdout.flush()
