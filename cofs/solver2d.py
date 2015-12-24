"""
Module for 2D solver class.

Tuomas Karna 2015-10-17
"""
from utility import *
import shallowwater_eq
import timeintegrator
import time as timeMod
from mpi4py import MPI
import exporter
from cofs.field_defs import fieldMetadata
from cofs.options import ModelOptions


class FlowSolver2d(FrozenClass):
    """Creates and solves 2D depth averaged equations with RT1-P1DG elements"""
    def __init__(self, mesh2d, bathymetry_2d, order=1, options={}):
        self._initialized = False

        # create 3D mesh
        self.mesh2d = mesh2d

        # Time integrator setup
        self.dt = None

        # 2d model specific default options
        options.setdefault('timeStepperType', 'SSPRK33')
        options.setdefault('timerLabels', ['mode2d'])
        options.setdefault('fields_to_export', ['elev_2d', 'uv_2d'])

        # override default options
        self.options = ModelOptions()
        self.options.update(options)

        # simulation time step bookkeeping
        self.simulation_time = 0
        self.iteration = 0
        self.iExport = 1

        self.visu_spaces = {}
        """Maps function space to a space where fields will be projected to for visualization"""

        self.fields = FieldDict()
        """Holds all functions needed by the solver object."""
        self.function_spaces = AttrDict()
        """Holds all function spaces needed by the solver object."""
        self.fields.bathymetry_2d = bathymetry_2d

        self.bnd_functions = {'shallow_water': {}}
        self._isfrozen = True  # disallow creating new attributes

    def setTimeStep(self):
        self.dt = self.options.dt
        if self.dt is None:
            mesh2d_dt = self.eq_sw.getTimeStep(Umag=self.options.uAdvection)
            dt = self.options.cfl_2d*float(mesh2d_dt.dat.data.min()/20.0)
            dt = comm.allreduce(dt, op=MPI.MIN)
            self.dt = dt
        if commrank == 0:
            print 'dt =', self.dt
            sys.stdout.flush()

    def createFunctionSpaces(self):
        """Creates function spaces"""
        self._isfrozen = False
        # ----- function spaces: elev in H, uv in U, mixed is W
        self.function_spaces.P0_2d = FunctionSpace(self.mesh2d, 'DG', 0)
        self.function_spaces.P1_2d = FunctionSpace(self.mesh2d, 'CG', 1)
        self.function_spaces.P1v_2d = VectorFunctionSpace(self.mesh2d, 'CG', 1)
        # 2D velocity space
        if self.options.mimetic:
            self.function_spaces.U_2d = FunctionSpace(self.mesh2d, 'RT', self.options.order+1)
        else:
            self.function_spaces.U_2d = VectorFunctionSpace(self.mesh2d, 'DG', self.options.order, name='U_2d')
        # TODO can this be omitted?
        # self.function_spaces.U_scalar_2d = FunctionSpace(self.mesh2d, 'DG', self.options.order)
        self.function_spaces.H_2d = FunctionSpace(self.mesh2d, 'DG', self.options.order)
        self.function_spaces.V_2d = MixedFunctionSpace([self.function_spaces.U_2d, self.function_spaces.H_2d])

        self.visu_spaces[self.function_spaces.U_2d] = self.function_spaces.P1v_2d
        self.visu_spaces[self.function_spaces.H_2d] = self.function_spaces.P1_2d
        self._isfrozen = True

    def createEquations(self):
        """Creates functions, equations and time steppers."""
        if not hasattr(self, 'U_2d'):
            self.createFunctionSpaces()
        self._isfrozen = False
        # ----- fields
        self.fields.solution_2d = Function(self.function_spaces.V_2d)

        # ----- Equations
        self.eq_sw = shallowwater_eq.ShallowWaterEquations(
            self.fields.solution_2d,
            self.fields.bathymetry_2d,
            lin_drag=self.options.lin_drag,
            viscosity_h=self.fields.get('hViscosity'),
            uvLaxFriedrichs=self.options.uvLaxFriedrichs,
            coriolis=self.options.coriolis,
            wind_stress=self.options.wind_stress,
            uv_source=self.options.uv_source_2d,
            elev_source=self.options.elev_source_2d,
            nonlin=self.options.nonlin)

        self.eq_sw.bnd_functions = self.bnd_functions['shallow_water']

        # ----- Time integrators
        self.setTimeStep()
        if self.options.timeStepperType.lower() == 'ssprk33':
            self.timeStepper = timeintegrator.SSPRK33Stage(self.eq_sw, self.dt,
                                                           self.eq_sw.solver_parameters)
        elif self.options.timeStepperType.lower() == 'ssprk33semi':
            self.timeStepper = timeintegrator.SSPRK33StageSemiImplicit(self.eq_sw,
                                                                       self.dt, self.eq_sw.solver_parameters)
        elif self.options.timeStepperType.lower() == 'forwardeuler':
            self.timeStepper = timeintegrator.ForwardEuler(self.eq_sw, self.dt,
                                                           self.eq_sw.solver_parameters)
        elif self.options.timeStepperType.lower() == 'cranknicolson':
            self.timeStepper = timeintegrator.CrankNicolson(self.eq_sw, self.dt,
                                                            self.eq_sw.solver_parameters)
        elif self.options.timeStepperType.lower() == 'sspimex':
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
            self.timeStepper = timeintegrator.SSPIMEX(self.eq_sw, self.dt,
                                                      solver_parameters=sp_expl,
                                                      solver_parameters_dirk=sp_impl)
        else:
            raise Exception('Unknown time integrator type: '+str(self.options.timeStepperType))

        # ----- File exporters
        # correct treatment of the split 2d functions
        uv_2d, elev_2d = self.fields.solution_2d.split()
        self.fields.uv_2d = uv_2d
        self.fields.elev_2d = elev_2d
        self.visu_spaces[uv_2d.function_space()] = self.function_spaces.P1v_2d
        self.visu_spaces[elev_2d.function_space()] = self.function_spaces.P1_2d
        self.exporters = {}
        e = exporter.ExportManager(self.options.outputdir,
                                   self.options.fields_to_export,
                                   self.fields,
                                   self.visu_spaces,
                                   fieldMetadata,
                                   export_type='vtk',
                                   verbose=self.options.verbose > 0)
        self.exporters['vtk'] = e
        numpyDir = os.path.join(self.options.outputdir, 'numpy')
        e = exporter.ExportManager(numpyDir,
                                   self.options.fields_to_exportNumpy,
                                   self.fields,
                                   self.visu_spaces,
                                   fieldMetadata,
                                   export_type='numpy',
                                   verbose=self.options.verbose > 0)
        self.exporters['numpy'] = e
        hdf5Dir = os.path.join(self.options.outputdir, 'hdf5')
        e = exporter.ExportManager(hdf5Dir,
                                   self.options.fields_to_exportHDF5,
                                   self.fields,
                                   self.visu_spaces,
                                   fieldMetadata,
                                   export_type='hdf5',
                                   verbose=self.options.verbose > 0)
        self.exporters['hdf5'] = e

        self._initialized = True
        self._isfrozen = True  # disallow creating new attributes

    def assignInitialConditions(self, elev=None, uv_init=None):
        if not self._initialized:
            self.createEquations()
        uv_2d, elev_2d = self.fields.solution_2d.split()
        if elev is not None:
            elev_2d.project(elev)
        if uv_init is not None:
            uv_2d.project(uv_init)

        self.timeStepper.initialize(self.fields.solution_2d)

    def export(self):
        for key in self.exporters:
            self.exporters[key].export()

    def loadState(self, iExport, t, iteration):
        """Loads simulation state from hdf5 outputs."""
        uv_2d, elev_2d = self.fields.solution_2d.split()
        self.exporters['hdf5'].exporters['uv_2d'].load(iExport, uv_2d)
        self.exporters['hdf5'].exporters['elev_2d'].load(iExport, elev_2d)
        self.assignInitialConditions(elev=elev_2d, uv_init=uv_2d)
        self.iExport = iExport
        self.simulation_time = t
        self.iteration = iteration
        self.printState(0.0)
        self.iExport += 1
        for k in self.exporters:
            self.exporters[k].setNextExportIx(self.iExport)

    def printState(self, cputime):
        norm_h = norm(self.fields.solution_2d.split()[1])
        norm_u = norm(self.fields.solution_2d.split()[0])

        if commrank == 0:
            line = ('{iexp:5d} {i:5d} T={t:10.2f} '
                    'eta norm: {e:10.4f} u norm: {u:10.4f} {cpu:5.2f}')
            print(bold(line.format(iexp=self.iExport, i=self.iteration, t=self.simulation_time, e=norm_h,
                                   u=norm_u, cpu=cputime)))
            sys.stdout.flush()

    def iterate(self, update_forcings=None,
                exportFunc=None):
        if not self._initialized:
            self.createEquations()

        T_epsilon = 1.0e-5
        cputimestamp = timeMod.clock()
        next_export_t = self.simulation_time + self.options.TExport

        # initialize conservation checks
        if self.options.checkVolConservation2d:
            eta = self.fields.solution_2d.split()[1]
            Vol2d_0 = comp_volume_2d(eta, self.fields.bathymetry_2d)
            print_info('Initial volume 2d {0:f}'.format(Vol2d_0))

        # initial export
        self.export()
        if exportFunc is not None:
            exportFunc()
        self.exporters['vtk'].export_bathymetry(self.fields.bathymetry_2d)

        while self.simulation_time <= self.options.T + T_epsilon:

            self.timeStepper.advance(self.simulation_time, self.dt, self.fields.solution_2d,
                                     update_forcings)

            # Move to next time step
            self.iteration += 1
            self.simulation_time = self.iteration*self.dt

            # Write the solution to file
            if self.simulation_time >= next_export_t - T_epsilon:
                cputime = timeMod.clock() - cputimestamp
                cputimestamp = timeMod.clock()
                self.printState(cputime)
                if self.options.checkVolConservation2d:
                    Vol2d = comp_volume_2d(self.fields.solution_2d.split()[1],
                                           self.fields.bathymetry_2d)
                if commrank == 0:
                    line = 'Rel. {0:s} error {1:11.4e}'
                    if self.options.checkVolConservation2d:
                        print(line.format('vol 2d', (Vol2d_0 - Vol2d)/Vol2d_0))
                    sys.stdout.flush()

                self.export()
                if exportFunc is not None:
                    exportFunc()

                next_export_t += self.options.TExport
                self.iExport += 1

                if commrank == 0 and len(self.options.timerLabels) > 0:
                    cost = {}
                    relcost = {}
                    totcost = 0
                    for label in self.options.timerLabels:
                        value = timing(label, reset=True)
                        cost[label] = value
                        totcost += value
                    for label in self.options.timerLabels:
                        c = cost[label]
                        relcost = c/max(totcost, 1e-6)
                        print '{0:25s} : {1:11.6f} {2:11.2f}'.format(
                            label, c, relcost)
                        sys.stdout.flush()
