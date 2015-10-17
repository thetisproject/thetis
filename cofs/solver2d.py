"""
Module for 2D solver class.

Tuomas Karna 2015-10-17
"""
from utility import *
import shallowWaterEq
import timeIntegrator as timeIntegrator
import limiter
import time as timeMod
from mpi4py import MPI
import exporter
import ufl
import weakref
from cofs.fieldDefs import fieldMetadata
from cofs.options import modelOptions


class flowSolver2d(object):
    """Creates and solves 2D depth averaged equations with RT1-P1DG elements"""
    def __init__(self, mesh2d, bathymetry2d, order=1, options={}):
        self._initialized = False

        # create 3D mesh
        self.mesh2d = mesh2d

        # Time integrator setup
        self.dt = None

        # 2d model specific default options
        options.setdefault('timeStepperType', 'SSPRK33')
        options.setdefault('timerLabels', ['mode2d'])
        options.setdefault('fieldsToExport', ['elev2d', 'uv2d'])

        # override default options
        self.options = modelOptions()
        self.options.update(options)

        self.visualizationSpaces = {}
        """Maps function space to a space where fields will be projected to for visualization"""

        self.fields = fieldDict()
        """Holds all functions needed by the solver object."""
        self.fields.bathymetry2d = bathymetry2d

        self.bnd_functions = {'shallow_water': {}}

    def setTimeStep(self):
        mesh2d_dt = self.eq_sw.getTimeStep(Umag=self.options.uAdvection)
        dt = self.options.cfl_2d*float(mesh2d_dt.dat.data.min()/20.0)
        dt = comm.allreduce(dt, op=MPI.MIN)
        if self.dt is None:
            self.dt = dt
        if commrank == 0:
            print 'dt =', self.dt
            sys.stdout.flush()

    def mightyCreator(self):
        """Creates function spaces, functions, equations and time steppers."""
        # ----- function spaces: elev in H, uv in U, mixed is W
        self.P0_2d = FunctionSpace(self.mesh2d, 'DG', 0)
        self.P1_2d = FunctionSpace(self.mesh2d, 'CG', 1)
        #self.U_2d = FunctionSpace(self.mesh2d, 'RT', self.options.order+1)
        self.U_2d = VectorFunctionSpace(self.mesh2d, 'DG', self.options.order)
        self.U_visu_2d = VectorFunctionSpace(self.mesh2d, 'CG', self.options.order)
        self.U_scalar_2d = FunctionSpace(self.mesh2d, 'DG', self.options.order)
        self.H_2d = FunctionSpace(self.mesh2d, 'DG', self.options.order)
        self.H_visu_2d = self.P1_2d
        self.V_2d = MixedFunctionSpace([self.U_2d, self.H_2d])

        # ----- fields
        self.fields.solution2d = Function(self.V_2d, name='solution2d')

        # ----- Equations
        self.eq_sw = shallowWaterEq.shallowWaterEquations(
            self.fields.solution2d,
            self.fields.bathymetry2d,
            lin_drag=self.options.lin_drag,
            viscosity_h=self.fields.get('hViscosity'),
            uvLaxFriedrichs=self.options.uvLaxFriedrichs,
            coriolis=self.options.coriolis,
            wind_stress=self.options.wind_stress,
            nonlin=self.options.nonlin)

        self.eq_sw.bnd_functions = self.bnd_functions['shallow_water']

        # ----- Time integrators
        self.setTimeStep()
        if self.options.timeStepperType.lower() == 'ssprk33':
            self.timeStepper = timeIntegrator.SSPRK33Stage(self.eq_sw, self.dt,
                                                            self.eq_sw.solver_parameters)
        elif self.options.timeStepperType.lower() == 'ssprk33semi':
            self.timeStepper = timeIntegrator.SSPRK33StageSemiImplicit(self.eq_sw,
                                                            self.dt, self.eq_sw.solver_parameters)
        elif self.options.timeStepperType.lower() == 'forwardeuler':
            self.timeStepper = timeIntegrator.ForwardEuler(self.eq_sw, self.dt,
                                                            self.eq_sw.solver_parameters)
        elif self.options.timeStepperType.lower() == 'cranknicolson':
            self.timeStepper = timeIntegrator.CrankNicolson(self.eq_sw, self.dt,
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
            self.timeStepper = timeIntegrator.SSPIMEX(self.eq_sw, self.dt,
                                                      solver_parameters=sp_expl,
                                                      solver_parameters_dirk=sp_impl)
        else:
            raise Exception('Unknown time integrator type: '+str(self.options.timeStepperType))

        # ----- File exporters
        uv2d, eta2d = self.fields.solution2d.split()
        self.exporter = exporter.exportManager(self.options.outputDir,
                                               self.options.fieldsToExport,
                                               self.fields,
                                               self.visualizationSpaces,
                                               fieldMetadata,
                                               verbose=self.options.verbose > 0)
        self._initialized = True

    def assignInitialConditions(self, elev=None, uv_init=None):
        if not self._initialized:
            self.mightyCreator()
        uv2d, eta2d = self.fields.solution2d.split()
        if elev is not None:
            eta2d.project(elev)
        if uv_init is not None:
            uv2d.project(uv_init)

        self.timeStepper.initialize(self.fields.solution2d)

    def iterate(self, updateForcings=None,
                exportFunc=None):
        if not self._initialized:
            self.mightyCreator()

        T_epsilon = 1.0e-5
        cputimestamp = timeMod.clock()
        t = 0
        i = 0
        iExp = 1
        next_export_t = t + self.options.TExport

        # initialize conservation checks
        if self.options.checkVolConservation2d:
            eta = self.fields.solution2d.split()[1]
            Vol2d_0 = compVolume2d(eta, self.fields.bathymetry2d)
            printInfo('Initial volume 2d {0:f}'.format(Vol2d_0))

        # initial export
        self.exporter.export()
        if exportFunc is not None:
            exportFunc()
        self.exporter.exportBathymetry(self.fields.bathymetry2d)

        while t <= self.options.T + T_epsilon:

            self.timeStepper.advance(t, self.dt, self.fields.solution2d,
                                     updateForcings)

            # Move to next time step
            t += self.dt
            i += 1

            # Write the solution to file
            if t >= next_export_t - T_epsilon:
                cputime = timeMod.clock() - cputimestamp
                cputimestamp = timeMod.clock()
                norm_h = norm(self.fields.solution2d.split()[1])
                norm_u = norm(self.fields.solution2d.split()[0])

                if self.options.checkVolConservation2d:
                    Vol2d = compVolume2d(self.fields.solution2d.split()[1],
                                         self.fields.bathymetry2d)
                if commrank == 0:
                    line = ('{iexp:5d} {i:5d} T={t:10.2f} '
                            'eta norm: {e:10.4f} u norm: {u:10.4f} {cpu:5.2f}')
                    print(bold(line.format(iexp=iExp, i=i, t=t, e=norm_h,
                                           u=norm_u, cpu=cputime)))
                    line = 'Rel. {0:s} error {1:11.4e}'
                    if self.options.checkVolConservation2d:
                        print(line.format('vol 2d', (Vol2d_0 - Vol2d)/Vol2d_0))
                    sys.stdout.flush()

                self.exporter.export()
                if exportFunc is not None:
                    exportFunc()

                next_export_t += self.options.TExport
                iExp += 1

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
