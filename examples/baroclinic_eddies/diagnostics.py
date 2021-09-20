"""
Implements diagnostic calculators for lock exchange test.
"""
from thetis import *


class RPECalculator(DiagnosticCallback):
    """
    Computes reference potential energy (RPE) from density field.

    RPE is stands for potential energy that is not available for the dynamics,
    it is a metric of mixing.

    RPE = int(density*z)*dx

    where density is sorted over the vertical by density: the heaviest water
    mass lies on the bottom of the domain.

    Relative RPE is given by

    \bar{RPE}(t) = (RPE(t) - RPE(0))/RPE(0)

    \bar{RPE} measures the fraction of initial potential energy that has been lost due
    to mixing.
    """
    name = 'rpe'
    variable_names = ['rpe', 'rel_rpe']

    def _initialize(self):
        # compute area of 2D mesh
        one_2d = Constant(1.0, domain=self.solver_obj.mesh2d.coordinates.ufl_domain())
        self.area_2d = assemble(one_2d*dx)
        self.rho = self.solver_obj.fields.density_3d
        fs = self.rho.function_space()
        self.test = TestFunction(fs)
        self.nodal_volume = assemble(self.test*dx)
        self.initial_rpe = None
        self._initialized = True

    def __call__(self):
        if not hasattr(self, '_initialized') or self._initialized is False:
            self._initialize()
        self.nodal_volume = assemble(self.test*dx)
        # NOTE naive parallelization: gather numpy arrays at rank 0 process
        comm = self.solver_obj.comm
        rho_array = comm.gather(self.rho.dat.data[:], root=0)
        volume_array = comm.gather(self.nodal_volume.dat.data[:], root=0)
        if comm.rank == 0:
            rho_array = numpy.hstack(rho_array)
            volume_array = numpy.hstack(volume_array)
            sorted_ix = numpy.argsort(rho_array)[::-1]
            rho0 = float(physical_constants['rho0'])
            rho_array = rho_array[sorted_ix] + rho0
            volume_array = volume_array[sorted_ix]
            z = (numpy.cumsum(volume_array) - 0.5*volume_array)/self.area_2d
            g = float(physical_constants['g_grav'])
            rpe = g*numpy.sum(rho_array*volume_array*z)
        else:
            rpe = None
        rpe = comm.bcast(rpe, root=0)
        if self.initial_rpe is None:
            self.initial_rpe = rpe
        rel_rpe = (rpe - self.initial_rpe)/numpy.abs(self.initial_rpe)
        return rpe, rel_rpe

    def message_str(self, *args):
        line = 'RPE: {:16.10e}, rel. RPE: {:14.8e}'.format(args[0], args[1])
        return line
