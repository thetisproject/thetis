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
            rho_array = np.hstack(rho_array)
            volume_array = np.hstack(volume_array)
            sorted_ix = np.argsort(rho_array)[::-1]
            rho_array = rho_array[sorted_ix] + physical_constants['rho0'].dat.data[0]
            volume_array = volume_array[sorted_ix]
            z = (np.cumsum(volume_array) - 0.5*volume_array)/self.area_2d
            g = physical_constants['g_grav'].dat.data[0]
            rpe = g*np.sum(rho_array*volume_array*z)
        else:
            rpe = None
        rpe = comm.bcast(rpe, root=0)
        if self.initial_rpe is None:
            self.initial_rpe = rpe
        rel_rpe = (rpe - self.initial_rpe)/np.abs(self.initial_rpe)
        return rpe, rel_rpe

    def message_str(self, *args):
        line = 'RPE: {:16.10e}, rel. RPE: {:14.8e}'.format(args[0], args[1])
        return line


class KineticEnergyCalculator(DiagnosticCallback):
    """
    Computes kinetic energy from the velocity field.

    E_kin = integral( 1/2 u^2 )

    """
    name = 'ekin'
    variable_names = ['ekin']

    def _initialize(self):
        self.uv = self.solver_obj.fields.uv_3d
        self.uv_dav = self.solver_obj.fields.uv_dav_3d
        self.w = self.solver_obj.fields.w_3d
        self._initialized = True

    def __call__(self):
        if not hasattr(self, '_initialized') or self._initialized is False:
            self._initialize()
        u = self.uv + self.uv_dav + self.w
        value = assemble(0.5*dot(u, u)*dx)
        return (value, )

    def message_str(self, *args):
        line = 'Ekin: {:16.10e}'.format(args[0])
        return line


class EnstrophyCalculator(DiagnosticCallback):
    """
    Computes enstrophy from horizontal velocity field.

    E = integral( (-du/dy + dv/dx)^2 )

    """
    name = 'enstrophy'
    variable_names = ['enstrophy']

    def _initialize(self):
        self.uv = self.solver_obj.fields.uv_3d
        self.uv_dav = self.solver_obj.fields.uv_dav_3d
        self._initialized = True

    def __call__(self):
        if not hasattr(self, '_initialized') or self._initialized is False:
            self._initialize()
        u = self.uv + self.uv_dav
        omega = -Dx(u[0], 1) + Dx(u[1], 0)
        value = assemble(omega**2 * dx)
        return (value, )

    def message_str(self, *args):
        line = 'Enstrophy: {:16.10e}'.format(args[0])
        return line


class SurfEnstrophyCalculator(DiagnosticCallback):
    """
    Computes enstrophy from horizontal velocity field on the surface.

    E = integral( (-du/dy + dv/dx)^2 )

    """
    name = 'surface-enstrophy'
    variable_names = ['enstrophy']

    def _initialize(self):
        self.uv = self.solver_obj.fields.uv_3d
        self.uv_dav = self.solver_obj.fields.uv_dav_3d
        self._initialized = True

    def __call__(self):
        if not hasattr(self, '_initialized') or self._initialized is False:
            self._initialize()
        u = self.uv + self.uv_dav
        omega = -Dx(u[0], 1) + Dx(u[1], 0)
        value = assemble(omega**2 * ds_surf)
        return (value, )

    def message_str(self, *args):
        line = 'Surf. Enstrophy: {:16.10e}'.format(args[0])
        return line
