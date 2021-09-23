"""
Implements diagnostic calculators for lock exchange test.
"""
from thetis import *


class FrontLocationCalculator(DiagnosticCallback):
    """
    Calculates the location of the propagating front at the top/bottom of the
    domain.

    This is a useful metric for assessing the accuracy of the model.
    Theoretical front location speed is

    U = 1/2*sqrt(g*H*delta_rho/rho_0)
    """
    name = 'front'
    variable_names = ['front_bot', 'front_top']

    def _initialize(self):
        one_2d = Constant(1.0, domain=self.solver_obj.mesh2d.coordinates.ufl_domain())
        self.area_2d = assemble(one_2d*dx)
        # store density difference
        self.rho = self.solver_obj.fields.density_3d
        self.rho_lim = [self.rho.dat.data.min(), self.rho.dat.data.max()]
        self.delta_rho = self.rho_lim[1] - self.rho_lim[0]
        self.mean_rho = 0.5*(self.rho_lim[1] + self.rho_lim[0])
        fs_2d = self.solver_obj.function_spaces.H_2d
        mesh2d = fs_2d.mesh()
        self.xyz = mesh2d.coordinates
        x = self.xyz.dat.data[:, 0]
        self.x_lim = [x.min(), x.max()]
        self.domain_center = 0.5*(self.x_lim[0] + self.x_lim[1])
        # extractors for surface rho fields
        self.rho_2d = Function(fs_2d, name='Density 2d')
        self.extractor_surf = SubFunctionExtractor(self.rho,
                                                   self.rho_2d,
                                                   boundary='top',
                                                   elem_facet='top')
        self.extractor_bot = SubFunctionExtractor(self.rho,
                                                  self.rho_2d,
                                                  boundary='bottom',
                                                  elem_facet='bottom')
        # project to higher order cg for interpolation
        fs_ho = get_functionspace(mesh2d, 'CG', 2)
        self.rho_ho = Function(fs_ho, name='Density 2d ho')
        self.ho_projector = Projector(self.rho_2d, self.rho_ho)
        self._initialized = True

    def compute_front_location(self):
        self.ho_projector.project()
        r = self.rho_ho.dat.data[:]
        off = 0.25*(self.rho_lim[1] - self.rho_lim[0])
        up_limit = self.rho_lim[1] - off
        low_limit = self.rho_lim[0] + off
        if (r > up_limit).all():  # no front all nodes above limits
            return self.x_lim[1]
        if (r < low_limit).all():  # no front all nodes below limits
            return self.x_lim[0]
        ix = (r > low_limit) * (r < up_limit)
        self.rho_ho.dat.data[:] = 0
        self.rho_ho.dat.data[ix] = 1
        mass = assemble(self.rho_ho*dx)
        if mass < 1e-20:
            return numpy.nan
        center_x = assemble(self.rho_ho*self.xyz[0]*dx)/mass
        return center_x

    def __call__(self):
        if not hasattr(self, '_initialized') or self._initialized is False:
            self._initialize()
        # compute x center of mass on top/bottom surfaces
        self.extractor_bot.solve()
        x_bot = self.compute_front_location()
        self.extractor_surf.solve()
        x_top = self.compute_front_location()
        return x_bot, x_top

    def message_str(self, *args):
        line = 'front bottom: {:12.4f}, top: {:12.4f}'.format(*args)
        return line


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
        rho_array = self.rho.dat.data[:]
        sorted_ix = numpy.argsort(rho_array)[::-1]
        rho0 = float(physical_constants['rho0'])
        rho_array = rho_array[sorted_ix] + rho0
        volume_array = self.nodal_volume.dat.data[:][sorted_ix]
        z = (numpy.cumsum(volume_array) - 0.5*volume_array)/self.area_2d
        g = float(physical_constants['g_grav'])
        rpe = g*numpy.sum(rho_array*volume_array*z)
        if self.initial_rpe is None:
            self.initial_rpe = rpe
        rel_rpe = (rpe - self.initial_rpe)/numpy.abs(self.initial_rpe)
        return rpe, rel_rpe

    def message_str(self, *args):
        line = 'RPE: {:16.10e}, rel. RPE: {:14.8e}'.format(args[0], args[1])
        return line
