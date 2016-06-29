"""
Implements diagnostic calculators for lock exchange test.
"""
from thetis import *

class FrontLocationCalculator(object):
    """
    Calculates the location of the propagating front at the top/bottom of the
    domain.

    This is a useful metric for assessing the accuracy of the model.
    Theoretical front location speed is

    U = 1/2*sqrt(g*H*delta_rho/rho_0)
    """
    def __init__(self, solver_obj):
        self.solver_obj = solver_obj
        self._initialized = False

    def _initialize(self):
        self.outfile = os.path.join(self.solver_obj.options.outputdir, 'diagnostic_front.txt')
        # flush old file
        with open(self.outfile, 'w'):
            pass
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
        fs_ho = FunctionSpace(mesh2d, 'CG', 2)
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
            return np.nan
        center_x = assemble(self.rho_ho*self.xyz[0]*dx)/mass
        return center_x

    def export(self):
        if not self._initialized:
            self._initialize()
        # compute x center of mass on top/bottom surfaces
        self.extractor_bot.solve()
        x_bot = self.compute_front_location()
        self.extractor_surf.solve()
        x_surf = self.compute_front_location()
        t = self.solver_obj.simulation_time
        with open(self.outfile, 'a') as f:
            f.write('{:20.8f} {:28.8f} {:20.8f}\n'.format(t, x_bot, x_surf))


class RPECalculator(object):
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
    def __init__(self, solver_obj):
        if COMM_WORLD.size > 1:
            raise NotImplementedError('RPE calculator has not been parallelized yet')
        self.solver_obj = solver_obj
        self._initialized = False

    def _initialize(self):
        self.outfile = os.path.join(self.solver_obj.options.outputdir, 'diagnostic_rpe.txt')
        # flush old file
        with open(self.outfile, 'w'):
            pass
        # compute area of 2D mesh
        one_2d = Constant(1.0, domain=self.solver_obj.mesh2d.coordinates.ufl_domain())
        self.area_2d = assemble(one_2d*dx)
        self.rho = self.solver_obj.fields.density_3d
        fs = self.rho.function_space()
        test = TestFunction(fs)
        self.nodal_volume = assemble(test*dx)
        self.depth = self.solver_obj.fields.bathymetry_2d.dat.data.mean()
        self.initial_rpe = None
        self.initial_mean_rho = None
        self._initialized = True

    def export(self):
        if not self._initialized:
            self._initialize()
        rho_array = self.rho.dat.data[:]
        sorted_ix = np.argsort(rho_array)[::-1]
        rho_array = rho_array[sorted_ix] + physical_constants['rho0'].dat.data[0]
        volume_array = self.nodal_volume.dat.data[:][sorted_ix]
        z = (np.cumsum(volume_array) - 0.5*volume_array)/self.area_2d
        g = physical_constants['g_grav'].dat.data[0]
        rpe = g*np.sum(rho_array*volume_array*z)
        if self.initial_rpe is None:
            self.initial_rpe = rpe
        rel_rpe = (rpe - self.initial_rpe)/np.abs(self.initial_rpe)
        t = self.solver_obj.simulation_time
        with open(self.outfile, 'a') as f:
            f.write('{:20.8f} {:28.8f} {:20.8e}\n'.format(t, rpe, rel_rpe))
