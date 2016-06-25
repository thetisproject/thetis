# Lock Exchange Test case
# =======================
#
# Solves hydrostatic flow in a closed rectangular channel.
#
# Dianeutral mixing depends on mesh Reynolds number [1]
# Re_h = U dx / nu
# U = 0.5 m/s characteristic velocity ~ 0.5*sqrt(g_h drho/rho_0)
# dx = horizontal mesh size
# nu = background viscosity
#
#
# Smagorinsky factor should be C_s = 1/sqrt(Re_h)
#
# Mesh resolutions:
# - ilicak [1]:  dx =  500 m,  20 layers
# COMODO lock exchange benchmark [2]:
# - coarse:      dx = 2000 m,  10 layers
# - coarse2 (*): dx = 1000 m,  20 layers
# - medium:      dx =  500 m,  40 layers
# - medium2 (*): dx =  250 m,  80 layers
# - fine:        dx =  125 m, 160 layers
# (*) not part of the original benchmark
#
# [1] Ilicak et al. (2012). Spurious dianeutral mixing and the role of
#     momentum closure. Ocean Modelling, 45-46(0):37-58.
#     http://dx.doi.org/10.1016/j.ocemod.2011.10.003
# [2] COMODO Lock Exchange test.
#     http://indi.imag.fr/wordpress/?page_id=446
# [3] Petersen et al. (2015). Evaluation of the arbitrary Lagrangian-Eulerian
#     vertical coordinate method in the MPAS-Ocean model. Ocean Modelling,
#     86:93-113.
#     http://dx.doi.org/10.1016/j.ocemod.2014.12.004
#
# Tuomas Karna 2015-03-03

from thetis import *

# TODO implement front location callback DONE
# TODO implement runtime plotting DONE
# TODO add option to use constant viscosity or smag scheme DONE
# TODO implement automatic dt estimation for v_adv
# TODO test effect/necessity of lax_friedrichs
# TODO test computing smag nu with weak form uv gradients
# TODO add option for changing time integrator?
# TODO also plot u and w at runtime

import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Plotter(object):
    """
    Plots density field at runtime
    """
    def __init__(self, solver_obj, imgdir):
        self.solver_obj = solver_obj
        self.imgdir = create_directory(imgdir)
        self.export_count = itertools.count()
        self._initialized = False

    def _initialize(self):
        # store density difference
        self.rho = self.solver_obj.fields.density_3d
        self.rho_lim = [self.rho.dat.data.min(), self.rho.dat.data.max()]

        # construct mesh points for plotting
        layers = self.solver_obj.mesh.topology.layers - 1
        mesh2d = self.solver_obj.mesh2d
        depth = self.solver_obj.fields.bathymetry_2d.dat.data.mean()
        self.xyz = mesh2d.coordinates
        x = self.xyz.dat.data[:, 0]
        x_max = x.max()
        x_min = x.min()
        delta_x = self.solver_obj.fields.h_elem_size_2d.dat.data.mean()*np.sqrt(2)
        n_x = np.round((x_max - x_min)/delta_x)
        npoints = layers*4
        z_max = -(depth - 1e-10)
        z = np.linspace(0, z_max, npoints)
        npoints = n_x + 1
        x = np.linspace(x_min, x_max, npoints)
        xx, zz = np.meshgrid(x, z)
        yy = np.zeros_like(xx)
        self.mesh_shape = xx.shape
        self.xyz = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T
        self.x_plot = x/1000.0  # to km
        self.z_plot = z

        self.cmap = plt.get_cmap('RdBu_r')
        self.cmap.set_over('#ffa500')
        self.cmap.set_under('#00e639')

    def plot_on_ax(self, ax, color_array, clim, titlestr, cbartitle, cmap):
        """Plots a field on given axis"""
        cmin, cmax = clim
        overshoot_tol = 1e-6
        # divide each integer in color array to this many levels
        cbar_reso = max(min(int(np.round(5.0/(cmax - cmin)*4)), 6), 1)
        levels = np.linspace(cmin - overshoot_tol, cmax + overshoot_tol,
                             cbar_reso*(cmax-cmin) + 1)
        p = ax.contourf(self.x_plot, self.z_plot, color_array, levels, cmap=cmap, extend='both')
        ax.set_xlabel('x [km]')
        ax.set_ylabel('z [m]')
        ax.set_title(titlestr)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=0.3, pad=0.1)
        cb = plt.colorbar(p, cax=cax, orientation='vertical', format='%4.1f')
        cb.set_label(cbartitle)
        cticks = np.linspace(cmin, cmax, (cmax-cmin) + 1)
        cb.set_ticks(cticks)
        cax.invert_yaxis()
        return p

    def export(self):
        if not self._initialized:
            self._initialize()

        # evaluate function on regular grid
        func = self.rho
        arr = np.array(func.at(tuple(self.xyz)))
        arr = arr.reshape(self.mesh_shape)

        nplots = 1
        fix, ax = plt.subplots(nrows=nplots, figsize=(12, 3.5*nplots))

        iexp = self.export_count.next()
        title = 'Time {:.2f} h'.format(self.solver_obj.simulation_time/3600.0)
        fname = 'plot_density_{0:06d}.png'.format(iexp)
        fname = os.path.join(self.imgdir, fname)
        varstr = 'Density'
        unit = 'kg m-3'
        clim = self.rho_lim
        clabel = '{:} [{:}]'.format(varstr, unit)
        self.plot_on_ax(ax, arr, clim, title, clabel, self.cmap)
        plt.savefig(fname, dpi=240, bbox_inches='tight')
        plt.close()


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


def run_lockexchange(reso_str='coarse', poly_order=1, element_family='dg-dg',
                     reynolds_number=1.0, use_limiter=True, dt=None,
                     viscosity='const'):
    """
    Runs lock exchange problem with a bunch of user defined options.
    """
    comm = COMM_WORLD

    print_output('Running lock exchange problem with options:')
    print_output('Resolution: {:}'.format(reso_str))
    print_output('Element family: {:}'.format(element_family))
    print_output('Polynomial order: {:}'.format(poly_order))
    print_output('Reynolds number: {:}'.format(reynolds_number))
    print_output('Use slope limiters: {:}'.format(use_limiter))
    print_output('Number of cores: {:}'.format(comm.size))

    refinement = {'huge': 0.6, 'coarse': 1, 'coarse2': 2, 'medium': 4,
                  'medium2': 8, 'fine': 16, 'ilicak': 4}
    # set mesh resolution
    depth = 20.0
    delta_x = 2000.0/refinement[reso_str]
    layers = int(round(10*refinement[reso_str]))
    if reso_str == 'ilicak':
        layers = 20
    delta_z = depth/layers
    print_output('Mesh resolution dx={:} nlayers={:} dz={:}'.format(delta_x, layers, delta_z))

    # generate unit mesh and transform its coords
    x_max = 32.0e3
    x_min = -32.0e3
    n_x = (x_max - x_min)/delta_x
    mesh2d = UnitSquareMesh(n_x, 2)
    coords = mesh2d.coordinates
    # x in [x_min, x_max], y in [-dx, dx]
    coords.dat.data[:, 0] = coords.dat.data[:, 0]*(x_max - x_min) + x_min
    coords.dat.data[:, 1] = coords.dat.data[:, 1]*2*delta_x - delta_x

    nnodes = mesh2d.topology.num_vertices()
    ntriangles = mesh2d.topology.num_cells()
    nprisms = ntriangles*layers
    print_output('Number of 2D nodes={:}, triangles={:}, prisms={:}'.format(nnodes, ntriangles, nprisms))

    lim_str = '_lim' if use_limiter else ''
    dt_str = '_dt{:}'.format(dt) if dt is not None else ''
    options_str = '_'.join([reso_str,
                            element_family,
                            'p{:}'.format(poly_order),
                            'visc-{:}'.format(viscosity),
                            'Re{:}'.format(reynolds_number),
                            ]) + lim_str + dt_str
    outputdir = 'outputs_' + options_str
    print_output('Exporting to {:}'.format(outputdir))

    # temperature and salinity, for linear eq. of state (from Petersen, 2015)
    temp_left = 5.0
    temp_right = 30.0
    salt_const = 35.0
    rho_0 = 1000.0
    physical_constants['rho0'].assign(rho_0)

    # compute horizontal viscosity
    uscale = 0.5
    nu_scale = uscale * delta_x / reynolds_number
    print_output('Horizontal viscosity: {:}'.format(nu_scale))

    dt_adv = 1.0/20.0*delta_x/np.sqrt(2)/1.0
    dt_visc = 1.0/120.0*(delta_x/np.sqrt(2))**2/nu_scale
    print_output('Max dt for advection: {:}'.format(dt_adv))
    print_output('Max dt for viscosity: {:}'.format(dt_visc))

    t_end = 25 * 3600
    t_export = 15*60.0
    if dt is None:
        # take smallest stable dt that fits the export intervals
        max_dt = min(dt_adv, dt_visc)
        ntime = int(np.ceil(t_export/max_dt))
        dt = t_export/ntime

    # bathymetry
    p1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(p1_2d, name='Bathymetry')
    bathymetry_2d.assign(depth)

    # create solver
    solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, layers)
    options = solver_obj.options
    options.order = poly_order
    options.element_family = element_family
    options.solve_salt = False
    options.constant_salt = Constant(salt_const)
    options.solve_temp = True
    options.solve_vert_diffusion = False
    options.use_bottom_friction = False
    options.use_ale_moving_mesh = False
    # options.use_imex = True
    # options.use_semi_implicit_2d = False
    # options.use_mode_split = False
    options.baroclinic = True
    options.uv_lax_friedrichs = Constant(1.0)
    options.tracer_lax_friedrichs = Constant(1.0)
    options.salt_jump_diff_factor = None  # Constant(1.0)
    options.salt_range = Constant(5.0)
    options.use_limiter_for_tracers = use_limiter
    # To keep const grid Re_h, viscosity scales with grid: nu = U dx / Re_h
    if viscosity == 'smag':
        options.smagorinsky_factor = Constant(1.0/np.sqrt(reynolds_number))
    elif viscosity == 'const':
        options.h_viscosity = Constant(nu_scale)
    else:
        raise Exception('Unknow viscosity type {:}'.format(viscosity))
    options.v_viscosity = Constant(1e-4)
    options.h_diffusivity = None
    options.dt = dt
    # if options.use_mode_split:
    #     options.dt = dt
    options.t_export = t_export
    options.t_end = t_end
    options.outputdir = outputdir
    options.u_advection = Constant(1.0)
    options.check_vol_conservation_2d = True
    options.check_vol_conservation_3d = True
    options.check_temp_conservation = True
    options.check_temp_overshoot = True
    options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                                'w_3d', 'w_mesh_3d', 'temp_3d', 'density_3d',
                                'uv_dav_2d', 'uv_dav_3d', 'baroc_head_3d',
                                'baroc_head_2d', 'smag_visc_3d']
    options.fields_to_export_hdf5 = list(options.fields_to_export)
    options.equation_of_state = 'linear'
    options.lin_equation_of_state_params = {
        'rho_ref': rho_0,
        's_ref': 35.0,
        'th_ref': 5.0,
        'alpha': 0.2,
        'beta': 0.0,
    }

    # Use direct solver for 2D
    # options.solver_parameters_sw = {
    #     'ksp_type': 'preonly',
    #     'pc_type': 'lu',
    #     'pc_factor_mat_solver_package': 'mumps',
    #     'snes_monitor': False,
    #     'snes_type': 'newtonls',
    # }

    if comm.size == 1:
        rpe_calc = RPECalculator(solver_obj)
        rpe_callback = rpe_calc.export
        front_calc = FrontLocationCalculator(solver_obj)
        front_callback = front_calc.export
        plotter = Plotter(solver_obj, imgdir=solver_obj.options.outputdir + '/plots')
        plot_callback = plotter.export
    else:
        rpe_callback = None

    def callback():
        if comm.size == 1:
            rpe_callback()
            front_callback()
            plot_callback()

    solver_obj.create_equations()
    esize = solver_obj.fields.h_elem_size_2d
    min_elem_size = comm.allreduce(np.min(esize.dat.data), op=MPI.MIN)
    max_elem_size = comm.allreduce(np.max(esize.dat.data), op=MPI.MAX)
    print_output('Elem size: {:} {:}'.format(min_elem_size, max_elem_size))

    temp_init3d = Function(solver_obj.function_spaces.H, name='initial temperature')
    # vertical barrier
    # temp_init3d.interpolate(Expression('(x[0] > 0.0) ? v_r : v_l',
    #                                    v_l=temp_left, v_r=temp_right))
    # smooth condition
    temp_init3d.interpolate(Expression('v_l - (v_l - v_r)*0.5*(tanh(x[0]/sigma) + 1.0)',
                                       sigma=10.0, v_l=temp_left, v_r=temp_right))

    solver_obj.assign_initial_conditions(temp=temp_init3d)
    solver_obj.iterate(export_func=callback)


def get_argparser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reso_str', type=str,
                        help='mesh resolution string',
                        default='coarse',
                        choices=['huge', 'coarse', 'coarse2', 'medium', 'medium2', 'fine', 'ilicak'])
    parser.add_argument('--no-limiter', action='store_false', dest='use_limiter',
                        help='do not use slope limiter for tracers')
    parser.add_argument('-p', '--poly_order', type=int, default=1,
                        help='order of finite element space')
    parser.add_argument('-f', '--element-family', type=str,
                        help='finite element family', default='dg-dg')
    parser.add_argument('-re', '--reynolds-number', type=float, default=1.0,
                        help='mesh Reynolds number for Smagorinsky scheme')
    parser.add_argument('-dt', '--dt', type=float,
                        help='force value for 3D time step')
    parser.add_argument('-visc', '--viscosity', type=str,
                        help='Type of horizontal viscosity',
                        default='const',
                        choices=['const', 'smag'])
    return parser


def parse_options():
    parser = get_argparser()
    args = parser.parse_args()
    args_dict = vars(args)
    run_lockexchange(**args_dict)

if __name__ == '__main__':
    parse_options()
