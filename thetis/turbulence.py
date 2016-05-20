"""
Generic Length Scale Turbulence Closure model [1].

This model solves two dynamic equations, for turbulent kinetic energy (tke, k)
and additional variable psi.

dk/dt + \nabla_h(uv*k) + d(w*k)/dz = d/dz(\nu_h/\sigma_k dk/dz) + P + B - eps
dpsi/dt + \nabla_h(uv*psi) + d(w*psi)/dz = d/dz(\nu_h/\sigma_psi dpsi/dz) +
   psi/k*(c1*P + c3*B - c2*eps*f_wall)

P = viscosity M**2             (production)
B = - diffusivity N**2         (byoyancy production)
M**2 = (du/dz)**2 + (dv/dz)**2 (shear frequency)
N**2 = -g\rho_0 (drho/dz)      (buoyancy frequency)

The additional variable is defined as
psi = (cmu0)**p * k**m * l**n
where p, m, n parameters and cmu0 is an empirical constant.

dpsi/dt + \nabla_h(uv*psi) + d(w*psi)dz = d/dz(\nu_h/\sigma_psi dpsi/dz) +
   psi/k*(c1*P + c3*B - c2*eps*f_wall)


Parameter c3 takes value c3_minus in stably stratified flows and c3_plus in
unstably stratified cases.

Turbulent length scale is obtained diagnostically as
l = (cmu0)**3 * k**(3/2) * eps**(-1)

TKE dissipation rate is given by
eps = (cmu0)**(3+p/n)*tke**(3/2+m/n)*psi**(-1/n)

Finally vertical eddy viscosity and diffusivity are obtained as

viscosity = c*sqrt(2*k)*l*stability_func_m
diffusivity = c*sqrt(2*k)*l*stability_func_rho

Stability functions are defined as obtained from [2] or [3].
Implementation follows [4].

[1] Umlauf, L. and Burchard, H. (2003). A generic length-scale equation for
    geophysical turbulence models. Journal of Marine Research, 61:235--265(31).
    http://dx.doi.org/10.1357/002224003322005087

[2] Kantha, L. H. and Clayson, C. A. (1994). An improved mixed layer model for
    geophysical applications. Journal of Geophysical Research: Oceans,
    99(C12):25235--25266.
    http://dx.doi.org/10.1029/94JC02257

[3] Canuto et al. (2001). Ocean Turbulence. Part I: One-Point Closure Model -
    Momentum and Heat Vertical Diffusivities. Journal of Physical Oceanography,
    31(6):1413-1426.
    http://dx.doi.org/10.1175/1520-0485(2001)031

[4] Warner et al. (2005). Performance of four turbulence closure models
    implemented using a generic length scale method. Ocean Modelling,
    8(1-2):81--113.
    http://dx.doi.org/10.1016/j.ocemod.2003.12.003

[5] Umlauf, L. and Burchard, H. (2005). Second-order turbulence closure models
    for geophysical boundary layers. A review of recent work. Continental Shelf
    Research, 25(7-8):795--827.
    http://dx.doi.org/10.1016/j.csr.2004.08.004

Tuomas Karna 2015-09-07
"""
from tracer_eq import *
from utility import *
from stability_functions import *


class GLSModelOptions(AttrDict):
    """
    Options for Generic Lenght Scale turbulence model
    """
    def __init__(self):
        """
        Initialize with default options
        """
        super(GLSModelOptions, self).__init__()
        self.closure_name = 'k-epsilon'
        """
        str: name of common closures

        'k-epsilon': k-epsilon setup
        'k-omega': k-epsilon setup
        'gls-a': Generic Length Scale setup A

        Sets default values for parameters p, m, n, schmidt_nb_tke, schmidt_nb_psi, c1, c2, c3_plus, c3_minus,
        f_wall, k_min, psi_min
        """
        self.stability_name = 'CA'
        """str: name of used stability function family

        'CA': Canuto A
        'CB': Canuto B
        'KC': Kantha-Clayson
        'CH': Cheng
        """
        self.p = 3.0
        """float: parameter p for the definition of psi"""
        self.m = 1.5
        """float: parameter m for the definition of psi"""
        self.n = -1.0
        """float: parameter n for the definition of psi"""
        self.schmidt_nb_tke = 1.0
        """float: turbulent kinetic energy Schmidt number"""
        self.schmidt_nb_psi = 1.3
        """float: psi Schmidt number"""
        self.cmu0 = 0.5477
        """float: cmu0 parameter"""
        self.compute_cmu0 = True
        """bool: compute cmu0 from stability function parameters"""
        self.c1 = 1.44
        """float: c1 parameter for Psi equations"""
        self.c2 = 1.92
        """float: c2 parameter for Psi equations"""
        self.c3_minus = -0.52
        """float: c3 parameter for Psi equations, stable stratification"""
        self.c3_plus = 1.0
        """float: c3 parameter for Psi equations, unstable stratification"""
        self.compute_c3_minus = True
        """bool: compute c3_minus from ri_st

        ri_st is the steady state gradient Richardson number"""
        self.f_wall = 1.0
        """float: wall function parameter"""
        self.ri_st = 0.25
        self.kappa = physical_constants['von_karman']
        """float: steady state gradient Richardson number

        Used to compute c3_minus"""
        self.compute_kappa = True
        """bool: compute von Karman constant from psi Schmidt number"""
        self.k_min = 3.7e-8
        """float: minimum value for turbulent kinetic energy"""
        self.psi_min = 1.0e-10
        """float: minimum value for psi"""
        self.eps_min = 1.0e-10
        """float: minimum value for epsilon"""
        self.len_min = 1.0e-10
        """float: minimum value for turbulent lenght scale"""
        self.compute_len_min = True
        """bool: compute min_len from k_min and psi_min"""
        self.compute_psi_min = True
        """bool: compute psi_len from k_min and eps_min"""
        self.visc_min = 1.0e-8
        """float: minimum value for eddy viscosity"""
        self.diff_min = 1.0e-8
        """float: minimum value for eddy diffusivity"""
        self.galperin_lim = 0.56
        """float: Galperin lenght scale limitation parameter"""

        self.limit_len = False
        """bool: apply Galperin lenght scale limit"""
        self.limit_psi = False
        """bool: apply Galperin lenght scale limit on psi"""
        self.limit_eps = False
        """bool: apply Galperin lenght scale limit on epsilon"""
        self.limit_len_min = True
        """bool: limit minimum turbulent length scale to len_min"""

    def apply_defaults(self, closure_name):
        """Applies default parameters for given closure name."""

        # standard values for different closures
        # from [3] tables 1 and 2
        kepsilon = {'p': 3,
                    'm': 1.5,
                    'n': -1.0,
                    'cmu0': 0.5477,
                    'schmidt_nb_tke': 1.0,
                    'schmidt_nb_psi': 1.3,
                    'c1': 1.44,
                    'c2': 1.92,
                    'c3_plus': 1.0,
                    'c3_minus': -0.52,
                    'f_wall': 1.0,
                    'k_min': 3.7e-8,
                    'psi_min': 1.0e-10,
                    'closure_name': 'k-epsilon',
                    }
        komega = {'p': -1.0,
                  'm': 0.5,
                  'n': -1.0,
                  'cmu0': 0.5477,
                  'schmidt_nb_tke': 2.0,
                  'schmidt_nb_psi': 2.0,
                  'c1': 0.555,
                  'c2': 0.833,
                  'c3_plus': 1.0,
                  'c3_minus': -0.52,
                  'f_wall': 1.0,
                  'k_min': 3.7e-8,
                  'eps_min': 1.0e-10,
                  'psi_min': 1.0e-10,
                  'closure_name': 'k-omega',
                  }
        gen = {'p': 2.0,
               'm': 1.0,
               'n': -0.67,
               'cmu0': 0.5477,
               'schmidt_nb_tke': 0.8,
               'schmidt_nb_psi': 1.07,
               'c1': 1.0,
               'c2': 1.22,
               'c3_plus': 1.0,
               'c3_minus': 0.05,
               'f_wall': 1.0,
               'k_min': 3.7e-8,
               'eps_min': 1.0e-10,
               'psi_min': 2.0e-7,
               'closure_name': 'gen',
               }
        if closure_name == 'k-epsilon':
            self.__dict__.update(kepsilon)
        elif closure_name == 'k-omega':
            self.__dict__.update(komega)
        elif closure_name == 'gen':
            self.__dict__.update(gen)

    def print_summary(self):
        """Prints all defined parameters and their values."""
        print_info('GLS Turbulence model parameters')
        for k in sorted(self.keys()):
            print_info('  {:16s} : {:}'.format(k, self[k]))


def set_func_min_val(f, minval):
    """
    Sets a minimum value to a function
    """
    f.dat.data[f.dat.data < minval] = minval


def set_func_max_val(f, maxval):
    """
    Sets a minimum value to a function
    """
    f.dat.data[f.dat.data > maxval] = maxval


class P1Average(object):
    """
    Takes a DP field and computes nodal averages and stores in P1 field.

    Source must be either a p0 or p1dg function.
    The averaging operation is both mass conservative and positivity preserving.
    """
    def __init__(self, p0, p1, p1dg):
        self.p0 = p0
        self.p1 = p1
        self.p1dg = p1dg
        self.vol_p1 = Function(self.p1, name='nodal volume p1')
        self.vol_p1dg = Function(self.p1dg, name='nodal volume p1dg')
        self.update_volumes()

    def update_volumes(self):
        assemble(TestFunction(self.p1)*dx, self.vol_p1)
        assemble(TestFunction(self.p1dg)*dx, self.vol_p1dg)

    def apply(self, source, solution):
        assert solution.function_space() == self.p1
        assert source.function_space() == self.p0 or source.function_space() == self.p1dg
        source_is_p0 = source.function_space() == self.p0

        source_str = 'source[0][c]' if source_is_p0 else 'source[d][c]'
        solution.assign(0.0)
        fs_source = source.function_space()
        self.kernel = op2.Kernel("""
            void my_kernel(double **p1_average, double **source, double **vol_p1, double **vol_p1dg) {
                for ( int d = 0; d < %(nodes)d; d++ ) {
                    for ( int c = 0; c < %(func_dim)d; c++ ) {
                        p1_average[d][c] += %(source_str)s * vol_p1dg[d][c] / vol_p1[d][c];
                    }
                }
            }""" % {'nodes': solution.cell_node_map().arity,
                    'func_dim': solution.function_space().dim,
                    'source_str': source_str},
            'my_kernel')

        op2.par_loop(
            self.kernel, self.p1.mesh().cell_set,
            solution.dat(op2.WRITE, self.p1.cell_node_map()),
            source.dat(op2.READ, fs_source.cell_node_map()),
            self.vol_p1.dat(op2.READ, self.p1.cell_node_map()),
            self.vol_p1dg.dat(op2.READ, self.p1dg.cell_node_map()),
            iterate=op2.ALL)


class VerticalGradSolver(object):
    """
    Computes vertical gradient in the weak sense.

    :arg source: A :class:`Function` or expression to differentiate.
    :arg source: A :class:`Function` where the solution will be stored.
        Must be in P0 space.
    """
    def __init__(self, source, solution, solver_parameters=None):
        if solver_parameters is None:
            solver_parameters = {}
        solver_parameters.setdefault('snes_type', 'ksponly')
        solver_parameters.setdefault('ksp_type', 'preonly')
        solver_parameters.setdefault('pc_type', 'bjacobi')
        solver_parameters.setdefault('sub_ksp_type', 'preonly')
        solver_parameters.setdefault('sub_pc_type', 'ilu')

        self.source = source
        self.solution = solution

        self.fs = self.solution.function_space()
        self.mesh = self.fs.mesh()

        # weak gradient evaluator
        test = TestFunction(self.fs)
        tri = TrialFunction(self.fs)
        normal = FacetNormal(self.mesh)
        a = inner(test, tri)*dx
        p = self.source
        l = -inner(p, Dx(test, 2))*dx
        l += avg(p)*jump(test, normal[2])*dS_h
        l += p*test*normal[2]*(ds_t + ds_b)
        prob = LinearVariationalProblem(a, l, self.solution, constant_jacobian=True)
        self.weak_grad_solver = LinearVariationalSolver(prob, solver_parameters=solver_parameters)

    def solve(self):
        self.weak_grad_solver.solve()


class SmoothVerticalGradSolver(object):
    """
    Computes vertical gradient in a smooth(er) way.

    :arg source: A :class:`Function` or expression to differentiate.
    :arg source: A :class:`Function` where the solution will be stored.
    """
    def __init__(self, source, solution):
        self.source = source
        self.solution = solution

        self.fs = self.solution.function_space()
        self.mesh = self.fs.mesh()

        p0 = FunctionSpace(self.mesh, 'DP', 0, vfamily='DP', vdegree=0)
        assert self.solution.function_space() == p0, 'solution must be in p0'

        self.source_p0 = Function(p0, name='p0 source')
        self.gradient_p0 = Function(p0, name='p0 gradient')

        self.p0_interpolator = Interpolator(self.source, self.source_p0)
        self.p0_grad_solver = VerticalGradSolver(self.source_p0, self.solution)
        self.grad_solver = VerticalGradSolver(self.source, self.gradient_p0)

        self.p0_copy_kernel = op2.Kernel("""
            void my_kernel(double **gradient, double **source) {
                gradient[0][0] = source[0][0];
            }""", 'my_kernel')

    def solve(self):
        # interpolate p1dg to prism centers
        self.p0_interpolator.interpolate()
        # compute weak gradine from source_p0
        self.p0_grad_solver.solve()
        # compute weak gradient directly
        self.grad_solver.solve()
        # compute mean of the two
        self.solution.assign(0.5*(self.solution + self.gradient_p0))
        # replace top/bottom values with weak gradient
        # FIXME how to combine ON_TOP + ON_BOTTOM in single call?
        op2.par_loop(self.p0_copy_kernel, self.mesh.cell_set,
                     self.solution.dat(op2.WRITE, self.fs.cell_node_map()),
                     self.gradient_p0.dat(op2.READ, self.fs.cell_node_map()),
                     iterate=op2.ON_TOP)
        op2.par_loop(self.p0_copy_kernel, self.mesh.cell_set,
                     self.solution.dat(op2.WRITE, self.fs.cell_node_map()),
                     self.gradient_p0.dat(op2.READ, self.fs.cell_node_map()),
                     iterate=op2.ON_BOTTOM)


class ShearFrequencySolver(object):
    """
    Computes vertical shear frequency squared form the given horizontal
    velocity field.

    M^2 = du/dz^2 + dv/dz^2
    """
    def __init__(self, uv, m2, mu, mv, mu_tmp, minval=1e-12):

        self.mu = mu
        self.mv = mv
        self.m2 = m2
        self.mu_tmp = mu_tmp
        self.minval = minval
        # relaxation coefficient between old and new mu or mv
        self.relaxation = 1.0

        self.var_solvers = {}
        for i_comp in range(2):
            solver = VerticalGradSolver(uv[i_comp], mu_tmp)
            self.var_solvers[i_comp] = solver

    def solve(self, init_solve=False):
        with timed_stage('shear_freq_solv'):
            mu_comp = [self.mu, self.mv]
            self.m2.assign(0.0)
            for i_comp in range(2):
                self.var_solvers[i_comp].solve()
                gamma = self.relaxation if not init_solve else 1.0
                mu_comp[i_comp].assign(gamma*self.mu_tmp +
                                       (1.0 - gamma)*mu_comp[i_comp])
                self.m2 += mu_comp[i_comp]*mu_comp[i_comp]
            # crop small/negative values
            set_func_min_val(self.m2, self.minval)


class BuoyFrequencySolver(object):
    """
    Computes buoyancy frequency squared form the given horizontal
    velocity field.

    N^2 = -g/rho0 drho/dz
    """
    def __init__(self, rho, n2, n2_tmp, minval=1e-12):
        self._no_op = False
        if rho is None:
            self._no_op = True

        if not self._no_op:

            self.n2 = n2
            self.n2_tmp = n2_tmp
            # relaxation coefficient between old and new mu or mv
            self.relaxation = 1.0

            self.var_solvers = {}

            g = physical_constants['g_grav']
            rho0 = physical_constants['rho0']
            p = -g/rho0 * rho
            solver = VerticalGradSolver(p, self.n2_tmp)
            self.var_solver = solver

    def solve(self, init_solve=False):
        with timed_stage('buoy_freq_solv'):
            if not self._no_op:
                self.var_solver.solve()
                gamma = self.relaxation if not init_solve else 1.0
                self.n2.assign(gamma*self.n2_tmp +
                               (1.0 - gamma)*self.n2)


class GenericLengthScaleModel(object):
    """
    Generic Length Scale turbulence closure model implementation
    """
    def __init__(self, solver, k_field, psi_field, uv_field, rho_field,
                 l_field, epsilon_field,
                 eddy_diffusivity, eddy_viscosity,
                 n2, m2, options=None):
        """
        Initialize GLS model

        Parameters
        ----------

        k_field : Function
            turbulent kinetic energy (TKE) field
        psi_field : Function
            field for the accompanying GLS variable psi
        uv_field : Function
            horizontal velocity field
        rho_field : Function
            water density field
        epsilon_field : Function
            TKE dissipation rate field
        l_field : Function
            turbulence length scale field
        eddy_viscosity, eddy_diffusivity : Function
            eddy viscosity/diffusivity fields
        n2, m2 : Function
            buoyancy and vertical shear frequency squared
        """
        self.solver = solver
        # 3d model fields
        self.uv = uv_field
        self.rho = rho_field
        # prognostic fields
        self.k = k_field
        self.psi = psi_field
        # diagnostic fields
        # NOTE redundant for k-epsilon model where psi==epsilon
        self.epsilon = epsilon_field
        self.l = l_field
        self.viscosity = eddy_viscosity
        self.diffusivity = eddy_diffusivity
        self.n2 = n2
        self.m2 = m2
        self.mu_tmp = Function(self.m2.function_space(),
                               name='tmp Shear frequency')
        self.mu = Function(self.m2.function_space(), name='Shear frequency X')
        self.mv = Function(self.m2.function_space(), name='Shear frequency Y')
        self.n2_tmp = Function(self.n2.function_space(),
                               name='tmp buoyancy frequency')
        self.n2_pos = Function(self.n2.function_space(),
                               name='positive buoyancy frequency')
        self.n2_neg = Function(self.n2.function_space(),
                               name='negative buoyancy frequency')

        if self.solver.options.use_smooth_eddy_viscosity:
            self.viscosity_native = Function(self.n2.function_space(),
                                             name='GLS viscosity')
            self.diffusivity_native = Function(self.n2.function_space(),
                                               name='GLS diffusivity')
            self.p1_averager = P1Average(solver.function_spaces.P0,
                                         solver.function_spaces.P1,
                                         solver.function_spaces.P1DG)
        else:
            self.viscosity_native = self.viscosity
            self.diffusivity_native = self.diffusivity

        # parameter to mix old and new viscosity values (1 => new only)
        self.relaxation = 1.0

        self.options = GLSModelOptions()
        if options is not None:
            self.options.update(options)

        o = self.options
        stability_name = o.stability_name
        stab_args = {'lim_alpha_shear': True,
                     'lim_alpha_buoy': True,
                     'smooth_alpha_buoy_lim': False}
        if stability_name == 'KC':
            self.stability_func = StabilityFunctionKanthaClayson(**stab_args)
        elif stability_name == 'CA':
            self.stability_func = StabilityFunctionCanutoA(**stab_args)
        elif stability_name == 'CB':
            self.stability_func = StabilityFunctionCanutoB(**stab_args)
        elif stability_name == 'CH':
            self.stability_func = StabilityFunctionCheng(**stab_args)
        else:
            raise Exception('Unknown stability function type: ' +
                            stability_name)

        if o.compute_cmu0:
            o.cmu0 = self.stability_func.compute_cmu0()
        if o.compute_kappa:
            kappa = self.stability_func.compute_kappa(o.schmidt_nb_psi, o.n, o.c1, o.c2)
            o.kappa = kappa
            # update mean flow model value as well
            physical_constants['von_karman'].assign(kappa)
        if o.compute_c3_minus:
            o.c3_minus = self.stability_func.compute_c3_minus(o.c1, o.c2, o.ri_st)
        if o.compute_psi_min:
            o.psi_min = (o.cmu0**(3.0 + o.p/o.n)*o.k_min**(3.0/2.0 + o.m/o.n)*o.eps_min**(-1.0))**o.n
        else:
            o.eps_min = o.cmu0**(3.0 + o.p/o.n)*o.k_min**(3.0/2.0 + o.m/o.n)*o.psi_min**(-1.0/o.n)
        # minimum length scale
        if o.compute_len_min:
            o.len_min = o.cmu0**3 * o.k_min**1.5 / o.eps_min

        self.shear_frequency_solver = ShearFrequencySolver(self.uv, self.m2,
                                                           self.mu, self.mv,
                                                           self.mu_tmp)
        if self.rho is not None:
            self.buoy_frequency_solver = BuoyFrequencySolver(self.rho, self.n2,
                                                             self.n2_tmp)

        self.initialize()
        self.options.print_summary()

    def initialize(self):
        """Initializes fields"""
        self.n2.assign(1e-12)
        self.n2_pos.assign(1e-12)
        self.n2_neg.assign(0.0)
        self.preprocess(init_solve=True)
        self.postprocess()

    def preprocess(self, init_solve=False):
        """
        To be called before evaluating the equations.

        Update all fields that depend on velocity and density.
        """
        # update m2 and N2

        self.shear_frequency_solver.solve(init_solve=init_solve)

        if self.rho is not None:
            self.buoy_frequency_solver.solve(init_solve=init_solve)
            # split to positive and negative parts
            self.n2_pos.assign(1e-12)
            self.n2_neg.assign(0.0)
            pos_ix = self.n2.dat.data[:] >= 0.0
            self.n2_pos.dat.data[pos_ix] = self.n2.dat.data[pos_ix]
            self.n2_neg.dat.data[~pos_ix] = self.n2.dat.data[~pos_ix]

    def postprocess(self):
        """
        To be called after evaluating the equations.

        Update all fields that depend on turbulence fields.
        """
        with timed_stage('turb_postproc'):
            o = self.options
            cmu0 = o.cmu0
            p = o.p
            n = o.n
            m = o.m

            # limit k
            set_func_min_val(self.k, o.k_min)

            k_arr = self.k.dat.data[:]
            n2_pos = self.n2_pos.dat.data[:]
            n2_pos_eps = 1e-12
            galp = o.galperin_lim
            if o.limit_psi:
                # impose Galperin limit on psi
                # psi^(1/n) <= sqrt(0.56)* (cmu0)^(p/n) *k^(m/n+0.5)* n2^(-0.5)
                val = (np.sqrt(galp) * (cmu0)**(p / n) * k_arr**(m / n + 0.5) * (n2_pos + n2_pos_eps)**(-0.5))**n
                if n > 0:
                    # impose max value
                    np.minimum(self.psi.dat.data, val, self.psi.dat.data)
                else:
                    # impose min value
                    np.maximum(self.psi.dat.data, val, self.psi.dat.data)
            set_func_min_val(self.psi, o.psi_min)

            # udpate epsilon
            self.epsilon.assign(cmu0**(3.0 + p/n)*self.k**(3.0/2.0 + m/n)*self.psi**(-1.0/n))
            if o.limit_eps:
                # impose Galperin limit on eps
                eps_min = cmu0**3.0/np.sqrt(galp)*np.sqrt(n2_pos)*k_arr
                np.maximum(self.epsilon.dat.data, eps_min, self.epsilon.dat.data)
            # impose minimum value
            # FIXME this should not be need because psi is limited
            set_func_min_val(self.epsilon, o.eps_min)

            # update L
            self.l.assign(cmu0**3.0 * self.k**(3.0/2.0) / self.epsilon)
            if o.limit_len_min:
                set_func_min_val(self.l, o.len_min)
            if o.limit_len:
                # Galperin length scale limitation
                len_max = np.sqrt(galp*k_arr/(n2_pos + n2_pos_eps))
                np.minimum(self.l.dat.data, len_max, self.l.dat.data)
            if self.l.dat.data.max() > 10.0:
                print ' * large L: {:f}'.format(self.l.dat.data.max())

            # update stability functions
            s_m, s_h = self.stability_func.evaluate(self.m2.dat.data,
                                                    self.n2.dat.data,
                                                    self.k.dat.data,
                                                    self.epsilon.dat.data)
            # update diffusivity/viscosity
            b = np.sqrt(self.k.dat.data[:])*self.l.dat.data[:]
            lam = self.relaxation
            new_visc = b*s_m/cmu0**3
            new_diff = b*s_h/cmu0**3
            self.viscosity_native.dat.data[:] = lam*new_visc + (1.0 - lam)*self.viscosity_native.dat.data[:]
            self.diffusivity_native.dat.data[:] = lam*new_diff + (1.0 - lam)*self.diffusivity_native.dat.data[:]

            if self.solver.options.use_smooth_eddy_viscosity:
                self.p1_averager.apply(self.viscosity_native, self.viscosity)
                self.p1_averager.apply(self.diffusivity_native, self.diffusivity)
            set_func_min_val(self.viscosity, o.visc_min)
            set_func_min_val(self.diffusivity, o.diff_min)

            # print '{:8s} {:10.3e} {:10.3e}'.format('k', self.k.dat.data.min(), self.k.dat.data.max())
            # print '{:8s} {:10.3e} {:10.3e}'.format('eps', self.epsilon.dat.data.min(), self.epsilon.dat.data.max())
            # print '{:8s} {:10.3e} {:10.3e}'.format('L', self.l.dat.data.min(), self.l.dat.data.max())
            # print '{:8s} {:10.3e} {:10.3e}'.format('M2', self.m2.dat.data.min(), self.m2.dat.data.max())
            # print '{:8s} {:10.3e} {:10.3e}'.format('N2', self.n2.dat.data.min(), self.n2.dat.data.max())
            # print '{:8s} {:10.3e} {:10.3e}'.format('N2+', self.n2_pos.dat.data.min(), self.n2_pos.dat.data.max())
            # print '{:8s} {:10.3e} {:10.3e}'.format('N2-', self.n2_neg.dat.data.min(), self.n2_neg.dat.data.max())
            # print '{:8s} {:10.3e} {:10.3e}'.format('s_h', s_h.min(), s_h.max())
            # print '{:8s} {:10.3e} {:10.3e}'.format('s_m', s_m.min(), s_m.max())
            # print '{:8s} {:10.3e} {:10.3e}'.format('nuv', self.viscosity.dat.data.min(), self.viscosity.dat.data.max())
            # print '{:8s} {:10.3e} {:10.3e}'.format('muv', self.diffusivity.dat.data.min(), self.diffusivity.dat.data.max())


class TKESourceTerm(TracerTerm):
    """
    Production and destruction terms of the TKE equation
    """
    def __init__(self, function_space, gls_model,
                 bathymetry=None, v_elem_size=None, h_elem_size=None):
        super(TKESourceTerm, self).__init__(function_space,
                                            bathymetry, v_elem_size, h_elem_size)
        self.gls_model = gls_model

    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        # TKE: P + B - eps
        # P = viscosity M**2           (production)
        # B = - diffusivity N**2       (byoyancy production)
        # M**2 = (du/dz)**2 + (dv/dz)**2 (shear frequency)
        # N**2 = -g\rho_0 (drho/dz)      (buoyancy frequency)
        # eps = (cmu0)**(3+p/n)*tke**(3/2+m/n)*psi**(-1/n)
        #                                (tke dissipation rate)
        eddy_viscosity = fields_old['viscosity_v']
        eddy_diffusivity = fields_old['diffusivity_v']
        epsilon = fields_old['epsilon']
        shear_freq2 = fields_old['shear_freq2']
        buoy_freq2_neg = fields_old['buoy_freq2_neg']
        buoy_freq2_pos = fields_old['buoy_freq2_pos']
        p = eddy_viscosity * shear_freq2
        b_source = - eddy_diffusivity * buoy_freq2_neg
        b_sink = - eddy_diffusivity * buoy_freq2_pos

        source = p + b_source + (b_sink - epsilon)/solution_old*solution  # patankar
        f = inner(source, self.test)*dx
        return f


class PsiSourceTerm(TracerTerm):
    """
    Production and destruction terms of the TKE equation
    """
    def __init__(self, function_space, gls_model,
                 bathymetry=None, v_elem_size=None, h_elem_size=None):
        super(PsiSourceTerm, self).__init__(function_space,
                                            bathymetry, v_elem_size, h_elem_size)
        self.gls_model = gls_model

    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        # psi: psi/k*(c1*P + c3*B - c2*eps*f_wall)
        # P = viscosity M**2           (production)
        # B = - diffusivity N**2       (byoyancy production)
        # M**2 = (du/dz)**2 + (dv/dz)**2 (shear frequency)
        # N**2 = -g\rho_0 (drho/dz)      (buoyancy frequency)
        # eps = (cmu0)**(3+p/n)*tke**(3/2+m/n)*psi**(-1/n)
        #                                (tke dissipation rate)
        eddy_viscosity = fields_old['viscosity_v']
        eddy_diffusivity = fields_old['diffusivity_v']
        diffusivity_v = eddy_viscosity/self.gls_model.options.schmidt_nb_psi
        k = fields_old['k']
        epsilon = fields_old['epsilon']
        shear_freq2 = fields_old['shear_freq2']
        buoy_freq2_neg = fields_old['buoy_freq2_neg']
        buoy_freq2_pos = fields_old['buoy_freq2_pos']
        p = eddy_viscosity * shear_freq2
        c1 = self.gls_model.options.c1
        c2 = self.gls_model.options.c2
        # c3 switch: c3 = c3_minus if n2 > 0 else c3_plus
        c3_minus = self.gls_model.options.c3_minus
        c3_plus = self.gls_model.options.c3_plus  # > 0
        assert c3_plus >= 0, 'c3_plus has unexpected sign'
        b_shear = c3_plus * -eddy_diffusivity * buoy_freq2_neg
        b_buoy = c3_minus * -eddy_diffusivity * buoy_freq2_pos
        if c3_minus > 0:
            b_source = b_shear
            b_sink = b_buoy
        else:
            b_source = b_shear + b_buoy
            b_sink = 0
        f_wall = self.gls_model.options.f_wall
        source = solution_old/k*(c1*p + b_source) + solution/k*(b_sink - c2*f_wall*epsilon)  # patankar
        f = inner(source, self.test)*dx

        # add bottom/top boundary condition for psi
        # (nuv_v/sigma_psi * dpsi/dz)_b = n * nuv_v/sigma_psi * (cmu0)^p * k^m * kappa^n * z_b^(n-1)
        # z_b = distance_from_bottom + z_0 (Burchard and Petersen, 1999)
        cmu0 = self.gls_model.options.cmu0
        p = self.gls_model.options.p
        m = self.gls_model.options.m
        n = self.gls_model.options.n
        z0_friction = physical_constants['z0_friction']
        kappa = physical_constants['von_karman']
        if self.v_elem_size is None:
            raise Exception('v_elem_size required')
        # bottom condition
        z_b = 0.5*self.v_elem_size + z0_friction
        diff_flux = (n*diffusivity_v*(cmu0)**p *
                     k**m * kappa**n * z_b**(n - 1.0))
        f += diff_flux*self.test*self.normal[2]*ds_bottom
        # surface condition
        z0_surface = 0.5*self.v_elem_size + Constant(0.02)  # TODO generalize
        z_s = self.v_elem_size + z0_surface
        diff_flux = -(n*diffusivity_v*(cmu0)**p *
                      k**m * kappa**n * z_s**(n - 1.0))
        f += diff_flux*self.test*self.normal[2]*ds_surf

        return f


class GLSVerticalDiffusionTerm(VerticalDiffusionTerm):
    """
    Vertical diffusion term where diffusivity is replaced by viscosity/Schmidt number.
    """
    def __init__(self, function_space, schmidt_nb,
                 bathymetry=None, v_elem_size=None, h_elem_size=None):
        super(GLSVerticalDiffusionTerm, self).__init__(function_space,
                                                       bathymetry, v_elem_size, h_elem_size)
        self.schmidt_nb = schmidt_nb

    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        d = {'diffusivity_v': fields_old['viscosity_v']/self.schmidt_nb}
        f = super(GLSVerticalDiffusionTerm, self).residual(solution, solution_old,
                                                           d, d, bnd_conditions=None)
        return f


class TKEEquation(Equation):
    """
    Turbulent kinetic energy equation without advection terms.
    """
    def __init__(self, function_space, gls_model,
                 bathymetry=None, v_elem_size=None, h_elem_size=None):
        super(TKEEquation, self).__init__(function_space)

        diff = GLSVerticalDiffusionTerm(function_space,
                                        gls_model.options.schmidt_nb_tke,
                                        bathymetry, v_elem_size, h_elem_size)
        source = TKESourceTerm(function_space,
                               gls_model,
                               bathymetry, v_elem_size, h_elem_size)
        self.add_term(source, 'implicit')
        self.add_term(diff, 'implicit')


class PsiEquation(Equation):
    """
    Psi equation without advection terms.
    """
    def __init__(self, function_space, gls_model,
                 bathymetry=None, v_elem_size=None, h_elem_size=None):
        super(PsiEquation, self).__init__(function_space)

        diff = GLSVerticalDiffusionTerm(function_space,
                                        gls_model.options.schmidt_nb_psi,
                                        bathymetry, v_elem_size, h_elem_size)
        source = PsiSourceTerm(function_space,
                               gls_model,
                               bathymetry, v_elem_size, h_elem_size)
        self.add_term(diff, 'implicit')
        self.add_term(source, 'implicit')
