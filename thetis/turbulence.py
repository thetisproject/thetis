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
psi = (cm0)**p * k**m * l**n
where p, m, n parameters and cm0 is an empirical constant.

dpsi/dt + \nabla_h(uv*psi) + d(w*psi)dz = d/dz(\nu_h/\sigma_psi dpsi/dz) +
   psi/k*(c1*P + c3*B - c2*eps*f_wall)


Parameter c3 takes value c3_minus in stably stratified flows and c3_plus in
unstably stratified cases.

Turbulent length scale is obtained diagnostically as
l = (cm0)**3 * k**(3/2) * eps**(-1)

TKE dissipation rate is given by
eps = (cm0)**(3+p/n)*tke**(3/2+m/n)*psi**(-1/n)

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



Tuomas Karna 2015-09-07
"""
from tracer_eq import *
from utility import *


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


class VerticalGradSolver(object):
    """
    Computes vertical gradient in the weak sense.

    :arg source: A :class:`Function` or expression to differentiate.
    :arg source: A :class:`Function` where the solution will be stored.
        Must be in P0 space.
    """
    def __init__(self, source, solution):
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
        prob = LinearVariationalProblem(a, l, self.solution)
        self.weak_grad_solver = LinearVariationalSolver(prob)

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
    def __init__(self, uv, m2, mu, mv, mu_tmp, minval=1e-12, solver_parameters={}):
        solver_parameters.setdefault('ksp_atol', 1e-12)
        solver_parameters.setdefault('ksp_rtol', 1e-16)

        self.mu = mu
        self.mv = mv
        self.m2 = m2
        self.mu_tmp = mu_tmp
        self.minval = minval
        # relaxation coefficient between old and new mu or mv
        self.relaxation = 0.5

        self.var_solvers = {}
        for i_comp in range(2):
            solver = SmoothVerticalGradSolver(uv[i_comp], mu_tmp)
            self.var_solvers[i_comp] = solver

    def solve(self, init_solve=False):
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
    def __init__(self, rho, n2, n2_tmp, minval=1e-12, solver_parameters={}):
        self._no_op = False
        if rho is None:
            self._no_op = True

        if not self._no_op:
            solver_parameters.setdefault('ksp_atol', 1e-12)
            solver_parameters.setdefault('ksp_rtol', 1e-16)

            self.n2 = n2
            self.n2_tmp = n2_tmp
            # relaxation coefficient between old and new mu or mv
            self.relaxation = 0.5

            self.var_solvers = {}

            g = physical_constants['g_grav']
            rho0 = physical_constants['rho0']
            p = -g/rho0 * rho
            solver = SmoothVerticalGradSolver(p, self.n2_tmp)
            self.var_solver = solver

    def solve(self, init_solve=False):
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
                 n2, m2,
                 closure_name='k-epsilon',
                 p=3.0, m=1.5, n=-1.0,
                 schmidt_nb_tke=1.0, schmidt_nb_psi=1.3,
                 c1=1.44, c2=1.92, c3_minus=-0.52, c3_plus=1.0,
                 f_wall=1.0, k_min=3.7e-8, psi_min=1.0e-10,
                 eps_min=1e-10, visc_min=1.0e-8, diff_min=1.0e-8,
                 galperin_lim=0.56, ri_st=0.25,
                 stability_type='CA',
                 ):
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
        p, m, n : float
            parameters defining psi variable
        c, c2, c3_minus, c3_plus : float
            parameters for psi production terms
        f_wall : float
            wall proximity function for k-kl type models
        k_min, psi_min : float
            minimum values for k and psi
        stability_type : string
            stability function type:
            'KC': Kantha and Clayson (1994)
            'CA': Canuto et al. (2001) version A
            'CB': Canuto et al. (2001) version B
            'CH': Cheng et al. (2002)
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

        # parameter to mix old and new viscosity values (1 => new only)
        self.relaxation = 0.5

        self.stability_type = stability_type
        if self.stability_type == 'KC':
            self.stability_func = StabilityFunctionKanthaClayson()
        elif self.stability_type == 'CA':
            self.stability_func = StabilityFunctionCanutoA()
        elif self.stability_type == 'CB':
            self.stability_func = StabilityFunctionCanutoB()
        elif self.stability_type == 'CH':
            self.stability_func = StabilityFunctionCheng()
        else:
            raise Exception('Unknown stability function type: ' +
                            self.stability_type)

        cm0 = self.stability_func.compute_cmu0()
        kappa = self.stability_func.compute_kappa(schmidt_nb_psi, n, c1, c2)
        physical_constants['von_karman'].assign(kappa)  # update global value
        c3_minus = self.stability_func.compute_c3_minus(c1, c2, ri_st)

        # minimum length scale
        len_min = cm0**3 * k_min**1.5 / eps_min

        # parameters
        self.params = {
            'p': p,
            'm': m,
            'n': n,
            'c1': c1,
            'c2': c2,
            'c3_minus': c3_minus,
            'c3_plus': c3_plus,
            'ri_st': ri_st,
            'f_wall': f_wall,
            'schmidt_nb_tke': schmidt_nb_tke,
            'schmidt_nb_psi': schmidt_nb_psi,
            'k_min': k_min,
            'psi_min': psi_min,
            'eps_min': eps_min,
            'len_min': len_min,
            'visc_min': visc_min,
            'diff_min': diff_min,
            'galperin_lim': galperin_lim,
            'cm0': cm0,
            'von_karman': kappa,
            'limit_psi': False,
            'limit_eps': False,
            'limit_len': False,
            'limit_len_min': True,
            'closure_name': closure_name,
            'stability_type': stability_type,
        }

        self.shear_frequency_solver = ShearFrequencySolver(self.uv, self.m2,
                                                           self.mu, self.mv,
                                                           self.mu_tmp)
        if self.rho is not None:
            self.buoy_frequency_solver = BuoyFrequencySolver(self.rho, self.n2,
                                                             self.n2_tmp)

        self.initialize()
        self._print_summary()

    def _print_summary(self):
        """Prints all defined parameters and their values."""
        print_info('GLS Turbulence model parameters')
        for k in sorted(self.params.keys()):
            print_info('  {:16s} : {:}'.format(k, self.params[k]))

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

        cm0 = self.params['cm0']
        p = self.params['p']
        n = self.params['n']
        m = self.params['m']

        # limit k
        set_func_min_val(self.k, self.params['k_min'])

        k_arr = self.k.dat.data[:]
        n2_pos = self.n2_pos.dat.data[:]
        n2_pos_eps = 1e-12
        galp = self.params['galperin_lim']
        if self.params['limit_psi']:
            # impose Galperin limit on psi
            # psi^(1/n) <= sqrt(0.56)* (cm0)^(p/n) *k^(m/n+0.5)* n2^(-0.5)
            val = (np.sqrt(galp) * (cm0)**(p / n) * k_arr**(m / n + 0.5) * (n2_pos + n2_pos_eps)**(-0.5))**n
            if n > 0:
                # impose max value
                np.minimum(self.psi.dat.data, val, self.psi.dat.data)
            else:
                # impose min value
                np.maximum(self.psi.dat.data, val, self.psi.dat.data)
        set_func_min_val(self.psi, self.params['psi_min'])

        # udpate epsilon
        self.epsilon.assign(cm0**(3.0 + p/n)*self.k**(3.0/2.0 + m/n)*self.psi**(-1.0/n))
        if self.params['limit_eps']:
            # impose Galperin limit on eps
            eps_min = cm0**3.0/np.sqrt(galp)*np.sqrt(n2_pos)*k_arr
            np.maximum(self.epsilon.dat.data, eps_min, self.epsilon.dat.data)
        # impose minimum value
        set_func_min_val(self.epsilon, self.params['eps_min'])

        # update L
        self.l.assign(cm0**3.0 * self.k**(3.0/2.0) / self.epsilon)
        if self.params['limit_len_min']:
            set_func_min_val(self.l, self.params['len_min'])
        if self.params['limit_len']:
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
        new_visc = b*s_m/cm0**3
        new_diff = b*s_h/cm0**3
        self.viscosity.dat.data[:] = lam*new_visc + (1.0 - lam)*self.viscosity.dat.data[:]
        self.diffusivity.dat.data[:] = lam*new_diff + (1.0 - lam)*self.diffusivity.dat.data[:]

        set_func_min_val(self.viscosity, self.params['visc_min'])
        set_func_min_val(self.diffusivity, self.params['diff_min'])
        set_func_max_val(self.viscosity, 0.05)
        set_func_max_val(self.diffusivity, 0.05)
        # print '{:8s} {:8.3e} {:8.3e}'.format('k', self.k.dat.data.min(), self.k.dat.data.max())
        # print '{:8s} {:8.3e} {:8.3e}'.format('eps', self.epsilon.dat.data.min(), self.epsilon.dat.data.max())
        # print '{:8s} {:8.3e} {:8.3e}'.format('L', self.l.dat.data.min(), self.l.dat.data.max())
        # print '{:8s} {:8.3e} {:8.3e}'.format('s_h', s_h.min(), s_h.max())
        # print '{:8s} {:8.3e} {:8.3e}'.format('s_m', s_m.min(), s_m.max())
        # print '{:8s} {:8.3e} {:8.3e}'.format('nuv', self.viscosity.dat.data.min(), self.viscosity.dat.data.max())
        # print '{:8s} {:8.3e} {:8.3e}'.format('M2', self.m2.dat.data.min(), self.m2.dat.data.max())
        # print '{:8s} {:8.3e} {:8.3e}'.format('N2', self.n2.dat.data.min(), self.n2.dat.data.max())
        # print '{:8s} {:8.3e} {:8.3e}'.format('N2+', self.n2_pos.dat.data.min(), self.n2_pos.dat.data.max())
        # print '{:8s} {:8.3e} {:8.3e}'.format('N2-', self.n2_neg.dat.data.min(), self.n2_neg.dat.data.max())


def compute_normalized_frequencies(shear2, buoy2, k, eps):
    """
    Computes normalized buoyancy and shear frequency squared.
    Follows Burchard and Bolding JPO (2001).
    """
    alpha_buoy = k**2/eps**2*buoy2
    alpha_shear = k**2/eps**2*shear2
    # print '{:8s} {:8.3e} {:8.3e}'.format('M2', shear2.min(), shear2.max())
    # print '{:8s} {:8.3e} {:8.3e}'.format('N2', buoy2.min(), buoy2.max())
    # print_info('{:8s} {:10.3e} {:10.3e}'.format('a_buoy', alpha_buoy.min(), alpha_buoy.max()))
    # print_info('{:8s} {:10.3e} {:10.3e}'.format('a_shear', alpha_shear.min(), alpha_shear.max()))
    return alpha_buoy, alpha_shear


class StabilityFunction(object):
    """Base class for all stability functions"""
    def __init__(self, lim_alpha_shear=True, lim_alpha_buoy=True,
                 smooth_alpha_buoy_lim=True, alpha_buoy_crit=-1.2):
        self.lim_alpha_shear = lim_alpha_shear
        self.lim_alpha_buoy = lim_alpha_buoy
        self.smooth_alpha_buoy_lim = smooth_alpha_buoy_lim
        self.alpha_buoy_crit = alpha_buoy_crit
        # for plotting and such
        self.description = []
        if self.lim_alpha_shear:
            self.description += ['lim', 'shear']
        if self.lim_alpha_buoy:
            self.description += ['lim', 'alpha']
            if self.smooth_alpha_buoy_lim:
                self.description += ['smooth']
                self.description += ['ac'+str(self.alpha_buoy_crit)]
        self.description = ' '.join(self.description)

        # Umlauf and Buchard (2005) eq A.10
        self.a1 = 2.0/3.0 - 0.5*self.cc2
        self.a2 = 1.0 - 0.5*self.cc3
        self.a3 = 1.0 - 0.5*self.cc4
        self.a4 = 0.5*self.cc5
        self.a5 = 0.5 - 0.5*self.cc6

        self.ab1 = 1.0 - self.cb2
        self.ab2 = 1.0 - self.cb3
        self.ab3 = 2.0*(1.0 - self.cb4)
        self.ab4 = 2.0*(1.0 - self.cb5)
        self.ab5 = 2.0*self.cbb*(1.0 - self.cb5)

        # Umlauf and Buchard (2005) eq A.12
        self.nn = 0.5*self.cc1
        self.nb = self.cb1

        # Umlauf and Buchard (2005) eq A.9
        self.d0 = 36.0*self.nn**3*self.nb**2
        self.d1 = 84.0*self.a5*self.ab3*self.nn**2*self.nb + 36.0*self.ab5*self.nn**3*self.nb
        self.d2 = 9.0*(self.ab2**2 - self.ab1**2)*self.nn**3 - 12.0*(self.a2**2 - 3.0*self.a3**2)*self.nn*self.nb**2
        self.d3 = 12.0*self.a5*self.ab3*(self.a2*self.ab1 - 3.0*self.a3*self.ab2)*self.nn +\
            12.0*self.a5*self.ab3*(self.a3**2 - self.a2**2)*self.nb +\
            12.0*self.ab5*(3.0*self.a3**2 - self.a2**2)*self.nn*self.nb
        self.d4 = 48.0*self.a5**2*self.ab3**2*self.nn + 36.0*self.a5*self.ab3*self.ab5*self.nn**2
        self.d5 = 3.0*(self.a2**2 - 3.0*self.a3**2)*(self.ab1**2 - self.ab2**2)*self.nn
        self.n0 = 36.0*self.a1*self.nn**2*self.nb**2
        self.n1 = -12.0*self.a5*self.ab3*(self.ab1 + self.ab2)*self.nn**2 +\
            8.0*self.a5*self.ab3*(6.0*self.a1 - self.a2 - 3.0*self.a3)*self.nn*self.nb +\
            36.0*self.a1*self.ab5*self.nn**2*self.nb
        self.n2 = 9.0*self.a1*(self.ab2**2 - self.ab1**2)*self.nn**2
        self.nb0 = 12.0*self.ab3*self.nn**3*self.nb  # NOTE ab3 is not squared!
        self.nb1 = 12.0*self.a5*self.ab3**2*self.nn**2
        self.nb2 = 9.0*self.a1*self.ab3*(self.ab1 - self.ab2)*self.nn**2 +\
            (6.0*self.a1*(self.a2 - 3.0*self.a3) - 4.0*(self.a2**2 - 3.0*self.a3**2))*self.ab3*self.nn*self.nb

    def compute_c3_minus(self, c1, c2, ri_st):
        """
        Compute c3_minus parameter from c1, c2 and stability functions.

        c3_minus is solved from equation

        ri_st = s_m/s_h*(c2 - c1)/(c2 - c3_minus)   [1]

        where ri_st is the steady state gradient Richardson number.
        (Burchard and Bolding, 2001, eq 32)
        """

        # A) solve numerically
        # use equilibrium equation (Umlauf and Buchard, 2005, eq A.15)
        # s_m*a_shear - s_h*a_shear*ri_st = 1.0
        # to solve a_shear at equilibrium
        from scipy.optimize import minimize

        def cost(a_shear):
            a_buoy = ri_st*a_shear
            s_m, s_h = self.eval_funcs(a_buoy, a_shear)
            res = s_m*a_shear - s_h*a_shear*ri_st - 1.0
            return res**2
        p = minimize(cost, 1.0)
        assert p.success, 'solving alpha_shear failed'
        a_shear = p.x[0]

        # # B) solve analytically
        # # compute alpha_shear for equilibrium condition (Umlauf and Buchard, 2005, eq A.19)
        # # aM^2 (-d5 + n2 - (d3 - n1 + nb2 )Ri - (d4 + nb1)Ri^2) + aM (-d2 + n0 - (d1 + nb0)Ri) - d0 = 0
        # a = -self.d5 + self.n2 - (self.d3 - self.n1 + self.nb2)*ri_st - (self.d4 + self.nb1)*ri_st**2
        # b = -self.d2 + self.n0 - (self.d1 + self.nb0)*ri_st
        # c = -self.d0
        # a_shear = (-b + np.sqrt(b**2 - 4*a*c))/2/a

        # compute aN from Ri_st and aM, Ri_st = aN/aM
        a_buoy = ri_st*a_shear
        # evaluate stability functions for equilibrium conditions
        s_m, s_h = self.eval_funcs(a_buoy, a_shear)

        # compute c3 minus from [1]
        c3_minus = c2 - (c2 - c1)*s_m/s_h/ri_st

        # check error in ri_st
        err = s_m/s_h*(c2 - c1)/(c2 - c3_minus) - ri_st
        assert np.abs(err) < 1e-5, 'steady state gradient Richardson number does not match'

        return c3_minus

    def compute_cmu0(self):
        """Computes the paramenter c_mu_0"""
        cm0 = ((self.a2**2 - 3*self.a3**2 + 3*self.a1*self.nn)/(3 * self.nn**2))**0.25
        return cm0

    def compute_kappa(self, sigma_psi, n, c1, c2):
        """
        Computes von Karman constant from the Psi Schmidt number.

        n, c1, c2 are GLS model parameters.
s
        from Umlauf and Burchard (2003) eq (14)
        """
        kappa = np.sqrt(sigma_psi * self.compute_cmu0()**2 * (c2 - c1)/(n**2))
        return kappa

    def get_alpha_buoy_min(self):
        """
        Compute minimum alpha buoy

        from Umlauf and Buchard (2005) table 3
        """
        # G = epsilon case, this is used in GOTM
        an_min = 0.5*(np.sqrt((self.d1 + self.nb0)**2. - 4.*self.d0*(self.d4 + self.nb1)) - (self.d1 + self.nb0))/(self.d4 + self.nb1)
        # eq. (47)
        # an_min = self.alpha_buoy_min
        return an_min

    def get_alpha_shear_max(self, alpha_buoy, alpha_shear):
        """
        Compute maximum alpha shear

        from Umlauf and Buchard (2005) eq (44)
        """
        as_max_n = (self.d0*self.n0 +
                    (self.d0*self.n1 + self.d1*self.n0)*alpha_buoy +
                    (self.d1*self.n1 + self.d4*self.n0)*alpha_buoy**2 +
                    self.d4*self.n1*alpha_buoy**3)
        as_max_d = (self.d2*self.n0 +
                    (self.d2*self.n1 + self.d3*self.n0)*alpha_buoy +
                    self.d3*self.n1*alpha_buoy**2)
        return as_max_n/as_max_d

    def get_alpha_buoy_smooth_min(self, alpha_buoy):
        """
        Compute smoothed alpha_buoy minimum

        from Burchard and Petersen (1999) eq (19)
        """
        return alpha_buoy - (alpha_buoy - self.alpha_buoy_crit)**2/(alpha_buoy + self.get_alpha_buoy_min() - 2*self.alpha_buoy_crit)

    def eval_funcs(self, alpha_buoy, alpha_shear):
        """
        Evaluate (unlimited) stability functions

        from Burchard and Petersen (1999) eqns (30) and (31)
        """
        den = self.d0 + self.d1*alpha_buoy + self.d2*alpha_shear + self.d3*alpha_buoy*alpha_shear + self.d4*alpha_buoy**2 + self.d5*alpha_shear**2
        c_mu = (self.n0 + self.n1*alpha_buoy + self.n2*alpha_shear) / den
        c_mu_p = (self.nb0 + self.nb1*alpha_buoy + self.nb2*alpha_shear) / den
        return c_mu, c_mu_p

    def evaluate(self, shear2, buoy2, k, eps):
        """
        Evaluates stability functions. Applies limiters on alpha_buoy and alpha_shear.
        """
        alpha_buoy, alpha_shear = compute_normalized_frequencies(shear2, buoy2, k, eps)
        if self.lim_alpha_buoy:
            # limit min (negative) alpha_buoy (Umlauf and Burchard, 2005)
            if not self.smooth_alpha_buoy_lim:
                # crop at minimum value
                np.maximum(alpha_buoy, self.get_alpha_buoy_min(), alpha_buoy)
            else:
                # do smooth limiting instead (Buchard and Petersen, 1999, eq 19)
                ab_smooth_min = self.get_alpha_buoy_smooth_min(alpha_buoy)
                # NOTE this must be applied to values alpha_buoy < ab_crit only!
                ix = alpha_buoy < self.alpha_buoy_crit
                alpha_buoy[ix] = ab_smooth_min[ix]

        if self.lim_alpha_shear:
            # limit max alpha_shear (Umlauf and Burchard, 2005, eq 44)
            as_max = self.get_alpha_shear_max(alpha_buoy, alpha_shear)
            np.minimum(alpha_shear, as_max, alpha_shear)

        return self.eval_funcs(alpha_buoy, alpha_shear)


class StabilityFunctionCanutoA(StabilityFunction):
    """Canuto et al. (2001) version A stability functions"""
    def __init__(self, lim_alpha_shear=True, lim_alpha_buoy=True,
                 smooth_alpha_buoy_lim=True, alpha_buoy_crit=-1.2):
        # parameters from Umlauf and Buchard (2005), Table 1
        self.cc1 = 5.0000
        self.cc2 = 0.8000
        self.cc3 = 1.9680
        self.cc4 = 1.1360
        self.cc5 = 0.0000
        self.cc6 = 0.4000

        self.cb1 = 5.9500
        self.cb2 = 0.6000
        self.cb3 = 1.0000
        self.cb4 = 0.0000
        self.cb5 = 0.3333
        self.cbb = 0.7200

        self.alpha_buoy_min = -2.324    # (table 3 in Umlauf and Burchard, 2005)
        self.name = 'Canuto A'

        super(StabilityFunctionCanutoA, self).__init__(lim_alpha_shear, lim_alpha_buoy,
                                                       smooth_alpha_buoy_lim, alpha_buoy_crit)


class StabilityFunctionCanutoB(StabilityFunction):
    """Canuto et al. (2001) version B stability functions"""
    def __init__(self, lim_alpha_shear=True, lim_alpha_buoy=True,
                 smooth_alpha_buoy_lim=True, alpha_buoy_crit=-1.2):
        # parameters from Umlauf and Buchard (2005), Table 1
        self.cc1 = 5.0000
        self.cc2 = 0.6983
        self.cc3 = 1.9664
        self.cc4 = 1.0940
        self.cc5 = 0.0000
        self.cc6 = 0.4950

        self.cb1 = 5.6000
        self.cb2 = 0.6000
        self.cb3 = 1.0000
        self.cb4 = 0.0000
        self.cb5 = 0.3333
        self.cbb = 0.4770

        self.alpha_buoy_min = -3.093    # (table 3 in Umlauf and Burchard, 2005)
        self.name = 'Canuto B'

        super(StabilityFunctionCanutoB, self).__init__(lim_alpha_shear, lim_alpha_buoy,
                                                       smooth_alpha_buoy_lim, alpha_buoy_crit)


class StabilityFunctionKanthaClayson(StabilityFunction):
    """Kantha and Clayson (1994) quasi-equilibrium stability functions"""
    def __init__(self, lim_alpha_shear=True, lim_alpha_buoy=True,
                 smooth_alpha_buoy_lim=True, alpha_buoy_crit=-1.2):
        # parameters from Umlauf and Buchard (2005), Table 1
        self.cc1 = 6.0000
        self.cc2 = 0.3200
        self.cc3 = 0.0000
        self.cc4 = 0.0000
        self.cc5 = 0.0000
        self.cc6 = 0.0000

        self.cb1 = 3.7280
        self.cb2 = 0.7000
        self.cb3 = 0.7000
        self.cb4 = 0.0000
        self.cb5 = 0.2000
        self.cbb = 0.6102

        self.alpha_buoy_min = -1.312  # (table 3 in Umlauf and Burchard, 2005)
        self.name = 'Kantha Clayson'

        super(StabilityFunctionKanthaClayson, self).__init__(lim_alpha_shear, lim_alpha_buoy,
                                                             smooth_alpha_buoy_lim, alpha_buoy_crit)


class StabilityFunctionCheng(StabilityFunction):
    """Cheng et al. (2002) quasi-equilibrium stability functions"""
    def __init__(self, lim_alpha_shear=True, lim_alpha_buoy=True,
                 smooth_alpha_buoy_lim=True, alpha_buoy_crit=-1.2):
        # parameters from Umlauf and Buchard (2005), Table 1
        self.cc1 = 5.0000
        self.cc2 = 0.7983
        self.cc3 = 1.9680
        self.cc4 = 1.1360
        self.cc5 = 0.0000
        self.cc6 = 0.5000

        self.cb1 = 5.5200
        self.cb2 = 0.2134
        self.cb3 = 0.3570
        self.cb4 = 0.0000
        self.cb5 = 0.3333
        self.cbb = 0.8200

        self.alpha_buoy_min = -2.029  # (table 3 in Umlauf and Burchard, 2005)
        self.name = 'Cheng'

        super(StabilityFunctionCheng, self).__init__(lim_alpha_shear, lim_alpha_buoy,
                                                     smooth_alpha_buoy_lim, alpha_buoy_crit)


class TKEEquation(TracerEquation):
    """
    Advection-diffusion equation for turbulent kinetic energy (tke).

    Inherited from TracerEquation so only turbulence related source terms
    and boundary conditions need to be implemented.
    """
    def __init__(self, solution, eta, uv, w,
                 w_mesh=None, dw_mesh_dz=None,
                 diffusivity_h=None, diffusivity_v=None,
                 uv_mag=None, uv_p1=None, lax_friedrichs_factor=None,
                 bnd_markers=None, bnd_len=None, v_elem_size=None,
                 h_elem_size=None,
                 viscosity_v=None, gls_model=None):
        self.schmidt_number = gls_model.params['schmidt_nb_tke']
        # NOTE vertical diffusivity must be divided by the TKE Schmidt number
        diffusivity_eff = viscosity_v/self.schmidt_number
        # call parent constructor
        super(TKEEquation, self).__init__(solution, eta, uv, w,
                                          w_mesh, dw_mesh_dz,
                                          diffusivity_h=diffusivity_h,
                                          diffusivity_v=diffusivity_eff,
                                          uv_mag=uv_mag, uv_p1=uv_p1,
                                          lax_friedrichs_factor=lax_friedrichs_factor,
                                          bnd_markers=bnd_markers,
                                          bnd_len=bnd_len,
                                          v_elem_size=v_elem_size,
                                          h_elem_size=h_elem_size)
        # additional functions to pass to RHS functions
        new_kwargs = {
            'eddy_diffusivity': diffusivity_v,
            'eddy_viscosity': viscosity_v,
            'buoyancy_freq2_pos': gls_model.n2_pos,
            'buoyancy_freq2_neg': gls_model.n2_neg,
            'shear_freq2': gls_model.m2,
            'epsilon': gls_model.epsilon,
            'k': gls_model.k,
        }
        self.kwargs.update(new_kwargs)

    def rhs_implicit(self, solution, eta, uv, w, eddy_viscosity,
                     eddy_diffusivity, shear_freq2, buoyancy_freq2_pos,
                     buoyancy_freq2_neg, epsilon,
                     **kwargs):
        """Returns the right hand side of the source terms.
        These terms do not depend on the solution."""

        # TKE: P + B - eps
        # P = viscosity M**2           (production)
        # B = - diffusivity N**2       (byoyancy production)
        # M**2 = (du/dz)**2 + (dv/dz)**2 (shear frequency)
        # N**2 = -g\rho_0 (drho/dz)      (buoyancy frequency)
        # eps = (cm0)**(3+p/n)*tke**(3/2+m/n)*psi**(-1/n)
        #                                (tke dissipation rate)
        solution_old = kwargs['solution_old']
        p = eddy_viscosity * shear_freq2
        b_source = - eddy_diffusivity * buoyancy_freq2_neg
        b_sink = - eddy_diffusivity * buoyancy_freq2_pos

        source = p + b_source + (b_sink - epsilon)/solution_old*solution  # patankar
        f = inner(source, self.test)*dx
        return f


class PsiEquation(TracerEquation):
    """
    Advection-diffusion equation for additional GLS model variable (psi).

    Inherited from TracerEquation so only turbulence related source terms
    and boundary conditions need to be implemented.
    """
    def __init__(self, solution, eta, uv, w,
                 w_mesh=None, dw_mesh_dz=None,
                 diffusivity_h=None, diffusivity_v=None,
                 uv_mag=None, uv_p1=None, lax_friedrichs_factor=None,
                 bnd_markers=None, bnd_len=None, v_elem_size=None,
                 h_elem_size=None,
                 viscosity_v=None, gls_model=None):
        # NOTE vertical diffusivity must be divided by the TKE Schmidt number
        self.schmidt_number = gls_model.params['schmidt_nb_psi']
        diffusivity_eff = viscosity_v/self.schmidt_number
        # call parent constructor
        super(PsiEquation, self).__init__(solution, eta, uv, w,
                                          w_mesh, dw_mesh_dz,
                                          diffusivity_h=diffusivity_h,
                                          diffusivity_v=diffusivity_eff,
                                          uv_mag=uv_mag, uv_p1=uv_p1,
                                          lax_friedrichs_factor=lax_friedrichs_factor,
                                          bnd_markers=bnd_markers,
                                          bnd_len=bnd_len,
                                          v_elem_size=v_elem_size,
                                          h_elem_size=h_elem_size)
        self.gls_model = gls_model
        # additional functions to pass to RHS functions
        new_kwargs = {
            'eddy_diffusivity': diffusivity_v,
            'eddy_viscosity': viscosity_v,
            'buoyancy_freq2_pos': gls_model.n2_pos,
            'buoyancy_freq2_neg': gls_model.n2_neg,
            'shear_freq2': gls_model.m2,
            'epsilon': gls_model.epsilon,
            'k': gls_model.k,
        }
        self.kwargs.update(new_kwargs)

    def rhs_implicit(self, solution, eta, uv, w, eddy_viscosity, eddy_diffusivity,
                     shear_freq2, buoyancy_freq2_pos, buoyancy_freq2_neg, epsilon, k, diffusivity_v,
                     **kwargs):
        """Returns the right hand side of the source terms.
        These terms do not depend on the solution."""

        # psi: psi/k*(c1*P + c3*B - c2*eps*f_wall)
        # P = viscosity M**2           (production)
        # B = - diffusivity N**2       (byoyancy production)
        # M**2 = (du/dz)**2 + (dv/dz)**2 (shear frequency)
        # N**2 = -g\rho_0 (drho/dz)      (buoyancy frequency)
        # eps = (cm0)**(3+p/n)*tke**(3/2+m/n)*psi**(-1/n)
        #                                (tke dissipation rate)
        solution_old = kwargs['solution_old']
        p = eddy_viscosity * shear_freq2
        c1 = self.gls_model.params['c1']
        c2 = self.gls_model.params['c2']
        # c3 switch: c3 = c3_minus if n2 > 0 else c3_plus
        c3_minus = self.gls_model.params['c3_minus']  # < 0
        c3_plus = self.gls_model.params['c3_plus']  # > 0
        assert c3_minus < 0, 'c3_minus has unexpected sign'
        assert c3_plus >= 0, 'c3_plus has unexpected sign'
        b_source = c3_minus * -eddy_diffusivity * buoyancy_freq2_pos  # source
        b_source += c3_plus * -eddy_diffusivity * buoyancy_freq2_neg  # source
        f_wall = self.gls_model.params['f_wall']
        source = solution_old/k*(c1*p + b_source) - solution/k*(c2*f_wall*epsilon)  # patankar
        f = inner(source, self.test)*dx

        if self.compute_vert_diffusion:
            # add bottom/top boundary condition for psi
            # (nuv_v/sigma_psi * dpsi/dz)_b = n * nuv_v/sigma_psi * (cm0)^p * k^m * kappa^n * z_b^(n-1)
            # z_b = distance_from_bottom + z_0 (Burchard and Petersen, 1999)
            cm0 = self.gls_model.params['cm0']
            p = self.gls_model.params['p']
            m = self.gls_model.params['m']
            n = self.gls_model.params['n']
            z0_friction = physical_constants['z0_friction']
            kappa = physical_constants['von_karman']
            if self.v_elem_size is None:
                raise Exception('v_elem_size required')
            # bottom condition
            z_b = self.v_elem_size + z0_friction
            diff_flux = (n*diffusivity_v*(cm0)**p *
                         k**m * kappa**n * z_b**(n - 1.0))
            f += diff_flux*self.test*self.normal[2]*ds_bottom
            # surface condition
            z0_surface = Constant(0.01)  # TODO generalize
            z_s = self.v_elem_size + z0_surface
            diff_flux = -(n*diffusivity_v*(cm0)**p *
                          k**m * kappa**n * z_s**(n - 1.0))
            f += diff_flux*self.test*self.normal[2]*ds_surf

        return f
