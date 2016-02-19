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


class ShearFrequencySolver(object):
    """
    Computes vertical shear frequency squared form the given horizontal
    velocity field.

    M^2 = du/dz^2 + dv/dz^2
    """
    def __init__(self, uv, m2, mu, mv, mu_tmp, minval=1e-12, solver_parameters={}):
        # NOTE m2 should be computed from DG uv field ?
        # solver_parameters.setdefault('ksp_type', 'cg')
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
            fs = m2.function_space()
            test = TestFunction(fs)
            tri = TrialFunction(fs)
            normal = FacetNormal(fs.mesh())
            a = inner(test, tri)*dx
            # # jump penalty -- smooth m2 -- this may blow up??
            # alpha = Constant(2.0)*abs(avg(uv[i_comp]))
            # a += alpha*jump(test)*jump(tri)*dS_h
            l = -inner(uv[i_comp], Dx(test, 2))*dx
            l += avg(uv[i_comp])*jump(test, normal[2])*dS_h
            l += uv[i_comp]*test*normal[2]*(ds_surf + ds_bottom)

            prob = LinearVariationalProblem(a, l, mu_tmp)
            solver = LinearVariationalSolver(prob)
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

            fs = n2.function_space()
            test = TestFunction(fs)
            tri = TrialFunction(fs)
            normal = FacetNormal(fs.mesh())
            a = inner(test, tri)*dx
            p = -g/rho0 * rho
            l = -inner(p, Dx(test, 2))*dx
            l += avg(p)*jump(test, normal[2])*dS_h
            l += p*test*normal[2]*(ds_surf + ds_bottom)

            prob = LinearVariationalProblem(a, l, self.n2_tmp)
            solver = LinearVariationalSolver(prob)
            self.var_solver = solver

    def solve(self, init_solve=False):
        if not self._no_op:
            self.var_solver.solve()
            gamma = self.relaxation if not init_solve else 1.0
            self.n2.assign(gamma*self.n2_tmp +
                           (1.0 - gamma)*self.n2)


class SmootherP1(object):
    """Applies p1 projection on p1dg fields in-place."""
    def __init__(self, p1dg, p1, v_elem_size):
        # TODO assert spaces
        self.p1dg = p1dg
        self.p1 = p1
        self.v_elem_size = v_elem_size
        self.tmp_func_p1 = Function(self.p1, name='tmp_p1_func')

    def apply(self, input, output=None):
        if output is None:
            output = input
        assert input.function_space() == self.p1dg
        assert output.function_space() == self.p1dg
        # project to p1
        # self.tmp_func_p1.project(input)
        # NOTE projection *must* be monotonic, add diffusion operator?
        test = TestFunction(self.p1)
        tri = TrialFunction(self.p1)
        a = inner(tri, test) * dx
        l = inner(input, test) * dx
        mu = Constant(1.0e-2)  # TODO can this be estimated??
        a += mu*inner(Dx(tri, 2), Dx(test, 2)) * dx
        prob = LinearVariationalProblem(a, l, self.tmp_func_p1)
        solver = LinearVariationalSolver(prob)
        solver.solve()
        # copy nodal values to original field
        par_loop("""
    for (int i=0; i<input.dofs; i++) {
        input[i][0] = p1field[i][0];  // TODO is this mapping valid?
    }
    """,
                 dx,
                 {'input': (output, WRITE),
                  'p1field': (self.tmp_func_p1, READ)})


class GenericLengthScaleModel(object):
    """
    Generic length scale implementation


    """
    def __init__(self, solver, k_field, psi_field, uv_field, rho_field,
                 l_field, epsilon_field,
                 eddy_diffusivity, eddy_viscosity,
                 n2, m2,
                 p=3.0, m=1.5, n=-1.0,
                 schmidt_nb_tke=1.0, schmidt_nb_psi=1.3,
                 c1=1.44, c2=1.92, c3_minus=-0.52, c3_plus=1.0,
                 f_wall=1.0, k_min=1.0e-10, psi_min=1.0e-14,
                 eps_min=1e-14, visc_min=1.0e-8, diff_min=1.0e-8,
                 galperin_lim=0.56,
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
            'CA': Canuto (2001) model A
            'CB': Canuto (2001) model B
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
        self.c3 = Function(self.n2.function_space(),
                           name='c3 parameter')
        self.n2_tmp = Function(self.n2.function_space(),
                               name='tmp buoyancy frequency')
        self.tmp_field_p1 = Function(solver.function_spaces.P1,
                                     name='tmp_p1_field')
        self.tmp_field_p0 = Function(solver.function_spaces.P0,
                                     name='tmp_p0_field')
        self.smoother = SmootherP1(self.solver.function_spaces.P1DG, self.solver.function_spaces.P1,
                                   self.solver.fields.v_elem_size_3d)
        # discusting HACK
        if self.rho is not None:
            self.rho_cg = Function(self.solver.function_spaces.P1,
                                   name='density p1')
            self.rho_p1_proj = Projector(self.rho, self.rho_cg)
        self.uv_cg = Function(self.solver.function_spaces.P1v,
                              name='uv p1')
        self.uv_p1_proj = Projector(self.uv, self.uv_cg)
        # parameter to mix old and new viscosity values (1 => new only)
        self.relaxation = 0.5

        cc1 = 5.0000
        cc2 = 0.8000
        cc3 = 1.9680
        cc4 = 1.1360
        # cc5 = 0.0000
        # cc6 = 0.4000
        # ct1 = 5.9500
        # ct2 = 0.6000
        # ct3 = 1.0000
        # ct4 = 0.0000
        # ct5 = 0.3333
        # ctt = 0.720

        # compute the a_i's for the Algebraic Stress Model
        a1 = 2.0/3.0 - cc2/2.0
        a2 = 1.0 - cc3/2.0
        a3 = 1.0 - cc4/2.0
        # a4 = cc5/2.00
        # a5 = 0.5 - cc6/2.0

        # at1 = 1.0 - ct2
        # at2 = 1.0 - ct3
        # at3 = 2.0 * (1.0 - ct4)
        # at4 = 2.0 * (1.0 - ct5)
        # at5 = 2.0 * ctt * (1.0 - ct5)

        # compute cm0
        nn = cc1/2.0
        cm0 = ((a2**2 - 3*a3**2 + 3*a1*nn)/(3 * nn**2))**0.25
        # cmsf = a1/N/cm0**3
        rad = schmidt_nb_psi * (c2 - c1)/(n**2)
        kappa = cm0*sqrt(rad)
        # rcm = cm0/cmsf
        # cde = cm0**3.

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
            'limit_psi': False,  # NOTE noisy
            'limit_eps': False,  # NOTE noisy
            'limit_len': True,  # NOTE introduces less noise
            'cg_gradients': True,  # NOTE needed for kato-phillips
        }
        self.stability_type = stability_type
        if self.stability_type == 'KC':
            self.stability_func = StabilityFuncKanthaClayson()
        elif self.stability_type == 'CA':
            self.stability_func = StabilityFuncCanutoA()
        else:
            raise Exception('Unknown stability function type: ' +
                            self.stability_type)
        # compute c3_minus
        c3_minus = self.stability_func.compute_c3_minus(c1, c2)
        self.params['c3_minus'] = c3_minus

        print_info('GLS Turbulence model parameters')
        for k in sorted(self.params.keys()):
            print_info('  {:16s} : {:12.8g}'.format(k, self.params[k]))

        uv = self.uv_cg if self.params['cg_gradients'] else self.uv
        self.shear_frequency_solver = ShearFrequencySolver(uv, self.m2,
                                                           self.mu, self.mv,
                                                           self.mu_tmp)
        if self.rho is not None:
            rho = self.rho_cg if self.params['cg_gradients'] else self.rho
            self.buoy_frequency_solver = BuoyFrequencySolver(rho, self.n2,
                                                             self.n2_tmp)

        self.initialize()

    def update_c3(self):
        """Assigns c3 to c3_minus or c3_plus"""
        # c3 switch: c3 = c3_minus if n2 > 0 else c3_plus
        self.c3.assign(self.params['c3_minus'])
        tol = -1.0e-12
        ix = self.n2.dat.data < tol
        self.c3.dat.data[ix] = self.params['c3_plus']

    def initialize(self):
        """Initializes fields"""
        self.n2.assign(1e-12)
        self.update_c3()
        self.preprocess(init_solve=True)
        self.postprocess()

    def preprocess(self, init_solve=False):
        """
        To be called before evaluating the equations.

        Update all fields that depend on velocity and density.
        """
        # update m2 and N2

        if self.params['cg_gradients']:
            self.uv_p1_proj.project()
        self.shear_frequency_solver.solve(init_solve=init_solve)

        if self.rho is not None:
            if self.params['cg_gradients']:
                self.rho_p1_proj.project()
            self.buoy_frequency_solver.solve(init_solve=init_solve)
        self.update_c3()

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
        n2_arr = self.n2.dat.data[:]
        n2_pos = n2_arr.copy()
        n2_min = 1e-12
        n2_pos[n2_pos < n2_min] = n2_min
        galp = self.params['galperin_lim']
        if self.params['limit_psi']:
            # impose Galperin limit on psi
            # psi^(1/n) <= sqrt(0.56)* (cm0)^(p/n) *k^(m/n+0.5)* n2^(-0.5)
            val = (np.sqrt(galp) * (cm0)**(p / n) * k_arr**(m / n + 0.5) * (n2_pos)**(-0.5))**n
            if n > 0:
                # impose max value
                self.psi.dat.data[:] = np.minimum(self.psi.dat.data[:], val)
            else:
                # impose min value
                self.psi.dat.data[:] = np.maximum(self.psi.dat.data[:], val)
        set_func_min_val(self.psi, self.params['psi_min'])

        # self.tmp_field_p0.project(self.k)
        # self.tmp_field_p1.project(self.tmp_field_p0)
        # self.k.project(self.tmp_field_p1)
        # self.tmp_field_p0.project(self.psi)
        # self.tmp_field_p1.project(self.tmp_field_p0)
        # self.psi.project(self.tmp_field_p1)
        # self.solver.tracer_limiter.apply(self.k)
        # self.solver.tracer_limiter.apply(self.psi)

        # udpate epsilon
        self.epsilon.assign(cm0**(3.0 + p/n)*self.k**(3.0/2.0 + m/n)*self.psi**(-1.0/n))
        if self.params['limit_eps']:
            # impose Galperin limit on eps
            eps_min = cm0**3.0/np.sqrt(galp)*np.sqrt(n2_pos)*k_arr
            self.epsilon.dat.data[:] = np.maximum(self.epsilon.dat.data, eps_min)
        # HACK special case for k-eps model
        # self.epsilon.assign(self.psi)
        # Galperin limitation as in GOTM
        # galp = self.params['galperin_lim']
        # epslim = cm0**3.0/np.sqrt(2.)/galp*k_arr*np.sqrt(n2_pos)
        # self.epsilon.dat.data[:] = np.maximum(self.epsilon.dat.data[:], epslim)
        # impose minimum value
        set_func_min_val(self.epsilon, self.params['eps_min'])

        # update L
        self.l.assign(cm0**3.0 * self.k**(3.0/2.0) / self.epsilon)
        set_func_min_val(self.l, self.params['len_min'])
        if self.params['limit_len']:
            # Galperin length scale limitation
            len_max = np.sqrt(galp*k_arr/n2_pos)
            self.l.dat.data[:] = np.minimum(self.l.dat.data, len_max)
        if self.l.dat.data.max() > 10.0:
            print ' * large L: {:f}'.format(self.l.dat.data.max())

        # update stability functions
        s_m, s_h = self.stability_func.get_functions(self.m2.dat.data,
                                                     self.n2.dat.data,
                                                     self.k.dat.data,
                                                     self.epsilon.dat.data,
                                                     self.l.dat.data)
        # c = self.stability_func.c
        # update diffusivity/viscosity
        b = np.sqrt(self.k.dat.data[:])*self.l.dat.data[:]
        lam = self.relaxation
        new_visc = b*s_m/cm0**3
        new_diff = b*s_h/cm0**3
        self.viscosity.dat.data[:] = lam*new_visc + (1.0 - lam)*self.viscosity.dat.data[:]
        self.diffusivity.dat.data[:] = lam*new_diff + (1.0 - lam)*self.diffusivity.dat.data[:]

        # self.smoother.apply(self.viscosity)
        # self.smoother.apply(self.diffusivity)
        set_func_min_val(self.viscosity, self.params['visc_min'])
        set_func_min_val(self.diffusivity, self.params['diff_min'])
        # print '{:8s} {:8.3e} {:8.3e}'.format('k', self.k.dat.data.min(), self.k.dat.data.max())
        # print '{:8s} {:8.3e} {:8.3e}'.format('eps', self.epsilon.dat.data.min(), self.epsilon.dat.data.max())
        # print '{:8s} {:8.3e} {:8.3e}'.format('L', self.l.dat.data.min(), self.l.dat.data.max())
        # print '{:8s} {:8.3e} {:8.3e}'.format('s_h', s_h.min(), s_h.max())
        # print '{:8s} {:8.3e} {:8.3e}'.format('s_m', s_m.min(), s_m.max())
        # print '{:8s} {:8.3e} {:8.3e}'.format('nuv', self.viscosity.dat.data.min(), self.viscosity.dat.data.max())
        # print '{:8s} {:8.3e} {:8.3e}'.format('M2', self.m2.dat.data.min(), self.m2.dat.data.max())
        # print '{:8s} {:8.3e} {:8.3e}'.format('N2', self.n2.dat.data.min(), self.n2.dat.data.max())


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


class StabilityFuncKanthaClayson(object):
    """
    Implementation of Kantha-Clayson stability functions.

    Implementation follows Burchard and Bolding JPO (2001).
    """
    def __init__(self):
        # parameters
        self.s0 = 0.1682
        self.s1 = 0.03269
        self.s2 = 0.1783
        self.s3 = 0.01586
        self.s4 = 0.003173
        self.t1 = 0.4679
        self.t2 = 0.07372
        self.t3 = 0.01761
        self.t4 = 0.03371

        self.c = 1.0

    def compute_c3_minus(self, c1, c2):
        """
        Compute c3_minus parameter from c1, c2 and stability functions.

        From Warner (2005) equation (47).
        """
        return 5.08*c1 - 4.08*c2

    def get_functions(self, shear2, buoy2, k, eps, l):
        """
        Computes the values of the stability functions
        """
        alpha_buoy, alpha_shear = compute_normalized_frequencies(shear2, buoy2, k, eps)
        den = 1 + self.t1*alpha_buoy + self.t2*alpha_shear + self.t3*alpha_buoy*alpha_shear + self.t4*alpha_buoy**2
        c_mu = (self.s0 + self.s1*alpha_buoy) / den
        c_mu_p = (self.s2 + self.s3*alpha_buoy + self.s4*alpha_shear) / den
        return c_mu, c_mu_p


class StabilityFuncCanutoA(object):
    """
    Implementation of Canuto model A stability functions.

    Implementation follows Burchard and Bolding JPO (2001).
    """
    def __init__(self):
        # parameters
        self.s0 = 0.1070
        self.s1 = 0.01741
        self.s2 = -0.00012
        self.s3 = 0.1120
        self.s4 = 0.004519
        self.s5 = 0.00088
        self.t1 = 0.2555
        self.t2 = 0.02872
        self.t3 = 0.008677
        self.t4 = 0.005222
        self.t5 = -0.0000337

        self.c = 1.0

    def compute_c3_minus(self, c1, c2):
        """
        Compute c3_minus parameter from c1, c2 and stability functions.

        Burchard and Bolding (2001)
        """
        c_mu_ratio = 1.32339
        ri_st = 0.25
        return c_mu_ratio*(c1 - c2)/ri_st + c2

    def get_functions(self, shear2, buoy2, k, eps, l):
        """
        Computes the values of the stability functions
        """
        alpha_buoy, alpha_shear = compute_normalized_frequencies(shear2, buoy2, k, eps)
        # limit max alpha_shear (Umlauf and Burchard, 2005)
        # as_max =

        # limit min (negative) alpha_buoy (Umlauf and Burchard, 2005)
        # ab_min = (sqrt((self.t1 + self.s3)**2. - 4.*1*(self.t3 + self.s4)) - (self.t1 + self.s3))/(self.t3 + self.s4)*0.5
        ab_min = -3.0  # practical value (from Fig 3 in Burchard and Bolding, 2001)
        alpha_buoy = np.maximum(alpha_buoy, ab_min)

        den = 1 + self.t1*alpha_buoy + self.t2*alpha_shear + self.t3*alpha_buoy**2 + self.t4*alpha_buoy*alpha_shear + self.t5*alpha_shear**2
        c_mu = (self.s0 + self.s1*alpha_buoy + self.s2*alpha_shear) / den
        c_mu_p = (self.s3 + self.s4*alpha_buoy + self.s5*alpha_shear) / den
        return c_mu, c_mu_p


class StabilityFuncCanutoB(object):
    """
    Implementation of Canuto model A stability functions.

    Implementation follows Burchard and Bolding JPO (2001).
    """
    def __init__(self):
        # parameters
        self.s0 = 0.1270
        self.s1 = 0.01526
        self.s2 = -0.00016
        self.s3 = 0.1190
        self.s4 = 0.00429
        self.s5 = 0.00066
        self.t1 = 0.1977
        self.t2 = 0.03154
        self.t3 = 0.005832
        self.t4 = 0.004124
        self.t5 = -0.000042

        self.c = 1.0

    def compute_c3_minus(self, c1, c2):
        """
        Compute c3_minus parameter from c1, c2 and stability functions.

        From Warner (2005) equation (48).
        """
        return 4.09*c1 - 4.00*c2

    def get_functions(self, shear2, buoy2, k, eps, l):
        """
        Computes the values of the stability functions
        """
        alpha_buoy, alpha_shear = compute_normalized_frequencies(shear2, buoy2, k, eps)
        den = 1 + self.t1*alpha_buoy + self.t2*alpha_shear + self.t3*alpha_buoy**2 + self.t4*alpha_buoy*alpha_shear + self.t5*alpha_shear**2
        c_mu = (self.s0 + self.s1*alpha_buoy + self.s2*alpha_shear) / den
        c_mu_p = (self.s3 + self.s4*alpha_buoy + self.s5*alpha_shear) / den
        return c_mu, c_mu_p


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
            'buoyancy_freq2': gls_model.n2,
            'shear_freq2': gls_model.m2,
            'epsilon': gls_model.epsilon,
            'k': gls_model.k,
        }
        self.kwargs.update(new_kwargs)

    def source(self, eta, uv, w, eddy_viscosity, eddy_diffusivity,
               shear_freq2, buoyancy_freq2, epsilon,
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
        p = eddy_viscosity * shear_freq2
        b = - eddy_diffusivity * buoyancy_freq2

        source = p + b - epsilon
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
            'buoyancy_freq2': gls_model.n2,
            'shear_freq2': gls_model.m2,
            'epsilon': gls_model.epsilon,
            'k': gls_model.k,
            'c3': gls_model.c3,
        }
        self.kwargs.update(new_kwargs)

    def rhs_implicit(self, solution, eta, uv, w, eddy_viscosity, eddy_diffusivity,
                     shear_freq2, buoyancy_freq2, epsilon, k, diffusivity_v, c3,
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
        p = eddy_viscosity * shear_freq2
        b = -eddy_diffusivity * buoyancy_freq2
        c1 = self.gls_model.params['c1']
        c2 = self.gls_model.params['c2']
        f_wall = self.gls_model.params['f_wall']
        # NOTE source term must be implicit for stability
        source = solution/k*(c1*p + c3*b - c2*f_wall*epsilon)
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
            z_b = 0.5*self.v_elem_size + z0_friction
            diff_flux = (n*diffusivity_v*(cm0)**p *
                         k**m * kappa**n * z_b**(n - 1.0))
            f += diff_flux*self.test*self.normal[2]*ds_bottom
            # surface condition
            z0_surface = Constant(0.01)  # TODO generalize
            z_s = 0.5*self.v_elem_size + z0_surface
            diff_flux = -(n*diffusivity_v*(cm0)**p *
                          k**m * kappa**n * z_s**(n - 1.0))
            f += diff_flux*self.test*self.normal[2]*ds_surf

        return f
