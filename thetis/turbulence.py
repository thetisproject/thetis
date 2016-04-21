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

    def solve(self):
        mu_comp = [self.mu, self.mv]
        self.m2.assign(0.0)
        for i_comp in range(2):
            self.var_solvers[i_comp].solve()
            mu_comp[i_comp].assign(self.relaxation*self.mu_tmp +
                                   (1.0 - self.relaxation)*mu_comp[i_comp])
            self.m2 += mu_comp[i_comp]*mu_comp[i_comp]
        # crop small/negative values
        # set_func_min_val(m2, minval)


class SmootherP1(object):
    """Applies p1 projection on p1dg fields in-place."""
    def __init__(self, p1dg, p1, v_elem_size):
        # TODO assert spaces
        self.p1dg = p1dg
        self.p1 = p1
        self.v_elem_size = v_elem_size
        self.tmp_func_p1 = Function(self.p1, name='tmp_p1_func')

    def apply(self, input):
        assert input.function_space() == self.p1dg
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
                 {'input': (input, WRITE),
                  'p1field': (self.tmp_func_p1, READ)})


class GenericLengthScaleModel(object):
    """
    Generic length scale implementation


    """
    def __init__(self, solver, k_field, psi_field, uv_field,
                 l_field, epsilon_field,
                 eddy_diffusivity, eddy_viscosity,
                 n2, m2,
                 p=3.0, m=1.5, n=-1.0,
                 schmidt_nb_tke=1.0, schmidt_nb_psi=1.3,
                 c1=1.44, c2=1.92, c3_minus=-0.52, c3_plus=1.0,
                 f_wall=1.0, k_min=1.0e-10, psi_min=1.0e-14,
                 eps_min=1e-14, visc_min=1.0e-7, diff_min=1.0e-7,
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
        self.tmp_field_p1 = Function(solver.function_spaces.P1,
                                     name='tmp_p1_field')
        self.tmp_field_p0 = Function(solver.function_spaces.P0,
                                     name='tmp_p0_field')
        self.smoother = SmootherP1(self.solver.function_spaces.P1DG, self.solver.function_spaces.P1,
                                   self.solver.fields.v_elem_size_3d)
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
        parameters['von_karman'] = kappa
        # rcm = cm0/cmsf
        # cde = cm0**3.

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
            'visc_min': visc_min,
            'diff_min': diff_min,
            'galperin_lim': galperin_lim,
        }
        self.params['cm0'] = cm0
        self.stability_type = stability_type
        if self.stability_type == 'KC':
            self.stability_func = StabilityFuncKanthaClayson()
        elif self.stability_type == 'CA':
            self.stability_func = StabilityFuncCanutoA()
        else:
            raise Exception('Unknown stability function type: ' +
                            self.stability_type)
        # compute c3_minus
        # c3_minus = self.stability_func.compute_c3_minus(c1, c2)
        self.params['c3_minus'] = -0.63  # NOTE depends on model and stab func

        self.shear_frequency_solver = ShearFrequencySolver(self.uv, self.m2,
                                                           self.mu, self.mv,
                                                           self.mu_tmp)

        self.initialize()

    def initialize(self):
        """Initializes fields"""
        self.n2.assign(1e-12)
        self.postprocess()

    def preprocess(self,):
        """
        To be called before evaluating the equations.

        Update all fields that depend on velocity and density.
        """
        # update m2 and N2
        self.shear_frequency_solver.solve()
        # self.smoother.apply(self.m2)
        set_func_min_val(self.m2, 1e-12)

    def postprocess(self):
        """
        To be called after evaluating the equations.

        Update all fields that depend on turbulence fields.
        """

        cm0 = self.params['cm0']
        # p = self.params['p']
        # n = self.params['n']
        # m = self.params['m']

        # smooth k
        # self.smoother.apply(self.k)
        # limit k
        set_func_min_val(self.k, self.params['k_min'])

        # smooth psi
        # self.smoother.apply(self.psi)
        # limit psi
        # psi^(1/n) <= sqrt(0.56)* (cm0)^(p/n) *k^(m/n+0.5)* n2^(-0.5)
        k_arr = self.k.dat.data[:]
        n2_arr = self.n2.dat.data[:]
        n2_pos = n2_arr.copy()
        n2_pos[n2_pos < 0.0] = 0.0
        # val = (np.sqrt(0.56) * (cm0)**(p / n) * k_arr**(m / n + 0.5) * (n2_pos + 1e-12)**(-0.5))**n
        # if n > 0:
        #    # impose max value
        #    self.psi.dat.data[:] = np.minimum(self.psi.dat.data[:], val)
        # else:
        #    # impose min value
        #    self.psi.dat.data[:] = np.maximum(self.psi.dat.data[:], val)
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
        # self.epsilon.assign(cm0**(3+p/n)*self.k**(3/2+m/n)*self.psi**(-1/n))
        # HACK special case for k-eps model
        self.epsilon.assign(self.psi)
        # # Galperin limitation as in GOTM
        # galp = self.params['galperin_lim']
        # epslim = cm0**3.0/np.sqrt(2.)/galp*k_arr*np.sqrt(n2_pos)
        # self.epsilon.dat.data[:] = np.maximum(self.epsilon.dat.data[:], epslim)
        # impose minimum value
        set_func_min_val(self.epsilon, self.params['eps_min'])

        # update L
        # self.l.assign(cm0**3 * self.k**(3.0/2.0) / self.epsilon)  # TODO why this doesn't work
        self.l.dat.data[:] = cm0**3.0 * np.sqrt(np.power(k_arr, 3)) / self.epsilon.dat.data[:]
        # self.smoother.apply(self.l)
        # set_func_max_val(self.l, 10.0)  # HACK limit L to something meaningful
        set_func_min_val(self.l, 1e-12)
        if self.l.dat.data.max() > 10.0:
            print ' * large L: {:f}'.format(self.l.dat.data.max())
        # # Galperin L limitation

        # update stability functions
        s_m, s_h = self.stability_func.get_functions(self.m2.dat.data,
                                                     self.n2.dat.data,
                                                     self.epsilon.dat.data,
                                                     self.k.dat.data)
        c = self.stability_func.c
        # update diffusivity/viscosity
        # NOTE should this the sqrt(k) or sqrt(2*k)?
        b = np.sqrt(2*self.k.dat.data[:])*self.l.dat.data[:]
        lam = self.relaxation
        self.viscosity.dat.data[:] = lam*c*b*s_m + (1.0 - lam)*self.viscosity.dat.data[:]
        self.diffusivity.dat.data[:] = lam*c*b*s_h + (1.0 - lam)*self.diffusivity.dat.data[:]
        # self.smoother.apply(self.viscosity)
        # self.smoother.apply(self.diffusivity)
        set_func_min_val(self.viscosity, self.params['visc_min'])
        set_func_min_val(self.diffusivity, self.params['diff_min'])
        print '{:8s} {:8.3e} {:8.3e}'.format('k', self.k.dat.data.min(), self.k.dat.data.max())
        print '{:8s} {:8.3e} {:8.3e}'.format('eps', self.epsilon.dat.data.min(), self.epsilon.dat.data.max())
        print '{:8s} {:8.3e} {:8.3e}'.format('L', self.l.dat.data.min(), self.l.dat.data.max())
        print '{:8s} {:8.3e} {:8.3e}'.format('s_h', s_h.min(), s_h.max())
        print '{:8s} {:8.3e} {:8.3e}'.format('s_m', s_m.min(), s_m.max())
        print '{:8s} {:8.3e} {:8.3e}'.format('nuv', self.viscosity.dat.data.min(), self.viscosity.dat.data.max())


class StabilityFuncKanthaClayson(object):
    """
    Implementation of Kantha-Clayson stability functions.

    Implementation follows Burchard and Deleersnijder (2001).
    """
    def __init__(self):

        # parameters
        self.A1 = 0.92
        self.A2 = 0.74
        self.B1 = 16.6
        self.B2 = 10.1
        self.C2 = 0.7
        self.C3 = 0.2
        self.gh_max = 0.029
        self.gh_min = -0.28
        self.gh0 = 0.0233
        self.gh_crit = 0.02
        self.cm0 = 0.5544
        self.c = 1.0
        self.C1 = (1.0 - (self.A1*self.B1**(1.0/3.0))**-1 - 6*self.A1/self.B1)/3.0
        self.a0 = self.A1*(1.0 - 3*self.C1)
        self.a1 = 3*self.A1*self.A2*(4*self.A1 + 3*self.A2*(1.0 - self.C2) - (1.0 - 3*self.C1)*(self.B2*(1.0-self.C3) + 4*self.A1))
        self.a4 = self.A2
        self.a5 = 18*self.A1**2*self.A2*self.C1
        self.a6 = -9*self.A1*self.A2**2
        self.b1 = -3*self.A2*(3*self.A1 + self.B2*(1.0 - self.C3) + 4*self.A1)
        self.b2 = 6*self.A1
        self.b3 = 27*self.A1*self.A2**2*(self.B2*(1.0 - self.C3) + 4*self.A1)
        self.b4 = 18*self.A1**2*self.A2*(3*self.A2*(1.0 - self.C2) + self.B2*(1 - self.C3))

    def compute_c3_minus(self, c1, c2):
        """
        Compute c3_minus parameter from c1, c2 and stability functions.

        From Warner (2005) equation (47).
        """
        return 5.08*c1 - 4.08*c2

    def get_functions(self, m2, n2, l, k):
        """
        Computes the values of the stability functions
        """
        b = l**2/(2*k)
        gh = - n2 * b
        gm = m2 * b
        print 'gh', gh.min(), gh.max()
        print 'gm', gm.min(), gm.max()
        # smoothing (from ROMS)
        gh = np.minimum(gh, self.gh0)
        gh = np.minimum(gh, gh-(gh-self.gh_crit)**2/(gh + self.gh0 - 2*self.gh_crit))
        gh = np.maximum(gh, self.gh_min)
        gm_max = (1 + self.b1*gh + self.b3*gh**2)/(self.b2 + self.b4*gh)
        gm = np.minimum(gm, gm_max)
        print 'gh2', gh.min(), gh.max()
        print 'gm2', gm.min(), gm.max()
        d = 1.0 + self.b1*gh + self.b2*gm + self.b3*gh**2 + self.b4*gm*gh
        s_m = np.maximum((self.a0 + self.a1*gh)/d, 0.0)
        s_h = np.maximum((self.a4 + self.a5*gm + self.a6*gh)/d, 0.0)
        print 's_h', s_h.min(), s_h.max()
        print 's_m', s_m.min(), s_m.max()
        return s_m, s_h


class StabilityFuncCanutoA(object):
    """
    Implementation of Canuto model A stability functions.

    Implementation follows Burchard and Deleersnijder (2001).
    """
    def __init__(self):
        # parameters
        self.s0 = 0.5168
        self.s1 = -7.848
        self.s2 = -0.0545
        self.s4 = 0.5412
        self.s5 = -2.04
        self.s6 = 0.3964
        self.t1 = -23.84
        self.t2 = 2.68
        self.t3 = 75.574
        self.t4 = -45.48
        self.t5 = -0.2937
        self.gh_max = 0.0673
        self.cm0 = 0.5270
        self.c = 1.0
        self.gh_min = -0.28
        self.gh0 = 0.0233
        self.gh_crit = 0.02

    def compute_c3_minus(self, c1, c2):
        """
        Compute c3_minus parameter from c1, c2 and stability functions.

        From Warner (2005) equation (48).
        """
        return 4.09*c1 - 4.00*c2

    def get_functions(self, m2, n2, eps, k):
        """
        Computes the values of the stability functions
        """
        # b = l**2/(2*k)
        # gh = - n2 * b
        # gm = m2 * b
        # #print 'gh', gh.min(), gh.max()
        # #print 'gm', gm.min(), gm.max()
        # # smoothing (from ROMS)
        # gh = np.minimum(gh, self.gh0)
        # gh = np.minimum(gh, gh-(gh-self.gh_crit)**2/(gh + self.gh0 - 2*self.gh_crit))
        # gh = np.maximum(gh, self.gh_min)
        # alpha = 1.0
        # gm_max = alpha*(1 + self.t1*gh + self.t3*gh**2)/(self.t2 + self.t4*gh)
        # gm = np.minimum(gm, gm_max)
        # gh = np.minimum(gh, self.gh_max)
        # #print 'gh2', gh.min(), gh.max()
        # #print 'gm2', gm.min(), gm.max()
        # d = 1.0 + self.t1*gh + self.t2*gm + self.t3*gh**2 + self.t4*gm*gh + self.t5*gm**2
        # s_m = (self.s0 + self.s1*gh + self.s2*gm)/d
        # s_h = (self.s4 + self.s5*gh + self.s6*gm)/d
        # return s_m, s_h

        # TODO get the params from gls class
        # schmidt_nb_psi = 1.3
        # c1 = 1.44
        # c2 = 1.92
        # n = -1.0

        an_limit_fact = 0.5

        cc1 = 5.0000
        cc2 = 0.8000
        cc3 = 1.9680
        cc4 = 1.1360
        # cc5 = 0.0000
        cc6 = 0.4000
        ct1 = 5.9500
        ct2 = 0.6000
        ct3 = 1.0000
        ct4 = 0.0000
        ct5 = 0.3333
        ctt = 0.720

        # compute the a_i's for the Algebraic Stress Model
        a1 = 2.0/3.0 - cc2/2.0
        a2 = 1.0 - cc3/2.0
        a3 = 1.0 - cc4/2.0
        # a4 = cc5/2.00
        a5 = 0.5 - cc6/2.0

        at1 = 1.0 - ct2
        at2 = 1.0 - ct3
        at3 = 2.0 * (1.0 - ct4)
        # at4 = 2.0 * (1.0 - ct5)
        at5 = 2.0 * ctt * (1.0 - ct5)

        # compute cm0
        nn = cc1/2.0
        cm0 = ((a2**2 - 3*a3**2 + 3*a1*nn)/(3 * nn**2))**0.25
        # cmsf = a1/N/cm0**3
        # rad = schmidt_nb_psi * (c2 - c1)/(n**2)
        # kappa = cm0*sqrt(rad)
        # rcm = cm0/cmsf
        # cde = cm0**3.

        # This is written out verbatim as in GOTM v4.3.1 (also GNU GPL)
        nn = 0.5 * cc1
        nt = ct1
        d0 = 36 * nn**3 * nt**2.
        d1 = 84.*a5*at3 * nn**2. * nt + 36.*at5 * nn**3. * nt
        d2 = 9.*(at2**2.-at1**2.) * nn**3. - 12.*(a2**2.-3.*a3**2.) * nn * nt**2.
        d3 = 12.*a5*at3*(a2*at1-3.*a3*at2) * nn + 12.*a5*at3*(a3**2.-a2**2.) * \
            nt + 12.*at5*(3.*a3**2.-a2**2.) * nn * nt
        d4 = 48.*a5**2.*at3**2. * nn + 36.*a5*at3*at5 * nn**2.
        d5 = 3.*(a2**2.-3.*a3**2.)*(at1**2.-at2**2.) * nn
        n0 = 36.*a1 * nn**2. * nt**2.
        n1 = - 12.*a5*at3*(at1+at2) * nn**2. + 8.*a5*at3*(6.*a1-a2-3.*a3) * \
            nn * nt + 36.*a1*at5 * nn**2. * nt
        n2 = 9.*a1*(at2**2.-at1**2.) * nn**2.
        nt0 = 12.*at3 * nn**3. * nt
        nt1 = 12.*a5*at3**2. * nn**2.
        nt2 = 9.*a1*at3*(at1-at2) * nn**2. + (6.*a1*(a2-3.*a3) - 4.*(a2**2.-3.*a3**2.))*at3 * nn * nt
        cm3_inv = 1./cm0**3

        # TODO move all above to init

        # mininum value of an to ensure as > 0 in equilibrium
        an_min_num = -(d1 + nt0) + np.sqrt((d1+nt0)**2. - 4.*d0*(d4+nt1))
        an_min_den = 2.*(d4+nt1)
        an_min = an_min_num / an_min_den

        # (special treatment to  avoid a singularity)
        tau2 = k*k / (eps*eps)
        an = tau2 * n2
        # clip an at minimum value
        np.maximum(an, an_limit_fact*an_min, an)
        # compute the equilibrium value of as
        tmp0 = -d0 - (d1 + nt0)*an - (d4 + nt1)*an*an
        tmp1 = -d2 + n0 + (n1-d3-nt2)*an

        # compute the equilibrium value of as
        tmp0 = -d0 - (d1 + nt0)*an - (d4 + nt1)*an*an
        tmp1 = -d2 + n0 + (n1-d3-nt2)*an
        tmp2 = n2-d5
        if (np.abs(tmp2) < 1e-7):
            ass = -tmp0 / tmp1
        else:
            ass = (-tmp1 + np.sqrt(tmp1*tmp1 - 4.*tmp0*tmp2)) / (2.*tmp2)

        # compute stability function
        dcm = d0 + d1*an + d2*ass + d3*an*ass + d4*an*an + d5*ass*ass
        ncm = n0 + n1*an + n2*ass
        ncmp = nt0 + nt1*an + nt2*ass
        s_m = cm3_inv * ncm / dcm
        s_h = cm3_inv * ncmp / dcm
        return s_m, s_h


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
        b = 0.0  # - eddy_diffusivity * buoyancy_freq2

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
        }
        self.kwargs.update(new_kwargs)

    def rhs_implicit(self, solution, eta, uv, w, eddy_viscosity, eddy_diffusivity,
                     shear_freq2, buoyancy_freq2, epsilon, k, diffusivity_v,
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
        b = 0.0  # - eddy_diffusivity * buoyancy_freq2
        c1 = self.gls_model.params['c1']
        c2 = self.gls_model.params['c2']
        # c3_plus = self.gls_model.params['c3_plus']
        c3_minus = self.gls_model.params['c3_minus']
        f_wall = self.gls_model.params['f_wall']
        # TODO implement c3 switch: c3 = c3_minus if n2 > 0 else c3_plus
        c3 = c3_minus
        # NOTE seems more stable explicitly with epsilon
        source = epsilon/k*(c1*p + c3*b - c2*f_wall*epsilon)
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
            z0_surface = Constant(0.001)  # TODO generalize
            z_s = 0.5*self.v_elem_size + z0_surface
            diff_flux = -(n*diffusivity_v*(cm0)**p *
                          k**m * kappa**n * z_s**(n - 1.0))
            f += diff_flux*self.test*self.normal[2]*ds_surf

        return f
