"""
Generic Length Scale Turbulence Closure model [1].

This model solves two dynamic equations, for turbulent kinetic energy (tke, k)
and additional variable psi.

dk/dt + \nabla_h(uv*k) + d(w*k)/dz = d/dz(\nu_h/\sigma_k dk/dz) + P + B - eps
dpsi/dt + \nabla_h(uv*psi) + d(w*psi)/dz = d/dz(\nu_h/\sigma_psi dpsi/dz) +
   psi/k*(c1*P + c3*B - c2*eps*F_wall)

P = viscosity M**2             (production)
B = - diffusivity N**2         (byoyancy production)
M**2 = (du/dz)**2 + (dv/dz)**2 (shear frequency)
N**2 = -g\rho_0 (drho/dz)      (buoyancy frequency)

The additional variable is defined as
psi = (c0_mu)**p * k**m * l**n
where p, m, n parameters and c0_mu is an empirical constant.

dpsi/dt + \nabla_h(uv*psi) + d(w*psi)dz = d/dz(\nu_h/\sigma_psi dpsi/dz) +
   psi/k*(c1*P + c3*B - c2*eps*F_wall)


Parameter c3 takes value c3_minus in stably stratified flows and c3_plus in
unstably stratified cases.

Turbulent length scale is obtained diagnostically as
l = (c0_mu)**3 * k**(3/2) * eps**(-1)

TKE dissipation rate is given by
eps = (c0_mu)**(3+p/n)*tke**(3/2+m/n)*psi**(-1/n)

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
from tracerEquation import *
from utility import *

# NOTE advection/diffusion should be treated similarly
#      to other tracers.
# NOTE vertical diffusion should be treated implicitly
# NOTE horizontal advection should be treated explicitly


def setFuncMinVal(f, minval):
    """
    Sets a minimum value to a function
    """
    f.dat.data[f.dat.data < minval] = minval


def computeShearFrequency(uv, M2, Mu_tmp, minval=1e-12,
                          solver_parameters={}):
    """
    Computes vertical shear frequency squared form the given horizontal
    velocity field.

    M^2 = du/dz^2 + dv/dz^2
    """
    solver_parameters.setdefault('ksp_type', 'cg')
    solver_parameters.setdefault('ksp_atol', 1e-12)
    solver_parameters.setdefault('ksp_rtol', 1e-16)

    M2.assign(0.0)
    for iComp in range(2):
        key = '-'.join(('M2', M2.name(), uv.name(), str(iComp)))
        if key not in linProblemCache:
            H = M2.function_space()
            test = TestFunction(H)
            tri = TrialFunction(H)
            dx = H.mesh()._dx
            dS_h = H.mesh()._dS_h
            ds_surf = H.mesh()._ds_b
            ds_bottom = H.mesh()._ds_t
            normal = FacetNormal(H.mesh())
            a = inner(test, tri)*dx
            L = - inner(uv[iComp], Dx(test, 2))*dx
            L += avg(uv[iComp])*jump(test, normal[2])*dS_h
            L += uv[iComp]*test*normal[2]*(ds_surf + ds_bottom)
            prob = LinearVariationalProblem(a, L, Mu_tmp)
            solver = LinearVariationalSolver(prob)
            linProblemCache.add(key, solver, 'updateCoords')
        linProblemCache[key].solve()
        M2 += Mu_tmp*Mu_tmp
    # crop small/negative values
    setFuncMinVal(M2, minval)


class genericLengthScaleModel(object):
    """
    Generic length scale implementation


    """
    def __init__(self, k_field, psi_field, uv_field,
                 l_field, epsilon_field,
                 eddy_diffusivity, eddy_viscosity,
                 N2, M2,
                 p=3.0, m=1.5, n=-1.0,
                 schmidt_nb_tke=1.0, schmidt_nb_psi=1.3,
                 c1=1.44, c2=1.92, c3_minus=-0.52, c3_plus=1.0,
                 F_wall=1.0, k_min=1.0e-10, psi_min=1.0e-12,
                 eps_min=1e-14,
                 galperin_lim=0.56,
                 stabilityType='KC',
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
        N2, M2 : Function
            buoyancy and vertical shear frequency squared
        p, m, n : float
            parameters defining psi variable
        c, c2, c3_minus, c3_plus : float
            parameters for psi production terms
        F_wall : float
            wall proximity function for k-kl type models
        k_min, psi_min : float
            minimum values for k and psi
        stabilityType : string
            stability function type:
            'KC': Kantha and Clayson (1994)
            'CA': Canuto (2001) model A
            'CB': Canuto (2001) model B
        """
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
        self.N2 = N2
        self.M2 = M2
        self.Mu_tmp = Function(self.M2.function_space(),
                               name='X/Y Shear frequency')
        # parameters
        self.params = {
            'p': p,
            'm': m,
            'n': n,
            'c1': c1,
            'c2': c2,
            'c3_minus': c3_minus,
            'c3_plus': c3_plus,
            'F_wall': F_wall,
            'schmidt_nb_tke': schmidt_nb_tke,
            'schmidt_nb_psi': schmidt_nb_psi,
            'k_min': k_min,
            'psi_min': psi_min,
            'eps_min': eps_min,
            'galperin_lim': galperin_lim,
            }
        self.stabilityType = stabilityType
        if self.stabilityType == 'KC':
            self.stabilityFunc = stabilityFuncKanthaClayson()
        else:
            raise Exception('Unknown stability function type: ' +
                            self.stabilityType)
        self.params['c0_mu'] = self.stabilityFunc.c0_mu

        self.initialize()

    def initialize(self):
        """Initializes fields"""
        self.N2.assign(1e-12)
        self.postprocess()

    def getEddyViscosity(self,):
        """
        Computes eddy viscosity from turbulence model fields

        viscosity = c*sqrt(2*tke)*L*stability_func_m + nu_0
        """
        return self.viscosity

    def getEddyDiffusivity(self,):
        """
        Computes eddy diffusivity from turbulence model fields

        diffusivity = c*sqrt(2*tke)*L*stability_func_rho + mu_0
        or
        diffusivity = Schmidt_number**a*viscosity (CHECK)
        """
        return self.diffusivity

    def preprocess(self,):
        """
        To be called before evaluating the equations.

        Update all fields that depend on velocity and density.
        """
        # update M2 and N2
        computeShearFrequency(self.uv, self.M2, self.Mu_tmp)

    def postprocess(self):
        """
        To be called after evaluating the equations.

        Update all fields that depend on turbulence fields.
        """

        c0_mu = self.params['c0_mu']
        p = self.params['p']
        n = self.params['n']
        m = self.params['m']
        # impose limits on k and psi
        setFuncMinVal(self.k, self.params['k_min'])
        setFuncMinVal(self.psi, self.params['psi_min'])
        # TODO limit psi
        # psi^(1/n) <= sqrt(0.56)* (c0_mu)^(p/n) *k^(m/n+0.5)* N2^(-0.5)
        k_arr = self.k.dat.data[:]
        N2_arr = self.N2.dat.data[:]
        val = (sqrt(0.56) * (c0_mu)**(p / n) * k_arr**(m / n + 0.5) * N2_arr**(-0.5))**n
        if n > 0:
            # impose max value
            self.psi.dat.data[:] = np.minimum(self.psi.dat.data[:], val)
        else:
            # impose min value
            self.psi.dat.data[:] = np.maximum(self.psi.dat.data[:], val)
        # udpate l and eps
        #self.epsilon.assign(c0_mu**(3+p/n)*self.k**(3/2+m/n)*self.psi**(-1/n))
        # HACK special case for k-eps model
        self.epsilon.assign(self.psi)
        # Galperin limitation as in GOTM
        N2_pos = self.N2.dat.data.copy()
        N2_tol = 0.0
        N2_pos[N2_pos < N2_tol] = N2_tol
        galp = self.params['galperin_lim']
        epslim = c0_mu**3.0/sqrt(2.)/galp*k_arr*np.sqrt(N2_pos)
        self.epsilon.dat.data[:] = np.maximum(self.epsilon.dat.data[:], epslim)
        # impose minimum value
        setFuncMinVal(self.epsilon, self.params['eps_min'])
        print 'eps', self.epsilon.dat.data.min(), self.epsilon.dat.data.max()

        #self.l.assign(c0_mu**3 * self.k**(3.0/2.0) / self.epsilon)  # TODO why this doesn't work
        self.l.dat.data[:] = c0_mu**3.0 * k_arr**(3.0/2.0) / self.epsilon.dat.data[:]
        ## Galperin L limitation
        ### FIXME use psi G_h limitation as discussed in Warner (2005)
        ##l_max = np.minimum(np.sqrt(0.56*self.k.dat.data[:]/N2_pos), 15.0)
        #self.l.dat.data[:] = np.minimum(self.l.dat.data[:], l_max)
        print 'L', self.l.dat.data.min(), self.l.dat.data.max()
        #setFuncMinVal(self.l, 1.0e-12)  # TODO this can be omitted?
        # update stability functions FIXME wrong
        S_M, S_H = self.stabilityFunc.getFunctions(self.M2.dat.data,
                                                   self.N2.dat.data,
                                                   self.l.dat.data,
                                                   self.k.dat.data)
        c = self.stabilityFunc.c
        # update diffusivity/viscosity
        #self.viscosity.assign(c*sqrt(2*self.k)*self.l*S_M)
        self.viscosity.dat.data[:] = c*np.sqrt(2*self.k.dat.data[:])*self.l.dat.data[:]*S_M
        #self.viscosity.dat.data[:] = S_M
        print 'nuv', self.viscosity.dat.data.min(), self.viscosity.dat.data.max()
        #self.viscosity.assign(self.l)
        #self.diffusivity.assign(c*sqrt(2*self.k)*self.l*S_H)


class stabilityFuncKanthaClayson(object):
    """
    Implementation of Kantha-Clayson stability functions.

    Following Burchard and Deleersnijder (2001).
    """
    def __init__(self):
        # parameters
        self.A1 = 0.92
        self.A2 = 0.74
        self.B1 = 16.6
        self.B2 = 10.1
        self.C2 = 0.7
        self.C3 = 0.2
        self.Gh_min = -0.28
        self.Gh0 = 0.0233
        self.Gh_crit = 0.02
        self.c0_mu = 0.5544
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

    def getFunctions(self, M2, N2, l, k):
        """
        Computes the values of the stability functions
        """
        b = l**2/(2*k)
        Gh = - N2 * b
        Gm = M2 * b
        print 'Gh', Gh.min(), Gh.max()
        print 'Gm', Gm.min(), Gm.max()
        # smoothing
        #Gh = ((Gh - (Gh - self.Gh_crit)**2) /
              #(Gh + self.Gh0 - 2*self.Gh_crit))
        Gh_max = 0.029
        Gm_max = (1 + self.b1*Gh + self.b3*Gh**2)/(self.b2 + self.b4*Gh)
        Gm = np.minimum(Gm, Gm_max)
        Gh = np.minimum(Gh, Gh_max)
        print 'Gh2', Gh.min(), Gh.max()
        print 'Gm2', Gm.min(), Gm.max()
        d = 1.0 + self.b1*Gh + self.b2*Gm + self.b3*Gh**2 + self.b4*Gm*Gh
        S_M = (self.a0 + self.a1*Gh)/d
        S_H = (self.a4 + self.a5*Gm + self.a6*Gh)/d
        print 'S_H', S_H.min(), S_H.max()
        print 'S_M', S_M.min(), S_M.max()
        return S_M, S_H


class tkeEquation(tracerEquation):
    """
    Advection-diffusion equation for turbulent kinetic energy (tke).

    Inherited from tracerEquation so only turbulence related source terms
    and boundary conditions need to be implemented.
    """
    def __init__(self, mesh, space, solution, eta, uv, w,
                 w_mesh=None, dw_mesh_dz=None,
                 diffusivity_h=None, diffusivity_v=None,
                 uvMag=None, uvP1=None, laxFriedrichsFactor=None,
                 bnd_markers=None, bnd_len=None, vElemSize=None,
                 viscosity_v=None, glsModel=None):
        self.schmidt_number = glsModel.params['schmidt_nb_tke']
        # NOTE vertical diffusivity must be divided by the TKE Schmidt number
        diffusivity_eff = viscosity_v/self.schmidt_number
        # call parent constructor
        super(tkeEquation, self).__init__(mesh, space, solution, eta, uv, w,
                                          w_mesh, dw_mesh_dz,
                                          diffusivity_h=diffusivity_h,
                                          diffusivity_v=diffusivity_eff,
                                          uvMag=uvMag, uvP1=uvP1,
                                          laxFriedrichsFactor=laxFriedrichsFactor,
                                          bnd_markers=bnd_markers,
                                          bnd_len=bnd_len,
                                          vElemSize=vElemSize)
        # additional functions to pass to RHS functions
        new_kwargs = {
            'eddy_diffusivity': diffusivity_v,
            'eddy_viscosity': viscosity_v,
            'buoyancy_freq2': glsModel.N2,
            'shear_freq2': glsModel.M2,
            'epsilon': glsModel.epsilon,
            'k': glsModel.k,
            }
        self.kwargs.update(new_kwargs)

    def Source(self, eta, uv, w, eddy_viscosity, eddy_diffusivity,
               shear_freq2, buoyancy_freq2, epsilon,
               **kwargs):
        """Returns the right hand side of the source terms.
        These terms do not depend on the solution."""

        # TKE: P + B - eps
        # P = viscosity M**2           (production)
        # B = - diffusivity N**2       (byoyancy production)
        # M**2 = (du/dz)**2 + (dv/dz)**2 (shear frequency)
        # N**2 = -g\rho_0 (drho/dz)      (buoyancy frequency)
        # eps = (c0_mu)**(3+p/n)*tke**(3/2+m/n)*psi**(-1/n)
        #                                (tke dissipation rate)
        P = eddy_viscosity * shear_freq2
        B = - eddy_diffusivity * buoyancy_freq2

        f = P + B - epsilon
        F = inner(f, self.test)*self.dx
        return F


class psiEquation(tracerEquation):
    """
    Advection-diffusion equation for additional GLS model variable (psi).

    Inherited from tracerEquation so only turbulence related source terms
    and boundary conditions need to be implemented.
    """
    def __init__(self, mesh, space, solution, eta, uv, w,
                 w_mesh=None, dw_mesh_dz=None,
                 diffusivity_h=None, diffusivity_v=None,
                 uvMag=None, uvP1=None, laxFriedrichsFactor=None,
                 bnd_markers=None, bnd_len=None, vElemSize=None,
                 viscosity_v=None, glsModel=None):
        # NOTE vertical diffusivity must be divided by the TKE Schmidt number
        self.schmidt_number = glsModel.params['schmidt_nb_psi']
        diffusivity_eff = viscosity_v/self.schmidt_number
        # call parent constructor
        super(psiEquation, self).__init__(mesh, space, solution, eta, uv, w,
                                          w_mesh, dw_mesh_dz,
                                          diffusivity_h=diffusivity_h,
                                          diffusivity_v=diffusivity_eff,
                                          uvMag=uvMag, uvP1=uvP1,
                                          laxFriedrichsFactor=laxFriedrichsFactor,
                                          bnd_markers=bnd_markers,
                                          bnd_len=bnd_len,
                                          vElemSize=vElemSize)
        self.glsModel = glsModel
        # additional functions to pass to RHS functions
        new_kwargs = {
            'eddy_diffusivity': diffusivity_v,
            'eddy_viscosity': viscosity_v,
            'buoyancy_freq2': glsModel.N2,
            'shear_freq2': glsModel.M2,
            'epsilon': glsModel.epsilon,
            'k': glsModel.k,
            }
        self.kwargs.update(new_kwargs)

    def RHS_implicit(self, solution, eta, uv, w, eddy_viscosity, eddy_diffusivity,
                     shear_freq2, buoyancy_freq2, epsilon, k, diffusivity_v,
                     **kwargs):
        """Returns the right hand side of the source terms.
        These terms do not depend on the solution."""

        # psi: psi/k*(c1*P + c3*B - c2*eps*F_wall)
        # P = viscosity M**2           (production)
        # B = - diffusivity N**2       (byoyancy production)
        # M**2 = (du/dz)**2 + (dv/dz)**2 (shear frequency)
        # N**2 = -g\rho_0 (drho/dz)      (buoyancy frequency)
        # eps = (c0_mu)**(3+p/n)*tke**(3/2+m/n)*psi**(-1/n)
        #                                (tke dissipation rate)
        P = eddy_viscosity * shear_freq2
        B = - eddy_diffusivity * buoyancy_freq2
        c1 = self.glsModel.params['c1']
        c2 = self.glsModel.params['c2']
        c3_plus = self.glsModel.params['c3_plus']
        c3_minus = self.glsModel.params['c3_minus']
        F_wall = self.glsModel.params['F_wall']
        # TODO implement c3 switch: c3 = c3_minus if N2 > 0 else c3_plus
        c3 = c3_minus
        f = solution/k*(c1*P + c3*B - c2*F_wall*epsilon)
        F = inner(f, self.test)*self.dx

        if self.computeVertDiffusion:
            # add bottom/top boundary condition for psi
            # (nuv_v/sigma_psi * dpsi/dz)_b = n * nuv_v/sigma_psi * (c0_mu)^p * k^m * kappa^n * z_b^(n-1)
            # z_b = distance_from_bottom + z_0 (Burchard and Petersen, 1999)
            c0_mu = self.glsModel.params['c0_mu']
            n = self.glsModel.params['n']
            m = self.glsModel.params['m']
            p = self.glsModel.params['p']
            z0_friction = physical_constants['z0_friction']
            kappa = physical_constants['von_karman']
            z_b = self.vElemSize + z0_friction
            if self.vElemSize is None:
                raise Exception('vElemSize required')
            diffFlux = n*diffusivity_v*(c0_mu)**p * k**m  * kappa**n *z_b**(n-1.0)
            F += diffFlux*self.test*self.normal[2]*self.ds_bottom

        return F
