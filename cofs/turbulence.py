"""
Generic Lenght Scale Turbulence Closure model [1].

This model solves two dynamic equations, for turbulent kinetic energy (tke, k)
and additional variable psi.

dk/dt + \nabla_h(uv*k) + d(w*k)\dz = d/dz(\nu_h/\sigma_k dk/dz) + P + B - eps
dpsi/dt + \nabla_h(uv*psi) + d(w*psi)\dz = d/dz(\nu_h/\sigma_psi dpsi/dz) +
   psi/k*(c1*P + c3*B - c2*eps*F_wall)

P = viscosity M**2             (production)
B = - diffusivity N**2         (byoyancy production)
M**2 = (du/dz)**2 + (dv/dz)**2 (shear frequency)
N**2 = -g\rho_0 (drho/dz)      (buoyancy frequency)

The additional variable is defined as
psi = (c0_mu)**p * k**m * l**n
where p, m, n parameters and c0_mu is an empirical constant.

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

# NOTE advection/diffusion should be treated similarly
#      to other tracers.
# NOTE vertical diffusion should be treated implicitly
# NOTE horizontal advection should be treated explicitly


class genericLengthScaleModel(object):
    """
    Generic lenght scale implementation


    """
    def __init__(self, k_field, psi_field, l_field, epsilon_field,
                 eddy_diffusivity, eddy_viscosity,
                 N2, M2,
                 p=3.0, m=1.5, n=-1.0,
                 schmidt_nb_k=1.0, schmidt_nb_psi=1.3,
                 c1=1.44, c2=1.92, c3_minus=-0.52, c3_plus=1.0,
                 F_wall=1.0, k_min=7.6e-6, psi_min=1.0e-12,
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
            turbulence lenght scale field
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
        # fields
        self.k = k_field
        self.psi = psi_field
        # NOTE redundant for k-epsilon model where psi==epsilon
        self.epsilon = epsilon_field
        self.l = l_field
        self.viscosity = eddy_viscosity
        self.diffusivity = eddy_diffusivity
        self.N2 = N2
        self.M2 = M2
        # parameters
        self.params = {
            'p': p,
            'm': m,
            'n': n,
            'c1': c1,
            'c1': c2,
            'c3_minus': c3_minus,
            'c3_plus': c3_plus,
            'F_wall': F_wall,
            }
        self.stabilityType = stabilityType
        if self.stabilityType == 'KC':
            self.stabilityFunc = stabilityFuncKanthaClayson()
        else:
            raise Exception('Unknown stability function type: ' +
                            self.stabilityType)
        self.params['c0_mu'] = self.stabilityFunc.c0_mu

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

    def _callback_preprocess(self,):
        """
        To be called before evaluating the equations.

        Update all fields that depend on velocity and density.
        """
        # update M2 and N2
        pass

    def _callback_postprocess(self,):
        """
        To be called after evaluating the equations.

        Update all fields that depend on turbulence fields.
        """

        def setMinVal(f, minval):
            f.dat.data[f.dat.data < minval] = minval
        c0_mu = self.params['c0_mu']
        p = self.params['p']
        n = self.params['n']
        m = self.params['m']
        # impose limits on k and psi
        setMinVal(self.k, self.params['k_min'])
        setMinVal(self.psi, self.params['psi_min'])
        # udpate l and eps
        # TODO limit l and epsilon correctly
        self.epsilon.assign(c0_mu**(3+p/n)*self.k**(3/2+m/n)*self.psi**(-1/n))
        setMinVal(self.epsilon, 1.0e-12)
        self.l.assign(c0_mu**3 * self.k**(3/2) / self.epsilon)
        setMinVal(self.l, 1.0e-12)
        # update stability functions
        # FIXME this will not work with the fields themselves
        # TODO compute with dat.data instead?
        S_M, S_H = self.stabilityFunc.getFunctions(self.N2, self.l, self.k)
        c = self.stabilityFunc.c
        # update diffusivity/viscosity
        self.viscosity.assign(c*sqrt(2*self.k)*self.l*S_M)
        self.diffusivity.assign(c*sqrt(2*self.k)*self.l*S_H)


class stabilityFuncKanthaClayson(object):
    """
    Implementation of Kantha-Clayson stability functions
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

    def getFunctions(self, N2, l, k):
        """
        Computes the values of the stability functions
        """
        Gh_unlim = - N2 * l**2 / (2*k)
        Gh = ((Gh_unlim - (Gh_unlim - self.Gh_crit)**2) /
              (Gh_unlim + self.Gh0 - 2*self.Gh_crit))
        S_H = ((self.A2 * (1.0 - 6*self.A1/self.B1)) /
               (1.0 - 3*self.A2*Gh*(6*self.A1 + self.B2*(1.0 - self.C3))))
        S_M = ((self.B1**(-1.0/3.0) + (18*self.A1**2 + 9*self.A1*self.A2*(1.0 - self.C2))*S_H*Gh) /
               (1.0 - 9*self.A1*self.A2*Gh))
        return S_H, S_M


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
                 bnd_markers=None, bnd_len=None,
                 viscosity_v=None, epsilon=None,
                 shear_freq2=None, buoyancy_freq2=None, glsModel=None):
        # call parent constructor
        super(tkeEquation, self).__init__(mesh, space, solution, eta, uv, w,
                                          w_mesh, dw_mesh_dz,
                                          diffusivity_h, diffusivity_v,
                                          uvMag, uvP1, laxFriedrichsFactor,
                                          bnd_markers, bnd_len)
        self.schmidt_number = glsModel.params['schmidt_nb_tke']
        # NOTE vertical diffusivity must be divided by the TKE Schmidt number
        viscosity_eff = viscosity_v/self.schmidt_number
        self.kwargs = {
            'viscosity_v': viscosity_eff,  # for vertical diffusion term
            'buoyancy_freq2': buoyancy_freq2,
            'shear_freq2': shear_freq2,
            'epsilon': epsilon,
            }

    def Source(self, eta, uv, w, viscosity_v, diffusivity_v,
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
        # NOTE needs original viscosity; scale by Schmidt number
        P = viscosity_v * self.schmidt_number * shear_freq2
        B = - diffusivity_v * buoyancy_freq2

        F = inner(P + B - epsilon, self.test)*self._dx
        return -F


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
                 bnd_markers=None, bnd_len=None,
                 viscosity_v=None, epsilon=None, k=None,
                 shear_freq2=None, buoyancy_freq2=None, glsModel=None):
        # call parent constructor
        super(psiEquation, self).__init__(mesh, space, solution, eta, uv, w,
                                          w_mesh, dw_mesh_dz,
                                          diffusivity_h, diffusivity_v,
                                          uvMag, uvP1, laxFriedrichsFactor,
                                          bnd_markers, bnd_len)
        self.glsModel = glsModel
        # NOTE vertical diffusivity must be divided by the TKE Schmidt number
        self.schmidt_number = glsModel.params['schmidt_nb_psi']
        viscosity_eff = viscosity_v/self.schmidt_number
        self.kwargs = {
            'viscosity_v': viscosity_eff,  # for vertical diffusion term
            'buoyancy_freq2': buoyancy_freq2,
            'shear_freq2': shear_freq2,
            'epsilon': epsilon,
            'k': k,
            }

    def Source(self, eta, uv, w, viscosity_v, diffusivity_v,
               shear_freq2, buoyancy_freq2, epsilon, k,
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
        # NOTE needs original viscosity; scale by Schmidt number
        P = viscosity_v * self.schmidt_number * shear_freq2
        B = - diffusivity_v * buoyancy_freq2
        # NOTE this depends on psi --> move to RHS_implicit?
        c1 = self.glsModel.params['c1']
        c2 = self.glsModel.params['c2']
        c3_plus = self.glsModel.params['c3_plus']
        c3_minus = self.glsModel.params['c3_minus']
        F_wall = self.glsModel.params['F_wall']
        # TODO implement c3 switch: c3 = c3_minus if N2 > 0 else c3_plus
        c3 = c3_minus
        f = psi/k*(c1*P + c3*B - c2*F_wall*epsilon)
        F = inner(P + B - epsilon, self.test)*self._dx
        return -F
