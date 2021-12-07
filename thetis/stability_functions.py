r"""
Implements turbulence closure model stability functions.

.. math::
    S_m &= S_m(\alpha_M, \alpha_N) \\
    S_\rho &= S_\rho(\alpha_M, \alpha_N)

where :math:`\alpha_M, \alpha_N` are the normalized shear and buoyancy frequency

    .. math::
        \alpha_M &= \frac{k^2}{\varepsilon^2} M^2 \\
        \alpha_N &= \frac{k^2}{\varepsilon^2} N^2

The following stability functions have been implemented

- Canuto A
- Canuto B
- Kantha-Clayson
- Cheng

References:

Umlauf, L. and Burchard, H. (2005). Second-order turbulence closure models
for geophysical boundary layers. A review of recent work. Continental Shelf
Research, 25(7-8):795--827. http://dx.doi.org/10.1016/j.csr.2004.08.004

Burchard, H. and Bolding, K. (2001). Comparative Analysis of Four
Second-Moment Turbulence Closure Models for the Oceanic Mixed Layer. Journal of
Physical Oceanography, 31(8):1943--1968.
http://dx.doi.org/10.1175/1520-0485(2001)031

Umlauf, L. and Burchard, H. (2003). A generic length-scale equation for
geophysical turbulence models. Journal of Marine Research, 61:235--265(31).
http://dx.doi.org/10.1357/002224003322005087

"""
import numpy
from abc import ABC, abstractmethod
from scipy.optimize import minimize
from .log import print_output


__all__ = [
    'StabilityFunctionBase',
    'StabilityFunctionCanutoA',
    'StabilityFunctionCanutoB',
    'StabilityFunctionCheng',
    'GOTMStabilityFunctionCanutoA',
    'GOTMStabilityFunctionCanutoB',
    'GOTMStabilityFunctionCheng',
    'GOTMStabilityFunctionKanthaClayson',
    'compute_normalized_frequencies'
]


def compute_normalized_frequencies(shear2, buoy2, k, eps, verbose=False):
    r"""
    Computes normalized buoyancy and shear frequency squared.

    .. math::
        \alpha_M &= \frac{k^2}{\varepsilon^2} M^2 \\
        \alpha_N &= \frac{k^2}{\varepsilon^2} N^2

    From Burchard and Bolding (2001).

    :arg shear2: :math:`M^2`
    :arg buoy2: :math:`N^2`
    :arg k: turbulent kinetic energy
    :arg eps: TKE dissipation rate
    """
    alpha_buoy = k**2/eps**2*buoy2
    alpha_shear = k**2/eps**2*shear2
    if verbose:
        print_output('{:8s} {:8.3e} {:8.3e}'.format('M2', shear2.min(), shear2.max()))
        print_output('{:8s} {:8.3e} {:8.3e}'.format('N2', buoy2.min(), buoy2.max()))
        print_output('{:8s} {:10.3e} {:10.3e}'.format('a_buoy', alpha_buoy.min(), alpha_buoy.max()))
        print_output('{:8s} {:10.3e} {:10.3e}'.format('a_shear', alpha_shear.min(), alpha_shear.max()))
    return alpha_buoy, alpha_shear


class StabilityFunctionBase(ABC):
    """
    Base class for all stability functions
    """
    @property
    @abstractmethod
    def name(self):
        pass

    def __init__(self, lim_alpha_shear=True, lim_alpha_buoy=True,
                 smooth_alpha_buoy_lim=True, alpha_buoy_crit=-1.2):
        r"""
        :kwarg bool lim_alpha_shear: limit maximum :math:`\alpha_M` values
            (see Umlauf and Burchard (2005) eq. 44)
        :kwarg bool lim_alpha_buoy: limit minimum (negative) :math:`\alpha_N` values
            (see Umlauf and Burchard (2005))
        :kwarg bool smooth_alpha_buoy_lim: if :math:`\alpha_N` is limited, apply a
            smooth limiter (see Burchard and Bolding (2001) eq. 19). Otherwise
            :math:`\alpha_N` is clipped at minimum value.
        :kwarg float alpha_buoy_crit: parameter for :math:`\alpha_N` smooth limiter
        """
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

        # denominator
        self.d0 = 1.0
        self.d1 = 1.0
        self.d2 = 1.0
        self.d3 = 1.0
        self.d4 = 1.0
        self.d5 = 1.0

        # c_mu
        self.n0 = 0.0
        self.n1 = 0.0
        self.n2 = 0.0

        # c_mu_p
        self.nb0 = 0.0
        self.nb1 = 0.0
        self.nb2 = 0.0

    def compute_alpha_shear_steady(self, ri_st, analytical=True):
        r"""
        Compute the steady-state :math:`\alpha_M`.

        Under steady-state conditions, the stability functions satisfy:

        .. math::
            S_m \alpha_M - S_\rho \alpha_M Ri_{st} = 1.0

        (Umlauf and Buchard, 2005, eq A.15) from which :math:`\alpha_M` can be
        solved for given gradient Richardson number :math:`Ri_{st}`.

        :arg float ri_st: Gradient Richardson number
        :kwarg bool analytical: If True (default), solve analytically using the
            coefficients of the stability function. Otherwise, solve
            :math:`\alpha_M` numerically from the equilibrium condition.
        """
        if not analytical:
            # A) solve numerically
            # use equilibrium equation (Umlauf and Buchard, 2005, eq A.15)
            # s_m*a_shear - s_h*a_shear*ri_st = 1.0
            # to solve a_shear at equilibrium
            # NOTE may fail/return incorrect solution for ri_st < -4

            def cost(a_shear):
                a_buoy = ri_st*a_shear
                s_m, s_h = self.eval_funcs(a_buoy, a_shear)
                res = s_m*a_shear - s_h*a_buoy - 1.0
                return res**2
            p = minimize(cost, 1.0)
            assert p.success, 'solving alpha_shear failed, Ri_st={:}'.format(ri_st)
            a_shear = p.x[0]
        else:
            # B) solve analytically
            # compute alpha_shear for equilibrium condition (Umlauf and Buchard, 2005, eq A.19)
            # aM^2 (-d5 + n2 - (d3 - n1 + nb2 )Ri - (d4 + nb1)Ri^2) + aM (-d2 + n0 - (d1 + nb0)Ri) - d0 = 0
            # NOTE this is more robust method
            a = -self.d5 + self.n2 - (self.d3 - self.n1 + self.nb2)*ri_st - (self.d4 + self.nb1)*ri_st**2
            b = -self.d2 + self.n0 - (self.d1 + self.nb0)*ri_st
            c = -self.d0
            a_shear = (-b + numpy.sqrt(b**2 - 4*a*c))/2/a

        return a_shear

    def compute_c3_minus(self, c1, c2, ri_st):
        r"""
        Compute c3_minus parameter from c1 and c2 parameters.

        c3_minus is solved from equation

        .. math::
            Ri_{st} = \frac{s_m}{s_h} \frac{c2 - c1}{c2 - c3_{-}}

        where :math:`Ri_{st}` is the steady state gradient Richardson number.
        (see Burchard and Bolding, 2001, eq 32)
        """
        a_shear = self.compute_alpha_shear_steady(ri_st, analytical=False)

        # compute aN from Ri_st and aM, Ri_st = aN/aM
        a_buoy = ri_st*a_shear

        # evaluate stability functions for equilibrium conditions
        s_m, s_h = self.eval_funcs(a_buoy, a_shear)

        # compute c3 minus from Umlauf and Burchard (2005)
        c3_minus = c2 - (c2 - c1)*s_m/s_h/ri_st

        # check error in ri_st
        err = s_m/s_h*(c2 - c1)/(c2 - c3_minus) - ri_st
        assert numpy.abs(err) < 1e-5, 'steady state gradient Richardson number does not match'

        return c3_minus

    def compute_cmu0(self, analytical=True):
        r"""
        Compute parameter :math:`c_\mu^0`

        See: Umlauf and Buchard (2005) eq A.22

        :kwarg bool analytical: If True (default), solve analytically using the
            coefficients of the stability function. Otherwise, solve
            :math:`\alpha_M` numerically from the equilibrium condition.
        """
        a_buoy = 0.0
        if analytical:
            a = self.d5 - self.n2
            b = self.d2 - self.n0
            c = self.d0
            a_shear = (-b - numpy.sqrt(b**2 - 4*a*c))/2/a
            s_m, s_h = self.eval_funcs(a_buoy, a_shear)
            cm0 = s_m**0.25
        else:

            def cost(a_shear):
                s_m, s_h = self.eval_funcs(a_buoy, a_shear)
                res = s_m*a_shear - 1.0
                return res**2
            p = minimize(cost, 1.0)
            assert p.success, 'solving alpha_shear failed'
            a_shear = p.x[0]
            s_m, s_h = self.eval_funcs(a_buoy, a_shear)
            cm0 = s_m**0.25
        return cm0

    def compute_kappa(self, sigma_psi, cmu0, n, c1, c2):
        r"""
        Computes von Karman constant from the Psi Schmidt number.

        See: Umlauf and Burchard (2003) eq (14)

        :arg sigma_psi: Psi Schmidt number
        :arg cmu0, n, c1, c2: GLS model parameters
        """
        return cmu0 / numpy.abs(n) * numpy.sqrt(sigma_psi * (c2 - c1))

    def compute_sigma_psi(self, kappa, cmu0, n, c1, c2):
        r"""
        Computes the Psi Schmidt number from von Karman constant.

        See: Umlauf and Burchard (2003) eq (14)

        :arg kappa: von Karman constant
        :arg cmu0, n, c1, c2: GLS model parameters
        """
        return (n * kappa)**2 / (cmu0**2 * (c2 - c1))

    def compute_length_clim(self, cmu0, ri_st):
        r"""
        Computes the Galpering length scale limit.

        :arg cmu0: parameter :math:`c_\mu^0`
        :arg ri_st: gradient Richardson number
        """
        a_shear = self.compute_alpha_shear_steady(ri_st)

        # compute aN from Ri_st and aM, Ri_st = aN/aM
        a_buoy = ri_st*a_shear

        clim = cmu0**3.0 * numpy.sqrt(a_buoy/2)
        return clim

    def get_alpha_buoy_min(self):
        r"""
        Compute minimum normalized buoyancy frequency :math:`\alpha_N`

        See: Umlauf and Buchard (2005), Table 3
        """
        # G = epsilon case, this is used in GOTM
        an_min = 0.5*(numpy.sqrt((self.d1 + self.nb0)**2. - 4.*self.d0*(self.d4 + self.nb1)) - (self.d1 + self.nb0))/(self.d4 + self.nb1)
        # eq. (47)
        # an_min = self.alpha_buoy_min
        return an_min

    def get_alpha_shear_max(self, alpha_buoy, alpha_shear):
        r"""
        Compute maximum normalized shear frequency :math:`\alpha_M`

        from Umlauf and Buchard (2005) eq (44)

        :arg alpha_buoy: normalized buoyancy frequency :math:`\alpha_N`
        :arg alpha_shear: normalized shear frequency :math:`\alpha_M`
        """
        as_max_n = (self.d0*self.n0
                    + (self.d0*self.n1 + self.d1*self.n0)*alpha_buoy
                    + (self.d1*self.n1 + self.d4*self.n0)*alpha_buoy**2
                    + self.d4*self.n1*alpha_buoy**3)
        as_max_d = (self.d2*self.n0
                    + (self.d2*self.n1 + self.d3*self.n0)*alpha_buoy
                    + self.d3*self.n1*alpha_buoy**2)
        return as_max_n/as_max_d

    def get_alpha_buoy_smooth_min(self, alpha_buoy):
        r"""
        Compute smoothed minimum for normalized buoyancy frequency

        See: Burchard and Petersen (1999), eq (19)

        :arg alpha_buoy: normalized buoyancy frequency :math:`\alpha_N`
        """
        return alpha_buoy - (alpha_buoy - self.alpha_buoy_crit)**2/(alpha_buoy + self.get_alpha_buoy_min() - 2*self.alpha_buoy_crit)

    def eval_funcs(self, alpha_buoy, alpha_shear):
        r"""
        Evaluate (unlimited) stability functions

        See: Burchard and Petersen (1999) eqns (30) and (31)

        :arg alpha_buoy: normalized buoyancy frequency :math:`\alpha_N`
        :arg alpha_shear: normalized shear frequency :math:`\alpha_M`
        :returns: :math:`S_m`, :math:`S_\rho`
        """
        den = self.d0 + self.d1*alpha_buoy + self.d2*alpha_shear + self.d3*alpha_buoy*alpha_shear + self.d4*alpha_buoy**2 + self.d5*alpha_shear**2
        c_mu = (self.n0 + self.n1*alpha_buoy + self.n2*alpha_shear) / den
        c_mu_p = (self.nb0 + self.nb1*alpha_buoy + self.nb2*alpha_shear) / den
        return c_mu, c_mu_p

    def evaluate(self, shear2, buoy2, k, eps):
        r"""
        Evaluate stability functions from dimensional variables.

        Applies limiters on :math:`\alpha_N` and :math:`\alpha_M`.

        :arg shear2: shear frequency squared, :math:`M^2`
        :arg buoy2: buoyancy frequency squared,:math:`N^2`
        :arg k: turbulent kinetic energy, :math:`k`
        :arg eps: TKE dissipation rate, :math:`\varepsilon`
        """
        alpha_buoy, alpha_shear = compute_normalized_frequencies(shear2, buoy2, k, eps)
        if self.lim_alpha_buoy:
            # limit min (negative) alpha_buoy (Umlauf and Burchard, 2005)
            if not self.smooth_alpha_buoy_lim:
                # crop at minimum value
                numpy.maximum(alpha_buoy, self.get_alpha_buoy_min(), alpha_buoy)
            else:
                # do smooth limiting instead (Buchard and Petersen, 1999, eq 19)
                ab_smooth_min = self.get_alpha_buoy_smooth_min(alpha_buoy)
                # NOTE this must be applied to values alpha_buoy < ab_crit only!
                ix = alpha_buoy < self.alpha_buoy_crit
                alpha_buoy[ix] = ab_smooth_min[ix]

        if self.lim_alpha_shear:
            # limit max alpha_shear (Umlauf and Burchard, 2005, eq 44)
            as_max = self.get_alpha_shear_max(alpha_buoy, alpha_shear)
            numpy.minimum(alpha_shear, as_max, alpha_shear)

        return self.eval_funcs(alpha_buoy, alpha_shear)


class GOTMStabilityFunctionBase(StabilityFunctionBase, ABC):
    """
    Base class for stability functions defined in Umlauf and Buchard (2005)
    """
    @property
    @abstractmethod
    def cc1(self):
        pass

    @property
    @abstractmethod
    def cc2(self):
        pass

    @property
    @abstractmethod
    def cc3(self):
        pass

    @property
    @abstractmethod
    def cc4(self):
        pass

    @property
    @abstractmethod
    def cc5(self):
        pass

    @property
    @abstractmethod
    def cc6(self):
        pass

    @property
    @abstractmethod
    def cb1(self):
        pass

    @property
    @abstractmethod
    def cb2(self):
        pass

    @property
    @abstractmethod
    def cb3(self):
        pass

    @property
    @abstractmethod
    def cb4(self):
        pass

    @property
    @abstractmethod
    def cb5(self):
        pass

    @property
    @abstractmethod
    def cbb(self):
        pass

    def __init__(self, lim_alpha_shear=True, lim_alpha_buoy=True,
                 smooth_alpha_buoy_lim=True, alpha_buoy_crit=-1.2):
        r"""
        :kwarg bool lim_alpha_shear: limit maximum :math:`\alpha_M` values
            (see Umlauf and Burchard (2005) eq. 44)
        :kwarg bool lim_alpha_buoy: limit minimum (negative) :math:`\alpha_N` values
            (see Umlauf and Burchard (2005))
        :kwarg bool smooth_alpha_buoy_lim: if :math:`\alpha_N` is limited, apply a
            smooth limiter (see Burchard and Bolding (2001) eq. 19). Otherwise
            :math:`\alpha_N` is clipped at minimum value.
        :kwarg float alpha_buoy_crit: parameter for :math:`\alpha_N` smooth limiter
        """
        super().__init__(lim_alpha_shear, lim_alpha_buoy,
                         smooth_alpha_buoy_lim, alpha_buoy_crit)

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
        self.nb0 = 12.0*self.ab3*self.nn**3*self.nb
        self.nb1 = 12.0*self.a5*self.ab3**2*self.nn**2
        self.nb2 = 9.0*self.a1*self.ab3*(self.ab1 - self.ab2)*self.nn**2 +\
            (6.0*self.a1*(self.a2 - 3.0*self.a3) - 4.0*(self.a2**2 - 3.0*self.a3**2))*self.ab3*self.nn*self.nb


class CanutoStabilityFunctionBase(StabilityFunctionBase, ABC):
    """
    Base class for original Canuto stability function.
    """
    @property
    @abstractmethod
    def l1(self):
        pass

    @property
    @abstractmethod
    def l2(self):
        pass

    @property
    @abstractmethod
    def l3(self):
        pass

    @property
    @abstractmethod
    def l4(self):
        pass

    @property
    @abstractmethod
    def l5(self):
        pass

    @property
    @abstractmethod
    def l6(self):
        pass

    @property
    @abstractmethod
    def l7(self):
        pass

    @property
    @abstractmethod
    def l8(self):
        pass

    def __init__(self, lim_alpha_shear=True, lim_alpha_buoy=True,
                 smooth_alpha_buoy_lim=True, alpha_buoy_crit=-1.2):
        r"""
        :kwarg bool lim_alpha_shear: limit maximum :math:`\alpha_M` values
            (see Umlauf and Burchard (2005) eq. 44)
        :kwarg bool lim_alpha_buoy: limit minimum (negative) :math:`\alpha_N` values
            (see Umlauf and Burchard (2005))
        :kwarg bool smooth_alpha_buoy_lim: if :math:`\alpha_N` is limited, apply a
            smooth limiter (see Burchard and Bolding (2001) eq. 19). Otherwise
            :math:`\alpha_N` is clipped at minimum value.
        :kwarg float alpha_buoy_crit: parameter for :math:`\alpha_N` smooth limiter
        """
        super().__init__(lim_alpha_shear, lim_alpha_buoy,
                         smooth_alpha_buoy_lim, alpha_buoy_crit)

        self.s0 = 1.5*self.l1*self.l5**2
        self.s1 = -self.l4*(self.l6 + self.l7) + 2*self.l4*self.l5*(self.l1 - self.l2/3.0 - self.l3) + 1.5*self.l1*self.l5*self.l8
        self.s2 = -3.0/8*self.l1*(self.l6**2 - self.l7**2)
        self.s4 = 2*self.l5
        self.s5 = 2*self.l4
        self.s6 = 2.0/3*self.l5*(3*self.l3**2 - self.l2**2) - 0.5*self.l5*self.l1*(3*self.l3 - self.l2) + 0.75*self.l1*(self.l6 - self.l7)

        self.dd0 = 3*self.l5**2
        self.dd1 = self.l5*(7*self.l4 + 3*self.l8)
        self.dd2 = self.l5**2*(3*self.l3**2 - self.l2**2) - 0.75*(self.l6**2 - self.l7**2)
        self.dd3 = self.l4*(4*self.l4 + 3*self.l8)
        self.dd5 = 0.25*(self.l2**2 - 3*self.l3**2)*(self.l6**2 - self.l7**2)
        self.dd4 = self.l4*(self.l2*self.l6 - 3*self.l3*self.l7 - self.l5*(self.l2**2 - self.l3**2)) + self.l5*self.l8*(3*self.l3**2 - self.l2**2)

        # unit conversion
        self.alpha_scalar = 4
        self.cu_scalar = 2

        self.d0 = self.dd0
        self.d1 = self.alpha_scalar*self.dd1
        self.d2 = self.alpha_scalar*self.dd2
        self.d3 = self.alpha_scalar**2*self.dd4
        self.d4 = self.alpha_scalar**2*self.dd3
        self.d5 = self.alpha_scalar**2*self.dd5
        self.n0 = self.cu_scalar*self.s0
        self.n1 = self.cu_scalar*self.alpha_scalar*self.s1
        self.n2 = self.cu_scalar*self.alpha_scalar*self.s2
        self.nb0 = self.cu_scalar*self.s4
        self.nb1 = self.cu_scalar*self.alpha_scalar*self.s5
        self.nb2 = self.cu_scalar*self.alpha_scalar*self.s6

    def eval_funcs_new(self, alpha_buoy, alpha_shear):
        r"""
        Evaluate (unlimited) stability functions

        From Canuto et al (2001)

        :arg alpha_buoy: normalized buoyancy frequency :math:`\alpha_N`
        :arg alpha_shear: normalized shear frequency :math:`\alpha_M`
        """
        tN2 = self.alpha_scalar*alpha_buoy
        tS2 = self.alpha_scalar*alpha_shear
        dsm = self.s0 + self.s1*tN2 + self.s2*tS2
        dsh = self.s4 + self.s5*tN2 + self.s6*tS2
        den = self.dd0 + self.dd1*tN2 + self.dd2*tS2 + self.dd3*tN2**2 + self.dd4*tN2*tS2 + self.dd5*tS2**2
        sm = self.cu_scalar*dsm/den
        sh = self.cu_scalar*dsh/den
        return sm, sh


class ChengStabilityFunctionBase(StabilityFunctionBase):
    """
    Base class for original Cheng stability function.
    """
    @property
    @abstractmethod
    def l1(self):
        pass

    @property
    @abstractmethod
    def l2(self):
        pass

    @property
    @abstractmethod
    def l3(self):
        pass

    @property
    @abstractmethod
    def l4(self):
        pass

    @property
    @abstractmethod
    def l5(self):
        pass

    @property
    @abstractmethod
    def l6(self):
        pass

    @property
    @abstractmethod
    def l7(self):
        pass

    @property
    @abstractmethod
    def l8(self):
        pass

    def __init__(self, lim_alpha_shear=True, lim_alpha_buoy=True,
                 smooth_alpha_buoy_lim=True, alpha_buoy_crit=-1.2):
        r"""
        :kwarg bool lim_alpha_shear: limit maximum :math:`\alpha_M` values
            (see Umlauf and Burchard (2005) eq. 44)
        :kwarg bool lim_alpha_buoy: limit minimum (negative) :math:`\alpha_N` values
            (see Umlauf and Burchard (2005))
        :kwarg bool smooth_alpha_buoy_lim: if :math:`\alpha_N` is limited, apply a
            smooth limiter (see Burchard and Bolding (2001) eq. 19). Otherwise
            :math:`\alpha_N` is clipped at minimum value.
        :kwarg float alpha_buoy_crit: parameter for :math:`\alpha_N` smooth limiter
        """
        super().__init__(lim_alpha_shear, lim_alpha_buoy,
                         smooth_alpha_buoy_lim, alpha_buoy_crit)

        self.s0 = 0.5*self.l1
        self.s1 = -1.0/3*self.l4*self.l5**-2*(self.l6 + self.l7) + 2.0/3*self.l4/self.l5*(self.l1 - 1.0/3*self.l2 - self.l3) + 0.5*self.l1/self.l5*self.l8
        self.s2 = 1.0/8*self.l1*self.l5**-2*(self.l6**2 - self.l7**2)
        self.s4 = 2.0/3/self.l5
        self.s5 = 2.0/3*self.l4*self.l5**-2
        self.s6 = 2.0/3/self.l5*(self.l3**2 - 1.0/3*self.l2**2) - 0.5*self.l1/self.l5*(self.l3 - 1.0/3*self.l2)

        self.dd0 = 1
        self.dd1 = (7.0/3*self.l4 + self.l8)/self.l5
        self.dd2 = (self.l3**2 - 1.0/3*self.l2**2) - 0.25*self.l5**-2*(self.l6**2 - self.l7**2)
        self.dd3 = 1.0/3*self.l4*self.l5**-2*(4*self.l4 + 3*self.l8)
        self.dd4 = 1.0/3*self.l4*self.l5**-2*(self.l2*self.l6 - 3*self.l3*self.l7 - self.l5*(self.l2**2 - self.l3**2)) + self.l8*(self.l3**2 - 1.0/3*self.l2**2)/self.l5
        self.dd5 = -1.0/4*self.l5**-2*(self.l3**2 - 1.0/3*self.l2**2)*(self.l6**2 - self.l7**2)

        # unit conversion
        self.alpha_scalar = 4
        self.cu_scalar = 2

        self.d0 = self.dd0
        self.d1 = self.alpha_scalar*self.dd1
        self.d2 = self.alpha_scalar*self.dd2
        self.d3 = self.alpha_scalar**2*self.dd4
        self.d4 = self.alpha_scalar**2*self.dd3
        self.d5 = self.alpha_scalar**2*self.dd5
        self.n0 = self.cu_scalar*self.s0
        self.n1 = self.cu_scalar*self.alpha_scalar*self.s1
        self.n2 = self.cu_scalar*self.alpha_scalar*self.s2
        self.nb0 = self.cu_scalar*self.s4
        self.nb1 = self.cu_scalar*self.alpha_scalar*self.s5
        self.nb2 = self.cu_scalar*self.alpha_scalar*self.s6

    def eval_funcs_new(self, alpha_buoy, alpha_shear):
        r"""
        Evaluate (unlimited) stability functions

        From Canuto et al (2001)

        :arg alpha_buoy: normalized buoyancy frequency :math:`\alpha_N`
        :arg alpha_shear: normalized shear frequency :math:`\alpha_M`
        """
        tN2 = self.alpha_scalar*alpha_buoy
        tS2 = self.alpha_scalar*alpha_shear
        dsm = self.s0 + self.s1*tN2 + self.s2*tS2
        dsh = self.s4 + self.s5*tN2 + self.s6*tS2
        den = self.dd0 + self.dd1*tN2 + self.dd2*tS2 + self.dd3*tN2**2 + self.dd4*tN2*tS2 + self.dd5*tS2**2
        sm = self.cu_scalar*dsm/den
        sh = self.cu_scalar*dsh/den
        return sm, sh


class StabilityFunctionCanutoA(CanutoStabilityFunctionBase):
    """
    Canuto A stability function as defined in the Canuto (2001) paper.
    """
    l1 = 0.107
    l2 = 0.0032
    l3 = 0.0864
    l4 = 0.12
    l5 = 11.9
    l6 = 0.4
    l7 = 0
    l8 = 0.48
    name = 'Canuto A'


class StabilityFunctionCanutoB(CanutoStabilityFunctionBase):
    """
    Canuto B stability function as defined in the Canuto (2001) paper.
    """
    l1 = 0.127
    l2 = 0.00336
    l3 = 0.0906
    l4 = 0.101
    l5 = 11.2
    l6 = 0.4
    l7 = 0
    l8 = 0.318
    name = 'Canuto B'


class StabilityFunctionCheng(ChengStabilityFunctionBase):
    """
    Cheng stability function as defined in the Cheng et al (2002) paper.
    """
    l1 = 0.107
    l2 = 0.0032
    l3 = 0.0864
    l4 = 0.1
    l5 = 11.04
    l6 = 0.786
    l7 = 0.643
    l8 = 0.547
    name = 'Cheng'


class GOTMStabilityFunctionCanutoA(GOTMStabilityFunctionBase):
    """
    Canuto et al. (2001) version A stability functions

    Parameters are from Umlauf and Buchard (2005), Table 1
    """
    cc1 = 5.0000
    cc2 = 0.8000
    cc3 = 1.9680
    cc4 = 1.1360
    cc5 = 0.0000
    cc6 = 0.4000
    cb1 = 5.9500
    cb2 = 0.6000
    cb3 = 1.0000
    cb4 = 0.0000
    cb5 = 0.3333
    cbb = 0.7200
    name = 'Canuto A'


class GOTMStabilityFunctionCanutoB(GOTMStabilityFunctionBase):
    """
    Canuto et al. (2001) version B stability functions

    Parameters are from Umlauf and Buchard (2005), Table 1
    """
    cc1 = 5.0000
    cc2 = 0.6983
    cc3 = 1.9664
    cc4 = 1.0940
    cc5 = 0.0000
    cc6 = 0.4950
    cb1 = 5.6000
    cb2 = 0.6000
    cb3 = 1.0000
    cb4 = 0.0000
    cb5 = 0.3333
    cbb = 0.4770
    name = 'Canuto B'


class GOTMStabilityFunctionKanthaClayson(GOTMStabilityFunctionBase):
    """
    Kantha and Clayson (1994) quasi-equilibrium stability functions

    Parameters are from Umlauf and Buchard (2005), Table 1
    """
    cc1 = 6.0000
    cc2 = 0.3200
    cc3 = 0.0000
    cc4 = 0.0000
    cc5 = 0.0000
    cc6 = 0.0000
    cb1 = 3.7280
    cb2 = 0.7000
    cb3 = 0.7000
    cb4 = 0.0000
    cb5 = 0.2000
    cbb = 0.6102
    name = 'Kantha Clayson'


class GOTMStabilityFunctionCheng(GOTMStabilityFunctionBase):
    """
    Cheng et al. (2002) quasi-equilibrium stability functions

    Parameters are from Umlauf and Buchard (2005), Table 1
    """
    cc1 = 5.0000
    cc2 = 0.7983
    cc3 = 1.9680
    cc4 = 1.1360
    cc5 = 0.0000
    cc6 = 0.5000
    cb1 = 5.5200
    cb2 = 0.2134
    cb3 = 0.3570
    cb4 = 0.0000
    cb5 = 0.3333
    cbb = 0.8200
    name = 'Cheng'
