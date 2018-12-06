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
from __future__ import absolute_import
import numpy as np

__all__ = ('StabilityFunction',
           'StabilityFunctionCanutoA',
           'StabilityFunctionCanutoB',
           'StabilityFunctionKanthaClayson',
           'StabilityFunctionCheng',
           'compute_normalized_frequencies'
           )


def compute_normalized_frequencies(shear2, buoy2, k, eps):
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
    # print_output('{:8s} {:8.3e} {:8.3e}'.format('M2', shear2.min(), shear2.max()))
    # print_output('{:8s} {:8.3e} {:8.3e}'.format('N2', buoy2.min(), buoy2.max()))
    # print_output('{:8s} {:10.3e} {:10.3e}'.format('a_buoy', alpha_buoy.min(), alpha_buoy.max()))
    # print_output('{:8s} {:10.3e} {:10.3e}'.format('a_shear', alpha_shear.min(), alpha_shear.max()))
    return alpha_buoy, alpha_shear


class StabilityFunction(object):
    """Base class for all stability functions"""
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

    def compute_alpha_shear_steady(self, ri_st, analytical=True):
        if not analytical:
            # A) solve numerically
            # use equilibrium equation (Umlauf and Buchard, 2005, eq A.15)
            # s_m*a_shear - s_h*a_shear*ri_st = 1.0
            # to solve a_shear at equilibrium
            # NOTE may fail/return incorrect solution for ri_st < -4
            from scipy.optimize import minimize

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
            a_shear = (-b + np.sqrt(b**2 - 4*a*c))/2/a

        return a_shear

    def compute_c3_minus(self, c1, c2, ri_st):
        r"""
        Compute c3_minus parameter from c1, c2 and stability functions.

        c3_minus is solved from equation

        .. math::
            Ri_{st} = \frac{s_m}{s_h} \frac{c2 - c1}{c2 - c3_minus}

        where :math:`Ri_{st}` is the steady state gradient Richardson number.
        (see Burchard and Bolding, 2001, eq 32)
        """
        a_shear = self.compute_alpha_shear_steady(ri_st)

        # compute aN from Ri_st and aM, Ri_st = aN/aM
        a_buoy = ri_st*a_shear

        # evaluate stability functions for equilibrium conditions
        s_m, s_h = self.eval_funcs(a_buoy, a_shear)

        # compute c3 minus from Umlauf and Burchard (2005)
        c3_minus = c2 - (c2 - c1)*s_m/s_h/ri_st

        # check error in ri_st
        err = s_m/s_h*(c2 - c1)/(c2 - c3_minus) - ri_st
        assert np.abs(err) < 1e-5, 'steady state gradient Richardson number does not match'

        return c3_minus

    def compute_cmu0(self):
        """
        Computes the paramenter c_mu_0 from stability function parameters

        Umlauf and Buchard (2005) eq A.22
        """
        cm0 = ((self.a2**2 - 3*self.a3**2 + 3*self.a1*self.nn)/(3 * self.nn**2))**0.25
        return cm0

    def compute_kappa(self, sigma_psi, n, c1, c2):
        """
        Computes von Karman constant from the Psi Schmidt number.

        n, c1, c2 are GLS model parameters.

        from Umlauf and Burchard (2003) eq (14)
        """
        kappa = np.sqrt(sigma_psi * self.compute_cmu0()**2 * (c2 - c1)/(n**2))
        return kappa

    def compute_length_clim(self, cm0, ri_st):
        """
        Computes the Galpering lenght scale limit.
        """
        a_shear = self.compute_alpha_shear_steady(ri_st)

        # compute aN from Ri_st and aM, Ri_st = aN/aM
        a_buoy = ri_st*a_shear

        clim = cm0**3.0 * np.sqrt(a_buoy/2)
        return clim

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
        r"""
        Compute maximum alpha shear

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
        Compute smoothed alpha_buoy minimum

        from Burchard and Petersen (1999) eq (19)

        :arg alpha_buoy: normalized buoyancy frequency :math:`\alpha_N`
        """
        return alpha_buoy - (alpha_buoy - self.alpha_buoy_crit)**2/(alpha_buoy + self.get_alpha_buoy_min() - 2*self.alpha_buoy_crit)

    def eval_funcs(self, alpha_buoy, alpha_shear):
        r"""
        Evaluate (unlimited) stability functions

        from Burchard and Petersen (1999) eqns (30) and (31)

        :arg alpha_buoy: normalized buoyancy frequency :math:`\alpha_N`
        :arg alpha_shear: normalized shear frequency :math:`\alpha_M`
        """
        den = self.d0 + self.d1*alpha_buoy + self.d2*alpha_shear + self.d3*alpha_buoy*alpha_shear + self.d4*alpha_buoy**2 + self.d5*alpha_shear**2
        c_mu = (self.n0 + self.n1*alpha_buoy + self.n2*alpha_shear) / den
        c_mu_p = (self.nb0 + self.nb1*alpha_buoy + self.nb2*alpha_shear) / den
        return c_mu, c_mu_p

    def evaluate(self, shear2, buoy2, k, eps):
        """
        Evaluates stability functions. Applies limiters on alpha_buoy and alpha_shear.

        :arg shear2: :math:`M^2`
        :arg buoy2: :math:`N^2`
        :arg k: turbulent kinetic energy
        :arg eps: TKE dissipation rate
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
    """
    Canuto et al. (2001) version A stability functions

    Parameters are from Umlauf and Buchard (2005), Table 1
    """
    def __init__(self, lim_alpha_shear=True, lim_alpha_buoy=True,
                 smooth_alpha_buoy_lim=True, alpha_buoy_crit=-1.2):
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
    """
    Canuto et al. (2001) version B stability functions

    Parameters are from Umlauf and Buchard (2005), Table 1
    """
    def __init__(self, lim_alpha_shear=True, lim_alpha_buoy=True,
                 smooth_alpha_buoy_lim=True, alpha_buoy_crit=-1.2):
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
    """
    Kantha and Clayson (1994) quasi-equilibrium stability functions

    Parameters are from Umlauf and Buchard (2005), Table 1
    """
    def __init__(self, lim_alpha_shear=True, lim_alpha_buoy=True,
                 smooth_alpha_buoy_lim=True, alpha_buoy_crit=-1.2):
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
    """
    Cheng et al. (2002) quasi-equilibrium stability functions

    Parameters are from Umlauf and Buchard (2005), Table 1
    """
    def __init__(self, lim_alpha_shear=True, lim_alpha_buoy=True,
                 smooth_alpha_buoy_lim=True, alpha_buoy_crit=-1.2):
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
