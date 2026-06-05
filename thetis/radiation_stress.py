from firedrake import *
from .utility import *
import numpy as np

class clrtxt:
    purple = '\033[95m'
    cyan = '\033[96m'
    darkcyan = '\033[36m'
    blue = '\033[94m'
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    bold = '\033[1m'
    underline = '\033[4m'
    end = '\033[0m'


class WaveEffectsOnCurrents(object):
    """
    Calculates the effects of waves on currents
    """
    def __init__(self, mesh2d, solver_options, solver_fields):
        """
        """
        self.mesh = mesh2d
        self.comm = mesh2d.comm
        self.options = solver_options
        self.fields = solver_fields

        self.depth = DepthExpression(
            self.fields.bathymetry_2d,
            use_nonlinear_equations=self.options.use_nonlinear_equations,
            use_wetting_and_drying=self.options.use_wetting_and_drying,
            wetting_and_drying_alpha=self.options.wetting_and_drying_alpha
        )

        self.rho0 = physical_constants['rho0']
        self.g_grav = physical_constants['g_grav']

        # Define functionspaces
        self.P1_2d = get_functionspace(self.mesh, "CG", 1)
        self.P1v_2d = VectorFunctionSpace(self.mesh, "CG", 1)


    def update(self, grad_rad_stress=None, l_wave=None, dir_wave = None,
                h_wave=None, qb=None, c1=1.0):
        """
        Update the wave effects on currents
        """

        # If using the gradient of radiation stress as calculated by SWAN
        if self.options.use_swan:
            message = clrtxt.bold + clrtxt.red + "Terminating program: " +\
                clrtxt.end + "The option to use the gradient of radiation " +\
                "as calculated by SWAN for the Wave Effects On Currents " +\
                "(WEOC) hasn't been implemented yet"
            raise SystemExit(message)

        # Use the Mellor (2015) formulation to calculate the gradient of radiation
        # stress from waves
        else:
            # Update the radiation stress formulation by multiplying with the
            # ramp coefficient
            h_wave.dat.data[:] = h_wave.dat.data[:] * c1
            dir_wave.dat.data[:] = dir_wave.dat.data[:] * c1
            l_wave.dat.data[:] = l_wave.dat.data[:] * c1
            qb.dat.data[:] = qb.dat.data[:] * c1

            # Multiply with coeff
            self.mellor(l_wave, dir_wave, h_wave)

        # Include wave roller effects
        if self.options.use_roller:
            self.roller_effects(l_wave, dir_wave, h_wave, qb)


    def swan_results(self, grad_rad_stress):
        """
        Not implemented yet. The plan is to use the already calculated SWAN's
        radiation stress directly and not calculate it from scratch

        Args:
            grad_rad_stress:

        Returns:
        """


        self.fields.rad_stress_2d.interpolate(Constant(1.)*grad_rad_stress)

        return
    

    def mellor(self, l_wave, dir_wave, h_wave):

        """
        Use the formulation found in
        Mellor, G. (2015), "A combined derivation of the integrated and \
        vertically resolved, coupled wave-current equations", Journal of \
        Physical Oceanography 45(6), 1453-1463
            $ S_{ij} = E * ( c_g/c * k_i*k_j/k^2 + \delta_ij*(c_g/c - 0.5) ) $
        where
            E is the wave energy;
            c_g : the group velocity;
            c : the phase velocity;
            k : the wavenumber;
            \delta_ij : the Kronecker delta function (=1 when i=j, else =0);
            i,j : the x- or y-direction;
            S_ij : the vertically integrated radiation stress
        to calculate the gradient of radiation stress and update the relevant
        Thetis field

        Note:
            It updates the self.fields.rad_stress_2d

        Args:
            l_wave: Wavelength [m]
            dir_wave : Wave direction [o]
            h_wave : Wave height [m]

        Returns:
            None
        """
        # Define limiters
        # Minimum wavelength [m]
        l_min = Constant(1.)
        # Maximum wavenumber*(total depth)
        kh_max = Constant(5.)

        # Apply limiter in the wavelength
        l_wave.interpolate(Max(l_wave, l_min))

        # Calculate the wavenumber
        k = Function(self.P1_2d).interpolate(Constant(2*pi)/l_wave)

        #Convert wave direction to radians
        dir_wave.interpolate(dir_wave*Constant(pi/180.))

        # Angle
        theta = Function(self.P1_2d).interpolate(dir_wave)

        ## Wavenumber components
        # x-component
        kx = Function(self.P1_2d).interpolate(k*cos(theta))
        # y-component
        ky = Function(self.P1_2d).interpolate(k*sin(theta))

        #Calculate the wave energy
        if self.options.use_monochromatic:
            # From Kudale & Bhalerao (2015) : "Design wave height should be
            # average of the highest 1/10 of the waves, i.e. H_(1/10), instead
            # of just H while representing waves by monochromatic wave trains
            # H_(1/10) = 1.27*H
            energy = Function(self.P1_2d).interpolate(
                Constant(1/8*1.27**2)*self.rho0*self.g_grav*h_wave**2
            )
        else:
            energy = Function(self.P1_2d).interpolate(
                Constant( 1 / 8 ) * self.rho0 * self.g_grav * h_wave ** 2
            )

        # Calculate total depth
        h_total = self.depth.get_total_depth(self.fields.elev_2d)
        #Calculate the (wavenumber)*(total_depth) and apply limiter
        kh = Function(self.P1_2d).interpolate(
            Min(k*h_total+Constant(10**(-14)), kh_max)
        )

        #Calculate (group velocity)/(phase velocity)
        n = Function(self.P1_2d).interpolate(
            Constant(0.5)+kh/sinh(Constant(2.)*kh)
        )

        #Calculate radiation stress
        sxx = Function(self.P1_2d).interpolate(
            energy*(n*(kx**2/k**2+Constant(1.))-Constant(0.5))
        )
        sxy = Function(self.P1_2d).interpolate(energy*(n*(kx*ky/k**2)))
        syy = Function(self.P1_2d).interpolate(
            energy*(n*(ky**2/k**2+Constant(1.))-Constant(0.5))
        )

        #Calculate gradient of radiation stress
        sxx_dx = Function(self.P1_2d).interpolate(sxx.dx(0))
        syy_dy = Function(self.P1_2d).interpolate(syy.dx(1))
        sxy_dx = Function(self.P1_2d).interpolate(sxy.dx(0))
        sxy_dy = Function(self.P1_2d).interpolate(sxy.dx(1))

        #Gradient of radiation stress components
        s_x = Function(self.P1_2d).interpolate(-sxx_dx-sxy_dy)
        s_y = Function(self.P1_2d).interpolate(-syy_dy-sxy_dx)

        # Update field
        self.fields.rad_stress_2d.interpolate(as_vector([s_x, s_y]))
        
        return


    def roller_effects(self, l_wave, dir_wave, h_wave, qb):
        """
        Calculate the gradient of wave roller effects according to Svendsen \
        (1984) and Reniers & Battjes (1987):
            $ R_{ij} = 2 * E_r * k_i*k_j/k^2 $
        where
            $ E_r = \\rho * g * A_r * \sin{\phi} $ is the energy due to wave \
            roller interface calculate according to Duncan (1981);
            $ A_r = 0.9 * H^2 $ is the roller area (Svendsen 1984);
            $ \phi$ is the roller angle with $ \tan{phi} ~ 0.1 $ (Reniers & \
            Battjes 1997; Martins et al. 2018)
            $ k $ is the wavenumber;
            $ i, j$ are the $x-$ and $y-$direction respectively

        Note:
            The self.fields.roller_2d is updated

        Args:
            l_wave : Wavelength [m]
            dir_wave : Wave direction [o]
            h_wave : Wave height [m]
            qb : Percentage of wave breking [-]

        Returns:
            None
        """
        # Define limiters
        # Minimum wavelength [m]
        l_min = Constant(1.)
        # Maximum wavenumber*(total depth)
        kh_max = Constant(5.)

        # Apply limiter in the wavelength
        l_wave.interpolate(Max(l_wave, l_min))

        # Calculate the wavenumber
        k = Function(self.P1_2d).interpolate(Constant(2 * pi) / l_wave)

        # Convert wave direction to radians
        dir_wave.interpolate(dir_wave * Constant(pi / 180.))

        # Angle
        theta = Function(self.P1_2d).interpolate(dir_wave)

        ## Wavenumber components
        # x-component
        kx = Function(self.P1_2d).interpolate(k * cos(theta))
        # y-component
        ky = Function(self.P1_2d).interpolate(k * sin(theta))

        # Define constants
        sinb = 0.1

        if self.options.use_monochromatic:
            # Calculate roller wave area according to Svendsen 1984 (see \
            # Martins et al. 2018)
            A = Function(self.P1_2d).interpolate(
                Constant(0.9 * 1.27 ** 2) * h_wave ** 2 * qb)
        else:
            A = Function(self.P1_2d).interpolate(
                Constant(0.9) * (h_wave) ** 2 * qb
            )

        # Calculate the roller wave energy
        # energy = Function(self.P1_2d).interpolate(-self.g_grav*A*cp**2/l_wave)
        roller_en = Function(self.P1_2d).interpolate(
            -self.rho0 * self.g_grav * A * sinb
        )
        # Calculate total depth
        h_total = self.depth.get_total_depth(self.fields.elev_2d)
        # Calculate the (wavenumber)*(total_depth) and apply limiter
        kh = Function(self.P1_2d).interpolate(
            Min(k * h_total + Constant(10 ** (-14)), kh_max))

        # Calculate (group velocity)/(phase velocity)
        n = Function(self.P1_2d).interpolate(
            Constant(0.5) + kh / sinh(Constant(2.) * kh))

        sxx = Function(self.P1_2d).interpolate(
            Constant(2) * roller_en * kx ** 2 / k ** 2
        )
        sxy = Function(self.P1_2d).interpolate(
            Constant(2) * roller_en * kx * ky / k ** 2
        )
        syy = Function(self.P1_2d).interpolate(
            Constant(2) * roller_en * ky ** 2 / k ** 2
        )

        # Calculate gradient of radiation stress
        sxx_dx = Function(self.P1_2d).interpolate(sxx.dx(0))
        syy_dy = Function(self.P1_2d).interpolate(syy.dx(1))
        sxy_dx = Function(self.P1_2d).interpolate(sxy.dx(0))
        sxy_dy = Function(self.P1_2d).interpolate(sxy.dx(1))

        # Gradient of radiation stress components
        s_x = Function(self.P1_2d).interpolate(-sxx_dx - sxy_dy)
        s_y = Function(self.P1_2d).interpolate(-syy_dy - sxy_dx)

        # Update field
        self.fields.roller_2d.interpolate(as_vector([s_x, s_y]))

        return

