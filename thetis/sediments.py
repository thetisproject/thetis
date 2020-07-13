from .utility import *


class Corrective_Velocity_Factor:
    def __init__(self, depth, ksp, bed_reference_height, settling_velocity, ustar):

        a = Constant(bed_reference_height/2)

        kappa = physical_constants['von_karman']

        # correction factor to advection velocity in sediment concentration equation
        Bconv = conditional(depth > Constant(1.1)*ksp, ksp/depth, Constant(1/1.1))
        Aconv = conditional(depth > Constant(1.1)*a, a/depth, Constant(1/1.1))

        # take max of value calculated either by ksp or depth
        Amax = conditional(Aconv > Bconv, Aconv, Bconv)

        r1conv = Constant(1) - (1/kappa)*conditional(settling_velocity/ustar < Constant(1),
                                                     settling_velocity/ustar, Constant(1))

        Ione = conditional(r1conv > Constant(1e-8), (Constant(1) - Amax**r1conv)/r1conv,
                           conditional(r1conv < Constant(- 1e-8), (Constant(1) - Amax**r1conv)/r1conv, ln(Amax)))

        Itwo = conditional(r1conv > Constant(1e-8), -(Ione + (ln(Amax)*(Amax**r1conv)))/r1conv,
                           conditional(r1conv < Constant(- 1e-8), -(Ione + (ln(Amax)*(Amax**r1conv)))/r1conv,
                                       Constant(-0.5)*ln(Amax)**2))

        self.alpha = -(Itwo - (ln(Amax) - ln(30))*Ione)/(Ione * ((ln(Amax) - ln(30)) + Constant(1)))

        # final correction factor
        self.corr_vel_factor = Function(depth.function_space()).interpolate(conditional(conditional(self.alpha > Constant(1),
                                                                                                    Constant(1), self.alpha) < Constant(0),
                                                                                        Constant(0), conditional(
                                                                                            self.alpha > Constant(1), Constant(1), self.alpha)))

    def update(self):

        # final correction factor
        self.corr_vel_factor.interpolate(conditional(conditional(self.alpha > Constant(1), Constant(1), self.alpha)
                                                     < Constant(0), Constant(0), conditional(self.alpha > Constant(1),
                                                                                             Constant(1), self.alpha)))


class SedimentModel(object):
    def __init__(self, options, mesh2d, erosion, deposition, uv_init, elev_init, bathymetry_2d,
                 beta_fn, surbeta2_fn, alpha_secc_fn, viscosity_morph, rhos):

        """

        Set up a full morphological model simulation using as an initial condition the results of a hydrodynamic only model.

        :arg options: Model options.
        :type options: :class:`.ModelOptions2d` instance
        :arg mesh2d: :class:`Mesh` object of the 2D mesh
        :arg erosion: string to choose whether use model defined, user defined or none
        :arg deposition: string to choose whether use model defined, user defined or none
        :arg uv_init: Initial velocity for the simulation.
        :type uv_init: :class:`Function`
        :arg elev_init: Initial velocity for the simulation.
        :type elev_init: :class:`Function`
        :arg bathymetry_2d: Bathymetry of the domain. Bathymetry stands for
            the mean water depth (positive downwards).
        :type bathymetry_2d: :class:`Function`
        :arg beta_fn: slope effect magnitude parameter
        :arg surbeta2_fn: slope effect angle correction parameter
        :arg alpha_secc_fn: secondary current parameter
        :arg viscosity_morph: viscosity value in morphodynamic equations
        :arg rhos: sediment density

        """

        self.options = options
        self.solve_suspended = options.sediment_model_options.solve_suspended
        self.use_conservative = options.sediment_model_options.use_sediment_conservative_form
        self.use_advective_velocity = options.sediment_model_options.use_advective_velocity
        self.solve_bedload = options.sediment_model_options.solve_bedload
        self.use_angle_correction = options.sediment_model_options.use_angle_correction
        self.use_slope_mag_correction = options.sediment_model_options.use_slope_mag_correction
        self.use_secondary_current = options.sediment_model_options.use_secondary_current
        self.use_wetting_and_drying = options.use_wetting_and_drying

        self.average_size = options.sediment_model_options.average_sediment_size
        self.bed_reference_height = options.sediment_model_options.bed_reference_height
        self.wetting_alpha = options.wetting_and_drying_alpha
        self.rhos = rhos

        self.bathymetry_2d = bathymetry_2d

        # define function spaces
        self.P1_2d = get_functionspace(mesh2d, "DG", 1)
        self.V = get_functionspace(mesh2d, "CG", 1)
        self.vector_cg = VectorFunctionSpace(mesh2d, "CG", 1)

        # define parameters
        self.g = physical_constants['g_grav']
        self.rhow = physical_constants['rho0']
        kappa = physical_constants['von_karman']

        ksp = Constant(3*self.average_size)
        self.a = Constant(self.bed_reference_height/2)
        self.viscosity = Constant(viscosity_morph)

        # magnitude slope effect parameter
        self.beta = Constant(beta_fn)
        # angle correction slope effect parameters
        self.surbeta2 = Constant(surbeta2_fn)
        # secondary current parameter
        self.alpha_secc = Constant(alpha_secc_fn)

        # calculate critical shields parameter thetacr
        self.R = Constant(self.rhos/self.rhow - 1)

        self.dstar = Constant(self.average_size*((self.g*self.R)/(self.viscosity**2))**(1/3))
        if max(self.dstar.dat.data[:] < 1):
            print('ERROR: dstar value less than 1')
        elif max(self.dstar.dat.data[:] < 4):
            self.thetacr = Constant(0.24*(self.dstar**(-1)))
        elif max(self.dstar.dat.data[:] < 10):
            self.thetacr = Constant(0.14*(self.dstar**(-0.64)))
        elif max(self.dstar.dat.data[:] < 20):
            self.thetacr = Constant(0.04*(self.dstar**(-0.1)))
        elif max(self.dstar.dat.data[:] < 150):
            self.thetacr = Constant(0.013*(self.dstar**(0.29)))
        else:
            self.thetacr = Constant(0.055)

        # critical bed shear stress
        self.taucr = Constant((self.rhos-self.rhow)*self.g*self.average_size*self.thetacr)

        # calculate settling velocity
        if self.average_size <= 1e-04:
            self.settling_velocity = Constant(self.g*(self.average_size**2)*self.R/(18*self.viscosity))
        elif self.average_size <= 1e-03:
            self.settling_velocity = Constant((10*self.viscosity/self.average_size) *
                                              (sqrt(1 + 0.01*((self.R*self.g*(self.average_size**3))
                                                              / (self.viscosity**2)))-1))
        else:
            self.settling_velocity = Constant(1.1*sqrt(self.g*self.average_size*self.R))

        self.uv_cg = Function(self.vector_cg).interpolate(uv_init)

        self.depth = Function(self.V).interpolate(DepthExpression(bathymetry_2d, use_wetting_and_drying=self.use_wetting_and_drying, wetting_and_drying_alpha=self.wetting_alpha).get_total_depth(elev_init))

        self.old_bathymetry_2d = Function(self.V).interpolate(self.bathymetry_2d)

        self.u = self.uv_cg[0]
        self.v = self.uv_cg[1]

        # define bed friction
        hc = conditional(self.depth > Constant(0.001), self.depth, Constant(0.001))
        aux = conditional(11.036*hc/self.bed_reference_height > Constant(1.001),
                          11.036*hc/self.bed_reference_height, Constant(1.001))
        self.qfc = Constant(2)/(ln(aux)/kappa)**2
        # skin friction coefficient
        cfactor = conditional(self.depth > ksp, Constant(2) *
                              (((1/kappa)*ln(11.036*self.depth/ksp))**(-2)), Constant(0.0))
        # mu - ratio between skin friction and normal friction
        self.mu = conditional(self.qfc > Constant(0), cfactor/self.qfc, Constant(0))

        # calculate bed shear stress
        self.unorm = (self.u**2) + (self.v**2)
        self.TOB = Function(self.V).interpolate(self.rhow*Constant(0.5)*self.qfc*self.unorm)

        options.sediment_model_options.solve_exner = True

        if self.solve_suspended:
            # deposition flux - calculating coefficient to account for stronger conc at bed
            B = conditional(self.a > self.depth, Constant(1.0), self.a/self.depth)
            ustar = sqrt(Constant(0.5)*self.qfc*self.unorm)
            exp1 = conditional((conditional((self.settling_velocity/(kappa*ustar)) - Constant(1) >
                                            Constant(0), (self.settling_velocity/(kappa*ustar)) - Constant(1),
                                            -(self.settling_velocity/(kappa*ustar)) + Constant(1))) > Constant(1e-04),
                               conditional((self.settling_velocity/(kappa*ustar)) - Constant(1) > Constant(3), Constant(3),
                                                (self.settling_velocity/(kappa*ustar))-Constant(1)), Constant(0))
            self.coefftest = conditional((conditional((self.settling_velocity/(kappa*ustar))
                                                      - Constant(1) > Constant(0), (self.settling_velocity/(kappa*ustar))
                                                      - Constant(1), -(self.settling_velocity/(kappa*ustar))
                                                      + Constant(1))) > Constant(1e-04), B *
                                         (Constant(1)-B**exp1)/exp1, -B*ln(B))
            self.coeff = Function(self.P1_2d).interpolate(conditional(conditional(self.coefftest >
                                                                                  Constant(1e-12), Constant(1)/self.coefftest,
                                                                                  Constant(1e12)) > Constant(1),
                                                                      conditional(self.coefftest > Constant(1e-12),
                                                                                  Constant(1)/self.coefftest, Constant(1e12)),
                                                                      Constant(1)))

            # erosion flux - above critical velocity bed is eroded
            self.s0 = (conditional(self.rhow*Constant(0.5)*self.qfc*self.unorm*self.mu > Constant(0),
                                   self.rhow*Constant(0.5)*self.qfc*self.unorm*self.mu,
                                   Constant(0)) - self.taucr)/self.taucr
            self.ceq = Function(self.P1_2d).interpolate(Constant(0.015)*(self.average_size/self.a)
                                                        * ((conditional(self.s0 < Constant(0),
                                                                        Constant(0), self.s0))**(1.5))
                                                        / (self.dstar**0.3))

            if self.use_advective_velocity:
                self.corr_factor_model = Corrective_Velocity_Factor(self.depth, ksp,
                                                                    self.bed_reference_height, self.settling_velocity, ustar)
            # update sediment rate to ensure equilibrium at inflow
            if self.use_conservative:
                self.equiltracer = Function(self.P1_2d).interpolate(self.depth*self.ceq/self.coeff)
            else:
                self.equiltracer = Function(self.P1_2d).interpolate(self.ceq/self.coeff)

            # get individual terms
            self.depo = self.settling_velocity*self.coeff
            self.ero = Function(self.P1_2d).interpolate(self.settling_velocity*self.ceq)

            self.depo_term = Function(self.P1_2d).interpolate(self.depo/self.depth)
            self.ero_term = Function(self.P1_2d).interpolate(self.ero/self.depth)

            if erosion == 'model_def':
                options.sediment_model_options.sediment_ero = self.ero_term
            elif erosion == 'depth_integrated':
                options.sediment_model_options.sediment_depth_integ_ero = self.ero
            elif not erosion == 'None' or 'user_defined':
                raise ValueError("Unrecognised string. Erosion must be 'model_def', 'depth_integrated', 'None' or 'user_defined'")

            if deposition == 'model_def':
                options.sediment_model_options.sediment_depo = self.depo_term
            elif deposition == 'depth_integrated':
                options.sediment_model_options.sediment_depth_integ_depo = self.depo_term
            elif not deposition == 'None' or 'user_defined':
                raise ValueError("Unrecognised string. Deposition must be 'model_def', 'depth_integrated', 'None' or 'user_defined'")

            self.options.sediment_model_options.solve_sediment = True
            if self.use_advective_velocity:
                self.options.sediment_model_options.sediment_advective_velocity_factor = self.corr_factor_model.corr_vel_factor
        else:
            self.options.sediment_model_options.solve_sediment = False

        if self.solve_bedload:
            # calculate angle of flow
            self.calfa = Function(self.V).interpolate(self.u/sqrt(self.unorm))
            self.salfa = Function(self.V).interpolate(self.v/sqrt(self.unorm))
            if self.use_angle_correction:
                # slope effect angle correction due to gravity
                self.stress = Function(self.V).interpolate(self.rhow*Constant(0.5)*self.qfc*self.unorm)

    def get_bedload_term(self, solution):

        if self.use_slope_mag_correction:
            # slope effect magnitude correction due to gravity where beta is a parameter normally set to 1.3
            # we use z_n1 and equals so that we can use an implicit method in Exner
            slopecoef = Constant(1) + self.beta*(solution.dx(0)*self.calfa + solution.dx(1)*self.salfa)
        else:
            slopecoef = Constant(1.0)

        if self.use_angle_correction:
            # slope effect angle correction due to gravity
            cparam = Constant((self.rhos-self.rhow)*self.g*self.average_size*(self.surbeta2**2))
            tt1 = conditional(self.stress > Constant(1e-10), sqrt(cparam/self.stress), sqrt(cparam/Constant(1e-10)))

            # define bed gradient
            self.dzdx = self.old_bathymetry_2d.dx(0)
            self.dzdy = self.old_bathymetry_2d.dx(1)

            # add on a factor of the bed gradient to the normal
            aa = self.salfa + tt1*self.dzdy
            bb = self.calfa + tt1*self.dzdx

            comb = sqrt(aa**2 + bb**2)
            angle_norm = conditional(comb > Constant(1e-10), comb, Constant(1e-10))

            # we use z_n1 and equals so that we can use an implicit method in Exner
            calfamod = (self.calfa + (tt1*solution.dx(0)))/angle_norm
            salfamod = (self.salfa + (tt1*solution.dx(1)))/angle_norm

        if self.use_secondary_current:
            # accounts for helical flow effect in a curver channel
            # use z_n1 and equals so can use an implicit method in Exner
            free_surface_dx = self.depth.dx(0) - solution.dx(0)
            free_surface_dy = self.depth.dx(1) - solution.dx(1)

            velocity_slide = (self.u*free_surface_dy)-(self.v*free_surface_dx)

            tandelta_factor = Constant(7)*self.g*self.rhow*self.depth*self.qfc\
                / (Constant(2)*self.alpha_secc*((self.u**2) + (self.v**2)))

            # accounts for helical flow effect in a curver channel
            if self.use_angle_correction:
                # if angle has already been corrected we must alter the corrected angle to obtain the corrected secondary current angle
                t_1 = (self.TOB*slopecoef*calfamod) + (self.v*tandelta_factor*velocity_slide)
                t_2 = (self.TOB*slopecoef*salfamod) - (self.u*tandelta_factor*velocity_slide)
            else:
                t_1 = (self.TOB*slopecoef*self.calfa) + (self.v*tandelta_factor*velocity_slide)
                t_2 = ((self.TOB*slopecoef*self.salfa) - (self.u*tandelta_factor*velocity_slide))

            # calculated to normalise the new angles
            t4 = sqrt((t_1**2) + (t_2**2))

            # updated magnitude correction and angle corrections
            slopecoef_secc = t4/self.TOB

            calfanew = t_1/t4
            salfanew = t_2/t4

        # implement meyer-peter-muller bedload transport formula
        thetaprime = self.mu*(self.rhow*Constant(0.5)*self.qfc*self.unorm)/((self.rhos-self.rhow)*self.g*self.average_size)

        # if velocity above a certain critical value then transport occurs
        phi = conditional(thetaprime < self.thetacr, 0, Constant(8)*(thetaprime-self.thetacr)**1.5)

        # bedload transport flux with magnitude correction
        if self.use_secondary_current:
            qb_total = slopecoef_secc*phi*sqrt(self.g*self.R*self.average_size**3)
        else:
            qb_total = slopecoef*phi*sqrt(self.g*self.R*self.average_size**3)

        # formulate bedload transport flux with correct angle depending on corrections implemented
        if self.use_angle_correction and self.use_secondary_current is False:
            qbx = qb_total*calfamod
            qby = qb_total*salfamod
        elif self.use_secondary_current:
            qbx = qb_total*calfanew
            qby = qb_total*salfanew
        else:
            qbx = qb_total*self.calfa
            qby = qb_total*self.salfa

        return qbx, qby

    def update(self, t_new, solution_2d):
        # update bathymetry
        self.old_bathymetry_2d.interpolate(self.bathymetry_2d)

        # extract new elevation and velocity and project onto CG space
        self.uv1, self.elev1 = solution_2d.split()
        self.uv_cg.project(self.uv1)

        self.depth.interpolate(DepthExpression(self.old_bathymetry_2d, use_wetting_and_drying=self.use_wetting_and_drying, wetting_and_drying_alpha=self.wetting_alpha).get_total_depth(self.elev1))

        self.TOB.interpolate(self.rhow*Constant(0.5)*self.qfc*self.unorm)

        if self.solve_suspended:
            # source term

            # deposition flux - calculating coefficient to account for stronger conc at bed
            self.coeff.interpolate(conditional(self.coefftest > Constant(0), Constant(1)/self.coefftest, Constant(0)))

            # erosion flux - above critical velocity bed is eroded
            self.ceq.interpolate(Constant(0.015)*(self.average_size/self.a) *
                                 ((conditional(self.s0 < Constant(0), Constant(0),
                                               self.s0))**(1.5))/(self.dstar**0.3))

            self.ero.interpolate(self.settling_velocity*self.ceq)
            self.ero_term.interpolate(self.ero/self.depth)
            self.depo_term.interpolate(self.depo/self.depth)

            if self.use_advective_velocity:
                self.corr_factor_model.update()

            # update sediment rate to ensure equilibrium at inflow
            if self.use_conservative:
                self.equiltracer.interpolate(self.depth*self.ceq/self.coeff)
            else:
                self.equiltracer.interpolate(self.ceq/self.coeff)

        if self.solve_bedload:
            # calculate angle of flow
            self.calfa.interpolate(self.u/sqrt(self.unorm))
            self.salfa.interpolate(self.v/sqrt(self.unorm))
            if self.use_angle_correction:
                # slope effect angle correction due to gravity
                self.stress.interpolate(self.rhow*Constant(0.5)*self.qfc*self.unorm)
