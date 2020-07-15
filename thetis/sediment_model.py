from .utility import *


class CorrectiveVelocityFactor:
    # FIXME: docstrings
    # explain how the interface works, at the moment a reader of the code only sees an __init__ and a update()
    # where's the output?
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
        self.velocity_correction_factor = Function(depth.function_space(), name='velocity correction factor')
        self.velocity_correction_factor_expr = max_value(min_value(self.alpha, Constant(1)), Constant(0.))
        self.update()

    def update(self):

        # final correction factor
        self.velocity_correction_factor.interpolate(self.velocity_correction_factor_expr)

class SedimentModel(object):
    def __init__(self, options, mesh2d, uv_init, elev_init, bathymetry_2d,
                 beta_fn, surbeta2_fn, alpha_secc_fn, viscosity_morph, rhos):

        """
        Set up a full morphological model simulation using as an initial condition the results of a hydrodynamic only model.

        :arg options: Model options.
        :type options: :class:`.ModelOptions2d` instance
        :arg mesh2d: :class:`Mesh` object of the 2D mesh
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
            self.settling_velocity = Constant((10*self.viscosity/self.average_size)
                                              * (sqrt(1 + 0.01*((self.R*self.g*(self.average_size**3))
                                                 / (self.viscosity**2)))-1))
        else:
            self.settling_velocity = Constant(1.1*sqrt(self.g*self.average_size*self.R))

        self.uv_cg = Function(self.vector_cg).interpolate(uv_init)

        # dictionary of steps (interpolate/project) to perform in order in update()
        self.update_steps = OrderedDict()
        # fields updated in these steps
        self.fields = AttrDict()

        self._add_interpolation_step('old_bathymetry_2d', self.bathymetry_2d)
        self._add_interpolation_step(
            'depth',
            DepthExpression(self.fields.old_bathymetry_2d, use_wetting_and_drying=self.use_wetting_and_drying, wetting_and_drying_alpha=self.wetting_alpha).get_total_depth(elev_init))

        self.u = self.uv_cg[0]
        self.v = self.uv_cg[1]

        # define bed friction
        hc = conditional(self.fields.depth > Constant(0.001), self.fields.depth, Constant(0.001))
        aux = conditional(11.036*hc/self.bed_reference_height > Constant(1.001),
                          11.036*hc/self.bed_reference_height, Constant(1.001))
        self.qfc = Constant(2)/(ln(aux)/kappa)**2
        # skin friction coefficient
        cfactor = conditional(self.fields.depth > ksp, Constant(2)
                              * (((1/kappa)*ln(11.036*self.fields.depth/ksp))**(-2)), Constant(0.0))
        # mu - ratio between skin friction and normal friction
        self.mu = conditional(self.qfc > Constant(0), cfactor/self.qfc, Constant(0))

        # calculate bed shear stress
        self.unorm = (self.u**2) + (self.v**2)
        # FIXME: bad name TOB
        self._add_interpolation_step('TOB', self.rhow*Constant(0.5)*self.qfc*self.unorm)

        # FIXME: to be discussed, we shouldn't be setting user options
        options.sediment_model_options.solve_exner = True

        if self.solve_suspended:
            # deposition flux - calculating coefficient to account for stronger conc at bed
            B = conditional(self.a > self.fields.depth, Constant(1.0), self.a/self.fields.depth)
            ustar = sqrt(Constant(0.5)*self.qfc*self.unorm)
            # FIXME: I can't make heads nor tails of this, the second conditional (first nested) seems to just try to implement abs
            # then then outside conditional check whether that's positive (?)
            # then the third conditional seems to be a cap, which you can better do with max_value()
            exp1 = conditional((conditional((self.settling_velocity/(kappa*ustar)) - Constant(1) >
                                            Constant(0), (self.settling_velocity/(kappa*ustar)) - Constant(1),
                                            -(self.settling_velocity/(kappa*ustar)) + Constant(1))) > Constant(1e-04),
                               conditional((self.settling_velocity/(kappa*ustar)) - Constant(1) > Constant(3), Constant(3),
                                                (self.settling_velocity/(kappa*ustar))-Constant(1)), Constant(0))
            # FIXME:same issue here
            self.coefftest = conditional((conditional((self.settling_velocity/(kappa*ustar))
                                                      - Constant(1) > Constant(0), (self.settling_velocity/(kappa*ustar))
                                                      - Constant(1), -(self.settling_velocity/(kappa*ustar))
                                                      + Constant(1))) > Constant(1e-04), B *
                                         (Constant(1)-B**exp1)/exp1, -B*ln(B))
            # FIXME: why is this different than what was in update?
            # FIXME: coeff is a bad name
            self._add_interpolation_step('coeff',
                    conditional(conditional(self.coefftest >
                        Constant(1e-12), Constant(1)/self.coefftest,
                        Constant(1e12)) > Constant(1),
                        conditional(self.coefftest > Constant(1e-12),
                            Constant(1)/self.coefftest, Constant(1e12)),
                        Constant(1)), V=self.P1_2d)

            # erosion flux - above critical velocity bed is eroded
            # FIXME:same issue here
            self.s0 = (conditional(self.rhow*Constant(0.5)*self.qfc*self.unorm*self.mu > Constant(0),
                                   self.rhow*Constant(0.5)*self.qfc*self.unorm*self.mu,
                                   Constant(0)) - self.taucr)/self.taucr
            # FIXME:same issue here
            self._add_interpolation_step('ceq',Constant(0.015)*(self.average_size/self.a)
                                                        * ((conditional(self.s0 < Constant(0),
                                                                        Constant(0), self.s0))**(1.5))
                                                        / (self.dstar**0.3),
                                                        V=self.P1_2d)

            if self.use_advective_velocity:
                self.corr_factor_model = CorrectiveVelocityFactor(self.fields.depth, ksp,
                                                                    self.bed_reference_height, self.settling_velocity, ustar)
                self.update_steps['correction_factor'] = self.corr_factor_model.update
                self.fields.velocity_correction_factor = self.corr_factor_model.velocity_correction_factor
            self._add_interpolation_step('equilibrium_tracer', self.fields.ceq/self.fields.coeff, V=self.P1_2d)

            # get individual terms
            self._deposition = self.settling_velocity*self.fields.coeff
            self._add_interpolation_step('erosion', self.settling_velocity*self.fields.ceq, V=self.P1_2d)

            self.options.sediment_model_options.solve_sediment = True
            if self.use_advective_velocity:
                self.options.sediment_model_options.sediment_advective_velocity_factor = self.fields.velocity_correction_factor
        else:
            self.options.sediment_model_options.solve_sediment = False

        if self.solve_bedload:
            # calculate angle of flow
            self._add_interpolation_step('calfa', self.u/sqrt(self.unorm))
            self._add_interpolation_step('salfa', self.v/sqrt(self.unorm))
            if self.use_angle_correction:
                # slope effect angle correction due to gravity
                self._add_interpolation_step('stress', self.rhow*Constant(0.5)*self.qfc*self.unorm)
        self.update(0.0, uv_init)

    def _add_interpolation_step(self, field_name, expr, V=None):
        """Add interpolation step to update

        :arg field_name: str name of new field to project into, stored in self.fields
        :arg expr: UFL expression to interpolate
        :kwarg V: FunctionSpace for new field (default is self.V)"""
        self.fields[field_name] = Function(V or self.V, name=field_name)
        self.update_steps[field_name] = Interpolator(expr, self.fields[field_name]).interpolate

    def get_bedload_term(self, solution):
        # FIXME: needs docstring
        # FIXME: bad name solution, means nothing in this context

        if self.use_slope_mag_correction:
            # slope effect magnitude correction due to gravity where beta is a parameter normally set to 1.3
            # we use z_n1 and equals so that we can use an implicit method in Exner
            slopecoef = Constant(1) + self.beta*(solution.dx(0)*self.fields.calfa + solution.dx(1)*self.fields.salfa)
        else:
            slopecoef = Constant(1.0)

        if self.use_angle_correction:
            # slope effect angle correction due to gravity
            cparam = Constant((self.rhos-self.rhow)*self.g*self.average_size*(self.surbeta2**2))
            tt1 = conditional(self.fields.stress > Constant(1e-10), sqrt(cparam/self.fields.stress), sqrt(cparam/Constant(1e-10)))

            # define bed gradient
            dzdx = self.fields.old_bathymetry_2d.dx(0)
            dzdy = self.fields.old_bathymetry_2d.dx(1)

            # add on a factor of the bed gradient to the normal
            aa = self.fields.salfa + tt1*dzdy
            bb = self.fields.calfa + tt1*dzdx

            comb = sqrt(aa**2 + bb**2)
            angle_norm = conditional(comb > Constant(1e-10), comb, Constant(1e-10))

            # we use z_n1 and equals so that we can use an implicit method in Exner
            calfamod = (self.fields.calfa + (tt1*solution.dx(0)))/angle_norm
            salfamod = (self.fields.salfa + (tt1*solution.dx(1)))/angle_norm

        if self.use_secondary_current:
            # accounts for helical flow effect in a curver channel
            # use z_n1 and equals so can use an implicit method in Exner
            free_surface_dx = self.fields.depth.dx(0) - solution.dx(0)
            free_surface_dy = self.fields.depth.dx(1) - solution.dx(1)

            velocity_slide = (self.u*free_surface_dy)-(self.v*free_surface_dx)

            tandelta_factor = Constant(7)*self.g*self.rhow*self.fields.depth*self.qfc\
                / (Constant(2)*self.alpha_secc*((self.u**2) + (self.v**2)))

            # accounts for helical flow effect in a curver channel
            if self.use_angle_correction:
                # if angle has already been corrected we must alter the corrected angle to obtain the corrected secondary current angle
                t_1 = (self.fields.TOB*slopecoef*calfamod) + (self.v*tandelta_factor*velocity_slide)
                t_2 = (self.fields.TOB*slopecoef*salfamod) - (self.u*tandelta_factor*velocity_slide)
            else:
                t_1 = (self.fields.TOB*slopecoef*self.fields.calfa) + (self.v*tandelta_factor*velocity_slide)
                t_2 = ((self.fields.TOB*slopecoef*self.fields.salfa) - (self.u*tandelta_factor*velocity_slide))

            # calculated to normalise the new angles
            t4 = sqrt((t_1**2) + (t_2**2))

            # updated magnitude correction and angle corrections
            slopecoef_secc = t4/self.fields.TOB

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
            qbx = qb_total*self.fields.calfa
            qby = qb_total*self.fields.salfa

        return qbx, qby

    def get_deposition_coefficient(self):
        """Returns coefficient C such that C/H*sediment is deposition term in sediment equation

        If sediment field is depth-averaged, C*sediment is (total) deposition (over the column)
        as it appears in the Exner equation, but deposition term in sediment equation needs
        averaging: C*sediment/H
        If sediment field is depth-itnegrated, C*sediment/H is (total) deposition (over the column)
        as it appears in the Exner equation, and is the same in the sediment equation."""
        return self._deposition

    def get_erosion_term(self):
        """Returns expression for (depth-integrated) erosion."""
        return self.fields.erosion

    def get_equilibrium_tracer(self):
        """Returns expression for (depth-averaged) equilibrium tracer."""
        return self.fields.equilibrium_tracer

    def update(self, t_new, uv):

        # velocity used in all expressions via self.u, self.v and self.unorm:
        self.uv_cg.project(uv)

        for step in self.update_steps.values():
            step()
