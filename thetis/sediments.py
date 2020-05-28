from thetis import *

class Corrective_Velocity_Factor:
    def __init__(self, depth, ksp, ks, settling_velocity, ustar):
        self.ksp = ksp 
        self.ks = ks
        self.settling_velocity = settling_velocity
        
        self.a = Constant(self.ks/2)
        
        self.kappa = physical_constants['von_karman']
        
        self.depth = depth
        self.ustar = ustar
        
        # correction factor to advection velocity in sediment concentration equation
        self.Bconv = conditional(self.depth > Constant(1.1)*self.ksp, self.ksp/self.depth, Constant(1/1.1))
        self.Aconv = conditional(self.depth > Constant(1.1)*self.a, self.a/self.depth, Constant(1/1.1))

        # take max of value calculated either by ksp or depth
        self.Amax = conditional(self.Aconv > self.Bconv, self.Aconv, self.Bconv)

        self.r1conv = Constant(1) - (1/self.kappa)*conditional(self.settling_velocity/self.ustar < Constant(1), self.settling_velocity/self.ustar, Constant(1))

        self.Ione = conditional(self.r1conv > Constant(1e-8), (Constant(1) - self.Amax**self.r1conv)/self.r1conv, conditional(self.r1conv < Constant(- 1e-8), (Constant(1) - self.Amax**self.r1conv)/self.r1conv, ln(self.Amax)))

        self.Itwo = conditional(self.r1conv > Constant(1e-8), -(self.Ione + (ln(self.Amax)*(self.Amax**self.r1conv)))/self.r1conv, conditional(self.r1conv < Constant(- 1e-8), -(self.Ione + (ln(self.Amax)*(self.Amax**self.r1conv)))/self.r1conv, Constant(-0.5)*ln(self.Amax)**2))

        self.alpha = -(self.Itwo - (ln(self.Amax) - ln(30))*self.Ione)/(self.Ione * ((ln(self.Amax) - ln(30)) + Constant(1)))

        # final correction factor
        self.corr_vel_factor = Function(depth.function_space()).interpolate(conditional(conditional(self.alpha > Constant(1), Constant(1), self.alpha) < Constant(0), Constant(0), conditional(self.alpha > Constant(1), Constant(1), self.alpha)))
    
    def update(self):

        # final correction factor
        self.corr_vel_factor.interpolate(conditional(conditional(self.alpha > Constant(1), Constant(1), self.alpha) < Constant(0), Constant(0), conditional(self.alpha > Constant(1), Constant(1), self.alpha)))


class SedimentModel(object):
    def __init__(self, morfac, suspendedload, convectivevel,
                  bedload, angle_correction, slope_eff, seccurrent,
                  mesh2d, bathymetry_2d, uv_init, elev_init, outputdir, ks, average_size, dt, final_time, 
                  beta_fn = 1.3, surbeta2_fn = 1/1.5, alpha_secc_fn = 0.75, viscosity_morph=1e-6,
                  wetting_and_drying=False, wetting_alpha=0.1, rhos=2650, cons_tracer=False):
        
        """
        Set up a full morphological model simulation using as an initial condition the results of a hydrodynamic only model.

        Inputs:
        morfac - morphological scale factor
        suspendedload - switch to turn on suspended sediment transport
        convectivevel - switch on convective velocity correction factor in sediment concentration equation
        bedload - switch to turn on bedload transport
        angle_correction - switch on slope effect angle correction
        slope_eff - switch on slope effect magnitude correction
        seccurrent - switch on secondary current for helical flow effect    
        mesh2d - define mesh working on
        bathymetry2d - define bathymetry of problem
        input_dir - folder containing results of hydrodynamics model which are used as initial conditions here
        ks - bottom friction coefficient for quadratic drag coefficient
        average_size - average sediment size
        beta_fn - magnitude slope effect parameter
        surbeta2_fn - angle correction slope effect parameter
        alpha_secc_fn - secondary current parameter    
        viscosity_morph - viscosity value in morphodynamic equations
        wetting_and_drying - wetting and drying switch
        wetting_alpha - wetting and drying parameter
        rhos - sediment density
        cons_tracer - conservative tracer switch

        """        

        self.suspendedload = suspendedload
        self.cons_tracer = cons_tracer
        self.convectivevel = convectivevel
        self.bedload = bedload
        self.angle_correction = angle_correction
        self.slope_eff = slope_eff
        self.seccurrent = seccurrent
        self.wetting_and_drying = wetting_and_drying        

        self.morfac = morfac
        self.average_size = average_size
        self.ks = ks
        self.wetting_alpha = wetting_alpha
        self.rhos = rhos
        self.dt = dt
        self.uv_init = uv_init
        self.elev_init = elev_init
        
        self.bathymetry_2d = bathymetry_2d

        # define bathymetry_file
        self.bathy_file = File(outputdir + "/bathy.pvd")

        # final time of simulation
        t_end = final_time/self.morfac

        self.t_old = Constant(0.0)

        # define function spaces
        self.P1_2d = get_function_space(mesh2d, "DG", 1)
        self.V = get_function_space(mesh2d, "CG", 1)
        self.vector_cg = get_function_space(mesh, "CG", 1, vector=True)

        # define test functions on mesh
        self.v = TestFunction(self.V)
        self.n = FacetNormal(mesh2d)
        self.z_n1 = Function(self.V, name="z^{n+1}")
        self.z_n = Function(self.V, name="z^{n}")

        # define parameters
        self.g = physical_constants['g_grav']
        self.rhow = physical_constants['rho0']
        self.kappa = physical_constants['von_karman']
        self.porosity = Constant(0.4)

        self.ksp = Constant(3*self.average_size)
        self.a = Constant(self.ks/2)
        self.viscosity = Constant(viscosity_morph)

        # magnitude slope effect parameter
        self.beta = Constant(beta_fn)
        # angle correction slope effect parameters
        self.surbeta2 = Constant(surbeta2_fn)
        self.cparam = Constant((self.rhos-self.rhow)*self.g*self.average_size*(self.surbeta2**2))
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
            self.settling_velocity = Constant((10*self.viscosity/self.average_size)*(sqrt(1 + 0.01*((self.R*self.g*(self.average_size**3))/(self.viscosity**2)))-1))
        else:
            self.settling_velocity = Constant(1.1*sqrt(self.g*self.average_size*self.R))

        self.uv_cg = Function(self.vector_cg).interpolate(self.uv_init)

        if self.wetting_and_drying:
            H = self.elev_init + self.bathymetry_2d
            self.depth = Function(self.V).project(H + (Constant(0.5) * (sqrt(H ** 2 + self.wetting_alpha ** 2) - H)))
        else:
            self.depth = Function(self.V).project(self.elev_init + self.bathymetry_2d)

        self.old_bathymetry_2d = Function(self.V).interpolate(self.bathymetry_2d)
        self.bathy_file.write(self.bathymetry_2d)    

        self.horizontal_velocity = self.uv_cg[0]
        self.vertical_velocity = self.uv_cg[1]

        # define bed friction
        self.hc = conditional(self.depth > Constant(0.001), self.depth, Constant(0.001))
        self.aux = conditional(11.036*self.hc/self.ks > Constant(1.001), 11.036*self.hc/self.ks, Constant(1.001))
        self.qfc = Constant(2)/(ln(self.aux)/self.kappa)**2
        # skin friction coefficient
        self.cfactor = conditional(self.depth > self.ksp, Constant(2)*(((1/self.kappa)*ln(11.036*self.depth/self.ksp))**(-2)), Constant(0.0))
        # mu - ratio between skin friction and normal friction
        self.mu = conditional(self.qfc > Constant(0), self.cfactor/self.qfc, Constant(0))

        # calculate bed shear stress
        self.unorm = (self.horizontal_velocity**2) + (self.vertical_velocity**2)
        self.TOB = Function(self.V).interpolate(self.rhow*Constant(0.5)*self.qfc*self.unorm)

        # define bed gradient
        self.dzdx = self.old_bathymetry_2d.dx(0)
        self.dzdy = self.old_bathymetry_2d.dx(1)

        if self.suspendedload:
            # deposition flux - calculating coefficient to account for stronger conc at bed
            self.B = conditional(self.a > self.depth, Constant(1.0), self.a/self.depth)
            self.ustar = sqrt(Constant(0.5)*self.qfc*self.unorm)
            self.exp1 = conditional((conditional((self.settling_velocity/(self.kappa*self.ustar)) - Constant(1) > Constant(0), (self.settling_velocity/(self.kappa*self.ustar)) - Constant(1), -(self.settling_velocity/(self.kappa*self.ustar)) + Constant(1))) > Constant(1e-04), conditional((self.settling_velocity/(self.kappa*self.ustar)) - Constant(1) > Constant(3), Constant(3), (self.settling_velocity/(self.kappa*self.ustar))-Constant(1)), Constant(0))
            self.coefftest = conditional((conditional((self.settling_velocity/(self.kappa*self.ustar)) - Constant(1) > Constant(0), (self.settling_velocity/(self.kappa*self.ustar)) - Constant(1), -(self.settling_velocity/(self.kappa*self.ustar)) + Constant(1))) > Constant(1e-04), self.B*(Constant(1)-self.B**self.exp1)/self.exp1, -self.B*ln(self.B))
            self.coeff = Function(self.P1_2d).interpolate(conditional(conditional(self.coefftest > Constant(1e-12), Constant(1)/self.coefftest, Constant(1e12)) > Constant(1), conditional(self.coefftest > Constant(1e-12), Constant(1)/self.coefftest, Constant(1e12)), Constant(1)))

            # erosion flux - above critical velocity bed is eroded
            self.s0 = (conditional(self.rhow*Constant(0.5)*self.qfc*self.unorm*self.mu > Constant(0), self.rhow*Constant(0.5)*self.qfc*self.unorm*self.mu, Constant(0)) - self.taucr)/self.taucr
            self.ceq = Function(self.P1_2d).interpolate(Constant(0.015)*(self.average_size/self.a) * ((conditional(self.s0 < Constant(0), Constant(0), self.s0))**(1.5))/(self.dstar**0.3))

            if self.convectivevel:
                self.corr_factor_model = Corrective_Velocity_Factor(self.depth, self.ksp, self.ks, self.settling_velocity, self.ustar)
            # update sediment rate to ensure equilibrium at inflow
            if self.cons_tracer:
                self.sediment_rate = Constant(self.depth.at([0, 0])*self.ceq.at([0, 0])/(self.coeff.at([0, 0])))
                self.testtracer = Function(self.P1_2d).interpolate(self.depth*self.ceq/self.coeff)
            else:
                self.sediment_rate = Constant(self.ceq.at([0, 0])/(self.coeff.at([0, 0])))
                self.testtracer = Function(self.P1_2d).interpolate(self.ceq/self.coeff)

            # get individual terms
            self.depo = self.settling_velocity*self.coeff
            self.ero = self.settling_velocity*self.ceq

            # calculate depth-averaged source term for sediment concentration equation
            if self.cons_tracer:
                self.source = Function(self.P1_2d).interpolate(-(self.depo*self.sediment_rate/(self.depth**2)) + (self.ero/self.depth))
                self.qbsourcedepth = -(self.depo*self.sediment_rate/self.depth) + self.ero
            else:
                self.source = Function(self.P1_2d).interpolate(-(self.depo*self.sediment_rate/self.depth) + (self.ero/self.depth))
                self.qbsourcedepth = -(self.depo*self.sediment_rate) + self.ero

        if self.bedload:
            # calculate angle of flow
            self.calfa = Function(self.V).interpolate(self.horizontal_velocity/sqrt(self.unorm))
            self.salfa = Function(self.V).interpolate(self.vertical_velocity/sqrt(self.unorm))

            if self.slope_eff:
                # slope effect magnitude correction due to gravity where beta is a parameter normally set to 1.3
                # we use z_n1 and equals so that we can use an implicit method in Exner
                self.slopecoef = Constant(1) + self.beta*(self.z_n1.dx(0)*self.calfa + self.z_n1.dx(1)*self.salfa)
            else:
                self.slopecoef = Constant(1.0)

            if self.angle_correction:
                # slope effect angle correction due to gravity
                self.tt1 = conditional(self.rhow*Constant(0.5)*self.qfc*self.unorm > Constant(1e-10), sqrt(self.cparam/(self.rhow*Constant(0.5)*self.qfc*self.unorm)), sqrt(self.cparam/Constant(1e-10)))
                # add on a factor of the bed gradient to the normal
                self.aa = self.salfa + self.tt1*self.dzdy
                self.bb = self.calfa + self.tt1*self.dzdx
                self.norm = Function(self.V).interpolate(conditional(sqrt(self.aa**2 + self.bb**2) > Constant(1e-10), sqrt(self.aa**2 + self.bb**2), Constant(1e-10)))

                # we use z_n1 and equals so that we can use an implicit method in Exner
                self.calfamod = (self.calfa + (self.tt1*self.z_n1.dx(0)))/self.norm
                self.salfamod = (self.salfa + (self.tt1*self.z_n1.dx(1)))/self.norm

            if self.seccurrent:
                # accounts for helical flow effect in a curver channel
                # use z_n1 and equals so can use an implicit method in Exner
                self.free_surface_dx = self.depth.dx(0) - self.z_n1.dx(0)
                self.free_surface_dy = self.depth.dx(1) - self.z_n1.dx(1)

                self.velocity_slide = (self.horizontal_velocity*self.free_surface_dy)-(self.vertical_velocity*self.free_surface_dx)

                self.tandelta_factor = Constant(7)*self.g*self.rhow*self.depth*self.qfc/(Constant(2)*self.alpha_secc*((self.horizontal_velocity**2) + (self.vertical_velocity**2)))

                if self.angle_correction:
                    # if angle has already been corrected we must alter the corrected angle to obtain the corrected secondary current angle
                    self.t_1 = (self.TOB*self.slopecoef*self.calfamod) + (self.vertical_velocity*self.tandelta_factor*self.velocity_slide)
                    self.t_2 = (self.TOB*self.slopecoef*self.salfamod) - (self.horizontal_velocity*self.tandelta_factor*self.velocity_slide)
                else:
                    self.t_1 = (self.TOB*self.slopecoef*self.calfa) + (self.vertical_velocity*self.tandelta_factor*self.velocity_slide)
                    self.t_2 = ((self.TOB*self.slopecoef*self.salfa) - (self.horizontal_velocity*self.tandelta_factor*self.velocity_slide))

                # calculated to normalise the new angles
                self.t4 = sqrt((self.t_1**2) + (self.t_2**2))

                # updated magnitude correction and angle corrections
                self.slopecoef_secc = self.t4/self.TOB
                
                self.calfanew = self.t_1/self.t4
                self.salfanew = self.t_2/self.t4                

            # implement meyer-peter-muller bedload transport formula
            self.thetaprime = self.mu*(self.rhow*Constant(0.5)*self.qfc*self.unorm)/((self.rhos-self.rhow)*self.g*self.average_size)

            # if velocity above a certain critical value then transport occurs
            self.phi = conditional(self.thetaprime < self.thetacr, 0, Constant(8)*(self.thetaprime-self.thetacr)**1.5)

            # bedload transport flux with magnitude correction
            if self.seccurrent:
                self.qb_total = self.slopecoef_secc*self.phi*sqrt(self.g*self.R*self.average_size**3)
                        
            else:
                self.qb_total = self.slopecoef*self.phi*sqrt(self.g*self.R*self.average_size**3)

            # add time derivative to exner equation with a morphological scale factor
            self.f = (((Constant(1)-self.porosity)*(self.z_n1 - self.z_n)/(self.dt*self.morfac)) * self.v)*dx

            # formulate bedload transport flux with correct angle depending on corrections implemented
            if self.angle_correction and self.seccurrent is False:
                self.qbx = self.qb_total*self.calfamod
                self.qby = self.qb_total*self.salfamod
            elif self.seccurrent:
                self.qbx = self.qb_total*self.calfanew
                self.qby = self.qb_total*self.salfanew
            else:
                self.qbx = self.qb_total*self.calfa
                self.qby = self.qb_total*self.salfa

            # add bedload transport to exner equation
            self.f += -(self.v*((self.qbx*self.n[0]) + (self.qby*self.n[1])))*ds(1) - (self.v*((self.qbx*self.n[0]) + (self.qby*self.n[1])))*ds(2) + (self.qbx*(self.v.dx(0)) + self.qby*(self.v.dx(1)))*dx
        else:
            # if no bedload transport component initialise exner equation with time derivative
            self.f = (((Constant(1)-self.porosity)*(self.z_n1 - self.z_n)/(self.dt*self.morfac)) * self.v)*dx

        if self.suspendedload:
            # add suspended sediment transport to exner equation multiplied by depth as the exner equation is not depth-averaged
            self.f += - (self.qbsourcedepth*self.v)*dx


    def update(self, t_new, solver_obj):
        # update bathymetry
        self.old_bathymetry_2d.assign(self.bathymetry_2d)

        # extract new elevation and velocity and project onto CG space
        self.uv1, self.elev1 = solver_obj.fields.solution_2d.split()
        self.uv_cg.project(self.uv1)


        if self.wetting_and_drying:
            self.depth.project(self.elev1 + solver_obj.depth.wd_bathymetry_displacement(self.elev1) + self.old_bathymetry_2d)
        else:
            self.depth.project(self.elev1 + self.old_bathymetry_2d) 
            
        # if include sediment then update_forcings is run twice but only want to update bathymetry once
        double_factor = False

        if self.suspendedload:
            if self.t_old.dat.data[:] == t_new:
                double_factor = True
        else:
            # if have no tracer then update_forcings is only run once so update bathymetry at each step
            double_factor = True                    

        if double_factor:
            self.z_n.assign(self.old_bathymetry_2d)

            self.TOB.interpolate(self.rhow*Constant(0.5)*self.qfc*self.unorm)

            if self.suspendedload:
                # source term

                # deposition flux - calculating coefficient to account for stronger conc at bed
                self.coeff.interpolate(conditional(self.coefftest > Constant(0), Constant(1)/self.coefftest, Constant(0)))

                # erosion flux - above critical velocity bed is eroded
                self.ceq.interpolate(Constant(0.015)*(self.average_size/self.a) * ((conditional(self.s0 < Constant(0), Constant(0), self.s0))**(1.5))/(self.dstar**0.3))

                # calculate depth-averaged source term for sediment concentration equation
                if self.cons_tracer:
                    self.source.interpolate(-(self.depo*solver_obj.fields.tracer_2d/(self.depth**2)) + (self.ero/self.depth))
                    self.qbsourcedepth = (-(self.depo*solver_obj.fields.tracer_2d/self.depth) + self.ero)
                else:
                    self.source.interpolate(-(self.depo*solver_obj.fields.tracer_2d/self.depth) + (self.ero/self.depth))
                    self.qbsourcedepth = -(self.depo*solver_obj.fields.tracer_2d) + self.ero

                if self.convectivevel:
                    self.corr_factor_model.update()

                # update sediment rate to ensure equilibrium at inflow
                if self.cons_tracer:
                    self.sediment_rate.assign(self.depth.at([0, 0])*self.ceq.at([0, 0])/(self.coeff.at([0, 0])))
                else:
                    self.sediment_rate.assign(self.ceq.at([0, 0])/(self.coeff.at([0, 0])))
                        
            if self.bedload:

                # calculate angle of flow
                self.calfa.interpolate(self.horizontal_velocity/sqrt(self.unorm))
                self.salfa.interpolate(self.vertical_velocity/sqrt(self.unorm))
                
                if self.angle_correction:
                    self.norm.interpolate(conditional(sqrt(self.aa**2 + self.bb**2) > Constant(1e-10), sqrt(self.aa**2 + self.bb**2), Constant(1e-10)))

            # solve exner equation using finite element methods
            solve(self.f == 0, self.z_n1)

            # update bed
            self.bathymetry_2d.assign(self.z_n1)

            if round(t_new, 2)%solver_obj.options.simulation_export_time == 0:
                self.bathy_file.write(self.bathymetry_2d)                        
        self.t_old.assign(t_new)      
