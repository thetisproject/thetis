"""
Migrating Trench Test case
=======================

Solves the test case of a migrating trench.

[1] Clare et al. 2020. “Hydro-morphodynamics 2D Modelling Using a Discontinuous
    Galerkin Discretisation.” EarthArXiv. January 9. doi:10.31223/osf.io/tpqvy.

"""

from thetis import *
import callback_cons_tracer as call

import numpy as np
import pandas as pd
import time

conservative = False

class Corrective_Velocity_Factor:
    def __init__(self, depth, ksp, ks, settling_velocity, ustar):
        self.ksp = ksp 
        self.ks = ks
        self.settling_velocity = settling_velocity
        
        self.a = Constant(self.ks/2)
        
        self.kappa = physical_constants['von_karman']
        
        # correction factor to advection velocity in sediment concentration equation
        Bconv = conditional(depth > Constant(1.1)*self.ksp, self.ksp/depth, Constant(1/1.1))
        Aconv = conditional(depth > Constant(1.1)*self.a, self.a/depth, Constant(1/1.1))

        # take max of value calculated either by ksp or depth
        Amax = conditional(Aconv > Bconv, Aconv, Bconv)

        r1conv = Constant(1) - (1/self.kappa)*conditional(self.settling_velocity/ustar < Constant(1), self.settling_velocity/ustar, Constant(1))

        Ione = conditional(r1conv > Constant(1e-8), (Constant(1) - Amax**r1conv)/r1conv, conditional(r1conv < Constant(- 1e-8), (Constant(1) - Amax**r1conv)/r1conv, ln(Amax)))

        Itwo = conditional(r1conv > Constant(1e-8), -(Ione + (ln(Amax)*(Amax**r1conv)))/r1conv, conditional(r1conv < Constant(- 1e-8), -(Ione + (ln(Amax)*(Amax**r1conv)))/r1conv, Constant(-0.5)*ln(Amax)**2))

        alpha = -(Itwo - (ln(Amax) - ln(30))*Ione)/(Ione * ((ln(Amax) - ln(30)) + Constant(1)))

        # final correction factor
        self.corr_vel_factor = Function(depth.function_space()).interpolate(conditional(conditional(alpha > Constant(1), Constant(1), alpha) < Constant(0), Constant(0), conditional(alpha > Constant(1), Constant(1), alpha)))
    
    def update(self, depth, ustar):
        
        # correction factor to advection velocity in sediment concentration equation
        Bconv = conditional(depth > Constant(1.1)*self.ksp, self.ksp/depth, Constant(1/1.1))
        Aconv = conditional(depth > Constant(1.1)*self.a, self.a/depth, Constant(1/1.1))

        # take max of value calculated either by ksp or depth
        Amax = conditional(Aconv > Bconv, Aconv, Bconv)

        r1conv = Constant(1) - (1/self.kappa)*conditional(self.settling_velocity/ustar < Constant(1), self.settling_velocity/ustar, Constant(1))

        Ione = conditional(r1conv > Constant(1e-8), (Constant(1) - Amax**r1conv)/r1conv, conditional(r1conv < Constant(- 1e-8), (Constant(1) - Amax**r1conv)/r1conv, ln(Amax)))

        Itwo = conditional(r1conv > Constant(1e-8), -(Ione + (ln(Amax)*(Amax**r1conv)))/r1conv, conditional(r1conv < Constant(- 1e-8), -(Ione + (ln(Amax)*(Amax**r1conv)))/r1conv, Constant(-0.5)*ln(Amax)**2))

        alpha = -(Itwo - (ln(Amax) - ln(30))*Ione)/(Ione * ((ln(Amax) - ln(30)) + Constant(1)))

        # final correction factor
        self.corr_vel_factor.interpolate(conditional(conditional(alpha > Constant(1), Constant(1), alpha) < Constant(0), Constant(0), conditional(alpha > Constant(1), Constant(1), alpha)))        

def initialise_fields(mesh2d, inputdir, outputdir,):
    """
    Initialise simulation with results from a previous simulation
    """
    DG_2d = FunctionSpace(mesh2d, 'DG', 1)
    # elevation
    with timed_stage('initialising elevation'):
        chk = DumbCheckpoint(inputdir + "/elevation", mode=FILE_READ)
        elev_init = Function(DG_2d, name="elevation")
        chk.load(elev_init)
        File(outputdir + "/elevation_imported.pvd").write(elev_init)
        chk.close()
    # velocity
    with timed_stage('initialising velocity'):
        chk = DumbCheckpoint(inputdir + "/velocity", mode=FILE_READ)
        V = VectorFunctionSpace(mesh2d, 'DG', 1)
        uv_init = Function(V, name="velocity")
        chk.load(uv_init)
        File(outputdir + "/velocity_imported.pvd").write(uv_init)
        chk.close()
        return elev_init, uv_init,

def morphological(morfac, morfac_transport, suspendedload, convectivevel,
                  bedload, angle_correction, slope_eff, seccurrent,
                  mesh2d, bathymetry_2d, input_dir, ks, average_size, dt, final_time, 
                  beta_fn = 1.3, surbeta2_fn = 1/1.5, alpha_secc_fn = 0.75, viscosity_hydro=1e-6, viscosity_morph=1e-6, 
                  wetting_and_drying=False, wetting_alpha=0.1, rhos=2650, cons_tracer=False, diffusivity=0.15):
    """
    Set up a full morphological model simulation using as an initial condition the results of a hydrodynamic only model.

    Inputs:
    boundary_consditions_fn - function defining boundary conditions for problem
    morfac - morphological scale factor
    morfac_transport - switch to turn on morphological component
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
    dt - timestep
    final_time - end time
    beta_fn - magnitude slope effect parameter
    surbeta2_fn - angle correction slope effect parameter
    alpha_secc_fn - secondary current parameter    
    viscosity_hydro - viscosity value in hydrodynamic equations
    viscosity_morph - viscosity value in morphodynamic equations
    wetting_and_drying - wetting and drying switch
    wetting_alpha - wetting and drying parameter
    rhos - sediment density
    cons_tracer - conservative tracer switch
    friction - choice of friction formulation - nikuradse and manning
    friction_coef - value of friction coefficient used in manning
    diffusivity - value of diffusivity coefficient
    tracer_init - initial tracer value

    Outputs:
    solver_obj - solver which we need to run to solve the problem
    update_forcings_tracer - function defining the updates to the model performed at each timestep
    outputdir - output directory
    """
    t_list = []

    def update_forcings_tracer(t_new):
        # update bathymetry
        old_bathymetry_2d.assign(bathymetry_2d)

        # extract new elevation and velocity and project onto CG space
        uv1, elev1 = solver_obj.fields.solution_2d.split()
        uv_cg.project(uv1)

        if wetting_and_drying:
            wd_bath_displacement = solver_obj.depth.wd_bathymetry_displacement
            depth.project(elev1 + wd_bath_displacement(elev1) + old_bathymetry_2d)
            elev_cg.project(elev1 + wd_bath_displacement(elev1))
        else:
            elev_cg.project(elev1)
            depth.project(elev_cg + old_bathymetry_2d)

        horizontal_velocity.interpolate(uv_cg[0])
        vertical_velocity.interpolate(uv_cg[1])

        # update bedfriction
        hc = conditional(depth > Constant(0.001), depth, Constant(0.001))
        aux = conditional(11.036*hc/ks > Constant(1.001), 11.036*hc/ks, Constant(1.001))
        qfc = Constant(2)/(ln(aux)/kappa)**2

        # calculate skin friction coefficient
        cfactor.interpolate(conditional(depth > ksp, 2*(((1/kappa)*ln(11.036*depth/ksp))**(-2)), Constant(0.0)))

        if morfac_transport:

            # if include sediment then update_forcings is run twice but only want to update bathymetry once
            t_list.append(t_new)
            double_factor = False

            if suspendedload:
                if len(t_list) > 1:
                    if t_list[len(t_list)-1] == t_list[len(t_list)-2]:
                        double_factor = True
            else:
                # if have no tracer then update_forcings is only run once so update bathymetry at each step
                double_factor = True                    

            if double_factor:
                z_n.assign(old_bathymetry_2d)

                # mu - ratio between skin friction and normal friction
                mu.interpolate(conditional(qfc > Constant(0), cfactor/qfc, Constant(0)))

                # bed shear stress
                unorm = ((horizontal_velocity**2) + (vertical_velocity**2))
                TOB.interpolate(rhow*Constant(0.5)*qfc*unorm)

                # calculate gradient of bed (noting bathymetry is -bed)
                dzdx.interpolate(old_bathymetry_2d.dx(0))
                dzdy.interpolate(old_bathymetry_2d.dx(1))

                if suspendedload:
                    # source term

                    # deposition flux - calculating coefficient to account for stronger conc at bed
                    B.interpolate(conditional(a > depth, a/a, a/depth))
                    ustar = sqrt(0.5*qfc*unorm)
                    exp1 = conditional((conditional((settling_velocity/(kappa*ustar)) - Constant(1) > Constant(0), (settling_velocity/(kappa*ustar)) - Constant(1), -(settling_velocity/(kappa*ustar)) + Constant(1))) > Constant(1e-4), conditional((settling_velocity/(kappa*ustar)) - Constant(1) > Constant(3), Constant(3), (settling_velocity/(kappa*ustar))-Constant(1)), Constant(0))
                    coefftest = conditional((conditional((settling_velocity/(kappa*ustar)) - Constant(1) > Constant(0), (settling_velocity/(kappa*ustar)) - Constant(1), -(settling_velocity/(kappa*ustar)) + Constant(1))) > Constant(1e-4), B*(Constant(1)-B**exp1)/exp1, -B*ln(B))
                    coeff.interpolate(conditional(coefftest > Constant(0), Constant(1)/coefftest, Constant(0)))

                    # erosion flux - above critical velocity bed is eroded
                    s0 = (conditional(rhow*Constant(0.5)*qfc*unorm*mu > Constant(0), rhow*Constant(0.5)*qfc*unorm*mu, Constant(0)) - taucr)/taucr
                    ceq.interpolate(Constant(0.015)*(average_size/a) * ((conditional(s0 < Constant(0), Constant(0), s0))**(1.5))/(dstar**0.3))

                    # calculate depth-averaged source term for sediment concentration equation
                    depo = settling_velocity*coeff
                    ero = settling_velocity*ceq

                    if cons_tracer:
                        source.interpolate(-(depo*solver_obj.fields.tracer_2d/(depth**2)) + (ero/depth))
                        qbsourcedepth.interpolate(-(depo*solver_obj.fields.tracer_2d/depth) + ero)
                    else:
                        source.interpolate(-(depo*solver_obj.fields.tracer_2d/depth) + (ero/depth))
                        qbsourcedepth.interpolate(-(depo*solver_obj.fields.tracer_2d) + ero)

                    if convectivevel:
                        x.update(depth, ustar)

                    # update sediment rate to ensure equilibrium at inflow
                    if cons_tracer:
                        sediment_rate.assign(depth.at([0, 0])*ceq.at([0, 0])/(coeff.at([0, 0])))
                    else:
                        sediment_rate.assign(ceq.at([0, 0])/(coeff.at([0, 0])))

                if bedload:

                    # calculate angle of flow
                    calfa.interpolate(horizontal_velocity/sqrt(unorm))
                    salfa.interpolate(vertical_velocity/sqrt(unorm))

                    if slope_eff:
                        # slope effect magnitude correction due to gravity where beta is a parameter normally set to 1.3
                        # we use z_n1 and equals so that we can use an implicit method in Exner
                        slopecoef = (Constant(1) + beta*(z_n1.dx(0)*calfa + z_n1.dx(1)*salfa))
                    else:
                        slopecoef = Constant(1.0)

                    if angle_correction:
                        # slope effect angle correction due to gravity
                        tt1 = conditional(rhow*Constant(0.5)*qfc*unorm > Constant(1e-10), sqrt(cparam/(rhow*Constant(0.5)*qfc*unorm)), sqrt(cparam/Constant(1e-10)))
                        # add on a factor of the bed gradient to the normal
                        aa = salfa + tt1*dzdy
                        bb = calfa + tt1*dzdx
                        norm = conditional(sqrt(aa**2 + bb**2) > Constant(1e-10), sqrt(aa**2 + bb**2), Constant(1e-10))
                        # we use z_n1 and equals so that we can use an implicit method in Exner
                        calfamod = (calfa + (tt1*z_n1.dx(0)))/norm
                        salfamod = (salfa + (tt1*z_n1.dx(1)))/norm

                    if seccurrent:
                        # accounts for helical flow effect in a curver channel

                        # again use z_n1 and equals so can use an implicit method in Exner
                        free_surface_dx = depth.dx(0) - z_n1.dx(0)
                        free_surface_dy = depth.dx(1) - z_n1.dx(1)

                        velocity_slide = (horizontal_velocity*free_surface_dy)-(vertical_velocity*free_surface_dx)

                        tandelta_factor.interpolate(Constant(7)*g*rhow*depth*qfc/(Constant(2)*alpha_secc*((horizontal_velocity**2) + (vertical_velocity**2))))

                        if angle_correction:
                            # if angle has already been corrected we must alter the corrected angle to obtain the corrected secondary current angle
                            t_1 = (TOB*slopecoef*calfamod) + (vertical_velocity*tandelta_factor*velocity_slide)
                            t_2 = (TOB*slopecoef*salfamod) - (horizontal_velocity*tandelta_factor*velocity_slide)
                        else:
                            t_1 = (TOB*slopecoef*calfa) + (vertical_velocity*tandelta_factor*velocity_slide)
                            t_2 = ((TOB*slopecoef*salfa) - (horizontal_velocity*tandelta_factor*velocity_slide))

                        # calculated to normalise the new angles
                        t4 = sqrt((t_1**2) + (t_2**2))

                        # updated magnitude correction and angle corrections
                        slopecoef = t4/TOB

                        calfanew = t_1/t4
                        salfanew = t_2/t4

                    # implement meyer-peter-muller bedload transport formula
                    thetaprime = mu*(rhow*Constant(0.5)*qfc*unorm)/((rhos-rhow)*g*average_size)

                    # if velocity above a certain critical value then transport occurs
                    phi.interpolate(conditional(thetaprime < thetacr, Constant(0), Constant(8)*(thetaprime-thetacr)**1.5))

                    # bedload transport flux with magnitude correction
                    qb_total = slopecoef*phi*sqrt(g*R*average_size**3)

                    # add time derivative to exner equation with a morphological scale factor
                    f = (((Constant(1)-porosity)*(z_n1 - z_n)/(dt*morfac)) * v)*dx

                    # formulate bedload transport flux with correct angle depending on corrections implemented
                    if angle_correction and seccurrent is False:
                        qbx = qb_total*calfamod
                        qby = qb_total*salfamod
                    elif seccurrent:
                        qbx = qb_total*calfanew
                        qby = qb_total*salfanew
                    else:
                        qbx = qb_total*calfa
                        qby = qb_total*salfa

                    # add bedload transport to exner equation
                    f += -(v*((qbx*n[0]) + (qby*n[1])))*ds(1) - (v*((qbx*n[0]) + (qby*n[1])))*ds(2) + (qbx*(v.dx(0)) + qby*(v.dx(1)))*dx

                else:
                    # if no bedload transport component initialise exner equation with time derivative
                    f = (((Constant(1)-porosity)*(z_n1 - z_n)/(dt*morfac)) * v)*dx

                if suspendedload:
                    # add suspended sediment transport to exner equation multiplied by depth as the exner equation is not depth-averaged

                    qbsourcedepth.interpolate(source*depth)
                    f += - (qbsourcedepth*v)*dx

                # solve exner equation using finite element methods
                solve(f == 0, z_n1)

                # update bed
                bathymetry_2d.assign(z_n1)

                if round(t_new, 2)%t_export == 0:
                    bathy_file.write(bathymetry_2d)                

    # choose directory to output results
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    outputdir = 'outputs' + st

    # define bathymetry_file
    bathy_file = File(outputdir + "/bathy.pvd")


    # final time of simulation
    t_end = final_time/morfac

    # export interval in seconds
    t_export = t_end/45

    print_output('Exporting to '+outputdir)

    x, y = SpatialCoordinate(mesh2d)

    # define function spaces
    P1_2d = FunctionSpace(mesh2d, 'DG', 1)
    V = FunctionSpace(mesh2d, 'CG', 1)
    vector_cg = VectorFunctionSpace(mesh2d, 'CG', 1)

    # define test functions on mesh
    v = TestFunction(V)
    n = FacetNormal(mesh2d)
    z_n1 = Function(V, name="z^{n+1}")
    z_n = Function(V, name="z^{n}")

    # define parameters
    g = physical_constants['g_grav']
    rhow = physical_constants['rho0']
    kappa = physical_constants['von_karman']
    porosity = Constant(0.4)

    ksp = Constant(3*average_size)
    a = Constant(ks/2)
    viscosity = Constant(viscosity_morph)
    
    # magnitude slope effect parameter
    beta = Constant(beta_fn)
    # angle correction slope effect parameters
    surbeta2 = Constant(surbeta2_fn)
    cparam = Constant((rhos-rhow)*g*average_size*(surbeta2**2))
    # secondary current parameter
    alpha_secc = Constant(alpha_secc_fn)    

    # calculate critical shields parameter thetacr
    R = Constant(rhos/rhow - 1)

    dstar = Constant(average_size*((g*R)/(viscosity**2))**(1/3))
    if max(dstar.dat.data[:] < 1):
        print('ERROR: dstar value less than 1')
    elif max(dstar.dat.data[:] < 4):
        thetacr = Constant(0.24*(dstar**(-1)))
    elif max(dstar.dat.data[:] < 10):
        thetacr = Constant(0.14*(dstar**(-0.64)))
    elif max(dstar.dat.data[:] < 20):
        thetacr = Constant(0.04*(dstar**(-0.1)))
    elif max(dstar.dat.data[:] < 150):
        thetacr = Constant(0.013*(dstar**(0.29)))
    else:
        thetacr = Constant(0.055)

    # critical bed shear stress
    taucr = Constant((rhos-rhow)*g*average_size*thetacr)

    # calculate settling velocity
    if average_size <= 1e-04:
        settling_velocity = Constant(g*(average_size**2)*R/(18*viscosity))
    elif average_size <= 1e-03:
        settling_velocity = Constant((10*viscosity/average_size)*(sqrt(1 + 0.01*((R*g*(average_size**3))/(viscosity**2)))-1))
    else:
        settling_velocity = Constant(1.1*sqrt(g*average_size*R))

    # initialise velocity, elevation and depth
    elev_init, uv_init = initialise_fields(mesh2d, input_dir, outputdir)

    uv_cg = Function(vector_cg).interpolate(uv_init)

    elev_cg = Function(V).interpolate(elev_init)

    if wetting_and_drying:
        H = Function(V).project(elev_cg + bathymetry_2d)
        depth = Function(V).project(H + (Constant(0.5) * (sqrt(H ** 2 + wetting_alpha ** 2) - H)))
    else:
        depth = Function(V).project(elev_cg + bathymetry_2d)

    old_bathymetry_2d = Function(V).interpolate(bathymetry_2d)
    #bathy_file.write(bathymetry_2d)    

    horizontal_velocity = Function(V).interpolate(uv_cg[0])
    vertical_velocity = Function(V).interpolate(uv_cg[1])

    # define bed friction
    hc = Function(P1_2d).interpolate(conditional(depth > Constant(0.001), depth, Constant(0.001)))
    aux = Function(P1_2d).interpolate(conditional(11.036*hc/ks > Constant(1.001), 11.036*hc/ks, Constant(1.001)))
    qfc = Function(P1_2d).interpolate(Constant(2)/(ln(aux)/kappa)**2)
    # skin friction coefficient
    cfactor = Function(P1_2d).interpolate(conditional(depth > ksp, Constant(2)*(((1/kappa)*ln(11.036*depth/ksp))**(-2)), Constant(0.0)))
    # mu - ratio between skin friction and normal friction
    mu = Function(P1_2d).interpolate(conditional(qfc > Constant(0), cfactor/qfc, Constant(0)))

    # calculate bed shear stress
    unorm = (horizontal_velocity**2) + (vertical_velocity**2)
    TOB = Function(V).interpolate(rhow*Constant(0.5)*qfc*unorm)

    # define bed gradient
    dzdx = Function(V).interpolate(old_bathymetry_2d.dx(0))
    dzdy = Function(V).interpolate(old_bathymetry_2d.dx(1))

    if suspendedload:
        # deposition flux - calculating coefficient to account for stronger conc at bed
        B = Function(P1_2d).interpolate(conditional(a > depth, a/a, a/depth))
        ustar = sqrt(Constant(0.5)*qfc*unorm)
        exp1 = conditional((conditional((settling_velocity/(kappa*ustar)) - Constant(1) > Constant(0), (settling_velocity/(kappa*ustar)) - Constant(1), -(settling_velocity/(kappa*ustar)) + Constant(1))) > Constant(1e-04), conditional((settling_velocity/(kappa*ustar)) - Constant(1) > Constant(3), Constant(3), (settling_velocity/(kappa*ustar))-Constant(1)), Constant(0))
        coefftest = conditional((conditional((settling_velocity/(kappa*ustar)) - Constant(1) > Constant(0), (settling_velocity/(kappa*ustar)) - Constant(1), -(settling_velocity/(kappa*ustar)) + Constant(1))) > Constant(1e-04), B*(Constant(1)-B**exp1)/exp1, -B*ln(B))
        coeff = Function(P1_2d).interpolate(conditional(conditional(coefftest > Constant(1e-12), Constant(1)/coefftest, Constant(1e12)) > Constant(1), conditional(coefftest > Constant(1e-12), Constant(1)/coefftest, Constant(1e12)), Constant(1)))

        # erosion flux - above critical velocity bed is eroded
        s0 = (conditional(rhow*Constant(0.5)*qfc*unorm*mu > Constant(0), rhow*Constant(0.5)*qfc*unorm*mu, Constant(0)) - taucr)/taucr
        ceq = Function(P1_2d).interpolate(Constant(0.015)*(average_size/a) * ((conditional(s0 < Constant(0), Constant(0), s0))**(1.5))/(dstar**0.3))

        if convectivevel:
            x = Corrective_Velocity_Factor(depth, ksp, ks, settling_velocity, ustar)
        # update sediment rate to ensure equilibrium at inflow
        if cons_tracer:
            sediment_rate = Constant(depth.at([0, 0])*ceq.at([0, 0])/(coeff.at([0, 0])))
            testtracer = Function(P1_2d).interpolate(depth*ceq/coeff)
        else:
            sediment_rate = Constant(ceq.at([0, 0])/(coeff.at([0, 0])))
            testtracer = Function(P1_2d).interpolate(ceq/coeff)

        # get individual terms
        depo = settling_velocity*coeff
        ero = settling_velocity*ceq

        # calculate depth-averaged source term for sediment concentration equation
        if cons_tracer:
            source = Function(P1_2d).interpolate(-(depo*sediment_rate/(depth**2)) + (ero/depth))
            qbsourcedepth = Function(P1_2d).interpolate(-(depo*sediment_rate/depth) + ero)
        else:
            source = Function(P1_2d).interpolate(-(depo*sediment_rate/depth) + (ero/depth))
            qbsourcedepth = Function(P1_2d).interpolate(-(depo*sediment_rate) + ero)
                
    if bedload:
        # calculate angle of flow
        calfa = Function(V).interpolate(horizontal_velocity/sqrt(unorm))
        salfa = Function(V).interpolate(vertical_velocity/sqrt(unorm))
        div_function = Function(vector_cg).interpolate(as_vector((calfa, salfa)))

        if slope_eff:
            # slope effect magnitude correction due to gravity where beta is a parameter normally set to 1.3
            slopecoef = Function(V).interpolate(Constant(1) + beta*(dzdx*calfa + dzdy*salfa))
        else:
            slopecoef = Function(V).interpolate(Constant(1.0))

        if angle_correction:
            # slope effect angle correction due to gravity
            tt1 = conditional(rhow*Constant(0.5)*qfc*unorm > Constant(1e-10), sqrt(cparam/(rhow*Constant(0.5)*qfc*unorm)), sqrt(cparam/Constant(1e-10)))
            # add on a factor of the bed gradient to the normal
            aa = salfa + tt1*dzdy
            bb = calfa + tt1*dzdx
            norm = conditional(sqrt(aa**2 + bb**2) > Constant(1e-10), sqrt(aa**2 + bb**2), Constant(1e-10))

        if seccurrent:
            # accounts for helical flow effect in a curver channel
            free_surface_dx = Function(V).interpolate(elev_cg.dx(0))
            free_surface_dy = Function(V).interpolate(elev_cg.dx(1))

            velocity_slide = (horizontal_velocity*free_surface_dy)-(vertical_velocity*free_surface_dx)

            tandelta_factor = Function(V).interpolate(Constant(7)*g*rhow*depth*qfc/(Constant(2)*alpha_secc*((horizontal_velocity**2) + (vertical_velocity**2))))

            t_1 = (TOB*slopecoef*calfa) + (vertical_velocity*tandelta_factor*velocity_slide)
            t_2 = ((TOB*slopecoef*salfa) - (horizontal_velocity*tandelta_factor*velocity_slide))

            # calculated to normalise the new angles
            t4 = sqrt((t_1**2) + (t_2**2))

            # updated magnitude correction and angle corrections
            slopecoef = t4/TOB

        # implement meyer-peter-muller bedload transport formula
        thetaprime = mu*(rhow*Constant(0.5)*qfc*unorm)/((rhos-rhow)*g*average_size)

        # if velocity above a certain critical value then transport occurs
        phi = Function(V).interpolate(conditional(thetaprime < thetacr, 0, Constant(8)*(thetaprime-thetacr)**1.5))

    # set up solver
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)

    options = solver_obj.options
    options.simulation_export_time = t_export
    options.simulation_end_time = t_end
    options.output_directory = outputdir
    options.check_volume_conservation_2d = True

    if suspendedload:
        # switch on tracer calculation if using sediment transport component
        options.solve_tracer = True
        options.use_tracer_conservative_form = cons_tracer
        options.fields_to_export = ['tracer_2d', 'uv_2d', 'elev_2d']
        options.tracer_advective_velocity_factor = x.corr_vel_factor
        options.tracer_source_2d = source
        options.check_tracer_conservation = True
        options.use_lax_friedrichs_tracer = False
    else:
        options.solve_tracer = False
        options.fields_to_export = ['uv_2d', 'elev_2d', 'bathymetry_2d']        

    # using nikuradse friction
    options.nikuradse_bed_roughness = ksp

    # set horizontal diffusivity parameter
    options.horizontal_diffusivity = Constant(diffusivity)
    options.horizontal_viscosity = Constant(viscosity_hydro)
    # crank-nicholson used to integrate in time system of ODEs resulting from application of galerkin FEM
    options.timestepper_type = 'CrankNicolson'
    options.timestepper_options.implicitness_theta = 1.0
    options.use_wetting_and_drying = wetting_and_drying
    options.wetting_and_drying_alpha = Constant(wetting_alpha)
    options.norm_smoother = Constant(wetting_alpha)

    if not hasattr(options.timestepper_options, 'use_automatic_timestep'):
        options.timestep = dt

    c = call.TracerTotalMassConservation2DCallback('tracer_2d',
                                                   solver_obj, export_to_hdf5=True, append_to_log=False)
    solver_obj.add_callback(c, eval_interval='timestep')

    # set boundary conditions

    left_bnd_id = 1
    right_bnd_id = 2

    swe_bnd = {}

    swe_bnd[left_bnd_id] = {'flux': Constant(-0.22)}
    swe_bnd[right_bnd_id] = {'elev': Constant(0.397)}    

    solver_obj.bnd_functions['shallow_water'] = swe_bnd

    if suspendedload:
        solver_obj.bnd_functions['tracer'] = {left_bnd_id: {'value': sediment_rate, 'flux': Constant(-0.22)}, right_bnd_id: {'elev': Constant(0.397)} }

        # set initial conditions
        solver_obj.assign_initial_conditions(uv=uv_init, elev=elev_init, tracer=testtracer)

    else:
        # set initial conditions
        solver_obj.assign_initial_conditions(uv=uv_init, elev=elev_init)

    return solver_obj, update_forcings_tracer, outputdir



## Note it is necessary to run trench_hydro first to get the hydrodynamics simulation

# define mesh
lx = 16
ly = 1.1
nx = lx*5
ny = 5
mesh2d = RectangleMesh(nx, ny, lx, ly)

x, y = SpatialCoordinate(mesh2d)

# define function spaces
V = FunctionSpace(mesh2d, 'CG', 1)
P1_2d = FunctionSpace(mesh2d, 'DG', 1)

# define underlying bathymetry
bathymetry_2d = Function(V, name='Bathymetry')
initialdepth = Constant(0.397)
depth_riv = Constant(initialdepth - 0.397)
depth_trench = Constant(depth_riv - 0.15)
depth_diff = depth_trench - depth_riv

trench = conditional(le(x, 5), depth_riv, conditional(le(x, 6.5), (1/1.5)*depth_diff*(x-6.5) + depth_trench,
                                                             conditional(le(x, 9.5), depth_trench, conditional(le(x, 11), -(1/1.5)*depth_diff*(x-11) + depth_riv,
                                                                                                                          depth_riv))))
bathymetry_2d.interpolate(-trench)

solver_obj, update_forcings_tracer, outputdir = morphological(morfac=100, morfac_transport=True, suspendedload=True, convectivevel=True,
                  bedload=True, angle_correction=False, slope_eff=True, seccurrent=False, wetting_and_drying = False,
                                                                        mesh2d=mesh2d, bathymetry_2d=bathymetry_2d, input_dir='hydrodynamics_trench', ks=0.025, average_size=160 * (10**(-6)), dt=0.3, final_time=15*3600, cons_tracer=conservative)#, wetting_alpha=wd_fn)

# run model
solver_obj.iterate(update_forcings=update_forcings_tracer)

# record final tracer and final bathymetry
xaxisthetis1 = []
tracerthetis1 = []
baththetis1 = []

for i in np.linspace(0, 15.8, 80):
    xaxisthetis1.append(i)
    if conservative:
        d = solver_obj.fields.bathymetry_2d.at([i, 0.55]) + solver_obj.fields.elev_2d.at([i, 0.55])
        tracerthetis1.append(solver_obj.fields.tracer_2d.at([i, 0.55])/d)
        baththetis1.append(solver_obj.fields.bathymetry_2d.at([i, 0.55]))
    else:
        tracerthetis1.append(solver_obj.fields.tracer_2d.at([i, 0.55]))
        baththetis1.append(solver_obj.fields.bathymetry_2d.at([i, 0.55]))

    # check tracer conservation
tracer_mass_int, tracer_mass_int_rerr = solver_obj.callbacks['timestep']['tracer_2d total mass']()
print("Tracer total mass error: %11.4e" % (tracer_mass_int_rerr))

# check tracer and bathymetry values using previous runs
tracer_solution = pd.read_csv('tracer.csv')
bed_solution = pd.read_csv('bed.csv')

assert max([abs((tracer_solution['Tracer'][i] - tracerthetis1[i])/tracer_solution['Tracer'][i]) for i in range(len(tracerthetis1))]) < 0.1, "error in tracer"

assert max([abs((bed_solution['Bathymetry'][i] - baththetis1[i])) for i in range(len(baththetis1))]) < 0.007, "error in bed level"
