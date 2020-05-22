"""
Base functions used to when modelling morphological changes.

For more details on equations see

[1] Clare et al. 2020. “Hydro-morphodynamics 2D Modelling Using a Discontinuous
    Galerkin Discretisation.” EarthArXiv. January 9. doi:10.31223/osf.io/tpqvy.

"""

from thetis import *
import time
import datetime
import numpy as np
from firedrake import *
import os
import callback_cons_tracer as call


def morphological(boundary_conditions_fn, morfac, morfac_transport, suspendedload, convectivevel,
                  bedload, angle_correction, slope_eff, seccurrent,
                  mesh2d, bathymetry_2d, input_dir, ks, average_size, dt, final_time,
                  beta_fn = 1.3, surbeta2_fn = 1/1.5, alpha_secc_fn = 0.75, viscosity_hydro=1e-6, viscosity_morph=1e-6,
                  wetting_and_drying=False, wetting_alpha=0.1, rhos=2650, cons_tracer=False, friction='nikuradse', friction_coef=0, diffusivity=0.15, tracer_init=None):
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
            wd_bath_displacement = solver_obj.eq_sw.bathymetry_displacement_mass_term.wd_bathymetry_displacement
            depth.project(elev1 + wd_bath_displacement(elev1) + old_bathymetry_2d)
            elev_cg.project(elev1 + wd_bath_displacement(elev1))
        else:
            elev_cg.project(elev1)
            depth.project(elev_cg + old_bathymetry_2d)

        horizontal_velocity.interpolate(uv_cg[0])
        vertical_velocity.interpolate(uv_cg[1])

        # update bedfriction
        hc.interpolate(conditional(depth > Constant(0.001), depth, Constant(0.001)))
        aux.assign(conditional(11.036*hc/ks > Constant(1.001), 11.036*hc/ks, Constant(1.001)))
        qfc.assign(Constant(2)/(ln(aux)/kappa)**2)

        # calculate skin friction coefficient
        hclip.interpolate(conditional(ksp > depth, ksp, depth))
        cfactor.interpolate(conditional(depth > ksp, 2*(((1/kappa)*ln(11.036*hclip/ksp))**(-2)), Constant(0.0)))

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
                mu.assign(conditional(qfc > Constant(0), cfactor/qfc, Constant(0)))

                # bed shear stress
                unorm.interpolate((horizontal_velocity**2) + (vertical_velocity**2))
                TOB.interpolate(rhow*0.5*qfc*unorm)

                # calculate gradient of bed (noting bathymetry is -bed)
                dzdx.interpolate(old_bathymetry_2d.dx(0))
                dzdy.interpolate(old_bathymetry_2d.dx(1))

                if suspendedload:
                    # source term

                    # deposition flux - calculating coefficient to account for stronger conc at bed
                    B.interpolate(conditional(a > depth, a/a, a/depth))
                    ustar.interpolate(sqrt(0.5*qfc*unorm))
                    exp1.assign(conditional((conditional((settling_velocity/(kappa*ustar)) - Constant(1) > Constant(0), (settling_velocity/(kappa*ustar)) - Constant(1), -(settling_velocity/(kappa*ustar)) + Constant(1))) > Constant(1e-4), conditional((settling_velocity/(kappa*ustar)) - Constant(1) > Constant(3), Constant(3), (settling_velocity/(kappa*ustar))-Constant(1)), Constant(0)))
                    coefftest.assign(conditional((conditional((settling_velocity/(kappa*ustar)) - Constant(1) > Constant(0), (settling_velocity/(kappa*ustar)) - Constant(1), -(settling_velocity/(kappa*ustar)) + Constant(1))) > Constant(1e-4), B*(Constant(1)-B**exp1)/exp1, -B*ln(B)))
                    coeff.assign(conditional(coefftest > Constant(0), Constant(1)/coefftest, Constant(0)))

                    # erosion flux - above critical velocity bed is eroded
                    s0.assign((conditional(rhow*Constant(0.5)*qfc*unorm*mu > Constant(0), rhow*Constant(0.5)*qfc*unorm*mu, Constant(0)) - taucr)/taucr)
                    ceq.assign(Constant(0.015)*(average_size/a) * ((conditional(s0 < Constant(0), Constant(0), s0))**(1.5))/(dstar**0.3))

                    # calculate depth-averaged source term for sediment concentration equation
                    depo.interpolate(settling_velocity*coeff)
                    ero.interpolate(settling_velocity*ceq)

                    if cons_tracer:
                        source.interpolate(-(depo*solver_obj.fields.tracer_2d/(depth**2)) + (ero/depth))
                        qbsourcedepth.interpolate(-(depo*solver_obj.fields.tracer_2d/depth) + ero)
                    else:
                        source.interpolate(-(depo*solver_obj.fields.tracer_2d/depth) + (ero/depth))
                        qbsourcedepth.interpolate(-(depo*solver_obj.fields.tracer_2d) + ero)

                    if convectivevel:
                        # correction factor to advection velocity in sediment concentration equation
                        Bconv.interpolate(conditional(depth > Constant(1.1)*ksp, ksp/depth, Constant(1/1.1)))
                        Aconv.interpolate(conditional(depth > Constant(1.1)*a, a/depth, Constant(1/1.1)))

                        # take max of value calculated either by ksp or depth
                        Amax.assign(conditional(Aconv > Bconv, Aconv, Bconv))

                        r1conv.assign(Constant(1) - Constant(1/kappa)*conditional(settling_velocity/ustar < Constant(1), settling_velocity/ustar, Constant(1)))

                        Ione.assign(conditional(r1conv > Constant(1e-8), (Constant(1) - Amax**r1conv)/r1conv, conditional(r1conv < Constant(- 1e-8), (Constant(1) - Amax**r1conv)/r1conv, ln(Amax))))

                        Itwo.assign(conditional(r1conv > Constant(1e-8), -(Ione + (ln(Amax)*(Amax**r1conv)))/r1conv, conditional(r1conv < - Constant(1e-8), -(Ione + (ln(Amax)*(Amax**r1conv)))/r1conv, -Constant(0.5)*ln(Amax)**2)))

                        alpha.assign(-(Itwo - (ln(Amax) - Constant(ln(30)))*Ione)/(Ione * ((ln(Amax) - Constant(ln(30))) + Constant(1))))

                        # final correction factor
                        alphatest2.assign(conditional(conditional(alpha > Constant(1), Constant(1), alpha) < Constant(0), Constant(0), conditional(alpha > Constant(1), Constant(1), alpha)))

                    else:
                        alphatest2.assign(Constant(1.0))

                    # update sediment rate to ensure equilibrium at inflow
                    if cons_tracer:
                        sediment_rate.assign(depth.at([0, 0])*ceq.at([0, 0])/(coeff.at([0, 0])))
                    else:
                        sediment_rate.assign(ceq.at([0, 0])/(coeff.at([0, 0])))

                if bedload:

                    # calculate angle of flow
                    calfa.interpolate(horizontal_velocity/sqrt(unorm))
                    salfa.interpolate(vertical_velocity/sqrt(unorm))
                    div_function.interpolate(as_vector((calfa, salfa)))

                    if slope_eff:
                        # slope effect magnitude correction due to gravity where beta is a parameter normally set to 1.3
                        # we use z_n1 and equals so that we can use an implicit method in Exner
                        slopecoef = (Constant(1) + beta*(z_n1.dx(0)*calfa + z_n1.dx(1)*salfa))
                    else:
                        slopecoef = Constant(1.0)

                    if angle_correction:
                        # slope effect angle correction due to gravity
                        tt1.interpolate(conditional(rhow*Constant(0.5)*qfc*unorm > Constant(1e-10), sqrt(cparam/(rhow*Constant(0.5)*qfc*unorm)), sqrt(cparam/Constant(1e-10))))
                        # add on a factor of the bed gradient to the normal
                        aa.assign(salfa + tt1*dzdy)
                        bb.assign(calfa + tt1*dzdx)
                        norm.assign(conditional(sqrt(aa**2 + bb**2) > Constant(1e-10), sqrt(aa**2 + bb**2), Constant(1e-10)))
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
                    thetaprime.interpolate(mu*(rhow*Constant(0.5)*qfc*unorm)/((rhos-rhow)*g*average_size))

                    # if velocity above a certain critical value then transport occurs
                    phi.assign(conditional(thetaprime < thetacr, Constant(0), Constant(8)*(thetaprime-thetacr)**1.5))

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

    # choose directory to output results
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    outputdir = 'outputs' + st

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

    horizontal_velocity = Function(V).interpolate(uv_cg[0])
    vertical_velocity = Function(V).interpolate(uv_cg[1])

    # define bed friction
    hc = Function(P1_2d).interpolate(conditional(depth > Constant(0.001), depth, Constant(0.001)))
    aux = Function(P1_2d).interpolate(conditional(11.036*hc/ks > Constant(1.001), 11.036*hc/ks, Constant(1.001)))
    qfc = Function(P1_2d).interpolate(Constant(2)/(ln(aux)/kappa)**2)
    # skin friction coefficient
    hclip = Function(P1_2d).interpolate(conditional(ksp > depth, ksp, depth))
    cfactor = Function(P1_2d).interpolate(conditional(depth > ksp, Constant(2)*(((1/kappa)*ln(11.036*hclip/ksp))**(-2)), Constant(0.0)))
    # mu - ratio between skin friction and normal friction
    mu = Function(P1_2d).interpolate(conditional(qfc > Constant(0), cfactor/qfc, Constant(0)))

    # calculate bed shear stress
    unorm = Function(P1_2d).interpolate((horizontal_velocity**2) + (vertical_velocity**2))
    TOB = Function(V).interpolate(rhow*Constant(0.5)*qfc*unorm)

    # define bed gradient
    dzdx = Function(V).interpolate(old_bathymetry_2d.dx(0))
    dzdy = Function(V).interpolate(old_bathymetry_2d.dx(1))

    if suspendedload:
        # deposition flux - calculating coefficient to account for stronger conc at bed
        B = Function(P1_2d).interpolate(conditional(a > depth, a/a, a/depth))
        ustar = Function(P1_2d).interpolate(sqrt(Constant(0.5)*qfc*unorm))
        exp1 = Function(P1_2d).interpolate(conditional((conditional((settling_velocity/(kappa*ustar)) - Constant(1) > Constant(0), (settling_velocity/(kappa*ustar)) - Constant(1), -(settling_velocity/(kappa*ustar)) + Constant(1))) > Constant(1e-04), conditional((settling_velocity/(kappa*ustar)) - Constant(1) > Constant(3), Constant(3), (settling_velocity/(kappa*ustar))-Constant(1)), Constant(0)))
        coefftest = Function(P1_2d).interpolate(conditional((conditional((settling_velocity/(kappa*ustar)) - Constant(1) > Constant(0), (settling_velocity/(kappa*ustar)) - Constant(1), -(settling_velocity/(kappa*ustar)) + Constant(1))) > Constant(1e-04), B*(Constant(1)-B**exp1)/exp1, -B*ln(B)))
        coeff = Function(P1_2d).interpolate(conditional(conditional(coefftest > Constant(1e-12), Constant(1)/coefftest, Constant(1e12)) > Constant(1), conditional(coefftest > Constant(1e-12), Constant(1)/coefftest, Constant(1e12)), Constant(1)))

        # erosion flux - above critical velocity bed is eroded
        s0 = Function(P1_2d).interpolate((conditional(rhow*Constant(0.5)*qfc*unorm*mu > Constant(0), rhow*Constant(0.5)*qfc*unorm*mu, Constant(0)) - taucr)/taucr)
        ceq = Function(P1_2d).interpolate(Constant(0.015)*(average_size/a) * ((conditional(s0 < Constant(0), Constant(0), s0))**(1.5))/(dstar**0.3))

        if convectivevel:
            # correction factor to advection velocity in sediment concentration equation

            Bconv = Function(P1_2d).interpolate(conditional(depth > Constant(1.1)*ksp, ksp/depth, Constant(1/1.1)))
            Aconv = Function(P1_2d).interpolate(conditional(depth > Constant(1.1)*a, a/depth, Constant(1/1.1)))

            # take max of value calculated either by ksp or depth
            Amax = Function(P1_2d).interpolate(conditional(Aconv > Bconv, Aconv, Bconv))

            r1conv = Function(P1_2d).interpolate(Constant(1) - (1/kappa)*conditional(settling_velocity/ustar < Constant(1), settling_velocity/ustar, Constant(1)))

            Ione = Function(P1_2d).interpolate(conditional(r1conv > Constant(1e-8), (Constant(1) - Amax**r1conv)/r1conv, conditional(r1conv < Constant(- 1e-8), (Constant(1) - Amax**r1conv)/r1conv, ln(Amax))))

            Itwo = Function(P1_2d).interpolate(conditional(r1conv > Constant(1e-8), -(Ione + (ln(Amax)*(Amax**r1conv)))/r1conv, conditional(r1conv < Constant(- 1e-8), -(Ione + (ln(Amax)*(Amax**r1conv)))/r1conv, Constant(-0.5)*ln(Amax)**2)))

            alpha = Function(P1_2d).interpolate(-(Itwo - (ln(Amax) - ln(30))*Ione)/(Ione * ((ln(Amax) - ln(30)) + Constant(1))))

            # final correction factor
            alphatest2 = Function(P1_2d).interpolate(conditional(conditional(alpha > Constant(1), Constant(1), alpha) < Constant(0), Constant(0), conditional(alpha > Constant(1), Constant(1), alpha)))

        else:
            alphatest2 = Function(P1_2d).interpolate(Constant(1.0))

        # update sediment rate to ensure equilibrium at inflow
        if cons_tracer:
            sediment_rate = Constant(depth.at([0, 0])*ceq.at([0, 0])/(coeff.at([0, 0])))
            testtracer = Function(P1_2d).interpolate(depth*ceq/coeff)
        else:
            sediment_rate = Constant(ceq.at([0, 0])/(coeff.at([0, 0])))
            testtracer = Function(P1_2d).interpolate(ceq/coeff)

        # get individual terms
        depo = Function(P1_2d).interpolate(settling_velocity*coeff)
        ero = Function(P1_2d).interpolate(settling_velocity*ceq)

        # calculate depth-averaged source term for sediment concentration equation
        if cons_tracer:
            if tracer_init is None:
                source = Function(P1_2d).interpolate(-(depo*sediment_rate/(depth**2)) + (ero/depth))
                qbsourcedepth = Function(P1_2d).interpolate(-(depo*sediment_rate/depth) + ero)
            else:
                source = Function(P1_2d).interpolate(-(depo*tracer_init/depth) + (ero/depth))
                qbsourcedepth = Function(P1_2d).interpolate(-(depo*tracer_init) + ero)
        else:
            if tracer_init is None:
                source = Function(P1_2d).interpolate(-(depo*sediment_rate/depth) + (ero/depth))
                qbsourcedepth = Function(P1_2d).interpolate(-(depo*sediment_rate) + ero)
            else:
                source = Function(P1_2d).interpolate(-(depo*tracer_init/depth) + (ero/depth))
                qbsourcedepth = Function(P1_2d).interpolate(-(depo*tracer_init) + ero)

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
            tt1 = Function(V).interpolate(conditional(rhow*Constant(0.5)*qfc*unorm > Constant(1e-10), sqrt(cparam/(rhow*Constant(0.5)*qfc*unorm)), sqrt(cparam/Constant(1e-10))))
            # add on a factor of the bed gradient to the normal
            aa = Function(V).interpolate(salfa + tt1*dzdy)
            bb = Function(V).interpolate(calfa + tt1*dzdx)
            norm = Function(V).interpolate(conditional(sqrt(aa**2 + bb**2) > Constant(1e-10), sqrt(aa**2 + bb**2), Constant(1e-10)))

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
        thetaprime = Function(V).interpolate(mu*(rhow*Constant(0.5)*qfc*unorm)/((rhos-rhow)*g*average_size))

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
        options.tracer_advective_velocity_factor = alphatest2
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
    swe_bnd, left_bnd_id, right_bnd_id, in_constant, out_constant, left_string, right_string = boundary_conditions_fn()

    for j in range(len(in_constant)):
        exec('constant_in' + str(j) + ' = Constant(' + str(in_constant[j]) + ')', globals())

    str1 = '{'
    if len(left_string) > 0:
        for i in range(len(left_string)):
            str1 += "'" + str(left_string[i]) + "': constant_in" + str(i) + ","
        str1 = str1[0:len(str1)-1] + "}"
        exec('swe_bnd[left_bnd_id] = ' + str1)

    for k in range(len(out_constant)):
        exec('constant_out' + str(k) + '= Constant(' + str(out_constant[k]) + ')', globals())

    str2 = '{'
    if len(right_string) > 0:
        for i in range(len(right_string)):
            str2 += "'" + str(right_string[i]) + "': constant_out" + str(i) + ","
        str2 = str2[0:len(str2)-1] + "}"
        exec('swe_bnd[right_bnd_id] = ' + str2)

    solver_obj.bnd_functions['shallow_water'] = swe_bnd

    if suspendedload:
        solver_obj.bnd_functions['tracer'] = {1: {'value': sediment_rate}}

        for i in solver_obj.bnd_functions['tracer'].keys():
            if i in solver_obj.bnd_functions['shallow_water'].keys():
                solver_obj.bnd_functions['tracer'][i].update(solver_obj.bnd_functions['shallow_water'][i])
        for i in solver_obj.bnd_functions['shallow_water'].keys():
            if i not in solver_obj.bnd_functions['tracer'].keys():
                solver_obj.bnd_functions['tracer'].update({i: solver_obj.bnd_functions['shallow_water'][i]})

        # set initial conditions
        if tracer_init is None:
            solver_obj.assign_initial_conditions(uv=uv_init, elev=elev_init, tracer=testtracer)
        else:
            if cons_tracer:
                tracer_init_int = Function(P1_2d).interpolate(tracer_init*depth)
                solver_obj.assign_initial_conditions(elev=elev_init, uv=uv_init, tracer=tracer_init_int)
            else:
                solver_obj.assign_initial_conditions(elev=elev_init, uv=uv_init, tracer=tracer_init)
    else:
        # set initial conditions
        solver_obj.assign_initial_conditions(uv=uv_init, elev=elev_init)

    return solver_obj, update_forcings_tracer, outputdir


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
