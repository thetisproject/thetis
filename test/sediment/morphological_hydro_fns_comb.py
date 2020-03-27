#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wednesday March 25 15:05:02 2019

@author: mc4117
"""

from thetis import *
import time
import datetime
import numpy as np
from firedrake import *
import os
import callback_cons_tracer as call


def hydrodynamics_only(boundary_conditions_fn, mesh2d, bathymetry_2d, uv_init, elev_init, average_size, dt, t_end, wetting_and_drying = False, wetting_alpha = 0.1, friction = 'nikuradse', friction_coef = 0, viscosity = 10**(-6)):
    """
    Sets up a simulation with only hydrodynamics until a quasi-steady state when it can be used as an initial
    condition for the full morphological model. We update the bed friction at each time step.
    The actual run of the model are done outside the function
    
    Inputs:
    boundary_consditions_fn - function defining boundary conditions for problem 
    mesh2d - define mesh working on
    bathymetry2d - define bathymetry of problem
    uv_init - initial velocity of problem
    elev_init - initial elevation of problem
    average_size - average sediment size
    dt - timestep
    t_end - end time
    wetting_and_drying - wetting and drying switch
    wetting_alpha - wetting and drying parameter
    friction - choice of friction formulation - nikuradse and manning
    friction_coef - value of friction coefficient used in manning    
    viscosity - viscosity of hydrodynamics. The default value should be 10**(-6)
    
    Outputs:
    solver_obj - solver which we need to run to solve the problem
    update_forcings_hydrodynamics - function defining the updates to the model performed at each timestep
    outputdir - directory of outputs
    """
    def update_forcings_hydrodynamics(t_new):
        # update bed friction
        if friction == 'nikuradse':
            uv1, elev1 = solver_obj.fields.solution_2d.split()
            
            if wetting_and_drying == True:
                wd_bath_displacement = solver_obj.eq_sw.bathymetry_displacement_mass_term.wd_bathymetry_displacement
                depth.project(elev1 + wd_bath_displacement(elev1) + bathymetry_2d)        
            else:
                depth.interpolate(elev1 + bathymetry_2d)       
   
            # calculate skin friction coefficient
            cfactor.interpolate(2*(0.4**2)/((ln(11.036*depth/(ksp)))**2))

    # choose directory to output results
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    outputdir = 'outputs'+ st

    # export interval in seconds
    
    if t_end < 100:
        t_export = 0.1
    else:
        t_export = np.round(t_end/200,0)

    print_output('Exporting to '+outputdir)

    # define function spaces
    V = FunctionSpace(mesh2d, 'CG', 1)
    P1_2d = FunctionSpace(mesh2d, 'DG', 1)

    # define parameters
    ksp = Constant(3*average_size)

    # define depth

    depth = Function(V).interpolate(elev_init + bathymetry_2d)

    # define bed friction
    cfactor = Function(P1_2d).interpolate(2*(0.4**2)/((ln(11.036*depth/(ksp)))**2))
    
    # set up solver 
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    options = solver_obj.options
    options.simulation_export_time = t_export
    options.simulation_end_time = t_end
    options.output_directory = outputdir

    options.check_volume_conservation_2d = True

    options.fields_to_export = ['uv_2d', 'elev_2d']
    options.solve_tracer = False
    options.use_lax_friedrichs_tracer = False
    if friction == 'nikuradse':
        options.quadratic_drag_coefficient = cfactor
    elif friction == 'manning':
        if friction_coef == 0:
            friction_coef = 0.02
        options.manning_drag_coefficient = Constant(friction_coef)
    else:
        print('Undefined friction')
    options.horizontal_viscosity = Constant(viscosity)

    # crank-nicholson used to integrate in time system of ODEs resulting from application of galerkin FEM
    options.timestepper_type = 'CrankNicolson'
    options.timestepper_options.implicitness_theta = 1.0
    options.use_wetting_and_drying = wetting_and_drying
    options.wetting_and_drying_alpha = Constant(wetting_alpha)  
    options.norm_smoother = Constant(wetting_alpha)

    if not hasattr(options.timestepper_options, 'use_automatic_timestep'):
        options.timestep = dt

    # set boundary conditions

    swe_bnd, left_bnd_id, right_bnd_id, in_constant, out_constant, left_string, right_string = boundary_conditions_fn(state = "initial")
    
    for j in range(len(in_constant)):
        exec('constant_in' + str(j) + ' = Constant(' + str(in_constant[j]) + ')', globals())

    str1 = '{'
    if len(left_string) > 0:
        for i in range(len(left_string)):
            str1 += "'"+ str(left_string[i]) + "': constant_in" + str(i) + ","
        str1 = str1[0:len(str1)-1] + "}"
        exec('swe_bnd[left_bnd_id] = ' + str1)

    for k in range(len(out_constant)):
        exec('constant_out' + str(k) + '= Constant(' + str(out_constant[k]) + ')', globals())

    str2 = '{'
    if len(right_string) > 0:
        for i in range(len(right_string)):
            str2 += "'"+ str(right_string[i]) + "': constant_out" + str(i) + ","
        str2 = str2[0:len(str2)-1] + "}"
        exec('swe_bnd[right_bnd_id] = ' + str2)
   
    solver_obj.bnd_functions['shallow_water'] = swe_bnd
  
    solver_obj.assign_initial_conditions(uv = uv_init, elev= elev_init)
    
    return solver_obj, update_forcings_hydrodynamics, outputdir


def morphological(boundary_conditions_fn, morfac, morfac_transport, convectivevel, \
                 mesh2d, bathymetry_2d, input_dir, ks, average_size, dt, final_time, viscosity_hydro = 10**(-6), viscosity_morph = 10**(-6), wetting_and_drying = False, wetting_alpha = 0.1, rhos = 2650, cons_tracer = False, friction = 'nikuradse', friction_coef = 0, diffusivity = 0.15, tracer_init = None):
    """
    Set up a full morphological model simulation using as an initial condition the results of a hydrodynamic only model.    
    
    Inputs:
    boundary_consditions_fn - function defining boundary conditions for problem 
    morfac - morphological scale factor 
    morfac_transport - switch to turn on morphological component
    convectivevel - switch on convective velocity correction factor in sediment concentration equation
    mesh2d - define mesh working on
    bathymetry2d - define bathymetry of problem
    input_dir - folder containing results of hydrodynamics model which are used as initial conditions here
    ks - bottom friction coefficient for quadratic drag coefficient
    average_size - average sediment size
    dt - timestep
    final_time - end time
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

        if wetting_and_drying == True:
            wd_bath_displacement = solver_obj.eq_sw.bathymetry_displacement_mass_term.wd_bathymetry_displacement
            depth.project(elev1 + wd_bath_displacement(elev1) + old_bathymetry_2d)  
            elev_cg.project(elev1 + wd_bath_displacement(elev1))	
        else:              
            elev_cg.project(elev1)
            depth.project(elev_cg + old_bathymetry_2d)
        
        horizontal_velocity.interpolate(uv_cg[0])
        vertical_velocity.interpolate(uv_cg[1])

        # update bedfriction 
        hc.interpolate(conditional(depth > 0.001, depth, 0.001))
        aux.assign(conditional(11.036*hc/ks > 1.001, 11.036*hc/ks, 1.001))
        qfc.assign(2/(ln(aux)/0.4)**2)
        
        # calculate skin friction coefficient
        hclip.interpolate(conditional(ksp > depth, ksp, depth))
        cfactor.interpolate(conditional(depth > ksp, 2*((2.5*ln(11.036*hclip/ksp))**(-2)), Constant(0.0)))


        if morfac_transport == True:

            # if include sediment then update_forcings is run twice but only want to update bathymetry once
            t_list.append(t_new)
            double_factor = False

            if len(t_list) > 1:
                    if t_list[len(t_list)-1] == t_list[len(t_list)-2]:
                        double_factor = True       
            
            if double_factor == True:
                z_n.assign(old_bathymetry_2d)
                    
        
                # mu - ratio between skin friction and normal friction
                mu.assign(conditional(qfc > 0, cfactor/qfc, 0))
            
                # bed shear stress
                unorm.interpolate((horizontal_velocity**2) + (vertical_velocity**2))
                TOB.interpolate(1000*0.5*qfc*unorm)
                
                # calculate gradient of bed (noting bathymetry is -bed)
                dzdx.interpolate(old_bathymetry_2d.dx(0))
                dzdy.interpolate(old_bathymetry_2d.dx(1))

            
                # source term
                
                # deposition flux - calculating coefficient to account for stronger conc at bed
                B.interpolate(conditional(a > depth, a/a, a/depth))
                ustar.interpolate(sqrt(0.5*qfc*unorm))
                exp1.assign(conditional((conditional((settling_velocity/(0.4*ustar)) - 1 > 0, (settling_velocity/(0.4*ustar)) -1, -(settling_velocity/(0.4*ustar)) + 1)) > 10**(-4), conditional((settling_velocity/(0.4*ustar)) -1 > 3, 3, (settling_velocity/(0.4*ustar))-1), 0))
                coefftest.assign(conditional((conditional((settling_velocity/(0.4*ustar)) - 1 > 0, (settling_velocity/(0.4*ustar)) -1, -(settling_velocity/(0.4*ustar)) + 1)) > 10**(-4), B*(1-B**exp1)/exp1, -B*ln(B)))
                coeff.assign(conditional(coefftest>0, 1/coefftest, 0))                    
        
                # erosion flux - above critical velocity bed is eroded
                s0.assign((conditional(1000*0.5*qfc*unorm*mu > 0, 1000*0.5*qfc*unorm*mu, 0) -taucr)/taucr)
                ceq.assign(0.015*(average_size/a) * ((conditional(s0 < 0, 0, s0))**(1.5))/(dstar**0.3))

                 
                # calculate depth-averaged source term for sediment concentration equation

                depo.interpolate(settling_velocity*coeff)
                ero.interpolate(settling_velocity*ceq)

                if cons_tracer:
                    source.interpolate(-(depo*solver_obj.fields.tracer_2d/(depth**2))+ (ero/depth))
                    qbsourcedepth.interpolate(-(depo*solver_obj.fields.tracer_2d/depth)+ ero)
                else:
                    source.interpolate(-(depo*solver_obj.fields.tracer_2d/depth)+ (ero/depth))
                    qbsourcedepth.interpolate(-(depo*solver_obj.fields.tracer_2d)+ ero)

                if convectivevel == True:
                        # correction factor to advection velocity in sediment concentration equation
                
                        Bconv.interpolate(conditional(depth > 1.1*ksp, ksp/depth, ksp/(1.1*ksp)))
                        Aconv.interpolate(conditional(depth > 1.1* a, a/depth, a/(1.1*a)))
                    
                        # take max of value calculated either by ksp or depth
                        Amax.assign(conditional(Aconv > Bconv, Aconv, Bconv))

                        r1conv.assign(1 - (1/0.4)*conditional(settling_velocity/ustar < 1, settling_velocity/ustar, 1))

                        Ione.assign(conditional(r1conv > 10**(-8), (1 - Amax**r1conv)/r1conv, conditional(r1conv < - 10**(-8), (1 - Amax**r1conv)/r1conv, ln(Amax))))

                        Itwo.assign(conditional(r1conv > 10**(-8), -(Ione + (ln(Amax)*(Amax**r1conv)))/r1conv, conditional(r1conv < - 10**(-8), -(Ione + (ln(Amax)*(Amax**r1conv)))/r1conv, -0.5*ln(Amax)**2)))

                        alpha.assign(-(Itwo - (ln(Amax) - ln(30))*Ione)/(Ione * ((ln(Amax) - ln(30)) + 1)))

                        # final correction factor
                        alphatest2.assign(conditional(conditional(alpha > 1, 1, alpha) < 0, 0, conditional(alpha > 1, 1, alpha)))

                else:
                        alphatest2.assign(Constant(1.0))
                        
                # update sediment rate to ensure equilibrium at inflow
                if cons_tracer:
                    sediment_rate.assign(depth.at([0,0])*ceq.at([0,0])/(coeff.at([0,0])))
                else:
                    sediment_rate.assign(ceq.at([0,0])/(coeff.at([0,0])))                 

                # initialise exner equation with time derivative
                f = (((1-porosity)*(z_n1 - z_n)/(dt*morfac)) * v)*dx 
                    
                # add suspended sediment transport to exner equation multiplied by depth as the exner equation is not depth-averaged
                f += - (qbsourcedepth*v)*dx
                
                # solve exner equation using finite element methods
                solve(f == 0, z_n1)
                
                # update bed
                bathymetry_2d.assign(z_n1)


    # choose directory to output results
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    outputdir = 'outputs'+ st


    # final time of simulation
    t_end = final_time/morfac

    # export interval in seconds
    t_export = t_end
    
    
    print_output('Exporting to '+outputdir)
    
    x, y = SpatialCoordinate(mesh2d)
    
    # define function spaces
    P1_2d = FunctionSpace(mesh2d, 'DG', 1)
    V = FunctionSpace(mesh2d, 'CG', 1)
    vector_cg = VectorFunctionSpace(mesh2d, 'CG', 1)

    # define test functions on mesh
    v = TestFunction(V)
    z_n1 = Function(V, name ="z^{n+1}")
    z_n = Function(V, name="z^{n}")

    # define parameters
    g = Constant(9.81)
    porosity = Constant(0.4)


    ksp = Constant(3*average_size)
    a = Constant(ks/2)
    viscosity = Constant(viscosity_morph)

    # calculate critical shields parameter thetacr
    R = Constant(rhos/1000 - 1)

    dstar = Constant(average_size*((g*R)/(viscosity**2))**(1/3))
    if max(dstar.dat.data[:] < 1):
        print('ERROR: dstar value less than 1')
    elif max(dstar.dat.data[:] < 4):
        thetacr = Constant(0.24*(dstar**(-1)))
    elif max(dstar.dat.data[:] < 10):
        thetacr = Constant(0.14*(dstar**(-0.64)))
    elif max(dstar.dat.data[:] < 20):
        thetacr =Constant(0.04*(dstar**(-0.1)))
    elif max(dstar.dat.data[:] < 150):
        thetacr =Constant(0.013*(dstar**(0.29)))        
    else:
        thetacr =Constant(0.055)


    # critical bed shear stress
    taucr = Constant((rhos-1000)*g*average_size*thetacr)  
    
    # calculate settling velocity
    if average_size <= 100*(10**(-6)):
            settling_velocity = Constant(9.81*(average_size**2)*((rhos/1000)-1)/(18*viscosity))
    elif average_size <= 1000*(10**(-6)):
            settling_velocity = Constant((10*viscosity/average_size)*(sqrt(1 + 0.01*((((rhos/1000) - 1)*9.81*(average_size**3))/(viscosity**2)))-1))
    else:
            settling_velocity = Constant(1.1*sqrt(9.81*average_size*((rhos/1000) - 1)))        
    
    
    # initialise velocity, elevation and depth
    elev_init, uv_init = initialise_fields(mesh2d, input_dir, outputdir)

    uv_cg = Function(vector_cg).interpolate(uv_init)

    elev_cg = Function(V).interpolate(elev_init)

    if wetting_and_drying:
        H = Function(V).project(elev_cg + bathymetry_2d)
        depth = Function(V).project(H + (0.5 * (sqrt(H ** 2 + wetting_alpha ** 2) - H)))
    else:
        depth = Function(V).project(elev_cg + bathymetry_2d)         

    old_bathymetry_2d = Function(V).interpolate(bathymetry_2d)

    horizontal_velocity = Function(V).interpolate(uv_cg[0])
    vertical_velocity = Function(V).interpolate(uv_cg[1])

    # define bed friction
    hc = Function(P1_2d).interpolate(conditional(depth > 0.001, depth, 0.001))
    aux = Function(P1_2d).interpolate(conditional(11.036*hc/ks > 1.001, 11.036*hc/ks, 1.001))
    qfc = Function(P1_2d).interpolate(2/(ln(aux)/0.4)**2)
    # skin friction coefficient
    hclip = Function(P1_2d).interpolate(conditional(ksp > depth, ksp, depth))
    cfactor = Function(P1_2d).interpolate(conditional(depth > ksp, 2*((2.5*ln(11.036*hclip/ksp))**(-2)), Constant(0.0)))
    # mu - ratio between skin friction and normal friction
    mu = Function(P1_2d).interpolate(conditional(qfc > 0, cfactor/qfc, 0))

    # calculate bed shear stress
    unorm = Function(P1_2d).interpolate((horizontal_velocity**2) + (vertical_velocity**2))
    TOB = Function(V).interpolate(1000*0.5*qfc*unorm)

    # define bed gradient
    dzdx = Function(V).interpolate(old_bathymetry_2d.dx(0))
    dzdy = Function(V).interpolate(old_bathymetry_2d.dx(1))

    
    # deposition flux - calculating coefficient to account for stronger conc at bed
    B = Function(P1_2d).interpolate(conditional(a > depth, a/a, a/depth))
    ustar = Function(P1_2d).interpolate(sqrt(0.5*qfc*unorm))
    exp1 = Function(P1_2d).interpolate(conditional((conditional((settling_velocity/(0.4*ustar)) - 1 > 0, (settling_velocity/(0.4*ustar)) -1, -(settling_velocity/(0.4*ustar)) + 1)) > 10**(-4), conditional((settling_velocity/(0.4*ustar)) -1 > 3, 3, (settling_velocity/(0.4*ustar))-1), 0))
    coefftest = Function(P1_2d).interpolate(conditional((conditional((settling_velocity/(0.4*ustar)) - 1 > 0, (settling_velocity/(0.4*ustar)) -1, -(settling_velocity/(0.4*ustar)) + 1)) > 10**(-4), B*(1-B**exp1)/exp1, -B*ln(B)))
    coeff = Function(P1_2d).interpolate(conditional(conditional(coefftest>10**(-12), 1/coefftest, 10**12)>1, conditional(coefftest>10**(-12), 1/coefftest, 10**12), 1))
        
    # erosion flux - above critical velocity bed is eroded
    s0 = Function(P1_2d).interpolate((conditional(1000*0.5*qfc*unorm*mu > 0, 1000*0.5*qfc*unorm*mu, 0) - taucr)/taucr)
    ceq = Function(P1_2d).interpolate(0.015*(average_size/a) * ((conditional(s0 < 0, 0, s0))**(1.5))/(dstar**0.3))

    if convectivevel == True:
            # correction factor to advection velocity in sediment concentration equation

            Bconv = Function(P1_2d).interpolate(conditional(depth > 1.1*ksp, ksp/depth, ksp/(1.1*ksp)))
            Aconv = Function(P1_2d).interpolate(conditional(depth > 1.1* a, a/depth, a/(1.1*a)))
                    
            # take max of value calculated either by ksp or depth
            Amax = Function(P1_2d).interpolate(conditional(Aconv > Bconv, Aconv, Bconv))

            r1conv = Function(P1_2d).interpolate(1 - (1/0.4)*conditional(settling_velocity/ustar < 1, settling_velocity/ustar, 1))

            Ione = Function(P1_2d).interpolate(conditional(r1conv > 10**(-8), (1 - Amax**r1conv)/r1conv, conditional(r1conv < - 10**(-8), (1 - Amax**r1conv)/r1conv, ln(Amax))))

            Itwo = Function(P1_2d).interpolate(conditional(r1conv > 10**(-8), -(Ione + (ln(Amax)*(Amax**r1conv)))/r1conv, conditional(r1conv < - 10**(-8), -(Ione + (ln(Amax)*(Amax**r1conv)))/r1conv, -0.5*ln(Amax)**2)))

            alpha = Function(P1_2d).interpolate(-(Itwo - (ln(Amax) - ln(30))*Ione)/(Ione * ((ln(Amax) - ln(30)) + 1)))

            # final correction factor
            alphatest2 = Function(P1_2d).interpolate(conditional(conditional(alpha > 1, 1, alpha) < 0, 0, conditional(alpha > 1, 1, alpha)))

    else:
            alphatest2 = Function(P1_2d).interpolate(Constant(1.0))        
        
    # update sediment rate to ensure equilibrium at inflow
    if cons_tracer:
        sediment_rate = Constant(depth.at([0,0])*ceq.at([0,0])/(coeff.at([0,0])))
        testtracer = Function(P1_2d).interpolate(depth*ceq/coeff)
    else:
        sediment_rate = Constant(ceq.at([0,0])/(coeff.at([0,0])))
        testtracer = Function(P1_2d).interpolate(ceq/coeff)        
        
    # get individual terms
    depo = Function(P1_2d).interpolate(settling_velocity*coeff)
    ero = Function(P1_2d).interpolate(settling_velocity*ceq)        
        
    # calculate depth-averaged source term for sediment concentration equation
    if cons_tracer:
        if tracer_init == None:
            source = Function(P1_2d).interpolate(-(depo*sediment_rate/(depth**2))+ (ero/depth))
            qbsourcedepth = Function(P1_2d).interpolate(-(depo*sediment_rate/depth)+ ero)
        else:
            source = Function(P1_2d).interpolate(-(depo*tracer_init/depth)+ (ero/depth))
            qbsourcedepth = Function(P1_2d).interpolate(-(depo*tracer_init)+ ero)        
    else:
        if tracer_init == None:
            source = Function(P1_2d).interpolate(-(depo*sediment_rate/depth)+ (ero/depth))
            qbsourcedepth = Function(P1_2d).interpolate(-(depo*sediment_rate)+ ero)
        else:
            source = Function(P1_2d).interpolate(-(depo*tracer_init/depth)+ (ero/depth))
            qbsourcedepth = Function(P1_2d).interpolate(-(depo*tracer_init)+ ero)

    # set up solver 
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    
    options = solver_obj.options
    options.simulation_export_time = t_export
    options.simulation_end_time = t_end
    options.output_directory = outputdir
    options.check_volume_conservation_2d = True
    
    # switch on tracer calculation if using sediment transport component
    options.solve_tracer = True
    options.use_tracer_conservative_form = cons_tracer
    options.fields_to_export = ['tracer_2d', 'uv_2d', 'elev_2d']
    options.tracer_advective_velocity_factor = alphatest2
    options.tracer_source_2d = source
    options.check_tracer_conservation = True
    options.use_lax_friedrichs_tracer = False
    # set bed friction
    if friction == 'nikuradse':
        options.quadratic_drag_coefficient = cfactor
    elif friction == 'manning':
        if friction_coef == 0:
            friction_coef = 0.02
        options.manning_drag_coefficient = Constant(friction_coef)
    else:
        print('Undefined friction')
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
            str1 += "'"+ str(left_string[i]) + "': constant_in" + str(i) + ","
        str1 = str1[0:len(str1)-1] + "}"
        exec('swe_bnd[left_bnd_id] = ' + str1)

    for k in range(len(out_constant)):
        exec('constant_out' + str(k) + '= Constant(' + str(out_constant[k]) + ')', globals())

    str2 = '{'
    if len(right_string) > 0:
        for i in range(len(right_string)):
            str2 += "'"+ str(right_string[i]) + "': constant_out" + str(i) + ","
        str2 = str2[0:len(str2)-1] + "}"
        exec('swe_bnd[right_bnd_id] = ' + str2)
         
    solver_obj.bnd_functions['shallow_water'] = swe_bnd
    
    
        
    solver_obj.bnd_functions['tracer'] = {1: {'value': sediment_rate}}

           
    for i in solver_obj.bnd_functions['tracer'].keys():
            if i in solver_obj.bnd_functions['shallow_water'].keys():
                solver_obj.bnd_functions['tracer'][i].update(solver_obj.bnd_functions['shallow_water'][i])
    for i in solver_obj.bnd_functions['shallow_water'].keys():
            if i not in solver_obj.bnd_functions['tracer'].keys():
                solver_obj.bnd_functions['tracer'].update({i:solver_obj.bnd_functions['shallow_water'][i]})        
        
    # set initial conditions
        
    if tracer_init == None:       
        solver_obj.assign_initial_conditions(uv=uv_init, elev= elev_init, tracer = testtracer)
    else:
        if cons_tracer:
            tracer_init_int = Function(P1_2d).interpolate(tracer_init*depth)
            solver_obj.assign_initial_conditions(elev= elev_init, uv = uv_init, tracer = tracer_init_int)
        else:
            solver_obj.assign_initial_conditions(elev= elev_init, uv = uv_init, tracer = tracer_init)            
    return solver_obj, update_forcings_tracer, outputdir

def export_final_state(inputdir, uv, elev,):
    """
    Export fields to be used in a subsequent simulation
    """
    if not os.path.exists(inputdir):
        os.makedirs(inputdir)
    print_output("Exporting fields for subsequent simulation")
    chk = DumbCheckpoint(inputdir + "/velocity", mode=FILE_CREATE)
    chk.store(uv, name="velocity")
    File(inputdir + '/velocityout.pvd').write(uv)
    chk.close()
    chk = DumbCheckpoint(inputdir + "/elevation", mode=FILE_CREATE)
    chk.store(elev, name="elevation")
    File(inputdir + '/elevationout.pvd').write(elev)
    chk.close()
 
    
    
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
        chk = DumbCheckpoint(inputdir + "/velocity" , mode=FILE_READ)
        V = VectorFunctionSpace(mesh2d, 'DG', 1)
        uv_init = Function(V, name="velocity")
        chk.load(uv_init)
        File(outputdir + "/velocity_imported.pvd").write(uv_init)
        chk.close()
        return  elev_init, uv_init,
