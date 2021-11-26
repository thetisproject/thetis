"""
MMS test for baroclinic 3D model
"""
from thetis import *
from scipy import stats
import pytest


field_metadata['uv_full'] = dict(field_metadata['uv_3d'])

# define some global constants
lx = 15e3
ly = 10e3
area = lx*ly
depth = 40.0
nu0 = 0.0
f0 = 1.0e-4
alpha = 0.2  # thermal expansion coeff
beta = 0.0  # haline contraction coeff
temp_ref = 5.0
salt_const = Constant(35.0)
temp_const = Constant(10.0)
rho_0 = 1000.0
physical_constants['rho0'] = rho_0
g_grav = physical_constants['g_grav']
eos_params = {
    'rho_ref': rho_0,
    's_ref': float(salt_const),
    'th_ref': temp_ref,
    'alpha': alpha,
    'beta': beta,
}
eos_alpha = eos_params['alpha']
eos_beta = eos_params['beta']
eos_t0 = eos_params['th_ref']
eos_s0 = eos_params['s_ref']


def setup1(xy, xyz):
    """
    Constant bathymetry and zero velocity, non-trivial active tracer
    """
    out = {}
    out['elev_2d'] = Constant(0)
    out['uv_full_3d'] = Constant((0, 0))
    out['uv_2d'] = as_vector((Constant(0), Constant(0)))
    out['uv_dav_3d'] = as_vector((Constant(0), Constant(0), Constant(0)))
    out['uv_3d'] = as_vector((Constant(0), Constant(0), Constant(0)))
    out['w_3d'] = as_vector((Constant(0), Constant(0), Constant(0)))
    out['temp_3d'] = 5*cos(xyz[2]/depth)*cos((2*xyz[0] + xyz[1])/lx) + 15
    out['density_3d'] = -eos_alpha*(-eos_t0 + 5*cos(xyz[2]/depth)*cos((2*xyz[0] + xyz[1])/lx) + 15) + eos_beta*(-eos_s0 + salt_const)
    out['baroc_head_3d'] = (-eos_alpha*(5*depth*sin(xyz[2]/depth)*cos((2*xyz[0] + xyz[1])/lx) - eos_t0*xyz[2] + 15*xyz[2]) + eos_beta*xyz[2]*(-eos_s0 + salt_const))/rho_0
    out['int_pg_3d'] = as_vector((-10*depth*eos_alpha*g_grav*sin(xyz[2]/depth)*sin((2*xyz[0] + xyz[1])/lx)/(lx*rho_0), -5*depth*eos_alpha*g_grav*sin(xyz[2]/depth)*sin((2*xyz[0] + xyz[1])/lx)/(lx*rho_0), Constant(0)))
    out['vol_source_2d'] = Constant(0)
    out['mom_source_2d'] = as_vector((Constant(0), Constant(0)))
    out['mom_source_3d'] = as_vector((-10*depth*eos_alpha*g_grav*sin(xyz[2]/depth)*sin((2*xyz[0] + xyz[1])/lx)/(lx*rho_0), -5*depth*eos_alpha*g_grav*sin(xyz[2]/depth)*sin((2*xyz[0] + xyz[1])/lx)/(lx*rho_0), Constant(0)))
    out['temp_source_3d'] = Constant(0)

    out['options'] = {}
    return out


def setup2(xy, xyz):
    """
    Constant bathymetry and zero velocity, non-trivial elevation
    """
    out = {}
    out['elev_2d'] = 5*cos((xy[0] + 5*xy[1]/2)/lx)
    out['uv_full_3d'] = Constant((0, 0))
    out['uv_2d'] = as_vector((Constant(0), Constant(0)))
    out['uv_dav_3d'] = as_vector((Constant(0), Constant(0), Constant(0)))
    out['uv_3d'] = as_vector((Constant(0), Constant(0), Constant(0)))
    out['w_3d'] = as_vector((Constant(0), Constant(0), Constant(0)))
    out['temp_3d'] = temp_const
    out['density_3d'] = -eos_alpha*(-eos_t0 + temp_const) + eos_beta*(-eos_s0 + salt_const)
    out['baroc_head_3d'] = xyz[2]*(-eos_alpha*(-eos_t0 + temp_const) + eos_beta*(-eos_s0 + salt_const))/rho_0 - 5*(-eos_alpha*(-eos_t0 + temp_const) + eos_beta*(-eos_s0 + salt_const))*cos((xyz[0] + 5*xyz[1]/2)/lx)/rho_0
    out['int_pg_3d'] = as_vector((-5*g_grav*(-eos_alpha*(-eos_t0 + temp_const) + eos_beta*(-eos_s0 + salt_const))*sin((xyz[0] + 5*xyz[1]/2)/lx)/(lx*rho_0), -25*g_grav*(-eos_alpha*(-eos_t0 + temp_const) + eos_beta*(-eos_s0 + salt_const))*sin((xyz[0] + 5*xyz[1]/2)/lx)/(2*lx*rho_0), Constant(0)))
    out['vol_source_2d'] = Constant(0)
    out['mom_source_2d'] = as_vector((-5*g_grav*sin((xy[0] + 5*xy[1]/2)/lx)/lx, -25*g_grav*sin((xy[0] + 5*xy[1]/2)/lx)/(2*lx)))
    out['mom_source_3d'] = as_vector((Constant(0), Constant(0), Constant(0)))
    out['temp_source_3d'] = Constant(0)

    out['options'] = {
        'use_baroclinic_formulation': False
    }
    return out


def setup2b(xy, xyz):
    """
    Constant bathymetry and zero velocity, non-trivial elevation, baroclinic
    """
    out = {}
    out['elev_2d'] = 5*cos((xy[0] + 5*xy[1]/2)/lx)
    out['uv_full_3d'] = Constant((0, 0))
    out['uv_2d'] = as_vector((Constant(0), Constant(0)))
    out['uv_dav_3d'] = as_vector((Constant(0), Constant(0), Constant(0)))
    out['uv_3d'] = as_vector((Constant(0), Constant(0), Constant(0)))
    out['w_3d'] = as_vector((Constant(0), Constant(0), Constant(0)))
    out['temp_3d'] = temp_const
    out['density_3d'] = -eos_alpha*(-eos_t0 + temp_const) + eos_beta*(-eos_s0 + salt_const)
    out['baroc_head_3d'] = xyz[2]*(-eos_alpha*(-eos_t0 + temp_const) + eos_beta*(-eos_s0 + salt_const))/rho_0 - 5*(-eos_alpha*(-eos_t0 + temp_const) + eos_beta*(-eos_s0 + salt_const))*cos((xyz[0] + 5*xyz[1]/2)/lx)/rho_0
    out['int_pg_3d'] = as_vector((-5*g_grav*(-eos_alpha*(-eos_t0 + temp_const) + eos_beta*(-eos_s0 + salt_const))*sin((xyz[0] + 5*xyz[1]/2)/lx)/(lx*rho_0), -25*g_grav*(-eos_alpha*(-eos_t0 + temp_const) + eos_beta*(-eos_s0 + salt_const))*sin((xyz[0] + 5*xyz[1]/2)/lx)/(2*lx*rho_0), Constant(0)))
    out['vol_source_2d'] = Constant(0)
    out['mom_source_2d'] = as_vector((-5*g_grav*sin((xy[0] + 5*xy[1]/2)/lx)/lx, -25*g_grav*sin((xy[0] + 5*xy[1]/2)/lx)/(2*lx)))
    out['mom_source_3d'] = as_vector((-5*g_grav*(-eos_alpha*(-eos_t0 + temp_const) + eos_beta*(-eos_s0 + salt_const))*sin((xyz[0] + 5*xyz[1]/2)/lx)/(lx*rho_0), -25*g_grav*(-eos_alpha*(-eos_t0 + temp_const) + eos_beta*(-eos_s0 + salt_const))*sin((xyz[0] + 5*xyz[1]/2)/lx)/(2*lx*rho_0), Constant(0)))
    out['temp_source_3d'] = Constant(0)

    out['options'] = {}
    return out


def setup3(xy, xyz):
    """
    Constant bathymetry, non-zero velocity
    """
    out = {}
    out['elev_2d'] = Constant(0)
    out['uv_full_3d'] = as_vector((cos((2*xyz[0] + xyz[1])/lx), Constant(0)))
    out['uv_2d'] = as_vector((cos((2*xy[0] + xy[1])/lx), Constant(0)))
    out['uv_dav_3d'] = as_vector((cos((2*xyz[0] + xyz[1])/lx), Constant(0), Constant(0)))
    out['uv_3d'] = as_vector((Constant(0), Constant(0), Constant(0)))
    out['w_3d'] = as_vector((Constant(0), Constant(0), 2*depth*sin((2*xyz[0] + xyz[1])/lx)/lx + 2*xyz[2]*sin((2*xyz[0] + xyz[1])/lx)/lx))
    out['temp_3d'] = temp_const
    out['density_3d'] = -eos_alpha*(-eos_t0 + temp_const) + eos_beta*(-eos_s0 + salt_const)
    out['baroc_head_3d'] = xyz[2]*(-eos_alpha*(-eos_t0 + temp_const) + eos_beta*(-eos_s0 + salt_const))/rho_0
    out['int_pg_3d'] = as_vector((Constant(0), Constant(0), Constant(0)))
    out['vol_source_2d'] = -2*depth*sin((2*xy[0] + xy[1])/lx)/lx
    out['mom_source_2d'] = as_vector((Constant(0), f0*cos((2*xy[0] + xy[1])/lx)))
    out['mom_source_3d'] = as_vector((-2*sin((2*xyz[0] + xyz[1])/lx)*cos((2*xyz[0] + xyz[1])/lx)/lx, Constant(0), Constant(0)))
    out['temp_source_3d'] = Constant(0)

    out['options'] = {
        'use_baroclinic_formulation': False
    }
    return out


def setup4(xy, xyz):
    """
    Constant bathymetry, non-zero velocity and passive temp
    """
    out = {}
    out['elev_2d'] = Constant(0)
    out['uv_full_3d'] = as_vector((cos(3*xyz[2]/depth)*cos((2*xyz[0] + xyz[1])/lx)/2, Constant(0)))
    out['uv_2d'] = as_vector((sin(3)*cos((2*xy[0] + xy[1])/lx)/6, Constant(0)))
    out['uv_dav_3d'] = as_vector((sin(3)*cos((2*xyz[0] + xyz[1])/lx)/6, Constant(0), Constant(0)))
    out['uv_3d'] = as_vector((cos(3*xyz[2]/depth)*cos((2*xyz[0] + xyz[1])/lx)/2 - sin(3)*cos((2*xyz[0] + xyz[1])/lx)/6, Constant(0), Constant(0)))
    out['w_3d'] = as_vector((Constant(0), Constant(0), depth*sin(3*xyz[2]/depth)*sin((2*xyz[0] + xyz[1])/lx)/(3*lx) + depth*sin(3)*sin((2*xyz[0] + xyz[1])/lx)/(3*lx)))
    out['temp_3d'] = 5*cos(xyz[2]/depth)*cos((xyz[0] + 2*xyz[1])/lx) + 15
    out['density_3d'] = -eos_alpha*(-eos_t0 + 5*cos(xyz[2]/depth)*cos((xyz[0] + 2*xyz[1])/lx) + 15) + eos_beta*(-eos_s0 + salt_const)
    out['baroc_head_3d'] = (-eos_alpha*(5*depth*sin(xyz[2]/depth)*cos((xyz[0] + 2*xyz[1])/lx) - eos_t0*xyz[2] + 15*xyz[2]) + eos_beta*xyz[2]*(-eos_s0 + salt_const))/rho_0
    out['int_pg_3d'] = as_vector((-5*depth*eos_alpha*g_grav*sin(xyz[2]/depth)*sin((xyz[0] + 2*xyz[1])/lx)/(lx*rho_0), -10*depth*eos_alpha*g_grav*sin(xyz[2]/depth)*sin((xyz[0] + 2*xyz[1])/lx)/(lx*rho_0), Constant(0)))
    out['vol_source_2d'] = -depth*sin(3)*sin((2*xy[0] + xy[1])/lx)/(3*lx)
    out['mom_source_2d'] = as_vector((Constant(0), f0*sin(3)*cos((2*xy[0] + xy[1])/lx)/6))
    out['mom_source_3d'] = as_vector((-sin((2*xyz[0] + xyz[1])/lx)*cos(3*xyz[2]/depth)**2*cos((2*xyz[0] + xyz[1])/lx)/(2*lx) - 3*(depth*sin(3*xyz[2]/depth)*sin((2*xyz[0] + xyz[1])/lx)/(3*lx) + depth*sin(3)*sin((2*xyz[0] + xyz[1])/lx)/(3*lx))*sin(3*xyz[2]/depth)*cos((2*xyz[0] + xyz[1])/lx)/(2*depth), Constant(0), Constant(0)))
    out['temp_source_3d'] = -5*sin((xyz[0] + 2*xyz[1])/lx)*cos(xyz[2]/depth)*cos(3*xyz[2]/depth)*cos((2*xyz[0] + xyz[1])/lx)/(2*lx) - 5*(depth*sin(3*xyz[2]/depth)*sin((2*xyz[0] + xyz[1])/lx)/(3*lx) + depth*sin(3)*sin((2*xyz[0] + xyz[1])/lx)/(3*lx))*sin(xyz[2]/depth)*cos((xyz[0] + 2*xyz[1])/lx)/depth

    out['options'] = {
        'use_baroclinic_formulation': False
    }
    return out


def setup4b(xy, xyz):
    """
    Constant bathymetry, non-zero velocity and temp, baroclinic
    """
    out = {}
    out['elev_2d'] = Constant(0)
    out['uv_full_3d'] = as_vector((cos(3*xyz[2]/depth)*cos((2*xyz[0] + xyz[1])/lx)/2, Constant(0)))
    out['uv_2d'] = as_vector((sin(3)*cos((2*xy[0] + xy[1])/lx)/6, Constant(0)))
    out['uv_dav_3d'] = as_vector((sin(3)*cos((2*xyz[0] + xyz[1])/lx)/6, Constant(0), Constant(0)))
    out['uv_3d'] = as_vector((cos(3*xyz[2]/depth)*cos((2*xyz[0] + xyz[1])/lx)/2 - sin(3)*cos((2*xyz[0] + xyz[1])/lx)/6, Constant(0), Constant(0)))
    out['w_3d'] = as_vector((Constant(0), Constant(0), depth*sin(3*xyz[2]/depth)*sin((2*xyz[0] + xyz[1])/lx)/(3*lx) + depth*sin(3)*sin((2*xyz[0] + xyz[1])/lx)/(3*lx)))
    out['temp_3d'] = 5*cos(xyz[2]/depth)*cos((xyz[0] + 2*xyz[1])/lx) + 15
    out['density_3d'] = -eos_alpha*(-eos_t0 + 5*cos(xyz[2]/depth)*cos((xyz[0] + 2*xyz[1])/lx) + 15) + eos_beta*(-eos_s0 + salt_const)
    out['baroc_head_3d'] = (-eos_alpha*(5*depth*sin(xyz[2]/depth)*cos((xyz[0] + 2*xyz[1])/lx) - eos_t0*xyz[2] + 15*xyz[2]) + eos_beta*xyz[2]*(-eos_s0 + salt_const))/rho_0
    out['int_pg_3d'] = as_vector((-5*depth*eos_alpha*g_grav*sin(xyz[2]/depth)*sin((xyz[0] + 2*xyz[1])/lx)/(lx*rho_0), -10*depth*eos_alpha*g_grav*sin(xyz[2]/depth)*sin((xyz[0] + 2*xyz[1])/lx)/(lx*rho_0), Constant(0)))
    out['vol_source_2d'] = -depth*sin(3)*sin((2*xy[0] + xy[1])/lx)/(3*lx)
    out['mom_source_2d'] = as_vector((Constant(0), f0*sin(3)*cos((2*xy[0] + xy[1])/lx)/6))
    out['mom_source_3d'] = as_vector((-5*depth*eos_alpha*g_grav*sin(xyz[2]/depth)*sin((xyz[0] + 2*xyz[1])/lx)/(lx*rho_0) - sin((2*xyz[0] + xyz[1])/lx)*cos(3*xyz[2]/depth)**2*cos((2*xyz[0] + xyz[1])/lx)/(2*lx) - 3*(depth*sin(3*xyz[2]/depth)*sin((2*xyz[0] + xyz[1])/lx)/(3*lx) + depth*sin(3)*sin((2*xyz[0] + xyz[1])/lx)/(3*lx))*sin(3*xyz[2]/depth)*cos((2*xyz[0] + xyz[1])/lx)/(2*depth), -10*depth*eos_alpha*g_grav*sin(xyz[2]/depth)*sin((xyz[0] + 2*xyz[1])/lx)/(lx*rho_0), Constant(0)))
    out['temp_source_3d'] = -5*sin((xyz[0] + 2*xyz[1])/lx)*cos(xyz[2]/depth)*cos(3*xyz[2]/depth)*cos((2*xyz[0] + xyz[1])/lx)/(2*lx) - 5*(depth*sin(3*xyz[2]/depth)*sin((2*xyz[0] + xyz[1])/lx)/(3*lx) + depth*sin(3)*sin((2*xyz[0] + xyz[1])/lx)/(3*lx))*sin(xyz[2]/depth)*cos((xyz[0] + 2*xyz[1])/lx)/depth

    out['options'] = {}
    return out


def setup5(xy, xyz):
    """
    Constant bathymetry, non-zero velocity and temp, baroclinic, symmetric
    """
    out = {}
    out['elev_2d'] = Constant(0)
    out['uv_full_3d'] = as_vector((sin(2*pi*xyz[0]/lx)*cos(3*xyz[2]/depth)/2, sin(xyz[2]/(2*depth))*cos(pi*xyz[1]/ly)/3, Constant(0)))
    out['uv_2d'] = as_vector((sin(3)*sin(2*pi*xy[0]/lx)/6, (-2*depth*cos(pi*xy[1]/ly)/3 + 2*depth*cos(1/2)*cos(pi*xy[1]/ly)/3)/depth))
    out['uv_dav_3d'] = as_vector((sin(3)*sin(2*pi*xyz[0]/lx)/6, (-2*depth*cos(pi*xyz[1]/ly)/3 + 2*depth*cos(1/2)*cos(pi*xyz[1]/ly)/3)/depth, Constant(0)))
    out['uv_3d'] = as_vector((sin(2*pi*xyz[0]/lx)*cos(3*xyz[2]/depth)/2 - sin(3)*sin(2*pi*xyz[0]/lx)/6, sin(xyz[2]/(2*depth))*cos(pi*xyz[1]/ly)/3 - (-2*depth*cos(pi*xyz[1]/ly)/3 + 2*depth*cos(1/2)*cos(pi*xyz[1]/ly)/3)/depth, Constant(0)))
    out['w_3d'] = as_vector((Constant(0), Constant(0), -2*pi*depth*sin(pi*xyz[1]/ly)*cos(xyz[2]/(2*depth))/(3*ly) + 2*pi*depth*sin(pi*xyz[1]/ly)*cos(1/2)/(3*ly) - pi*depth*sin(3*xyz[2]/depth)*cos(2*pi*xyz[0]/lx)/(3*lx) - pi*depth*sin(3)*cos(2*pi*xyz[0]/lx)/(3*lx)))
    out['temp_3d'] = 15*sin(pi*xyz[0]/lx)*sin(pi*xyz[1]/ly)*cos(xyz[2]/depth) + 15
    out['density_3d'] = -eos_alpha*(-eos_t0 + 15*sin(pi*xyz[0]/lx)*sin(pi*xyz[1]/ly)*cos(xyz[2]/depth) + 15) + eos_beta*(-eos_s0 + salt_const)
    out['baroc_head_3d'] = (-15*depth*eos_alpha*sin(xyz[2]/depth)*sin(pi*xyz[0]/lx)*sin(pi*xyz[1]/ly) + eos_alpha*eos_t0*xyz[2] - 15*eos_alpha*xyz[2] - eos_beta*eos_s0*xyz[2] + eos_beta*salt_const*xyz[2])/rho_0
    out['int_pg_3d'] = as_vector((15*pi*depth*eos_alpha*g_grav*sin(xyz[2]/depth)*sin(pi*xyz[1]/ly)*cos(pi*xyz[0]/lx)/(lx*rho_0), 15*pi*depth*eos_alpha*g_grav*sin(xyz[2]/depth)*sin(pi*xyz[0]/lx)*cos(pi*xyz[1]/ly)/(ly*rho_0), Constant(0)))
    out['vol_source_2d'] = -2*pi*depth*sin(pi*xy[1]/ly)*cos(1/2)/(3*ly) + 2*pi*depth*sin(pi*xy[1]/ly)/(3*ly) + pi*depth*sin(3)*cos(2*pi*xy[0]/lx)/(3*lx)
    out['mom_source_2d'] = as_vector((-f0*(-2*depth*cos(pi*xy[1]/ly)/3 + 2*depth*cos(1/2)*cos(pi*xy[1]/ly)/3)/depth, f0*sin(3)*sin(2*pi*xy[0]/lx)/6))
    out['mom_source_3d'] = as_vector((15*pi*depth*eos_alpha*g_grav*sin(xyz[2]/depth)*sin(pi*xyz[1]/ly)*cos(pi*xyz[0]/lx)/(lx*rho_0) - f0*(sin(xyz[2]/(2*depth))*cos(pi*xyz[1]/ly)/3 - (-2*depth*cos(pi*xyz[1]/ly)/3 + 2*depth*cos(1/2)*cos(pi*xyz[1]/ly)/3)/depth) + pi*sin(2*pi*xyz[0]/lx)*cos(3*xyz[2]/depth)**2*cos(2*pi*xyz[0]/lx)/(2*lx) - 3*(-2*pi*depth*sin(pi*xyz[1]/ly)*cos(xyz[2]/(2*depth))/(3*ly) + 2*pi*depth*sin(pi*xyz[1]/ly)*cos(1/2)/(3*ly) - pi*depth*sin(3*xyz[2]/depth)*cos(2*pi*xyz[0]/lx)/(3*lx) - pi*depth*sin(3)*cos(2*pi*xyz[0]/lx)/(3*lx))*sin(3*xyz[2]/depth)*sin(2*pi*xyz[0]/lx)/(2*depth), 15*pi*depth*eos_alpha*g_grav*sin(xyz[2]/depth)*sin(pi*xyz[0]/lx)*cos(pi*xyz[1]/ly)/(ly*rho_0) + f0*(sin(2*pi*xyz[0]/lx)*cos(3*xyz[2]/depth)/2 - sin(3)*sin(2*pi*xyz[0]/lx)/6) - pi*sin(xyz[2]/(2*depth))**2*sin(pi*xyz[1]/ly)*cos(pi*xyz[1]/ly)/(9*ly) + (-2*pi*depth*sin(pi*xyz[1]/ly)*cos(xyz[2]/(2*depth))/(3*ly) + 2*pi*depth*sin(pi*xyz[1]/ly)*cos(1/2)/(3*ly) - pi*depth*sin(3*xyz[2]/depth)*cos(2*pi*xyz[0]/lx)/(3*lx) - pi*depth*sin(3)*cos(2*pi*xyz[0]/lx)/(3*lx))*cos(xyz[2]/(2*depth))*cos(pi*xyz[1]/ly)/(6*depth), Constant(0)))
    out['temp_source_3d'] = 5*pi*sin(xyz[2]/(2*depth))*sin(pi*xyz[0]/lx)*cos(xyz[2]/depth)*cos(pi*xyz[1]/ly)**2/ly + 15*pi*sin(2*pi*xyz[0]/lx)*sin(pi*xyz[1]/ly)*cos(xyz[2]/depth)*cos(3*xyz[2]/depth)*cos(pi*xyz[0]/lx)/(2*lx) - 15*(-2*pi*depth*sin(pi*xyz[1]/ly)*cos(xyz[2]/(2*depth))/(3*ly) + 2*pi*depth*sin(pi*xyz[1]/ly)*cos(1/2)/(3*ly) - pi*depth*sin(3*xyz[2]/depth)*cos(2*pi*xyz[0]/lx)/(3*lx) - pi*depth*sin(3)*cos(2*pi*xyz[0]/lx)/(3*lx))*sin(xyz[2]/depth)*sin(pi*xyz[0]/lx)*sin(pi*xyz[1]/ly)/depth

    out['options'] = {}
    return out


def run(setup, refinement, polynomial_degree, do_export=True, **options):
    """Run single test and return L2 error"""
    print_output('--- running {:} refinement {:}'.format(setup.__name__, refinement))

    # domain dimensions
    dt = 25.0/refinement
    t_end = 50*dt
    if do_export:
        t_export = t_end/10
    else:
        t_export = t_end

    bath_expr = Constant(depth)

    # mesh
    n_layers = 2*refinement
    nx = 4*refinement
    ny = 4*refinement
    mesh2d = RectangleMesh(nx, ny, lx, ly)

    # bathymetry
    p1_2d = get_functionspace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(p1_2d, name='Bathymetry')
    bathymetry_2d.project(bath_expr)

    solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, n_layers)
    options = solver_obj.options
    options.element_family = 'dg-dg'
    options.polynomial_degree = polynomial_degree
    options.use_bottom_friction = False
    options.use_baroclinic_formulation = True
    options.solve_temperature = True
    options.solve_salinity = False
    options.coriolis_frequency = Constant(f0)
    options.use_quadratic_pressure = False
    options.use_bottom_friction = False
    options.use_lax_friedrichs_velocity = True
    options.lax_friedrichs_velocity_scaling_factor = Constant(1.0)
    options.use_limiter_for_tracers = False
    options.use_limiter_for_velocity = False
    options.constant_salinity = Constant(salt_const)
    options.horizontal_velocity_scale = Constant(2.0)
    options.horizontal_viscosity_scale = Constant(1e-2)
    options.timestepper_options.use_automatic_timestep = False
    options.timestep = dt
    options.no_exports = not do_export
    options.simulation_end_time = t_end
    options.simulation_export_time = t_export
    options.fields_to_export = ['elev_2d', 'uv_2d', 'salt_3d', 'uv_3d', 'w_3d',
                                'temp_3d',
                                'density_3d', 'baroc_head_3d', 'int_pg_3d']
    options.horizontal_viscosity_scale = Constant(nu0)
    options.equation_of_state_type = 'linear'
    options.equation_of_state_options.update(eos_params)
    xy = SpatialCoordinate(solver_obj.mesh2d)
    xyz = SpatialCoordinate(solver_obj.mesh)
    sdict = setup(xy, xyz)

    options.update(sdict['options'])
    options.update(options)
    # NOTE needed for 2d swe boundary conditions
    options.timestepper_options.swe_options.solver_parameters['snes_atol'] = 1e-10

    solver_obj.create_function_spaces()

    # analytical solution
    ana_w_3d = sdict['w_3d']
    f = Function(solver_obj.function_spaces.U, name='ana w').project(ana_w_3d)
    if do_export:
        File(options.output_directory + '/ana_w.pvd').write(f).write(f)

    ana_temp_3d = sdict['temp_3d']
    f = Function(solver_obj.function_spaces.H, name='ana temp').project(ana_temp_3d)
    if do_export:
        File(options.output_directory + '/ana_temp.pvd').write(f).write(f)

    ana_rho_3d = sdict['density_3d']
    f = Function(solver_obj.function_spaces.H, name='ana rho').project(ana_rho_3d)
    if do_export:
        File(options.output_directory + '/ana_rho.pvd').write(f).write(f)

    ana_bhead_3d = sdict['baroc_head_3d']
    f = Function(solver_obj.function_spaces.H, name='ana bhead').project(ana_bhead_3d)
    if do_export:
        File(options.output_directory + '/ana_bhead.pvd').write(f).write(f)

    ana_intpg_3d = sdict['int_pg_3d']
    f = Function(solver_obj.function_spaces.U, name='ana int pg').project(ana_intpg_3d)
    if do_export:
        File(options.output_directory + '/ana_intpg.pvd').write(f).write(f)

    options.volume_source_2d = sdict['vol_source_2d']
    options.momentum_source_2d = sdict['mom_source_2d']
    options.momentum_source_3d = sdict['mom_source_3d']
    options.temperature_source_3d = sdict['temp_source_3d']

    bnd_temp = {'value': sdict['temp_3d']}
    solver_obj.bnd_functions['temp'] = {1: bnd_temp, 2: bnd_temp,
                                        3: bnd_temp, 4: bnd_temp}
    # NOTE use symmetic uv condition to get correct w
    bnd_mom = {'uv': sdict['uv_full_3d']}
    solver_obj.bnd_functions['momentum'] = {1: bnd_mom, 2: bnd_mom,
                                            3: bnd_mom, 4: bnd_mom}
    bnd_swe = {'elev': sdict['elev_2d'], 'uv': sdict['uv_2d']}
    solver_obj.bnd_functions['shallow_water'] = {1: bnd_swe, 2: bnd_swe,
                                                 3: bnd_swe, 4: bnd_swe}

    solver_obj.create_equations()
    solver_obj.assign_initial_conditions(elev=sdict['elev_2d'],
                                         uv_2d=sdict['uv_2d'],
                                         uv_3d=sdict['uv_3d'],
                                         temp=sdict['temp_3d'])
    solver_obj.iterate()

    area = lx*ly
    var_list = ['elev_2d', 'uv_2d', 'uv_3d', 'temp_3d', 'w_3d']
    l2_err = {}
    for v in var_list:
        l2_err[v] = errornorm(sdict[v], solver_obj.fields[v])/numpy.sqrt(area)
    solver_obj.fields.uv_dav_2d.assign(solver_obj.fields.uv_2d)
    solver_obj.copy_uv_dav_to_uv_dav_3d.solve()
    f = solver_obj.function_spaces.U.get_work_function()
    f.assign(solver_obj.fields.uv_3d + solver_obj.fields.uv_dav_3d)
    l2_err['uv_full'] = errornorm(sdict['uv_full_3d'], f)/numpy.sqrt(area)
    solver_obj.function_spaces.U.restore_work_function(f)
    for k in sorted(l2_err):
        print_output('L2 error {:} {:.12f}'.format(k, l2_err[k]))

    return l2_err


def run_convergence(setup, ref_list, saveplot=False, **options):
    """Runs test for a list of refinements and computes error convergence rate"""
    if saveplot and COMM_WORLD.size > 1:
        raise Exception('Cannot use matplotlib in parallel')
    polynomial_degree = options.get('polynomial_degree', 1)
    space_str = options.get('element_family')
    l2_err = []
    for r in ref_list:
        l2_err.append(run(setup, r, **options))
    x_log = numpy.log10(numpy.array(ref_list, dtype=float)**-1)
    var_list = sorted(l2_err[0].keys())
    y_log = {}
    for k in var_list:
        y_log[k] = numpy.log10(numpy.array([e[k] for e in l2_err]))
    setup_name = setup.__name__

    def check_convergence(x_log, y_log, expected_slope, field_str, plot=False, ax=None):
        slope_rtol = 0.07
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
        if plot:
            # plot points
            ax.plot(x_log, y_log, 'k.')
            x_min = x_log.min()
            x_max = x_log.max()
            offset = 0.05*(x_max - x_min)
            npoints = 50
            xx = numpy.linspace(x_min - offset, x_max + offset, npoints)
            yy = intercept + slope*xx
            # plot line
            ax.plot(xx, yy, linestyle='--', linewidth=0.5, color='k')
            ax.text(xx[2*int(npoints/3)], yy[2*int(npoints/3)], '{:4.2f}'.format(slope),
                    verticalalignment='top',
                    horizontalalignment='left')
            ax.set_xlabel('log10(dx)')
            ax.set_ylabel('log10(L2 error)')
            ax.set_title(field_metadata[field_str]['name'])
            ax.grid(True, which='major')
        if expected_slope is not None:
            err_msg = '{:}: {:} Wrong convergence rate {:.4f}, expected {:.4f}'.format(setup_name, field_str, slope, expected_slope)
            assert slope > expected_slope*(1 - slope_rtol), err_msg
            print_output('{:}: {:} convergence rate {:.4f} PASSED'.format(setup_name, field_str, slope))
        else:
            print_output('{:}: {:} convergence rate {:.4f}'.format(setup_name, field_str, slope))
        return slope

    vars_to_plot = ['elev_2d', 'temp_3d', 'uv_full', 'w_3d']
    if saveplot:
        import matplotlib.pyplot as plt
        ncols = int(len(vars_to_plot)/2)
        nrows = 2
        fig, ax_list = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
        ax_iter = iter(ax_list.ravel())
        char_iter = iter(['a', 'b', 'c', 'd'])
        fig.subplots_adjust(wspace=0.3, hspace=0.33)

    def expected_rate(v):
        if v in ['w_3d']:
            return polynomial_degree
        return polynomial_degree + 1

    for v in var_list:
        do_plot = v in vars_to_plot and saveplot
        ax = next(ax_iter) if do_plot else None
        if do_plot:
            char = next(char_iter)
            ax.text(-0.07, 1.02, char+')', fontsize=16, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left')
        check_convergence(x_log, y_log[v], expected_rate(v), v, plot=do_plot, ax=ax)

    if saveplot:
        ref_str = 'ref-' + '-'.join([str(r) for r in ref_list])
        degree_str = 'o{:}'.format(polynomial_degree)
        test_name = 'baroclinic-mms'
        imgfile = '_'.join(['convergence', test_name, setup_name, ref_str, degree_str, space_str])
        imgfile += '.pdf'
        imgdir = create_directory('plots')
        imgfile = os.path.join(imgdir, imgfile)
        print_output('saving figure {:}'.format(imgfile))
        plt.savefig(imgfile, dpi=200, bbox_inches='tight')


@pytest.mark.parallel(nprocs=2)
def test_baroclinic_mms():
    run_convergence(setup5, [1, 2, 4],
                    polynomial_degree=1, element_family='dg-dg',
                    saveplot=False, do_export=False)


if __name__ == '__main__':
    run_convergence(setup5, [1, 2, 4, 6, 8, 10],
                    polynomial_degree=1, element_family='dg-dg',
                    saveplot=True, do_export=False)
