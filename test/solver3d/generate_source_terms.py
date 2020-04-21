"""
Generates setups for baroclinic MMS tests

"""
import sympy
import numbers
from sympy import init_printing
init_printing()

# coordinates
x, y, z = sympy.symbols('xyz[0] xyz[1] xyz[2]')
x_2d, y_2d = sympy.symbols('xy[0] xy[1]')
z_tilde = sympy.symbols('z_tilde')
# domain lenght, x in [0, Lx], y in [0, Ly]
lx, ly = sympy.symbols('lx ly')
# depth scale
depth = sympy.symbols('depth', positive=True)
# coriolis scale
f0 = sympy.symbols('f0', positive=True)
# viscosity scale
nu0 = sympy.symbols('nu0', positive=True)
# constant salinity
salt0 = sympy.symbols('salt_const', positive=True)
temp0 = sympy.symbols('temp_const', positive=True)
# gravitational acceleration
g = sympy.symbols('g_grav')
# time
t = sympy.symbols('t', positive=True)
T = sympy.symbols('T', positive=True)

eos_alpha, eos_beta, eos_t0, eos_s0 = sympy.symbols('eos_alpha eos_beta eos_t0 eos_s0')
rho0 = sympy.symbols('rho_0', positive=True)

# setup 1 -- temp only
# bath = depth
# elev = 0
# u = 0
# v = 0
# temp = 5*sympy.cos((2*x + y)/lx)*sympy.cos((z/depth)) + 15
# salt = salt0
# nu = nu0
# f = f0
# omit_int_pg = False

# setup 2 -- elevation only
# bath = depth
# elev = 5*sympy.cos((x + 5*y/2)/lx)
# u = 0
# v = 0
# temp = temp0
# salt = salt0
# nu = nu0
# f = f0
# #omit_int_pg = True
# omit_int_pg = False  # setup2b

# setup 3 -- velocity only
# bath = depth
# elev = 0
# u = sympy.cos((2*x + y)/lx)
# v = 0
# temp = temp0
# salt = salt0
# nu = nu0
# f = f0
# omit_int_pg = True

# setup 4 -- velocity and temp
# bath = depth
# elev = 0
# u = sympy.cos((2*x + y)/lx)*sympy.cos(3*(z/depth))/2
# v = 0
# temp = 5*sympy.cos((x + 2*y)/lx)*sympy.cos((z/depth)) + 15
# salt = salt0
# nu = nu0
# f = f0
# omit_int_pg = False  # setup4b

# setup 5 -- velocity and temp, symmetric
bath = depth
elev = 0
u = sympy.sin(2*sympy.pi*x/lx)*sympy.cos(3*(z/depth))/2
v = sympy.cos(sympy.pi*y/ly)*sympy.sin((z/depth/2))/3
temp = 15*sympy.sin(sympy.pi*x/lx)*sympy.sin(sympy.pi*y/ly)*sympy.cos(z/depth) + 15
salt = salt0
nu = nu0
f = f0
omit_int_pg = False


def split_velocity(u, v, eta, bath):
    u_2d = sympy.integrate(u, (z, -bath, eta))/(eta+bath)
    v_2d = sympy.integrate(v, (z, -bath, eta))/(eta+bath)
    u_3d = u - u_2d
    v_3d = v - v_2d
    return u_3d, v_3d, u_2d, v_2d


def evaluate_w(eta, u, v, bath):
    assert sympy.diff(bath, x) == 0 and sympy.diff(bath, y) == 0, 'bath source not implemented'
    return sympy.integrate(-(sympy.diff(u, x) + sympy.diff(v, y)), (z, -bath, z))


def evaluate_baroclinicity(elev, temp, salt):
    rho = - eos_alpha*(temp - eos_t0) + eos_beta*(salt - eos_s0)
    baroc_head = sympy.integrate(rho/rho0, (z, elev, z))
    return rho, baroc_head


def evaluate_tracer_source(eta, trac, u, v, w, bath, f, nu):
    adv_t_uv = sympy.diff(trac, x)*u + sympy.diff(trac, y)*v
    adv_t_w = sympy.diff(trac, z)*w
    adv_t = adv_t_uv + adv_t_w
    return adv_t, adv_t_uv, adv_t_w


def evaluate_mom_source(eta, baroc_head, u, v, w, u_3d, v_3d, bath, f, nu):
    int_pg_x = -g*sympy.diff(baroc_head, x)  # NOTE why the minus sign? BUG
    int_pg_y = -g*sympy.diff(baroc_head, y)
    adv_u = sympy.diff(u, x)*u + sympy.diff(u, y)*v
    adv_v = sympy.diff(v, x)*u + sympy.diff(v, y)*v
    adv_w_u = sympy.diff(u, z)*w
    adv_w_v = sympy.diff(v, z)*w
    cori_u = -f*v_3d
    cori_v = f*u_3d
    res_u = adv_u + adv_w_u + cori_u
    res_v = adv_v + adv_w_v + cori_v
    if not omit_int_pg:
        res_u += int_pg_x
        res_v += int_pg_y
    return res_u, res_v, int_pg_x, int_pg_y, adv_u, adv_v, adv_w_u, adv_w_v, cori_u, cori_v


def evaluate_swe_source(eta, u, v, bath, f, nu, nonlin=True):
    # evaluate shallow water equations
    if nonlin:
        tot_depth = eta + bath
    else:
        tot_depth = bath
    div_hu = sympy.diff(tot_depth*u, x) + sympy.diff(tot_depth*v, y)
    res_elev = sympy.diff(eta, t) + div_hu
    cori_u = -f*v
    cori_v = f*u
    pg_u = g*sympy.diff(eta, x)
    pg_v = g*sympy.diff(eta, y)
    visc_u = -(2*sympy.diff(nu*sympy.diff(u, x), x)
               + sympy.diff(nu*sympy.diff(u, y), y)
               + sympy.diff(nu*sympy.diff(v, x), y))
    visc_v = -(2*sympy.diff(nu*sympy.diff(v, y), y)
               + sympy.diff(nu*sympy.diff(v, x), x)
               + sympy.diff(nu*sympy.diff(u, y), x))
    visc_u += -sympy.diff(tot_depth, x)/tot_depth * nu * 2 * sympy.diff(u, x)
    visc_v += -sympy.diff(tot_depth, y)/tot_depth * nu * 2 * sympy.diff(v, y)
    # NOTE in the coupled system 2d mom eq has no advection/diffusion terms
    res_u = sympy.diff(u, t) + cori_u + pg_u
    res_v = sympy.diff(v, t) + cori_v + pg_v
    return res_elev, res_u, res_v, cori_u, cori_v


u_3d, v_3d, u_2d, v_2d = split_velocity(u, v, elev, bath)
w = evaluate_w(elev, u, v, bath)
rho, baroc_head = evaluate_baroclinicity(elev, temp, salt)

mom_source_x, mom_source_y, int_pg_x, int_pg_y, adv_u, adv_v, adv_w_u, adv_w_v, cori_u, cori_v = evaluate_mom_source(elev, baroc_head, u, v, w, u_3d, v_3d, bath, f, nu)

vol_source_2d, mom_source_2d_x, mom_source_2d_y, cori_u_2d, cori_v_2d = evaluate_swe_source(elev, u_2d, v_2d, bath, f, nu, nonlin=True)
temp_source_3d, adv_t_uv, adv_t_w = evaluate_tracer_source(elev, temp, u, v, w, bath, f, nu)


def expr2str(e):
    if isinstance(e, (numbers.Number, sympy.numbers.Zero)):
        return 'Constant({:})'.format(e)
    return str(e)


def print_expr(name, *expr, dict_fmt=True):
    if len(expr) == 1:
        expr_str = expr2str(expr[0])
    else:
        if all(isinstance(e, numbers.Number) for e in expr):
            comp_str = ', '.join([str(e) for e in expr])
            expr_str = 'Constant((' + comp_str + '))'
        else:
            comp_str = ', '.join([expr2str(e) for e in expr])
            expr_str = 'as_vector((' + comp_str + '))'
    if dict_fmt:
        print("    out['{:}'] = {:}".format(name, expr_str))
    else:
        print("    {:} = {:}".format(name, expr_str))


def to_2d_coords(expr):
    if isinstance(expr, numbers.Number):
        return expr
    return expr.subs(x, x_2d).subs(y, y_2d)


def print_ufl_expressions():
    """
    Print python expressions
    """
    print_expr('elev_2d', to_2d_coords(elev))
    print_expr('uv_full_3d', u, v, 0)
    print_expr('uv_2d', to_2d_coords(u_2d), to_2d_coords(v_2d))
    print_expr('uv_dav_3d', u_2d, v_2d, 0)
    print_expr('uv_3d', u_3d, v_3d, 0)
    print_expr('w_3d', 0, 0, w)
    print_expr('temp_3d', temp)
    print_expr('density_3d', rho)
    print_expr('baroc_head_3d', baroc_head)
    print_expr('int_pg_3d', int_pg_x, int_pg_y, 0)

    print_expr('vol_source_2d', to_2d_coords(vol_source_2d))
    print_expr('mom_source_2d', to_2d_coords(mom_source_2d_x), to_2d_coords(mom_source_2d_y))
    print_expr('mom_source_3d', mom_source_x, mom_source_y, 0)
    print_expr('temp_source_3d', temp_source_3d)


def print_latex_expressions():
    """
    Print expressions in Latex for publications
    """
    _x, _y, _z = sympy.symbols('x y z')
    _lx, _ly = sympy.symbols('L_x L_y')
    _depth = sympy.symbols('h')
    _eos_alpha, _eos_t0, _eos_s0 = sympy.symbols('alpha T_0 S_0')
    _eos_beta = 0
    _rho0 = sympy.symbols('rho_0', positive=True)
    _salt0 = sympy.symbols('S_a', positive=True)
    _g = sympy.symbols('g', positive=True)

    def for_latex(e):
        o = e.subs([(x, _x), (y, _y), (z, _z), (lx, _lx), (ly, _ly), (depth, _depth), (g, _g)])
        o = o.subs([(eos_alpha, _eos_alpha), (eos_beta, _eos_beta), (eos_t0, _eos_t0), (eos_s0, _eos_s0), (rho0, _rho0), (salt0, _salt0)])
        return sympy.simplify(o)

    print('\nAnalytical functions:\n')
    print('T_a &= ' + sympy.latex(for_latex(temp)))
    print('u_a &= ' + sympy.latex(for_latex(u)))
    print('v_a &= ' + sympy.latex(for_latex(v)))
    print('u_a\' &= ' + sympy.latex(for_latex(u_3d)))
    print('v_a\' &= ' + sympy.latex(for_latex(v_3d)))
    print('\\bar{u}_a &= ' + sympy.latex(for_latex(u_2d)))
    print('\\bar{v}_a &= ' + sympy.latex(for_latex(v_2d)))
    print('w_a &= ' + sympy.latex(for_latex(w)))
    print('\\rho_a\' &= ' + sympy.latex(for_latex(rho)))
    print('r_a &= ' + sympy.latex(for_latex(baroc_head)))
    print('(\\IPG)_{x} &= ' + sympy.latex(for_latex(int_pg_x)))
    print('(\\IPG)_{y} &= ' + sympy.latex(for_latex(int_pg_y)))
    print('(\\bnabla_h \\cdot (\\bu \\bu))_{x} &= ' + sympy.latex(for_latex(adv_u)))
    print('(\\bnabla_h \\cdot (\\bu \\bu))_{y} &= ' + sympy.latex(for_latex(adv_v)))
    print('\\pd{\\left(w u \\right)}{z} &= ' + sympy.latex(for_latex(adv_w_u)))
    print('\\pd{\\left(w v \\right)}{z} &= ' + sympy.latex(for_latex(adv_w_v)))
    print('f\\bm{e}_z\\wedge\\baru &= ' + sympy.latex(for_latex(cori_u_2d)))
    print('f\\bm{e}_z\\wedge\\baru &= ' + sympy.latex(for_latex(cori_v_2d)))
    print('f\\bm{e}_z\\wedge u\' &= ' + sympy.latex(for_latex(cori_u)))
    print('f\\bm{e}_z\\wedge v\' &= ' + sympy.latex(for_latex(cori_v)))
    print('\\bnabla_h\\cdot\\left(H\\bbaru\\right) &= ' + sympy.latex(for_latex(vol_source_2d)))
    print('\\bnabla_h \\cdot (\\bu T) &= ' + sympy.latex(for_latex(adv_t_uv)))
    print('\\pd{\\left(w T \\right)}{z} &= ' + sympy.latex(for_latex(adv_t_w)))


print_ufl_expressions()
