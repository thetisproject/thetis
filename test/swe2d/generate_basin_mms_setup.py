"""
Generates setups for 2d shallow water MMS tests

"""
import sympy
from sympy import init_printing

init_printing()

# coordinates
x, y, z = sympy.symbols('x[0] x[1] x[2]')
z_tilde = sympy.symbols('z_tilde')
# domain lenght, x in [0, Lx], y in [0, Ly]
lx, ly = sympy.symbols('lx ly')
# depth scale
h0 = sympy.symbols('h0', positive=True)
# coriolis scale
f0 = sympy.symbols('f0', positive=True)
# viscosity scale
nu0 = sympy.symbols('nu0', positive=True)
# gravitational acceleration
g = sympy.symbols('g')
# time
t = sympy.symbols('t', positive=True)
T = sympy.symbols('T', positive=True)


def get_ufl_expr(w):
    """
    Generates string that can be though to be a UFL Expression"""
    return str(w)


def get_scalar_entry(name, u=None, v=None, w=None):
    """Generates an entry for a scalar expression"""
    t = """    out['{name}'] = {u}\n"""

    def fds(u):
        return '0.0' if u is None else get_ufl_expr(u)
    return t.format(name=name, u=fds(u))


def get_vector_entry(name, u=None, v=None):
    """Generates an entry for a 2d vector expression"""
    t = """    out['{name}'] = as_vector(
        [
            {u},
            {v},
        ])\n"""

    def fds(u):
        return '0.0' if u is None else get_ufl_expr(u)
    return t.format(name=name, u=fds(u), v=fds(v))


def get_header(name, description):
    t = '''def {name}(x, lx, ly, h0, f0, nu0, g):
    """
    {txt}
    """
    out = {{}}\n'''
    return t.format(name=name, txt=description)


def get_footer():
    t = """
    # NOTE boundary condititions must be set manually to something  meaningful
    out['bnd_funcs'] = {1: {'uv': None},
                        2: {'uv': None},
                        3: {'uv': None},
                        4: {'uv': None},
                        }
    return out"""
    return t


def evaluate_source_term(eta, u, v, h, f, nu, nonlin=True):
    # evaluate equations
    if nonlin:
        depth = eta + h
    else:
        depth = h
    div_hu = sympy.diff(depth*u, x) + sympy.diff(depth*v, y)
    res_elev = sympy.diff(eta, t) + div_hu
    u_x = sympy.diff(u, x)
    u_y = sympy.diff(u, y)
    v_x = sympy.diff(v, x)
    v_y = sympy.diff(v, y)
    if nonlin:
        adv_u = u*u_x + v*u_y
        adv_v = u*v_x + v*v_y
    else:
        adv_u = adv_v = 0
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
    visc_u += -sympy.diff(depth, x)/depth * nu * 2 * sympy.diff(u, x)
    visc_v += -sympy.diff(depth, y)/depth * nu * 2 * sympy.diff(v, y)
    res_u = sympy.diff(u, t) + adv_u + cori_u + pg_u + visc_u
    res_v = sympy.diff(v, t) + adv_v + cori_v + pg_v + visc_v
    return res_elev, res_u, res_v


def generate_setup(name, description, h, f, eta, u, v, nu):
    """
    Generates setup function that can be copied to mms test.
    """
    res_elev, res_u, res_v = evaluate_source_term(eta, u, v, h, f, nu)
    txt = ''
    txt += get_header(name, description)
    txt += get_scalar_entry('bath_expr', h)
    if f != 0.0:
        txt += get_scalar_entry('cori_expr', f)
    if nu != 0.0:
        txt += get_scalar_entry('visc_expr', nu)
    txt += get_scalar_entry('elev_expr', eta)
    txt += get_vector_entry('uv_expr', u=u, v=v)
    txt += get_scalar_entry('res_elev_expr', res_elev)
    txt += get_vector_entry('res_uv_expr', u=res_u, v=res_v)
    txt += get_footer()
    print('')
    print('')
    print(txt)


name = 'setup7'
description = """Non-trivial Coriolis, bath, elev, u and v, tangential velocity is zero at bnd to test flux BCs"""
h = 4.0 + h0*sympy.sqrt(0.3*x*x + 0.2*y*y + 0.1)/lx
f = f0*sympy.cos(sympy.pi*(x + y)/lx)
nu = 0
eta = sympy.cos(sympy.pi*(3.0*x + 1.0*y)/lx)
u = sympy.sin(sympy.pi*(-2.0*x + 1.0*y)/lx)*sympy.sin(sympy.pi*y/ly)
v = 0.5*sympy.sin(sympy.pi*x/lx)*sympy.sin(sympy.pi*(-3.0*x + 1.0*y)/lx)
generate_setup(name, description, h, f, eta, u, v, nu)

name = 'setup8'
description = """Non-trivial Coriolis, bath, elev, u and v, tangential velocity is non-zero at bnd, must prescribe uv at boundary."""
h = 4.0 + h0*sympy.sqrt(0.3*x*x + 0.2*y*y + 0.1)/lx
f = f0*sympy.cos(sympy.pi*(x + y)/lx)
nu = 0
eta = sympy.cos(sympy.pi*(3.0*x + 1.0*y)/lx)
u = sympy.sin(sympy.pi*(-2.0*x + 1.0*y)/lx)
v = 0.5*sympy.sin(sympy.pi*(-3.0*x + 1.0*y)/lx)
generate_setup(name, description, h, f, eta, u, v, nu)

name = 'setup9'
description = 'No Coriolis, non-trivial bath, viscosity, elev, u and v.'
h = 4.0 + h0*sympy.sqrt(0.3*x*x + 0.2*y*y + 0.1)/lx
f = 0
eta = sympy.cos(sympy.pi*(3.0*x + 1.0*y)/lx)
u = sympy.sin(sympy.pi*(-2.0*x + 1.0*y)/lx)
v = 0.5*sympy.sin(sympy.pi*(-3.0*x + 1.0*y)/lx)
nu = nu0*(1.0 + x/lx)
generate_setup(name, description, h, f, eta, u, v, nu)
