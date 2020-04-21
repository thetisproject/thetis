"""
Generates setups for tracer advection-diffusion MMS test

"""
import sympy
from sympy import init_printing

init_printing()

# coordinates
x, y, z = sympy.symbols('x y z')
# domain lenght, x in [0, Lx], y in [0, Ly]
lx, ly = sympy.symbols('lx ly')


def is_constant(u):
    """
    True if u does not depend on x,y,z
    """
    out = 0
    for i in (x, y, z):
        out += sympy.diff(u, i)
    return out == 0


def get_ufl_expr(u):
    """Generates string that can be evaluated as a UFL expression"""
    fmt = 'Constant({:})' if is_constant(u) else '{:}'
    return fmt.format(str(u))


def get_scalar_entry(name, u, *args):
    """Generates an entry for a scalar expression"""
    t = """
    def {name}(self, {args}):
        return {u}\n"""

    args_str = ', '.join(args)
    return t.format(name=name, u=get_ufl_expr(u), args=args_str)


def get_vector_entry(name, u, v, w, *args):
    """Generates an entry for a 2d vector expression"""
    t = """
    def {name}(self, {args}):
        return as_vector(
            [
                {:},
                {:},
                {:},
            ])\n"""

    args_str = ', '.join(args)
    uvw = map(get_ufl_expr, (u, v, w))
    return t.format(*uvw, name=name, args=args_str)


def get_header(name, description):
    t = '''class {name}:
    """
    {txt}
    """'''
    return t.format(name=name, txt=description)


def compute_residual(h, eta, u, v, w, kappa, tracer):
    """Compute residual of advection-diffusion equation"""
    adv = sympy.diff(tracer*u, x) + sympy.diff(tracer*v, y) + sympy.diff(tracer*w, z)
    stress = kappa*(sympy.diff(tracer, x) + sympy.diff(tracer, y))
    diff = sympy.diff(stress, x) + sympy.diff(stress, y)
    res = adv + diff
    return res


def compute_w(eta, u, v, h):
    """Solves w from continuity equation"""
    div_uv = sympy.diff(u, x) + sympy.diff(v, y)
    c = u*sympy.diff(h, x) + v*sympy.diff(h, y)
    w = -sympy.integrate(div_uv, (z, -h, z)) - c
    return w


def generate_setup(name, description, h, eta, u, v, kappa, tracer):
    """
    Generates setup function that can be copied to mms test.
    """
    w = compute_w(eta, u, v, h)
    residual = compute_residual(h, eta, u, v, w, kappa, tracer)
    txt = ''
    txt += get_header(name, description)
    args_2d = 'x', 'y', 'lx', 'ly'
    args_3d = 'x', 'y', 'z', 'lx', 'ly'
    txt += get_scalar_entry('bath', h, *args_2d)
    txt += get_scalar_entry('elev', eta, *args_2d)
    txt += get_vector_entry('uv', u, v, 0, *args_3d)
    txt += get_vector_entry('w', 0, 0, w, *args_3d)
    txt += get_scalar_entry('kappa', kappa, *args_3d)
    txt += get_scalar_entry('tracer', tracer, *args_3d)
    txt += get_scalar_entry('residual', residual, *args_3d)
    print('')
    print(txt)


name = 'Setup1'
description = """Constant bathymetry and u velocty, zero diffusivity, non-trivial tracer"""
h = 40.0
eta = 0.0
u = 1.0
v = 0.0
kappa = 0.0
tracer = sympy.sin(0.2*sympy.pi*(3.0*x + 1.0*y)/lx)
generate_setup(name, description, h, eta, u, v, kappa, tracer)

name = 'Setup2'
description = """Constant bathymetry, zero velocity, constant kappa, x-varying T"""
h = 40.0
eta = 0.0
u = 1.0
v = 0.0
kappa = 50.0
tracer = sympy.sin(3*sympy.pi*x/lx)
generate_setup(name, description, h, eta, u, v, kappa, tracer)

name = 'Setup3'
description = """Constant bathymetry, zero kappa, non-trivial velocity and T"""
h = 40.0
eta = 0.0
u = sympy.sin(sympy.pi*(y/ly + 2*x/lx))*sympy.sin(sympy.pi*z/40)
v = sympy.sin(sympy.pi*(0.3*y/ly + 0.3*x/lx))*sympy.sin(sympy.pi*z/40)
kappa = 0.0
tracer = (0.8*sympy.cos(0.5*sympy.pi*z/40) + 0.2)*sympy.cos(sympy.pi*(0.75*y/ly + 1.5*x/lx))
generate_setup(name, description, h, eta, u, v, kappa, tracer)

name = 'Setup4'
description = """Constant bathymetry, constant kappa, non-trivial velocity and T"""
h = 40.0
eta = 0.0
u = sympy.sin(sympy.pi*(y/ly + 2*x/lx))*sympy.sin(sympy.pi*z/40)
v = sympy.sin(sympy.pi*(0.3*y/ly + 0.3*x/lx))*sympy.sin(sympy.pi*z/40)
kappa = 50.0
tracer = (0.8*sympy.cos(0.5*sympy.pi*z/40) + 0.2)*sympy.cos(sympy.pi*(0.75*y/ly + 1.5*x/lx))
generate_setup(name, description, h, eta, u, v, kappa, tracer)
