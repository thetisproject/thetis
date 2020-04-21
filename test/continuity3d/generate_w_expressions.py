"""
Generates setups for testing w computation

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


def compute_w(eta, u, v, h):
    """Solves w from continuity equation"""
    div_uv = sympy.diff(u, x) + sympy.diff(v, y)
    c = u*sympy.diff(h, x) + v*sympy.diff(h, y)
    w = -sympy.integrate(div_uv, (z, -h, z)) - c
    return w


def generate_setup(name, description, h, eta, u, v):
    """
    Generates setup function that can be copied to mms test.
    """
    w = compute_w(eta, u, v, h)
    txt = ''
    txt += get_header(name, description)
    args_2d = 'x', 'y', 'lx', 'ly'
    args_3d = 'x', 'y', 'z', 'lx', 'ly'
    txt += get_scalar_entry('bath', h, *args_2d)
    txt += get_scalar_entry('elev', eta, *args_2d)
    txt += get_vector_entry('uv', u, v, 0, *args_3d)
    txt += get_scalar_entry('w', w, *args_3d)
    print('')
    print(txt)


name = 'Setup1'
description = """linear bath and elev, constant u,v"""
h = 10.0 + 3*x/lx
eta = x*y/lx/lx
u = 1.0
v = 0.3
generate_setup(name, description, h, eta, u, v)

name = 'Setup2'
description = """Constant bath and elev, linear u"""
h = 20.0
eta = 0.0
u = x/lx
v = 0.0
generate_setup(name, description, h, eta, u, v)

name = 'Setup3'
description = """Non-trivial bath and elev, uv depends on (x,y)"""
h = 3 + 6.0*(sympy.cos(sympy.pi*sympy.sqrt(x*x + y*y + 1.0)/lx) + 3.0)
eta = 5.0*sympy.sin(0.4*sympy.pi*sympy.sqrt(1.5*x*x + y*y + 1.0)/lx)
u = sympy.sin(0.2*sympy.pi*(3.0*x + 1.0*y)/lx)
v = 0.2*sympy.sin(0.2*sympy.pi*(3.0*x + 1.0*y)/lx)
generate_setup(name, description, h, eta, u, v)
