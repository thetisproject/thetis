"""
MMS test for 2d shallow water equations.

- setuoXX functions define analytical expressions for fields and boundary
  conditions. Expressions were derived with sympy.
- run function runs the MMS setup with a single mesh resolution, returning
  L2 errors.
- run_convergence runs a scaling test, computes and asserts convergence rate.
- test_XX functions are the default test cases for continuous testing.

Tuomas Karna 2015-10-29
"""
from cofs import *
import numpy
from scipy import stats
import pytest


def setup1(Lx, Ly, depth, f0, g, mimetic=True):
    """
    Tests the pressure gradient only

    Constant bath, zero velocity, no Coriolis
    """
    out = {}
    out['bath_expr'] = Expression(
        'h0',
        Lx=Lx, h0=depth, f0=f0, g=g)
    out['cori_expr'] = Expression(
        '0',
        Lx=Lx, h0=depth, f0=f0, g=g)
    out['elev_expr'] = Expression(
        'cos(pi*(3.0*x[0] + 1.0*x[1])/Lx)',
        Lx=Lx, h0=depth, f0=f0, g=g)
    out['uv_expr'] = Expression(
        (
            '0',
            '0',
        ), Lx=Lx, h0=depth, f0=f0, g=g)
    out['res_elev_expr'] = Expression(
        '0',
        Lx=Lx, h0=depth, f0=f0, g=g)
    out['res_uv_expr'] = Expression(
        (
            '-3.0*pi*g*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx',
            '-1.0*pi*g*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx',
        ), Lx=Lx, h0=depth, f0=f0, g=g)
    out['options'] = {}
    out['options'] = {'mimetic': mimetic}
    out['bnd_funcs'] = {1: {'elev': None, 'uv': None},
                        2: {'elev': None, 'uv': None},
                        3: {'elev': None, 'uv': None},
                        4: {'elev': None, 'uv': None},
                        }
    return out


def setup1dg(Lx, Ly, depth, f0, g):
    """
    Tests the pressure gradient only

    Constant bath, zero velocity, no Coriolis
    """
    return setup1(Lx, Ly, depth, f0, g, mimetic=False)


def setup2(Lx, Ly, depth, f0, g):
    """
    Tests the advection and div(Hu) terms

    Constant bath, x velocity, zero elevation, no Coriolis
    """
    out = {}
    out['bath_expr'] = Expression(
        'h0',
        Lx=Lx, h0=depth, f0=f0, g=g)
    out['cori_expr'] = Expression(
        '0',
        Lx=Lx, h0=depth, f0=f0, g=g)
    out['elev_expr'] = Expression(
        '0',
        Lx=Lx, h0=depth, f0=f0, g=g)
    out['uv_expr'] = Expression(
        (
            'sin(2*pi*x[0]/Lx)',
            '0',
        ), Lx=Lx, h0=depth, f0=f0, g=g)
    out['res_elev_expr'] = Expression(
        '2*pi*h0*cos(2*pi*x[0]/Lx)/Lx',
        Lx=Lx, h0=depth, f0=f0, g=g)
    out['res_uv_expr'] = Expression(
        (
            '2*pi*sin(2*pi*x[0]/Lx)*cos(2*pi*x[0]/Lx)/Lx',
            '0',
        ), Lx=Lx, h0=depth, f0=f0, g=g)
    out['options'] = {}
    out['bnd_funcs'] = {1: {'elev': None, 'uv': None},
                        2: {'elev': None, 'uv': None},
                        3: {'elev': None, 'uv': None},
                        4: {'elev': None, 'uv': None},
                        }
    return bath_expr, elev_expr, uv_expr, cori_expr, res_elev_expr, res_uv_expr, options


def setup3(Lx, Ly, depth, f0, g):
    """
    Tests and div(Hu) terms for nonlin=False option

    Constant bath, x velocity, zero elevation, no Coriolis
    """
    out = {}
    out['bath_expr'] = Expression(
        'h0',
        Lx=Lx, h0=depth, f0=f0, g=g)
    out['cori_expr'] = Expression(
        '0',
        Lx=Lx, h0=depth, f0=f0, g=g)
    out['elev_expr'] = Expression(
        '0',
        Lx=Lx, h0=depth, f0=f0, g=g)
    out['uv_expr'] = Expression(
        (
            'sin(2*pi*x[0]/Lx)',
            '0',
        ), Lx=Lx, h0=depth, f0=f0, g=g)
    out['res_elev_expr'] = Expression(
        '2*pi*h0*cos(2*pi*x[0]/Lx)/Lx',
        Lx=Lx, h0=depth, f0=f0, g=g)
    out['res_uv_expr'] = Expression(
        (
            '0',
            '0',
        ), Lx=Lx, h0=depth, f0=f0, g=g)
    out['options'] = {'nonlin': False}
    out['bnd_funcs'] = {1: {'elev': None, 'uv': None},
                        2: {'elev': None, 'uv': None},
                        3: {'elev': None, 'uv': None},
                        4: {'elev': None, 'uv': None},
                        }
    return out


def setup4(Lx, Ly, depth, f0, g):
    """
    Constant bath, no Coriolis, non-trivial elev and u
    """
    out = {}
    out['bath_expr'] = Expression(
        'h0',
        Lx=Lx, h0=depth, f0=f0, g=g)
    out['cori_expr'] = Expression(
        '0',
        Lx=Lx, h0=depth, f0=f0, g=g)
    out['elev_expr'] = Expression(
        'cos(pi*(3.0*x[0] + 1.0*x[1])/Lx)',
        Lx=Lx, h0=depth, f0=f0, g=g)
    out['uv_expr'] = Expression(
        (
            'sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx)',
            '0',
        ), Lx=Lx, h0=depth, f0=f0, g=g)
    out['res_elev_expr'] = Expression(
        '-2.0*pi*(h0 + cos(pi*(3.0*x[0] + 1.0*x[1])/Lx))*cos(pi*(-2.0*x[0] + 1.0*x[1])/Lx)/Lx - 3.0*pi*sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx)*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx',
        Lx=Lx, h0=depth, f0=f0, g=g)
    out['res_uv_expr'] = Expression(
        (
            '-3.0*pi*g*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx - 2.0*pi*sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx)*cos(pi*(-2.0*x[0] + 1.0*x[1])/Lx)/Lx',
            '-1.0*pi*g*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx',
        ), Lx=Lx, h0=depth, f0=f0, g=g)
    out['options'] = {}
    out['bnd_funcs'] = {1: {'elev': None, 'uv': None},
                        2: {'elev': None, 'uv': None},
                        3: {'elev': None, 'uv': None},
                        4: {'elev': None, 'uv': None},
                        }
    return out


def setup5(Lx, Ly, depth, f0, g):
    """
    No Coriolis, non-trivial bath, elev, u and v
    """
    out = {}
    out['bath_expr'] = Expression(
        '4.0 + h0*sqrt(0.3*x[0]*x[0] + 0.2*x[1]*x[1])/Lx',
        Lx=Lx, h0=depth, f0=f0, g=g)
    out['cori_expr'] = Expression(
        '0',
        Lx=Lx, h0=depth, f0=f0, g=g)
    out['elev_expr'] = Expression(
        'cos(pi*(3.0*x[0] + 1.0*x[1])/Lx)',
        Lx=Lx, h0=depth, f0=f0, g=g)
    out['uv_expr'] = Expression(
        (
            'sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx)',
            '0.5*sin(pi*(-3.0*x[0] + 1.0*x[1])/Lx)',
        ), Lx=Lx, h0=depth, f0=f0, g=g)
    out['res_elev_expr'] = Expression(
        '(0.3*h0*x[0]/(Lx*sqrt(0.3*x[0]*x[0] + 0.2*x[1]*x[1])) - 3.0*pi*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx)*sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx) + 0.5*(0.2*h0*x[1]/(Lx*sqrt(0.3*x[0]*x[0] + 0.2*x[1]*x[1])) - 1.0*pi*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx)*sin(pi*(-3.0*x[0] + 1.0*x[1])/Lx) + 0.5*pi*(cos(pi*(3.0*x[0] + 1.0*x[1])/Lx) + 4.0 + h0*sqrt(0.3*x[0]*x[0] + 0.2*x[1]*x[1])/Lx)*cos(pi*(-3.0*x[0] + 1.0*x[1])/Lx)/Lx - 2.0*pi*(cos(pi*(3.0*x[0] + 1.0*x[1])/Lx) + 4.0 + h0*sqrt(0.3*x[0]*x[0] + 0.2*x[1]*x[1])/Lx)*cos(pi*(-2.0*x[0] + 1.0*x[1])/Lx)/Lx',
        Lx=Lx, h0=depth, f0=f0, g=g)
    out['res_uv_expr'] = Expression(
        (
            '-3.0*pi*g*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx + 0.5*pi*sin(pi*(-3.0*x[0] + 1.0*x[1])/Lx)*cos(pi*(-2.0*x[0] + 1.0*x[1])/Lx)/Lx - 2.0*pi*sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx)*cos(pi*(-2.0*x[0] + 1.0*x[1])/Lx)/Lx',
            '-1.0*pi*g*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx + 0.25*pi*sin(pi*(-3.0*x[0] + 1.0*x[1])/Lx)*cos(pi*(-3.0*x[0] + 1.0*x[1])/Lx)/Lx - 1.5*pi*sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx)*cos(pi*(-3.0*x[0] + 1.0*x[1])/Lx)/Lx',
        ), Lx=Lx, h0=depth, f0=f0, g=g)
    out['options'] = {}
    out['bnd_funcs'] = {1: {'elev': None, 'uv': None},
                        2: {'elev': None, 'uv': None},
                        3: {'elev': None, 'uv': None},
                        4: {'elev': None, 'uv': None},
                        }
    return out


def setup6(Lx, Ly, depth, f0, g):
    """
    No Coriolis, non-trivial bath, elev, u and v,
    tangential velocity is zero at bnd to test flux BCs
    """
    out = {}
    out['bath_expr'] = Expression(
        '4.0 + h0*sqrt(0.3*x[0]*x[0] + 0.2*x[1]*x[1])/Lx',
        Lx=Lx, Ly=Ly, h0=depth, f0=f0, g=g)
    out['cori_expr'] = Expression(
        '0',
        Lx=Lx, Ly=Ly, h0=depth, f0=f0, g=g)
    out['elev_expr'] = Expression(
        'cos(pi*(3.0*x[0] + 1.0*x[1])/Lx)',
        Lx=Lx, Ly=Ly, h0=depth, f0=f0, g=g)
    out['uv_expr'] = Expression(
        (
            'sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx)*sin(pi*x[1]/Ly)',
            '0.5*sin(pi*x[0]/Lx)*sin(pi*(-3.0*x[0] + 1.0*x[1])/Lx)',
        ), Lx=Lx, Ly=Ly, h0=depth, f0=f0, g=g)
    out['res_elev_expr'] = Expression(
        '(0.3*h0*x[0]/(Lx*sqrt(0.3*x[0]*x[0] + 0.2*x[1]*x[1])) - 3.0*pi*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx)*sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx)*sin(pi*x[1]/Ly) + 0.5*(0.2*h0*x[1]/(Lx*sqrt(0.3*x[0]*x[0] + 0.2*x[1]*x[1])) - 1.0*pi*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx)*sin(pi*x[0]/Lx)*sin(pi*(-3.0*x[0] + 1.0*x[1])/Lx) + 0.5*pi*(cos(pi*(3.0*x[0] + 1.0*x[1])/Lx) + 4.0 + h0*sqrt(0.3*x[0]*x[0] + 0.2*x[1]*x[1])/Lx)*sin(pi*x[0]/Lx)*cos(pi*(-3.0*x[0] + 1.0*x[1])/Lx)/Lx - 2.0*pi*(cos(pi*(3.0*x[0] + 1.0*x[1])/Lx) + 4.0 + h0*sqrt(0.3*x[0]*x[0] + 0.2*x[1]*x[1])/Lx)*sin(pi*x[1]/Ly)*cos(pi*(-2.0*x[0] + 1.0*x[1])/Lx)/Lx',
        Lx=Lx, Ly=Ly, h0=depth, f0=f0, g=g)
    out['res_uv_expr'] = Expression(
        (
            '0.5*(pi*sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx)*cos(pi*x[1]/Ly)/Ly + 1.0*pi*sin(pi*x[1]/Ly)*cos(pi*(-2.0*x[0] + 1.0*x[1])/Lx)/Lx)*sin(pi*x[0]/Lx)*sin(pi*(-3.0*x[0] + 1.0*x[1])/Lx) - 3.0*pi*g*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx - 2.0*pi*sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx)*pow(sin(pi*x[1]/Ly), 2)*cos(pi*(-2.0*x[0] + 1.0*x[1])/Lx)/Lx',
            '(-1.5*pi*sin(pi*x[0]/Lx)*cos(pi*(-3.0*x[0] + 1.0*x[1])/Lx)/Lx + 0.5*pi*sin(pi*(-3.0*x[0] + 1.0*x[1])/Lx)*cos(pi*x[0]/Lx)/Lx)*sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx)*sin(pi*x[1]/Ly) - 1.0*pi*g*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx + 0.25*pi*pow(sin(pi*x[0]/Lx), 2)*sin(pi*(-3.0*x[0] + 1.0*x[1])/Lx)*cos(pi*(-3.0*x[0] + 1.0*x[1])/Lx)/Lx',
        ), Lx=Lx, Ly=Ly, h0=depth, f0=f0, g=g)
    out['options'] = {}
    out['bnd_funcs'] = {1: {'elev': None, 'flux_left': None},
                        2: {'flux_right': None},
                        3: {'elev': None, 'flux_lower': None},
                        4: {'un_upper': None},
                        }
    return out


def setup7(Lx, Ly, depth, f0, g, mimetic=True):
    """
    Non-trivial Coriolis, bath, elev, u and v,
    tangential velocity is zero at bnd to test flux BCs
    """
    out = {}
    out['bath_expr'] = Expression(
        '4.0 + h0*sqrt(0.3*x[0]*x[0] + 0.2*x[1]*x[1])/Lx',
        Lx=Lx, Ly=Ly, h0=depth, f0=f0, g=g)
    out['cori_expr'] = Expression(
        'f0*cos(pi*(x[0] + x[1])/Lx)',
        Lx=Lx, Ly=Ly, h0=depth, f0=f0, g=g)
    out['elev_expr'] = Expression(
        'cos(pi*(3.0*x[0] + 1.0*x[1])/Lx)',
        Lx=Lx, Ly=Ly, h0=depth, f0=f0, g=g)
    out['uv_expr'] = Expression(
        (
            'sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx)*sin(pi*x[1]/Ly)',
            '0.5*sin(pi*x[0]/Lx)*sin(pi*(-3.0*x[0] + 1.0*x[1])/Lx)',
        ), Lx=Lx, Ly=Ly, h0=depth, f0=f0, g=g)
    out['res_elev_expr'] = Expression(
        '(0.3*h0*x[0]/(Lx*sqrt(0.3*x[0]*x[0] + 0.2*x[1]*x[1])) - 3.0*pi*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx)*sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx)*sin(pi*x[1]/Ly) + 0.5*(0.2*h0*x[1]/(Lx*sqrt(0.3*x[0]*x[0] + 0.2*x[1]*x[1])) - 1.0*pi*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx)*sin(pi*x[0]/Lx)*sin(pi*(-3.0*x[0] + 1.0*x[1])/Lx) + 0.5*pi*(cos(pi*(3.0*x[0] + 1.0*x[1])/Lx) + 4.0 + h0*sqrt(0.3*x[0]*x[0] + 0.2*x[1]*x[1])/Lx)*sin(pi*x[0]/Lx)*cos(pi*(-3.0*x[0] + 1.0*x[1])/Lx)/Lx - 2.0*pi*(cos(pi*(3.0*x[0] + 1.0*x[1])/Lx) + 4.0 + h0*sqrt(0.3*x[0]*x[0] + 0.2*x[1]*x[1])/Lx)*sin(pi*x[1]/Ly)*cos(pi*(-2.0*x[0] + 1.0*x[1])/Lx)/Lx',
        Lx=Lx, Ly=Ly, h0=depth, f0=f0, g=g)
    out['res_uv_expr'] = Expression(
        (
            '-0.5*f0*sin(pi*x[0]/Lx)*sin(pi*(-3.0*x[0] + 1.0*x[1])/Lx)*cos(pi*(x[0] + x[1])/Lx) + 0.5*(pi*sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx)*cos(pi*x[1]/Ly)/Ly + 1.0*pi*sin(pi*x[1]/Ly)*cos(pi*(-2.0*x[0] + 1.0*x[1])/Lx)/Lx)*sin(pi*x[0]/Lx)*sin(pi*(-3.0*x[0] + 1.0*x[1])/Lx) - 3.0*pi*g*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx - 2.0*pi*sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx)*pow(sin(pi*x[1]/Ly), 2)*cos(pi*(-2.0*x[0] + 1.0*x[1])/Lx)/Lx',
            'f0*sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx)*sin(pi*x[1]/Ly)*cos(pi*(x[0] + x[1])/Lx) + (-1.5*pi*sin(pi*x[0]/Lx)*cos(pi*(-3.0*x[0] + 1.0*x[1])/Lx)/Lx + 0.5*pi*sin(pi*(-3.0*x[0] + 1.0*x[1])/Lx)*cos(pi*x[0]/Lx)/Lx)*sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx)*sin(pi*x[1]/Ly) - 1.0*pi*g*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx + 0.25*pi*pow(sin(pi*x[0]/Lx), 2)*sin(pi*(-3.0*x[0] + 1.0*x[1])/Lx)*cos(pi*(-3.0*x[0] + 1.0*x[1])/Lx)/Lx',
        ), Lx=Lx, Ly=Ly, h0=depth, f0=f0, g=g)
    out['options'] = {'mimetic': mimetic}
    out['bnd_funcs'] = {1: {'elev': None, 'flux_left': None},
                        2: {'flux_right': None},
                        3: {'elev': None, 'flux_lower': None},
                        4: {'un_upper': None},
                        }
    return out


def setup7dg(Lx, Ly, depth, f0, g):
    """
    Non-trivial Coriolis, bath, elev, u and v,
    tangential velocity is zero at bnd to test flux BCs
    """
    return setup7(Lx, Ly, depth, f0, g, mimetic=False)


def setup8(Lx, Ly, depth, f0, g, mimetic=True):
    """
    Non-trivial Coriolis, bath, elev, u and v,
    tangential velocity is non-zero at bnd, must prescribe uv at boundary.
    """
    out = {}
    out['bath_expr'] = Expression(
        '4.0 + h0*sqrt(0.3*x[0]*x[0] + 0.2*x[1]*x[1])/Lx',
        Lx=Lx, Ly=Ly, h0=depth, f0=f0, g=g)
    out['cori_expr'] = Expression(
        'f0*cos(pi*(x[0] + x[1])/Lx)',
        Lx=Lx, Ly=Ly, h0=depth, f0=f0, g=g)
    out['elev_expr'] = Expression(
        'cos(pi*(3.0*x[0] + 1.0*x[1])/Lx)',
        Lx=Lx, Ly=Ly, h0=depth, f0=f0, g=g)
    out['uv_expr'] = Expression(
        (
            'sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx)',
            '0.5*sin(pi*(-3.0*x[0] + 1.0*x[1])/Lx)',
        ), Lx=Lx, Ly=Ly, h0=depth, f0=f0, g=g)
    out['res_elev_expr'] = Expression(
        '(0.3*h0*x[0]/(Lx*sqrt(0.3*x[0]*x[0] + 0.2*x[1]*x[1])) - 3.0*pi*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx)*sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx) + 0.5*(0.2*h0*x[1]/(Lx*sqrt(0.3*x[0]*x[0] + 0.2*x[1]*x[1])) - 1.0*pi*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx)*sin(pi*(-3.0*x[0] + 1.0*x[1])/Lx) + 0.5*pi*(cos(pi*(3.0*x[0] + 1.0*x[1])/Lx) + 4.0 + h0*sqrt(0.3*x[0]*x[0] + 0.2*x[1]*x[1])/Lx)*cos(pi*(-3.0*x[0] + 1.0*x[1])/Lx)/Lx - 2.0*pi*(cos(pi*(3.0*x[0] + 1.0*x[1])/Lx) + 4.0 + h0*sqrt(0.3*x[0]*x[0] + 0.2*x[1]*x[1])/Lx)*cos(pi*(-2.0*x[0] + 1.0*x[1])/Lx)/Lx',
        Lx=Lx, Ly=Ly, h0=depth, f0=f0, g=g)
    out['res_uv_expr'] = Expression(
        (
            '-0.5*f0*sin(pi*(-3.0*x[0] + 1.0*x[1])/Lx)*cos(pi*(x[0] + x[1])/Lx) - 3.0*pi*g*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx + 0.5*pi*sin(pi*(-3.0*x[0] + 1.0*x[1])/Lx)*cos(pi*(-2.0*x[0] + 1.0*x[1])/Lx)/Lx - 2.0*pi*sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx)*cos(pi*(-2.0*x[0] + 1.0*x[1])/Lx)/Lx',
            'f0*sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx)*cos(pi*(x[0] + x[1])/Lx) - 1.0*pi*g*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx + 0.25*pi*sin(pi*(-3.0*x[0] + 1.0*x[1])/Lx)*cos(pi*(-3.0*x[0] + 1.0*x[1])/Lx)/Lx - 1.5*pi*sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx)*cos(pi*(-3.0*x[0] + 1.0*x[1])/Lx)/Lx',
        ), Lx=Lx, Ly=Ly, h0=depth, f0=f0, g=g)
    out['options'] = {'mimetic': mimetic}
    # NOTE uv condition alone does not work
    out['bnd_funcs'] = {1: {'elev': None, 'uv': None},
                        2: {'elev': None, 'uv': None},
                        3: {'elev': None, 'uv': None},
                        4: {'elev': None, 'uv': None},
                        }
    return out


def setup8dg(Lx, Ly, depth, f0, g):
    """
    Non-trivial Coriolis, bath, elev, u and v,
    tangential velocity is non-zero at bnd, must prescribe uv at boundary.
    """
    return setup8(Lx, Ly, depth, f0, g, mimetic=False)


def run(setup, refinement, order, export=True):
    """Run single test and return L2 error"""
    print '--- running {:} refinement {:}'.format(setup.__name__, refinement)
    # domain dimensions
    Lx = 15e3
    Ly = 10e3
    area = Lx*Ly
    f0 = 5e-3  # NOTE large value to make Coriolis terms larger
    g = physical_constants['g_grav']
    depth = 40.0
    T_period = 5000.0        # period of signals
    T = 1000.0  # 500.0  # 3*T_period           # simulation duration
    TExport = T_period/100.0  # export interval
    if not export:
        TExport = 1.0e12  # high enough

    SET = setup(Lx, Ly, depth, f0, g)

    # mesh
    nx = 5*refinement
    ny = 5*refinement
    mesh2d = RectangleMesh(nx, ny, Lx, Ly)
    dt = 4.0/refinement

    # outputs
    outputDir = create_directory('outputs')

    # bathymetry
    P1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(P1_2d, name='Bathymetry')
    bathymetry_2d.project(SET['bath_expr'])
    if bathymetry_2d.dat.data.min() < 0.0:
        print 'bath', bathymetry_2d.dat.data.min(), bathymetry_2d.dat.data.max()
        raise Exception('Negative bathymetry')

    solverObj = solver2d.flowSolver2d(mesh2d, bathymetry_2d)
    solverObj.options.order = order
    solverObj.options.mimetic = True
    solverObj.options.uAdvection = Constant(1.0)
    solverObj.options.outputDir = outputDir
    solverObj.options.T = T
    solverObj.options.dt = dt
    solverObj.options.TExport = TExport
    solverObj.options.timerLabels = []
    # solverObj.options.timeStepperType = 'cranknicolson'
    solverObj.options.update(SET['options'])

    solverObj.createFunctionSpaces()

    # analytical solution in high-order space for computing L2 norms
    H_2d_HO = FunctionSpace(solverObj.mesh2d, 'DG', order+3)
    U_2d_HO = VectorFunctionSpace(solverObj.mesh2d, 'DG', order+4)
    elev_ana_ho = Function(H_2d_HO, name='Analytical elevation')
    elev_ana_ho.project(SET['elev_expr'])
    uv_ana_ho = Function(U_2d_HO, name='Analytical velocity')
    uv_ana_ho.project(SET['uv_expr'])

    # functions for source terms
    source_uv = Function(solverObj.function_spaces.U_2d, name='momentum source')
    source_uv.project(SET['res_uv_expr'])
    source_elev = Function(solverObj.function_spaces.H_2d, name='continuity source')
    source_elev.project(SET['res_elev_expr'])
    coriolis_func = Function(solverObj.function_spaces.H_2d, name='coriolis')
    coriolis_func.project(SET['cori_expr'])
    solverObj.options.uv_source_2d = source_uv
    solverObj.options.elev_source_2d = source_elev
    solverObj.options.coriolis = coriolis_func

    # functions for boundary conditions
    # analytical elevation
    elev_ana = Function(solverObj.function_spaces.H_2d, name='Analytical elevation')
    elev_ana.project(SET['elev_expr'])
    # analytical uv
    uv_ana = Function(solverObj.function_spaces.U_2d, name='Analytical velocity')
    uv_ana.project(SET['uv_expr'])
    # normal velocity (scalar field, will be interpreted as un*normal vector)
    # left/right bnds
    un_ana_x = Function(solverObj.function_spaces.H_2d, name='Analytical normal velocity x')
    un_ana_x.project(uv_ana[0])
    # lower/uppser bnds
    un_ana_y = Function(solverObj.function_spaces.H_2d, name='Analytical normal velocity y')
    un_ana_y.project(uv_ana[1])
    # flux through left/right bnds
    flux_ana_x = Function(solverObj.function_spaces.H_2d, name='Analytical x flux')
    flux_ana_x.project(uv_ana[0]*(bathymetry_2d + elev_ana)*Ly)
    # flux through lower/upper bnds
    flux_ana_y = Function(solverObj.function_spaces.H_2d, name='Analytical x flux')
    flux_ana_y.project(uv_ana[1]*(bathymetry_2d + elev_ana)*Lx)

    # construct bnd conditions from setup
    bnd_funcs = SET['bnd_funcs']
    # correct values to replace None in bnd_funcs
    # NOTE: scalar velocity (flux|un) sign def: positive out of domain
    bnd_field_mapping = {'symm': None,
                         'elev': elev_ana,
                         'uv': uv_ana,
                         'un_left': -un_ana_x,
                         'un_right': un_ana_x,
                         'un_lower': -un_ana_y,
                         'un_upper': un_ana_y,
                         'flux_left': -flux_ana_x,
                         'flux_right': flux_ana_x,
                         'flux_lower': -flux_ana_y,
                         'flux_upper': flux_ana_y,
                         }
    for bnd_id in bnd_funcs:
        d = {}  # values for this bnd e.g. {'elev': elev_ana, 'uv': uv_ana}
        for bnd_field in bnd_funcs[bnd_id]:
            field_name = bnd_field.split('_')[0]
            d[field_name] = bnd_field_mapping[bnd_field]
        # set to the correct bnd_id
        solverObj.bnd_functions['shallow_water'][bnd_id] = d
        # # print a fancy description
        # bnd_str = ''
        # for k in d:
        #     if isinstance(d[k], ufl.algebra.Product):
        #         a, b = d[k].operands()
        #         name = '{:} * {:}'.format(a.value(), b.name())
        #     else:
        #         if d[k] is not None:
        #             name = d[k].name()
        #         else:
        #             name = str(d[k])
        #     bnd_str += '{:}: {:}, '.format(k, name)
        # print('bnd {:}: {:}'.format(bnd_id, bnd_str))

    solverObj.assignInitialConditions(elev=elev_ana, uv_init=uv_ana)
    solverObj.iterate()

    elev_L2_err = errornorm(elev_ana_ho, solverObj.fields.solution2d.split()[1])/numpy.sqrt(area)
    uv_L2_err = errornorm(uv_ana_ho, solverObj.fields.solution2d.split()[0])/numpy.sqrt(area)
    print 'elev L2 error {:.12f}'.format(elev_L2_err)
    print 'uv L2 error {:.12f}'.format(uv_L2_err)
    tmpFunctionCache.clear()  # NOTE must destroy all cached solvers for next simulation
    return elev_L2_err, uv_L2_err


def run_convergence(setup, ref_list, order, export=False, savePlot=False):
    """Runs test for a list of refinements and computes error convergence rate"""
    l2_err = []
    for r in ref_list:
        l2_err.append(run(setup, r, order, export=export))
    x_log = numpy.log10(numpy.array(ref_list, dtype=float)**-1)
    y_log = numpy.log10(numpy.array(l2_err))
    y_log_elev = y_log[:, 0]
    y_log_uv = y_log[:, 1]

    def check_convergence(x_log, y_log, expected_slope, field_str, savePlot):
        slope_rtol = 0.2
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
        setup_name = setup.__name__
        if savePlot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 5))
            # plot points
            ax.plot(x_log, y_log, 'k.')
            x_min = x_log.min()
            x_max = x_log.max()
            offset = 0.05*(x_max - x_min)
            N = 50
            xx = numpy.linspace(x_min - offset, x_max + offset, N)
            yy = intercept + slope*xx
            # plot line
            ax.plot(xx, yy, linestyle='--', linewidth=0.5, color='k')
            ax.text(xx[2*N/3], yy[2*N/3], '{:4.2f}'.format(slope),
                    verticalalignment='top',
                    horizontalalignment='left')
            ax.set_xlabel('log10(dx)')
            ax.set_ylabel('log10(L2 error)')
            ax.set_title(field_str)
            ref_str = 'ref-' + '-'.join([str(r) for r in ref_list])
            order_str = 'o{:}'.format(order)
            imgfile = '_'.join(['convergence', setup_name, field_str, ref_str, order_str])
            imgfile += '.png'
            imgDir = create_directory('plots')
            imgfile = os.path.join(imgDir, imgfile)
            print 'saving figure', imgfile
            plt.savefig(imgfile, dpi=200, bbox_inches='tight')
        if expected_slope is not None:
            err_msg = '{:}: Wrong convergence rate {:.4f}, expected {:.4f}'.format(setup_name, slope, expected_slope)
            assert abs(slope - expected_slope)/expected_slope < slope_rtol, err_msg
            print '{:}: convergence rate {:.4f} PASSED'.format(setup_name, slope)
        else:
            print '{:}: {:} convergence rate {:.4f}'.format(setup_name, field_str, slope)
        return slope

    check_convergence(x_log, y_log_elev, order+1, 'elev', savePlot)
    check_convergence(x_log, y_log_uv, order+1, 'uv', savePlot)

# NOTE nontrivial velocity implies slower convergence
# NOTE try time dependent solution: need to update source terms
# NOTE using Lax-Friedrichs stabilization in mom. advection term improves convergence of velocity

# ---------------------------
# standard tests for pytest
# ---------------------------


@pytest.fixture(params=[setup7, setup8], ids=["Setup7", "Setup8"])
def choose_setup(request):
    return request.param


@pytest.fixture(params=[False, True], ids=["DG", "mimetic"])
def setup_function(request, choose_setup):
    return lambda *args: choose_setup(*args, mimetic=request.param)


def test_steady_state_basin_convergence(setup_function):
    run_convergence(setup_function, [1, 2, 4, 6], 1, savePlot=False)

# ---------------------------
# run individual setup for debugging
# ---------------------------

# run(setup1, 2, 1)

# ---------------------------
# run individual scaling test
# ---------------------------

# run_convergence(setup8dg, [1, 2, 4], 1, savePlot=True)
