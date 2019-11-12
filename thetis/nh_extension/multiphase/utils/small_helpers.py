# Copyright (C) 2017-2019 Tormod Landet
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
import numpy
import dolfin


def create_vector_functions(simulation, vec_name, comp_name, V):
    """
    Create vector functions and add them to the simulation
    Example::

        create_vector_functions(simulation, 'u', 'u%d', Vu)

    This will create simulation.data['u'] and simulation.data['uX']
    """
    vec = []
    for d in range(simulation.ndim):
        name = comp_name % d
        f = dolfin.Function(V)
        f.rename(name, name)
        vec.append(f)
        assert name not in simulation.data
        simulation.data[name] = f
    simulation.data[vec_name] = dolfin.as_vector(vec)
    return simulation.data[vec_name]


def shift_fields(simulation, names):
    """
    Shift variables to next time step. Examples::

        shift_fields(sim, ['u%d', 'up%d', 'upp%d'])
        shift_fields(sim, ['rho', 'rho_p', 'rho_pp'])

    Will cause eg rho_pp = rho_p, rho_p = rho. This function can
    also be used for assigment, since a = b can be written::

        shift_fields(sim, ['b', 'a'])

    The function values are shifted one step right in the given
    list of functions. The first function is not changed and the
    last function's values will be lost
    """
    # Detect whether we are treating a vector field or a scalar
    ndim = None
    if '%d' in names[0]:
        ndim = simulation.ndim

    for i in range(len(names) - 1):
        target = names[-i - 1]
        source = names[-i - 2]
        if ndim is None:
            simulation.data[target].assign(simulation.data[source])
        else:
            for d in range(ndim):
                simulation.data[target % d].assign(simulation.data[source % d])


def velocity_change(u1, u2, ui_tmp):
    """
    Compute the relative difference between two vector fields using
    a temporary vector component function ui_tmp for calculations
    """
    diff = 0
    for d in range(u1.ufl_shape[0]):
        ui_tmp.assign(u1[d])
        ui_tmp.vector().axpy(-1, u2[d].vector())
        ui_tmp.vector().apply('insert')

        nd = ui_tmp.vector().norm('l2')
        n1 = u1[d].vector().norm('l2')
        if n1 != 0:
            diff += nd / n1
    return diff


def get_local(v, V=None, include_ghosts=True):
    """
    Get local values from vector v belonging to function space V

    You can pass a Function as first argument and V = None to
    automativally get v and V from the function
    """
    if V is None:
        # A Function was passed
        V = v.function_space()
        v = v.vector()

    if not include_ghosts:
        return v.get_local()
    else:
        im = V.dofmap().index_map()
        num_dofs_local_and_ghosts = im.size(im.MapSize.ALL)
        indices = numpy.arange(num_dofs_local_and_ghosts, dtype=numpy.intc)
        return v.get_local(indices)


def set_local(v, arr, V=None, apply=None):
    """
    Set local values from arr into vector v belonging to function space V

    You can pass a Function as first argument and V = None to
    automativally get v and V from the function
    """
    if V is None:
        # A Function was passed
        V = v.function_space()
        v = v.vector()

    im = V.dofmap().index_map()
    Nall = im.size(im.MapSize.ALL)
    Nown = im.size(im.MapSize.OWNED)
    assert len(arr.shape) == 1 and arr.shape[0] in (Nall, Nown)

    if arr.shape[0] != Nown:
        arr = arr[:Nown]
    v.set_local(arr)

    if apply is not None:
        v.apply(apply)


@contextmanager
def dolfin_log_level(level):
    old_level = dolfin.get_log_level()
    dolfin.set_log_level(level)
    yield
    dolfin.set_log_level(old_level)
