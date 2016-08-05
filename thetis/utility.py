"""
Utility functions and classes for 3D hydrostatic ocean model

Tuomas Karna 2015-02-21
"""
from __future__ import absolute_import
from .firedrake import *
import os
import numpy as np
import sys
from .physical_constants import physical_constants
from pyop2.profiling import timed_region, timed_function, timed_stage  # NOQA
from mpi4py import MPI  # NOQA
import ufl  # NOQA
import coffee.base as ast  # NOQA
from collections import OrderedDict, namedtuple  # NOQA
from .field_defs import field_metadata
from firedrake import Function as FiredrakeFunction
from firedrake import Constant as FiredrakeConstant

ds_surf = ds_t
ds_bottom = ds_b
# NOTE some functions now depend on FlowSolver object
# TODO move those functions in the solver class


class FrozenClass(object):
    """A class where creating a new attribute will raise an exception if _isfrozen == True"""
    _isfrozen = False

    def __setattr__(self, key, value):
        if self._isfrozen and not hasattr(self, key):
            raise TypeError('Adding new attribute "{:}" to {:} class is forbidden'.format(key, self.__class__.__name__))
        super(FrozenClass, self).__setattr__(key, value)


class SumFunction(object):
    """
    Helper class to keep track of sum of Coefficients.
    """
    def __init__(self):
        """
        Initialize empty sum.

        get operation returns Constant(0)
        """
        self.coeff_list = []

    def add(self, coeff):
        """
        Adds a coefficient to self
        """
        if coeff is None:
            return
        # classes = (Function, Constant, ufl.algebra.Sum, ufl.algebra.Product)
        # assert not isinstance(coeff, classes), \
            # ('bad argument type: ' + str(type(coeff)))
        self.coeff_list.append(coeff)

    def get_sum(self):
        """
        Returns a sum of all added Coefficients
        """
        if len(self.coeff_list) == 0:
            return None
        return sum(self.coeff_list)


class AttrDict(dict):
    """
    Dictionary that provides both self['key'] and self.key access to members.

    http://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute-in-python
    """
    def __init__(self, *args, **kwargs):
        if sys.version_info < (2, 7, 4):
            raise Exception('AttrDict requires python >= 2.7.4 to avoid memory leaks')
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class FieldDict(AttrDict):
    """
    AttrDict that checks that all added fields have proper meta data.

    Values can be either Function or Constant objects.
    """
    def _check_inputs(self, key, value):
        if key != '__dict__':
            from firedrake.functionspaceimpl import MixedFunctionSpace, WithGeometry
            if not isinstance(value, (FiredrakeFunction, FiredrakeConstant)):
                raise TypeError('Value must be a Function or Constant object')
            fs = value.function_space()
            is_mixed = (isinstance(fs, MixedFunctionSpace) or
                        (isinstance(fs, WithGeometry) and
                         isinstance(fs.topological, MixedFunctionSpace)))
            if not is_mixed and key not in field_metadata:
                msg = 'Trying to add a field "{:}" that has no metadata. ' \
                      'Add field_metadata entry to field_defs.py'.format(key)
                raise Exception(msg)

    def _set_functionname(self, key, value):
        """Set function.name to key to ensure consistent naming"""
        if isinstance(value, FiredrakeFunction):
            value.rename(name=key)

    def __setitem__(self, key, value):
        self._check_inputs(key, value)
        self._set_functionname(key, value)
        super(FieldDict, self).__setitem__(key, value)

    def __setattr__(self, key, value):
        self._check_inputs(key, value)
        self._set_functionname(key, value)
        super(FieldDict, self).__setattr__(key, value)


ElementContinuity = namedtuple("ElementContinuity", ["dg", "horizontal_dg", "vertical_dg"])
"""A named tuple describing the continuity of an element."""


def element_continuity(fiat_element):
    """Return an :class:`ElementContinuity` instance with the
    continuity of a given element.

    :arg fiat_element: The fiat element to determine the continuity
        of.
    :returns: A new :class:`ElementContinuity` instance.
    """
    import FIAT
    cell = fiat_element.get_reference_element()

    if isinstance(cell, FIAT.reference_element.TensorProductCell):
        # Pull apart
        horiz = element_continuity(fiat_element.A).dg
        vert = element_continuity(fiat_element.B).dg
        return ElementContinuity(dg=horiz and vert,
                                 horizontal_dg=horiz,
                                 vertical_dg=vert)
    else:
        edofs = fiat_element.entity_dofs()
        dim = cell.get_spatial_dimension()
        dg = True
        for i in range(dim - 1):
            if any(len(k) for k in edofs[i].values()):
                dg = False
                break
        return ElementContinuity(dg, dg, dg)


def create_directory(path, comm=COMM_WORLD):
    if comm.rank == 0:
        if os.path.exists(path):
            if not os.path.isdir(path):
                raise Exception('file with same name exists', path)
        else:
            os.makedirs(path)
    comm.barrier()
    return path


def extrude_mesh_sigma(mesh2d, n_layers, bathymetry_2d):
    """Extrudes 2d surface mesh with bathymetry data defined in a 2d field."""
    mesh = ExtrudedMesh(mesh2d, layers=n_layers, layer_height=1.0/n_layers)

    coordinates = mesh.coordinates
    fs_3d = coordinates.function_space()
    fs_2d = bathymetry_2d.function_space()
    new_coordinates = Function(fs_3d)

    # number of nodes in vertical direction
    n_vert_nodes = len(fs_3d.fiat_element.B.entity_closure_dofs()[1][0])

    nodes = fs_3d.bt_masks['geometric'][0]
    idx = op2.Global(len(nodes), nodes, dtype=np.int32, name='node_idx')
    kernel = op2.Kernel("""
        void my_kernel(double **new_coords, double **old_coords, double **bath2d, int *idx) {
            for ( int d = 0; d < %(nodes)d; d++ ) {
                for ( int e = 0; e < %(v_nodes)d; e++ ) {
                    new_coords[idx[d]+e][0] = old_coords[idx[d]+e][0];
                    new_coords[idx[d]+e][1] = old_coords[idx[d]+e][1];
                    new_coords[idx[d]+e][2] = -bath2d[d][0] * (1.0 - old_coords[idx[d]+e][2]);
                }
            }
        }""" % {'nodes': bathymetry_2d.cell_node_map().arity,
                'v_nodes': n_vert_nodes},
        'my_kernel')

    op2.par_loop(kernel, mesh.cell_set,
                 new_coordinates.dat(op2.WRITE, fs_3d.cell_node_map()),
                 coordinates.dat(op2.READ, fs_3d.cell_node_map()),
                 bathymetry_2d.dat(op2.READ, fs_2d.cell_node_map()),
                 idx(op2.READ),
                 iterate=op2.ALL)

    mesh.coordinates.assign(new_coordinates)

    return mesh


def comp_volume_2d(eta, bath):
    val = assemble((eta+bath)*dx)
    return val


def comp_volume_3d(mesh):
    one = Constant(1.0, domain=mesh.coordinates.ufl_domain())
    val = assemble(one*dx)
    return val


def comp_tracer_mass_3d(scalar_func):
    val = assemble(scalar_func*dx)
    return val


def get_zcoord_from_mesh(zcoord, solver_parameters={}):
    """Evaluates z coordinates from the 3D mesh"""
    solver_parameters.setdefault('ksp_atol', 1e-12)
    solver_parameters.setdefault('ksp_rtol', 1e-16)
    fs = zcoord.function_space()
    tri = TrialFunction(fs)
    test = TestFunction(fs)
    a = tri*test*dx
    l = fs.mesh().coordinates[2]*test*dx
    solve(a == l, zcoord, solver_parameters=solver_parameters)
    return zcoord


class VerticalVelocitySolver(object):
    """Computes vertical velocity from continuity equation"""
    def __init__(self, solution, uv, bathymetry, boundary_funcs={},
                 solver_parameters={}):
        solver_parameters.setdefault('snes_type', 'ksponly')
        solver_parameters.setdefault('ksp_type', 'preonly')
        solver_parameters.setdefault('pc_type', 'bjacobi')
        solver_parameters.setdefault('sub_ksp_type', 'preonly')
        solver_parameters.setdefault('sub_pc_type', 'ilu')
        fs = solution.function_space()
        mesh = fs.mesh()
        test = TestFunction(fs)
        tri = TrialFunction(fs)
        normal = FacetNormal(mesh)

        # define measures with a reasonable quadrature degree
        p, q = fs.ufl_element().degree()
        self.quad_degree = (2*p, 2*q)
        self.dx = dx(degree=self.quad_degree)
        self.dS_h = dS_h(degree=self.quad_degree)
        self.dS_v = dS_v(degree=self.quad_degree)
        self.ds_surf = ds_surf(degree=self.quad_degree)

        # NOTE weak dw/dz
        a = tri[2]*test[2]*normal[2]*ds_surf + \
            avg(tri[2])*jump(test[2], normal[2])*dS_h - Dx(test[2], 2)*tri[2]*self.dx

        # NOTE weak div(uv)
        uv_star = avg(uv)
        # NOTE in the case of mimetic uv the div must be taken over all components
        l = (inner(uv, nabla_grad(test[2]))*self.dx -
             (uv_star[0]*jump(test[2], normal[0]) +
              uv_star[1]*jump(test[2], normal[1]) +
              uv_star[2]*jump(test[2], normal[2])
              )*(self.dS_v) -
             (uv_star[0]*jump(test[2], normal[0]) +
              uv_star[1]*jump(test[2], normal[1]) +
              uv_star[2]*jump(test[2], normal[2])
              )*(self.dS_h) -
             (uv[0]*normal[0] +
              uv[1]*normal[1] +
              uv[2]*normal[2]
              )*test[2]*self.ds_surf
             )
        for bnd_marker in mesh.exterior_facets.unique_markers:
            funcs = boundary_funcs.get(bnd_marker)
            ds_bnd = ds_v(int(bnd_marker), degree=self.quad_degree)
            if funcs is None:
                # assume land boundary
                continue
            else:
                # use symmetry condition
                l += -(uv[0]*normal[0] + uv[1]*normal[1])*test[2]*ds_bnd

        # NOTE For ALE mesh constant_jacobian should be False
        # however the difference is very small as A is nearly independent of
        # mesh stretching: only the normals vary in time
        self.prob = LinearVariationalProblem(a, l, solution,
                                             constant_jacobian=True)
        self.solver = LinearVariationalSolver(self.prob,
                                              solver_parameters=solver_parameters)

    def solve(self):
        self.solver.solve()


class VerticalIntegrator(object):
    """
    Computes vertical integral (or average) of a field.

    :param input: 3D field to integrate
    :param output: 3D field where the integral is stored
    :param bottom_to_top: Defines the integration direction: If True integration is performed along the z axis, from bottom surface to top surface.
    :param bnd_value: Value of the integral at the bottom (top) boundary if bottom_to_top is True (False)
    :param average: If True computes the vertical average instead. Requires bathymetry and elevation fields
    :param bathymetry: 3D field defining the bathymetry
    :param elevation: 3D field defining the free surface elevation
    """
    def __init__(self, input, output, bottom_to_top=True,
                 bnd_value=Constant(0.0), average=False,
                 bathymetry=None, elevation=None, solver_parameters={}):
        solver_parameters.setdefault('snes_type', 'ksponly')
        solver_parameters.setdefault('ksp_type', 'preonly')
        solver_parameters.setdefault('pc_type', 'bjacobi')
        solver_parameters.setdefault('sub_ksp_type', 'preonly')
        solver_parameters.setdefault('sub_pc_type', 'ilu')

        self.output = output
        space = output.function_space()
        mesh = space.mesh()
        vertical_is_dg = element_continuity(space.fiat_element).vertical_dg
        tri = TrialFunction(space)
        phi = TestFunction(space)
        normal = FacetNormal(mesh)

        # define measures with a reasonable quadrature degree
        p, q = space.ufl_element().degree()
        self.quad_degree = (2*p, 2*q + 1)
        self.dx = dx(degree=self.quad_degree)
        self.dS_h = dS_h(degree=self.quad_degree)
        self.ds_surf = ds_surf(degree=self.quad_degree)
        self.ds_bottom = ds_bottom(degree=self.quad_degree)

        if bottom_to_top:
            bnd_term = normal[2]*inner(bnd_value, phi)*self.ds_bottom
            mass_bnd_term = normal[2]*inner(tri, phi)*self.ds_surf
        else:
            bnd_term = normal[2]*inner(bnd_value, phi)*self.ds_surf
            mass_bnd_term = normal[2]*inner(tri, phi)*self.ds_bottom

        self.a = -inner(Dx(phi, 2), tri)*self.dx + mass_bnd_term
        gamma = (normal[2] + abs(normal[2]))
        if bottom_to_top:
            up_value = avg(tri*gamma)
        else:
            up_value = avg(tri*(1 - gamma))
        if vertical_is_dg:
            if len(input.ufl_shape) > 0:
                dim = input.ufl_shape[0]
                for i in range(dim):
                    self.a += up_value[i]*jump(phi[i], normal[2])*self.dS_h
            else:
                self.a += up_value*jump(phi, normal[2])*self.dS_h
        source = input
        if average:
            source = input/(elevation + bathymetry)
        else:
            source = input
        self.l = inner(source, phi)*self.dx + bnd_term
        self.prob = LinearVariationalProblem(self.a, self.l, output, constant_jacobian=False)
        self.solver = LinearVariationalSolver(self.prob, solver_parameters=solver_parameters)

    def solve(self):
        """
        Computes the integral and stores it in the :arg:`output` field.
        """
        self.solver.solve()


class DensitySolver(object):
    """
    Computes density from salinity and temperature using the equation of state.

    Density is computed point-wise assuming that T,S and rho are in the same
    function space.
    """
    def __init__(self, salinity, temperature, density, eos_class):
        self.fs = density.function_space()
        self.eos = eos_class

        if isinstance(salinity, FiredrakeFunction):
            assert self.fs == salinity.function_space()
        if isinstance(temperature, FiredrakeFunction):
            assert self.fs == temperature.function_space()

        self.s = salinity
        self.t = temperature
        self.rho = density

    def _get_array(self, function):
        if isinstance(function, FiredrakeFunction):
            assert self.fs == function.function_space()
            return function.dat.data[:]
        if isinstance(function, FiredrakeConstant):
            return function.dat.data[0]
        # assume that function is a float
        return function

    def solve(self):
        s = self._get_array(self.s)
        th = self._get_array(self.t)
        p = 0.0  # NOTE ignore pressure for now
        rho0 = self._get_array(physical_constants['rho0'])
        self.rho.dat.data[:] = self.eos.compute_rho(s, th, p, rho0)


def compute_baroclinic_head(solver):
    """
    Computes baroclinic head from density field

    r = 1/rho_0 int_{z=-h}^{\eta} rho' dz
    """
    solver.density_solver.solve()
    solver.rho_integrator.solve()
    solver.fields.baroc_head_3d *= -physical_constants['rho0_inv']
    solver.baro_head_averager.solve()
    solver.extract_surf_baro_head.solve()


class VelocityMagnitudeSolver(object):
    """
    Computes magnitude of (u[0],u[1],w) and stores it in solution
    """
    def __init__(self, solution, u=None, w=None, min_val=1e-6,
                 solver_parameters={}):
        self.solution = solution
        self.min_val = min_val
        function_space = solution.function_space()
        test = TestFunction(function_space)
        tri = TrialFunction(function_space)

        a = test*tri*dx
        s = 0
        if u is not None:
            s += u[0]**2 + u[1]**2
        if w is not None:
            s += w**2
        l = test*sqrt(s)*dx
        self.prob = LinearVariationalProblem(a, l, solution)
        self.solver = LinearVariationalSolver(self.prob, solver_parameters=solver_parameters)

    def solve(self):
        self.solver.solve()
        np.maximum(self.solution.dat.data, self.min_val, self.solution.dat.data)


class HorizontalJumpDiffusivity(object):
    """Computes tracer jump diffusivity for horizontal advection."""
    def __init__(self, alpha, tracer, output, h_elem_size, umag,
                 tracer_mag, max_val, min_val=1e-6, solver_parameters={}):
        solver_parameters.setdefault('ksp_atol', 1e-6)
        solver_parameters.setdefault('ksp_rtol', 1e-8)
        if output.function_space() != max_val.function_space():
            raise Exception('output and max_val function spaces do not match')
        self.output = output
        self.min_val = min_val
        self.max_val = max_val

        fs = output.function_space()
        test = TestFunction(fs)
        tri = TrialFunction(fs)
        a = inner(test, tri)*dx + jump(test, tri)*dS_v
        tracer_jump = jump(tracer)
        # TODO jump scalar must depend on the tracer value scale
        # TODO can this be estimated automatically e.g. global_max(abs(S))
        maxjump = Constant(0.05)*tracer_mag
        l = alpha*avg(umag*h_elem_size)*(tracer_jump/maxjump)**2*avg(test)*dS_v
        self.prob = LinearVariationalProblem(a, l, output)
        self.solver = LinearVariationalSolver(self.prob, solver_parameters=solver_parameters)

    def solve(self):
        self.solver.solve()
        np.minimum(self.max_val.dat.data, self.output.dat.data, self.output.dat.data)
        self.output.dat.data[self.output.dat.data[:] < self.min_val] = self.min_val


class ExpandFunctionTo3d(object):
    """
    Copies a field from 2d mesh to 3d mesh, assigning the same value over the
    vertical dimension. Horizontal function space must be the same.
    """
    def __init__(self, input_2d, output_3d, elem_height=None):
        self.input_2d = input_2d
        self.output_3d = output_3d
        self.fs_2d = self.input_2d.function_space()
        self.fs_3d = self.output_3d.function_space()

        family_2d = self.fs_2d.ufl_element().family()
        ufl_elem = self.fs_3d.ufl_element()
        if isinstance(ufl_elem, ufl.VectorElement):
            # Unwind vector
            ufl_elem = ufl_elem.sub_elements()[0]
        if isinstance(ufl_elem, ufl.HDivElement):
            # RT case
            ufl_elem = ufl_elem._element
        if ufl_elem.family() == 'TensorProductElement':
            # a normal tensorproduct element
            family_3dh = ufl_elem.sub_elements()[0].family()
            if family_2d != family_3dh:
                raise Exception('2D and 3D spaces do not match: {0:s} {1:s}'.format(family_2d, family_3dh))
        if family_2d == 'Raviart-Thomas' and elem_height is None:
            raise Exception('elem_height must be provided for Raviart-Thomas spaces')
        self.do_rt_scaling = family_2d == 'Raviart-Thomas'

        self.iter_domain = op2.ALL

        # number of nodes in vertical direction
        n_vert_nodes = len(self.fs_3d.fiat_element.B.entity_closure_dofs()[1][0])

        nodes = self.fs_3d.bt_masks['geometric'][0]
        self.idx = op2.Global(len(nodes), nodes, dtype=np.int32, name='node_idx')
        self.kernel = op2.Kernel("""
            void my_kernel(double **func, double **func2d, int *idx) {
                for ( int d = 0; d < %(nodes)d; d++ ) {
                    for ( int c = 0; c < %(func_dim)d; c++ ) {
                        for ( int e = 0; e < %(v_nodes)d; e++ ) {
                            func[idx[d]+e][c] = func2d[d][c];
                        }
                    }
                }
            }""" % {'nodes': self.input_2d.cell_node_map().arity,
                    'func_dim': self.input_2d.function_space().dim,
                    'v_nodes': n_vert_nodes},
            'my_kernel')

        if self.do_rt_scaling:
            solver_parameters = {}
            solver_parameters.setdefault('ksp_atol', 1e-12)
            solver_parameters.setdefault('ksp_rtol', 1e-16)
            test = TestFunction(self.fs_3d)
            tri = TrialFunction(self.fs_3d)
            a = inner(tri, test)*dx
            l = inner(self.output_3d, test)*elem_height*dx
            prob = LinearVariationalProblem(a, l, self.output_3d)
            self.rt_scale_solver = LinearVariationalSolver(
                prob, solver_parameters=solver_parameters)

    def solve(self):
        with timed_stage('copy_2d_to_3d'):
            # execute par loop
            op2.par_loop(
                self.kernel, self.fs_3d.mesh().cell_set,
                self.output_3d.dat(op2.WRITE, self.fs_3d.cell_node_map()),
                self.input_2d.dat(op2.READ, self.fs_2d.cell_node_map()),
                self.idx(op2.READ),
                iterate=self.iter_domain)

            if self.do_rt_scaling:
                self.rt_scale_solver.solve()


class SubFunctionExtractor(object):
    """
    Extract a 2D subfunction from a 3D extruded mesh.

    input_3d: Function in 3d mesh
    output_2d: Function in 2d mesh
    boundary: 'top'|'bottom'
        Defines whether to extract from the surface or bottom 3D elements
    elem_facet: 'top'|'bottom'|'average'
        Defines which facet of the 3D element is extracted. The 'average'
        computes mean of the top and bottom facets of the 3D element.
    elem_height: Function in 2d mesh
        Function that defines the element heights
        (required for Raviart-Thomas spaces only)
    """
    def __init__(self, input_3d, output_2d,
                 boundary='top', elem_facet=None,
                 elem_height=None):
        self.input_3d = input_3d
        self.output_2d = output_2d
        self.fs_3d = self.input_3d.function_space()
        self.fs_2d = self.output_2d.function_space()

        if elem_facet is None:
            # extract surface/bottom face by default
            elem_facet = boundary

        family_2d = self.fs_2d.ufl_element().family()
        elem = self.fs_3d.ufl_element()
        if isinstance(elem, ufl.VectorElement):
            elem = elem.sub_elements()[0]
        if isinstance(elem, ufl.HDivElement):
            elem = elem._element
        if isinstance(elem, ufl.TensorProductElement):
            # a normal tensorproduct element
            family_3dh = elem.sub_elements()[0].family()
            if family_2d != family_3dh:
                raise Exception('2D and 3D spaces do not match: {0:s} {1:s}'.format(family_2d, family_3dh))
        if family_2d == 'Raviart-Thomas' and elem_height is None:
            raise Exception('elem_height must be provided for Raviart-Thomas spaces')
        self.do_rt_scaling = family_2d == 'Raviart-Thomas'

        if elem_facet == 'bottom':
            nodes = self.fs_3d.bt_masks['geometric'][0]
        elif elem_facet == 'top':
            nodes = self.fs_3d.bt_masks['geometric'][1]
        elif elem_facet == 'average':
            nodes = (self.fs_3d.bt_masks['geometric'][0] +
                     self.fs_3d.bt_masks['geometric'][1])
        else:
            raise Exception('Unsupported elem_facet: {:}'.format(elem_facet))
        if boundary == 'top':
            self.iter_domain = op2.ON_TOP
        elif boundary == 'bottom':
            self.iter_domain = op2.ON_BOTTOM

        out_nodes = self.fs_2d.fiat_element.space_dimension()

        if elem_facet == 'average':
            assert (len(nodes) == 2*out_nodes)
        else:
            assert (len(nodes) == out_nodes)

        self.idx = op2.Global(len(nodes), nodes, dtype=np.int32, name='node_idx')
        if elem_facet == 'average':
            # compute average of top and bottom elem nodes
            self.kernel = op2.Kernel("""
                void my_kernel(double **func, double **func3d, int *idx) {
                    int nnodes = %(nodes)d;
                    for ( int d = 0; d < nnodes; d++ ) {
                        for ( int c = 0; c < %(func_dim)d; c++ ) {
                            func[d][c] = 0.5*(func3d[idx[d]][c] +
                                              func3d[idx[d + nnodes]][c]);
                        }
                    }
                }""" % {'nodes': self.output_2d.cell_node_map().arity,
                        'func_dim': self.output_2d.function_space().dim},
                'my_kernel')
        else:
            self.kernel = op2.Kernel("""
                void my_kernel(double **func, double **func3d, int *idx) {
                    for ( int d = 0; d < %(nodes)d; d++ ) {
                        for ( int c = 0; c < %(func_dim)d; c++ ) {
                            func[d][c] = func3d[idx[d]][c];
                        }
                    }
                }""" % {'nodes': self.output_2d.cell_node_map().arity,
                        'func_dim': self.output_2d.function_space().dim},
                'my_kernel')

        if self.do_rt_scaling:
            solver_parameters = {}
            solver_parameters.setdefault('ksp_atol', 1e-12)
            solver_parameters.setdefault('ksp_rtol', 1e-16)
            test = TestFunction(self.fs_2d)
            tri = TrialFunction(self.fs_2d)
            a = inner(tri, test)*dx
            l = inner(self.output_2d, test)/elem_height*dx
            prob = LinearVariationalProblem(a, l, self.output_2d)
            self.rt_scale_solver = LinearVariationalSolver(
                prob, solver_parameters=solver_parameters)

    def solve(self):
        with timed_stage('copy_3d_to_2d'):
            # execute par loop
            op2.par_loop(self.kernel, self.fs_3d.mesh().cell_set,
                         self.output_2d.dat(op2.WRITE, self.fs_2d.cell_node_map()),
                         self.input_3d.dat(op2.READ, self.fs_3d.cell_node_map()),
                         self.idx(op2.READ),
                         iterate=self.iter_domain)

            if self.do_rt_scaling:
                self.rt_scale_solver.solve()


def compute_elem_height(zcoord, output):
    """
    Compute element heights on an extruded mesh.
    zcoord (P1CG) contains zcoordinates of the mesh
    element height is stored in output function (typ. P1DG).
    """
    fs_in = zcoord.function_space()
    fs_out = output.function_space()

    iterate = op2.ALL

    # NOTE height maybe <0 if mesh was extruded like that
    kernel = op2.Kernel("""
        void my_kernel(double **func, double **zcoord) {
            for ( int d = 0; d < %(nodes)d/2; d++ ) {
                for ( int c = 0; c < %(func_dim)d; c++ ) {
                    double dz = fabs(zcoord[2*d+1][c] - zcoord[2*d][c]);
                    func[2*d][c] = dz;
                    func[2*d+1][c] = dz;
                }
            }
        }""" % {'nodes': zcoord.cell_node_map().arity,
                'func_dim': zcoord.function_space().dim},
        'my_kernel')
    op2.par_loop(
        kernel, fs_out.mesh().cell_set,
        output.dat(op2.WRITE, fs_out.cell_node_map()),
        zcoord.dat(op2.READ, fs_in.cell_node_map()),
        iterate=iterate)

    return output


def compute_bottom_drag(z_bottom, drag):
    """Computes bottom drag coefficient (Cd) from boundary log layer."""
    von_karman = physical_constants['von_karman']
    z0_friction = physical_constants['z0_friction']
    drag.assign((von_karman / ln((z_bottom + z0_friction)/z0_friction))**2)
    return drag


def compute_bottom_friction(solver, uv_3d, uv_bottom_2d,
                            z_bottom_2d, bathymetry_2d,
                            bottom_drag_2d):
    # compute velocity at middle of bottom element
    solver.extract_uv_bottom.solve()
    solver.extract_z_bottom.solve()
    z_bottom_2d.assign((z_bottom_2d + bathymetry_2d))
    compute_bottom_drag(z_bottom_2d, bottom_drag_2d)
    if solver.options.use_parabolic_viscosity:
        solver.copy_uv_bottom_to_3d.solve()
        solver.copy_bottom_drag_to_3d.solve()


def get_horizontal_elem_size_2d(sol2d):
    """
    Computes horizontal element size from the 2D mesh, stores the output in
    the given field.
    """
    p1_2d = sol2d.function_space()
    mesh = p1_2d.mesh()
    test = TestFunction(p1_2d)
    tri = TrialFunction(p1_2d)
    a = inner(test, tri) * dx
    l = inner(test, sqrt(CellVolume(mesh))) * dx
    solve(a == l, sol2d)
    return sol2d


def get_horizontal_elem_size_3d(sol2d, sol3d):
    """
    Computes horizontal element size from the 2D mesh, then copies it on a 3D
    field.
    """
    get_horizontal_elem_size_2d(sol2d)
    ExpandFunctionTo3d(sol2d, sol3d).solve()


class ALEMeshUpdater(object):
    """
    Computes mesh velocity from changes in (continuous) 2D elevation field
    """
    def __init__(self, solver):
        self.solver = solver
        self.fields = solver.fields
        if self.solver.options.use_ale_moving_mesh:
            # continous elevation
            self.elev_cg_2d = Function(self.solver.function_spaces.P1_2d,
                                       name='elev cg 2d')
            # elevation in coordinate space
            self.proj_elev_to_cg_2d = Projector(self.fields.elev_2d,
                                                self.elev_cg_2d)
            self.proj_elev_cg_to_coords_2d = Projector(self.elev_cg_2d,
                                                       self.fields.elev_cg_2d)
            self.cp_elev_2d_to_3d = ExpandFunctionTo3d(self.fields.elev_cg_2d,
                                                       self.fields.elev_cg_3d)
            self.cp_w_mesh_surf_2d_to_3d = ExpandFunctionTo3d(self.fields.w_mesh_surf_2d,
                                                              self.fields.w_mesh_surf_3d)
            self.cp_v_elem_size_to_2d = SubFunctionExtractor(self.fields.v_elem_size_3d,
                                                             self.fields.v_elem_size_2d,
                                                             boundary='top', elem_facet='top')

    def initialize(self):
        """Set values for initial mesh (elevation at rest)"""
        get_zcoord_from_mesh(self.fields.z_coord_ref_3d)
        self.fields.z_coord_3d.assign(self.fields.z_coord_ref_3d)
        self.update_elem_height()

    def update_elem_height(self):
        """Updates vertical element size fields"""
        compute_elem_height(self.fields.z_coord_3d, self.fields.v_elem_size_3d)
        self.cp_v_elem_size_to_2d.solve()

    def compute_mesh_velocity_begin(self):
        """Stores the current 2D elevation state as the "old" field"""
        assert self.solver.options.use_ale_moving_mesh
        self.proj_elev_to_cg_2d.project()
        self.proj_elev_cg_to_coords_2d.project()

    def compute_mesh_velocity_finalize(self, c=1.0):
        """Stores the current 2D elevation state as the "new" fields,
        and compute w_mesh using the given time step factor."""
        # compute w_mesh_surf = (elev_new - elev_old)/dt/c
        assert self.solver.options.use_ale_moving_mesh
        self.fields.w_mesh_surf_2d.assign(self.fields.elev_cg_2d)
        self.proj_elev_to_cg_2d.project()
        self.proj_elev_cg_to_coords_2d.project()
        self.fields.w_mesh_surf_2d += -self.fields.elev_cg_2d
        self.fields.w_mesh_surf_2d *= -1.0/self.solver.dt/c
        # use that to compute w_mesh in whole domain
        self.cp_w_mesh_surf_2d_to_3d.solve()
        # solve w_mesh at nodes
        w_mesh_surf = self.fields.w_mesh_surf_3d.dat.data[:]
        z_ref = self.fields.z_coord_ref_3d.dat.data[:]
        h = self.fields.bathymetry_3d.dat.data[:]
        self.fields.w_mesh_3d.dat.data[:] = w_mesh_surf * (z_ref + h)/h

    def update_mesh_coordinates(self):
        """
        Updates 3D mesh coordinates to match current elev_2d field

        elev_2d is first projected to continous space"""
        assert self.solver.options.use_ale_moving_mesh
        self.proj_elev_to_cg_2d.project()
        self.proj_elev_cg_to_coords_2d.project()
        self.cp_elev_2d_to_3d.solve()

        eta = self.fields.elev_cg_3d.dat.data[:]
        z_ref = self.fields.z_coord_ref_3d.dat.data[:]
        bath = self.fields.bathymetry_3d.dat.data[:]
        new_z = eta*(z_ref + bath)/bath + z_ref
        self.solver.mesh.coordinates.dat.data[:, 2] = new_z
        self.fields.z_coord_3d.dat.data[:] = new_z
        self.update_elem_height()


class ParabolicViscosity(object):
    """Computes parabolic eddy viscosity profile assuming log layer flow
    nu = kappa * u_bf * (-z) * (bath + z0 + z) / (bath + z0)
    with
    u_bf = sqrt(Cd)*|uv_bottom|
    """
    def __init__(self, uv_bottom, bottom_drag, bathymetry, nu,
                 solver_parameters={}):
        solver_parameters.setdefault('ksp_atol', 1e-12)
        solver_parameters.setdefault('ksp_rtol', 1e-16)
        self.min_val = 1e-10
        self.solution = nu

        kappa = physical_constants['von_karman']
        z0 = physical_constants['z0_friction']
        fs = nu.function_space()
        x = fs.mesh().coordinates
        test = TestFunction(fs)
        tri = TrialFunction(fs)
        a = tri*test*dx
        uv_mag = sqrt(uv_bottom[0]**2 + uv_bottom[1]**2)
        parabola = -x[2]*(bathymetry + z0 + x[2])/(bathymetry + z0)
        l = kappa*sqrt(bottom_drag)*uv_mag*parabola*test*dx
        self.prob = LinearVariationalProblem(a, l, nu)
        self.solver = LinearVariationalSolver(self.prob, solver_parameters=solver_parameters)

    def solve(self):
        self.solver.solve()
        # remove negative values
        ix = self.solution.dat.data[:] < self.min_val
        self.solution.dat.data[ix] = self.min_val


def beta_plane_coriolis_params(latitude):
    """Computes beta plane parameters based on the latitude (given in degrees)."""
    omega = 7.2921150e-5  # rad/s Earth rotation rate
    r = 6371.e3  # Earth radius
    # Coriolis parameter f = 2 Omega sin(alpha)
    # Beta plane approximation f_beta = f_0 + Beta y
    # f_0 = 2 Omega sin(alpha_0)
    # Beta = df/dy|_{alpha=alpha_0}
    #      = (df/dalpha*dalpha/dy)_{alpha=alpha_0}
    #      = 2 Omega cos(alpha_0) /R
    alpha_0 = 2*np.pi*latitude/360.0
    f_0 = 2*omega*np.sin(alpha_0)
    beta = 2*omega*np.cos(alpha_0)/r
    return f_0, beta


def beta_plane_coriolis_function(degrees, out_function, y_offset=0.0):
    """Interpolates beta plane Coriolis parameter to the given functions."""
    # NOTE assumes that mesh y coordinate spans [-L_y, L_y]
    f0, beta = beta_plane_coriolis_params(45.0)
    out_function.interpolate(
        Expression('f0+beta*(x[1]-y_0)', f0=f0, beta=beta, y_0=y_offset))


class SmagorinskyViscosity(object):
    """
    Computes Smagorinsky subgrid scale viscosity.

    This formulation is according to [1] and [2].

    nu = (C_s L_x)**2 |S|
    |S| = sqrt(D_T**2 + D_S**2)
    D_T = du/dx - dv/dy
    D_S = du/dy + dv/dx
    L_x is the horizontal element size
    C_s is the Smagorinsky coefficient

    To match a certain mesh Reynolds number Re_h set
    C_s = 1/sqrt(Re_h)

    [1] Ilicak et al. (2012). Spurious dianeutral mixing and the role of
        momentum closure. Ocean Modelling, 45-46(0):37-58.
        http://dx.doi.org/10.1016/j.ocemod.2011.10.003
    [2] Griffies and Hallberg (2000). Biharmonic friction with a
        Smagorinsky-like viscosity for use in large-scale eddy-permitting
        ocean models. Monthly Weather Review, 128(8):2935-2946.
        http://dx.doi.org/10.1175/1520-0493(2000)128%3C2935:BFWASL%3E2.0.CO;2
    """
    def __init__(self, uv, output, c_s, h_elem_size, max_val, min_val=1e-10,
                 weak_form=True, solver_parameters={}):
        solver_parameters.setdefault('ksp_atol', 1e-12)
        solver_parameters.setdefault('ksp_rtol', 1e-16)
        assert max_val.function_space() == output.function_space(), \
            'max_val function must belong to the same space as output'
        self.max_val = max_val
        self.min_val = min_val
        self.output = output
        self.weak_form = weak_form

        if self.weak_form:
            # solve grad(u) weakly
            mesh = output.function_space().mesh()
            fs_grad = FunctionSpace(mesh, 'DP', 1, vfamily='DP', vdegree=1)
            self.grad = {}
            for icomp in [0, 1]:
                for j in [0, 1]:
                    self.grad[(icomp, j)] = Function(fs_grad, name='uv_grad({:},{:})'.format(icomp, j))

            tri_grad = TrialFunction(fs_grad)
            test_grad = TestFunction(fs_grad)

            normal = FacetNormal(mesh)
            a = inner(tri_grad, test_grad)*dx

            self.solver_grad = {}
            for icomp in [0, 1]:
                for j in [0, 1]:
                    a = inner(tri_grad, test_grad)*dx
                    # l = inner(Dx(uv[0], 0), test_grad)*dx
                    l = -inner(Dx(test_grad, j), uv[icomp])*dx
                    l += inner(avg(uv[icomp]), jump(test_grad, normal[j]))*dS_v
                    l += inner(uv[icomp], test_grad*normal[j])*ds_v
                    prob = LinearVariationalProblem(a, l, self.grad[(icomp, j)])
                    self.solver_grad[(icomp, j)] = LinearVariationalSolver(prob, solver_parameters=solver_parameters)

        fs = output.function_space()
        tri = TrialFunction(fs)
        test = TestFunction(fs)

        # rate of strain tensor
        if self.weak_form:
            d_t = self.grad[(0, 0)] - self.grad[(1, 1)]
            d_s = self.grad[(0, 1)] + self.grad[(1, 0)]
        else:
            d_t = Dx(uv[0], 0) - Dx(uv[1], 1)
            d_s = Dx(uv[0], 1) + Dx(uv[1], 0)
        nu = c_s**2*h_elem_size**2 * sqrt(d_t**2 + d_s**2)

        a = test*tri*dx
        l = test*nu*dx
        self.prob = LinearVariationalProblem(a, l, output)
        self.solver = LinearVariationalSolver(self.prob, solver_parameters=solver_parameters)

    def solve(self):
        if self.weak_form:
            for icomp in [0, 1]:
                for j in [0, 1]:
                    self.solver_grad[(icomp, j)].solve()
        self.solver.solve()
        # remove negative values
        ix = self.output.dat.data < self.min_val
        self.output.dat.data[ix] = self.min_val

        # crop too large values
        ix = self.output.dat.data > self.max_val.dat.data
        self.output.dat.data[ix] = self.max_val.dat.data[ix]


class EquationOfState(object):
    """
    Equation of State according to [1] for computing sea water density.

    [1] Jackett, D. R., McDougall, T. J., Feistel, R., Wright, D. G., and
        Griffies, S. M. (2006). Algorithms for Density, Potential Temperature,
        Conservative Temperature, and the Freezing Temperature of Seawater.
        Journal of Atmospheric and Oceanic Technology, 23(12):1709-1728.
    """
    def __init__(self):
        # polynomial coefficients
        self.a = np.array([9.9984085444849347e2, 7.3471625860981584e0, -5.3211231792841769e-2,
                           3.6492439109814549e-4, 2.5880571023991390e0, -6.7168282786692355e-3,
                           1.9203202055760151e-3, 1.1798263740430364e-2, 9.8920219266399117e-8,
                           4.6996642771754730e-6, -2.5862187075154352e-8, -3.2921414007960662e-12])
        self.b = np.array([1.0, 7.2815210113327091e-3, -4.4787265461983921e-5, 3.3851002965802430e-7,
                           1.3651202389758572e-10, 1.7632126669040377e-3, -8.8066583251206474e-6,
                           -1.8832689434804897e-10, 5.7463776745432097e-6, 1.4716275472242334e-9,
                           6.7103246285651894e-6, -2.4461698007024582e-17, -9.1534417604289062e-18])

    def compute_rho(self, s, th, p, rho0=0.0):
        """
        Computes sea water density.

        :param S: Salinity expressed on the Practical Salinity Scale 1978
        :param Th: Potential temperature in Celsius
        :param p: Pressure in decibars (1 dbar = 1e4 Pa)
        :param rho0: Optional reference density

        Th is referenced to pressure p_r = 0 dbar. All pressures are gauge
        pressures: they are the absolute pressures minus standard atmosperic
        pressure 10.1325 dbar.
        Last optional argument rho0 is for computing deviation
        rho' = rho(S, Th, p) - rho0.
        """
        s_pos = np.maximum(s, 0.0)  # ensure salinity is positive
        a = self.a
        b = self.b
        pn = (a[0] + th*a[1] + th*th*a[2] + th*th*th*a[3] + s_pos*a[4] +
              th*s_pos*a[5] + s_pos*s_pos*a[6] + p*a[7] + p*th * th*a[8] + p*s_pos*a[9] +
              p*p*a[10] + p*p*th*th * a[11])
        pd = (b[0] + th*b[1] + th*th*b[2] + th*th*th*b[3] +
              th*th*th*th*b[4] + s_pos*b[5] + s_pos*th*b[6] + s_pos*th*th*th*b[7] +
              pow(s_pos, 1.5)*b[8] + pow(s_pos, 1.5)*th*th*b[9] + p*b[10] +
              p*p*th*th*th*b[11] + p*p*p*th*b[12])
        rho = pn/pd - rho0
        return rho


class LinearEquationOfState(object):
    """
    Linear Equation of State.

    rho = rho_ref - alpha*(T - T_ref) + beta*(S - S_ref)
    """
    def __init__(self, rho_ref, alpha, beta, th_ref, s_ref):
        self.rho_ref = rho_ref
        self.alpha = alpha
        self.beta = beta
        self.th_ref = th_ref
        self.S_ref = s_ref

    def compute_rho(self, s, th, p, rho0=0.0):
        """
        Computes sea water density.

        :param S: Salinity expressed on the Practical Salinity Scale 1978
        :param Th: Potential temperature in Celsius
        :param p: Pressure in decibars (1 dbar = 1e4 Pa)
        :param rho0: Optional reference density

        Pressure is ingored in this equation of state.
        """
        rho = (self.rho_ref - rho0 -
               self.alpha*(th - self.th_ref) +
               self.beta*(s - self.S_ref))
        return rho


def tensor_jump(v, n):
    """Jump term for vector functions based on the tensor product.
    This is the discrete equivalent of grad(u) as opposed to the normal vectorial
    jump which represents div(u)."""
    return outer(v('+'), n('+')) + outer(v('-'), n('-'))


def compute_boundary_length(mesh2d):
    """
    Computes the length of the boundary segments in 2d mesh.
    """
    p1 = FunctionSpace(mesh2d, 'CG', 1)
    boundary_markers = mesh2d.exterior_facets.unique_markers
    boundary_len = {}
    for i in boundary_markers:
        ds_restricted = ds(int(i))
        one_func = Function(p1).assign(1.0)
        boundary_len[i] = assemble(one_func * ds_restricted)
    return boundary_len
