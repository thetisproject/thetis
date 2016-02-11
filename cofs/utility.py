"""
Utility functions and classes for 3D hydrostatic ocean model

Tuomas Karna 2015-02-21
"""
from firedrake import *
import os
import numpy as np
import sys
from physical_constants import physical_constants
import colorama
from pyop2.profiling import timed_region, timed_function, timing  # NOQA
from mpi4py import MPI  # NOQA
import ufl  # NOQA
import coffee.base as ast  # NOQA
from cofs.field_defs import field_metadata

comm = op2.MPI.comm
commrank = op2.MPI.comm.rank

ds_surf = ds_b
ds_bottom = ds_t
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


class Equation(object):
    """Base class for all equations"""
    # TODO move to equation.py
    def mass_matrix(self, *args, **kwargs):
        """Returns weak form for left hand side."""
        raise NotImplementedError(('This method must be implemented '
                                   'in the derived class'))

    def rhs(self, *args, **kwargs):
        """Returns weak form for the right hand side."""
        raise NotImplementedError(('This method must be implemented '
                                   'in the derived class'))

    def rhs_implicit(self, *args, **kwargs):
        """Returns weak form for the right hand side of all implicit terms"""
        raise NotImplementedError(('This method must be implemented '
                                   'in the derived class'))

    def source(self, *args, **kwargs):
        """Returns weak for for terms that do not depend on the solution."""
        raise NotImplementedError(('This method must be implemented '
                                   'in the derived class'))


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
            if not isinstance(value, (Function, Constant)):
                raise TypeError('Value must be a Function or Constant object')
            fs = value.function_space()
            if not isinstance(fs, MixedFunctionSpace) and key not in field_metadata:
                msg = 'Trying to add a field "{:}" that has no meta data. ' \
                      'Add field_metadata entry to field_defs.py'.format(key)
                raise Exception(msg)

    def _set_functionname(self, key, value):
        """Set function.name to key to ensure consistent naming"""
        if isinstance(value, Function):
            value.rename(name=key)

    def __setitem__(self, key, value):
        self._check_inputs(key, value)
        self._set_functionname(key, value)
        super(FieldDict, self).__setitem__(key, value)

    def __setattr__(self, key, value):
        self._check_inputs(key, value)
        self._set_functionname(key, value)
        super(FieldDict, self).__setattr__(key, value)


class TemporaryFunctionCache(object):
    """
    Holds temporary functions for needed function spaces in a dict.

    """
    def __init__(self):
        self.functions = {}

    def add(self, function_space):
        """Allocates a new function and adds it to the dict"""
        assert function_space not in self.functions
        if function_space.name is not None:
            name = 'tmp_func_' + function_space.name
        else:
            name = 'tmp_func_' + str(len(self.functions))
        f = Function(function_space, name=name)
        key = self._get_key(function_space)
        self.functions[key] = f

    def _get_key(self, function_space):
        """generate a unique str representation for function space"""
        ufl_elem = function_space.ufl_element()
        elems = [ufl_elem]
        if hasattr(ufl_elem, '_A'):
            elems += [ufl_elem._A, ufl_elem._B]
        elif isinstance(ufl_elem, EnrichedElement):
            for d in elems:
                elems.append(d)
                if hasattr(d, '_A'):
                    elems += [ufl_elem._A, ufl_elem._B]

        def get_elem_str(e):
            s = e.shortstr()
            d = e.degree()
            if not isinstance(d, tuple):
                s = s.replace('?', str(d))
            return s
        attrib = [get_elem_str(ee) for ee in elems]
        key = '_'.join(attrib)
        return key

    def get(self, function_space):
        """
        Returns a tmp function for the given function space.

        If it doesn't exist, will allocate one.
        """
        if function_space not in self.functions:
            self.add(function_space)
        key = self._get_key(function_space)
        return self.functions[key]

    def clear(self):
        self.functions = {}

tmp_function_cache = TemporaryFunctionCache()


def print_info(msg):
    if commrank == 0:
        print(msg)


def colorify_text(color):
    def painter(func):
        def func_wrapper(text):
            return color + func(text) + colorama.Style.RESET_ALL
        return func_wrapper
    return painter


@colorify_text(colorama.Back.RED + colorama.Fore.WHITE)
def red(text):
    """Returns given string in uppercase, white on red background."""
    return text.upper()


@colorify_text(colorama.Back.GREEN + colorama.Fore.BLACK)
def green(text):
    """Returns given string in uppercase, black on green background."""
    return text.upper()


@colorify_text(colorama.Style.BRIGHT + colorama.Fore.WHITE)
def bold(text):
    """Returns given string in uppercase, white on green background."""
    return text.upper()


def create_directory(path):
    if commrank == 0:
        if os.path.exists(path):
            if not os.path.isdir(path):
                raise Exception('file with same name exists', path)
        else:
            os.makedirs(path)
    return path


def extrude_mesh_linear(mesh2d, n_layers, xmin, xmax, dmin, dmax):
    """Extrudes 2d surface mesh with defined by two endpoints in x axis."""
    layer_height = 1.0/n_layers
    base_coords = mesh2d.coordinates
    extrusion_kernel = op2.Kernel("""
            void uniform_extrusion_kernel(double **base_coords,
                        double **ext_coords,
                        int **layer,
                        double *layer_height) {
                for ( int d = 0; d < %(base_map_arity)d; d++ ) {
                    for ( int c = 0; c < %(base_coord_dim)d; c++ ) {
                        ext_coords[2*d][c] = base_coords[d][c];
                        ext_coords[2*d+1][c] = base_coords[d][c];
                    }
                    double s = fmin(fmax((ext_coords[2*d][0]-%(xmin)f)/(%(xmax)f-%(xmin)f), 0.0), 1.0);
                    double depth = s*%(dmax)f + (1.0-s)*%(dmin)f;
                    ext_coords[2*d][%(base_coord_dim)d] = -depth*layer_height[0]*layer[0][0];
                    ext_coords[2*d+1][%(base_coord_dim)d] = -depth*layer_height[0]*(layer[0][0]+1);
                }
            }""" % {'base_map_arity': base_coords.cell_node_map().arity,
                    'base_coord_dim': base_coords.function_space().dim,
                    'xmin': xmin, 'xmax': xmax, 'dmin': dmin, 'dmax': dmax},
        'uniform_extrusion_kernel')
    mesh = ExtrudedMesh(mesh2d, layers=n_layers,
                        layer_height=layer_height,
                        extrusion_type='custom',
                        kernel=extrusion_kernel, gdim=3)
    return mesh


def extrude_mesh_sigma(mesh2d, n_layers, bathymetry_2d):
    """Extrudes 2d surface mesh with bathymetry data defined in a 2d field."""
    mesh = ExtrudedMesh(mesh2d, layers=n_layers, layer_height=1.0/n_layers)

    xyz = mesh.coordinates

    n_nodes_2d = bathymetry_2d.dat.data.shape[0]
    n_vert = xyz.dat.data.shape[0]/n_nodes_2d
    # TODO can the loop be circumvented?
    for i_node in range(n_nodes_2d):
        xyz.dat.data[i_node*n_vert:i_node*n_vert+n_vert, 2] *= -bathymetry_2d.dat.data[i_node]
    return mesh


def comp_volume_2d(eta, bath):
    val = assemble((eta+bath)*dx)
    return val


def comp_volume_3d(mesh):
    one = Constant(1.0, domain=mesh.coordinates.domain())
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
    def __init__(self, solution, uv, bathymetry,
                 boundary_markers=[], boundary_funcs={},
                 solver_parameters={}):
        solver_parameters.setdefault('ksp_atol', 1e-12)
        solver_parameters.setdefault('ksp_rtol', 1e-16)
        fs = solution.function_space()
        mesh = fs.mesh()
        test = TestFunction(fs)
        tri = TrialFunction(fs)
        normal = FacetNormal(mesh)

        # NOTE weak dw/dz
        a = tri[2]*test[2]*normal[2]*ds_surf + \
            avg(tri[2])*jump(test[2], normal[2])*dS_h - Dx(test[2], 2)*tri[2]*dx
        # NOTE weak div(uv)
        uv_star = avg(uv)
        # NOTE in the case of mimetic uv the div must be taken over all components
        l = (inner(uv, nabla_grad(test[2]))*dx -
             (uv_star[0]*jump(test[2], normal[0]) +
              uv_star[1]*jump(test[2], normal[1]) +
              uv_star[2]*jump(test[2], normal[2]))*(dS_v + dS_h) -
             (uv[0]*normal[0] +
              uv[1]*normal[1] +
              uv[2]*normal[2])*test[2]*ds_surf
             )
        for bnd_marker in boundary_markers:
            funcs = boundary_funcs.get(bnd_marker)
            ds_bnd = ds_v(int(bnd_marker))
            if funcs is None:
                # assume land boundary
                continue
            else:
                # use symmetry condition
                l += -(uv[0]*normal[0] + uv[1]*normal[1])*test[2]*ds_bnd

        self.prob = LinearVariationalProblem(a, l, solution)
        self.solver = LinearVariationalSolver(self.prob, solver_parameters=solver_parameters)

    def solve(self):
        self.solver.solve()


class VerticalIntegrator(object):
    """
    Computes vertical integral of the scalar field in the output
    function space.
    """
    def __init__(self, input, output, bottom_to_top=True,
                 bnd_value=Constant(0.0), average=False,
                 bathymetry=None, solver_parameters={}):
        solver_parameters.setdefault('ksp_atol', 1e-12)
        solver_parameters.setdefault('ksp_rtol', 1e-16)

        space = output.function_space()
        mesh = space.mesh()
        vertical_is_dg = False
        if (hasattr(space.ufl_element(), '_B') and
                space.ufl_element()._B.family() != 'Lagrange'):
            # a normal tensorproduct element
            vertical_is_dg = True
        if 'HDiv' in space.ufl_element().shortstr():
            # Hdiv vector space, assume DG in vertical
            vertical_is_dg = True
        tri = TrialFunction(space)
        phi = TestFunction(space)
        normal = FacetNormal(mesh)
        if bottom_to_top:
            bnd_term = normal[2]*inner(bnd_value, phi)*ds_bottom
            mass_bnd_term = normal[2]*inner(tri, phi)*ds_surf
        else:
            bnd_term = normal[2]*inner(bnd_value, phi)*ds_surf
            mass_bnd_term = normal[2]*inner(tri, phi)*ds_bottom

        a = -inner(Dx(phi, 2), tri)*dx + mass_bnd_term
        if vertical_is_dg:
            if len(input.ufl_shape) > 0:
                dim = input.ufl_shape[0]
                for i in range(dim):
                    a += avg(tri[i])*jump(phi[i], normal[2])*dS_h
            else:
                a += avg(tri)*jump(phi, normal[2])*dS_h
        source = input
        if average:
            # FIXME this should be H not h
            source = input/bathymetry
        l = inner(source, phi)*dx + bnd_term
        self.prob = LinearVariationalProblem(a, l, output)
        self.solver = LinearVariationalSolver(self.prob, solver_parameters=solver_parameters)

    def solve(self):
        with timed_region('func_vert_int'):
            self.solver.solve()


def compute_baroclinic_head(solver, salt, baroc_head_3d, baroc_head_2d,
                            baroc_head_int_3d, bath3d):
    """
    Computes baroclinic head from density field

    r = 1/rho_0 int_{z=-h}^{\eta} rho' dz
    """
    solver.rho_integrator.solve()
    baroc_head_3d *= -physical_constants['rho0_inv']
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
        self.solution.dat.data[:] = np.maximum(self.solution.dat.data[:], self.min_val)


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
        self.output.dat.data[:] = np.minimum(self.max_val.dat.data, self.output.dat.data)
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
        if hasattr(self.fs_3d.ufl_element(), '_A'):
            # a normal tensorproduct element
            family_3dh = self.fs_3d.ufl_element()._A.family()
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
        with timed_region('func_copy_2d_to_3d'):
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
    use_bottom_value: bool
        Extract function at bottom elements
    elem_bottom_nodes: bool
        Extract function at bottom mask of the element
        Typically elem_bottom_nodes = use_bottom_value to obtain values at
        surface/bottom faces of the extruded mesh
    elem_height: Function in 2d mesh
        Function that defines the element heights
        (required for Raviart-Thomas spaces only)
    """
    def __init__(self, input_3d, output_2d, use_bottom_value=True,
                 elem_bottom_nodes=None, elem_height=None):
        self.input_3d = input_3d
        self.output_2d = output_2d
        self.fs_3d = self.input_3d.function_space()
        self.fs_2d = self.output_2d.function_space()

        # NOTE top/bottom are defined differently than in firedrake
        sub_domain = 'bottom' if use_bottom_value else 'top'
        if elem_bottom_nodes is None:
            # extract surface/bottom face by default
            elem_bottom_nodes = use_bottom_value

        family_2d = self.fs_2d.ufl_element().family()
        if hasattr(self.fs_3d.ufl_element(), '_A'):
            # a normal tensorproduct element
            family_3dh = self.fs_3d.ufl_element()._A.family()
            if family_2d != family_3dh:
                raise Exception('2D and 3D spaces do not match: {0:s} {1:s}'.format(family_2d, family_3dh))
        if family_2d == 'Raviart-Thomas' and elem_height is None:
            raise Exception('elem_height must be provided for Raviart-Thomas spaces')
        self.do_rt_scaling = family_2d == 'Raviart-Thomas'

        if elem_bottom_nodes:
            nodes = self.fs_3d.bt_masks['geometric'][1]
        else:
            nodes = self.fs_3d.bt_masks['geometric'][0]
        if sub_domain == 'top':
            # 'top' means free surface, where extrusion started
            self.iter_domain = op2.ON_BOTTOM
        elif sub_domain == 'bottom':
            # 'bottom' means the bed, where extrusion ended
            self.iter_domain = op2.ON_TOP

        out_nodes = self.fs_2d.fiat_element.space_dimension()

        assert (len(nodes) == out_nodes)

        self.idx = op2.Global(len(nodes), nodes, dtype=np.int32, name='node_idx')
        self.kernel = op2.Kernel("""
            void my_kernel(double **func, double **func3d, int *idx) {
                for ( int d = 0; d < %(nodes)d; d++ ) {
                    for ( int c = 0; c < %(func_dim)d; c++ ) {
                        func[d][c] = func3d[idx[d]][c];
                        //func[d][c] = idx[d];
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
        with timed_region('func_copy_3d_to_2d'):
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
                    // NOTE is fabs required here??
                    double dz = zcoord[2*d+1][c]-zcoord[2*d][c];
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


def compute_bottom_drag(uv_bottom, z_bottom, bathymetry, drag):
    """Computes bottom drag coefficient (Cd) from boundary log layer."""
    von_karman = physical_constants['von_karman']
    z0_friction = physical_constants['z0_friction']
    drag.assign((von_karman / ln((z_bottom)/z0_friction))**2)
    return drag


def compute_bottom_friction(solver, uv_3d, uv_bottom_2d, uv_bottom_3d, z_coord_3d,
                            z_bottom_2d, bathymetry_2d,
                            bottom_drag_2d, bottom_drag_3d,
                            v_elem_size_2d=None, v_elem_size_3d=None):
    # compute velocity at middle of bottom element
    solver.extract_uv_bottom.solve()
    tmp = uv_bottom_2d.dat.data.copy()
    solver.extract_uv_bottom.solve()
    uv_bottom_2d.dat.data[:] = 0.5*(uv_bottom_2d.dat.data + tmp)
    solver.copy_uv_bottom_to_3d.solve()
    solver.extract_z_bottom.solve()
    z_bottom_2d.dat.data[:] += bathymetry_2d.dat.data[:]
    z_bottom_2d.dat.data[:] *= 0.5
    compute_bottom_drag(uv_bottom_2d, z_bottom_2d, bathymetry_2d, bottom_drag_2d)
    solver.copy_bottom_drag_to_3d.solve()


def get_horizontal_elem_size(sol2d, sol3d=None):
    """
    Computes horizontal element size from the 2D mesh, then copies it over a 3D
    field.
    """
    p1_2d = sol2d.function_space()
    mesh = p1_2d.mesh()
    cellsize = CellSize(mesh)
    test = TestFunction(p1_2d)
    tri = TrialFunction(p1_2d)
    sol2d = Function(p1_2d)
    a = test * tri * dx
    l = test * cellsize * dx
    solve(a == l, sol2d)
    if sol3d is None:
        return sol2d
    ExpandFunctionTo3d(sol2d, sol3d).solve()
    return sol3d


class ALEMeshCoordinateUpdater(object):
    """Updates extrusion so that free surface mathces eta3d value"""
    def __init__(self, mesh, eta, bathymetry, z_coord, z_coord_ref,
                 solver_parameters={}):
        solver_parameters.setdefault('ksp_atol', 1e-12)
        solver_parameters.setdefault('ksp_rtol', 1e-16)
        self.coords = mesh.coordinates
        self.z_coord = z_coord

        fs = z_coord.function_space()
        # sigma stretch function
        new_z = eta*(z_coord_ref + bathymetry)/bathymetry + z_coord_ref
        # update z_coord
        tri = TrialFunction(fs)
        test = TestFunction(fs)
        a = tri*test*dx
        l = new_z*test*dx
        self.prob = LinearVariationalProblem(a, l, z_coord)
        self.solver = LinearVariationalSolver(self.prob, solver_parameters=solver_parameters)

    def solve(self):
        self.solver.solve()
        # assign to mesh
        self.coords.dat.data[:, 2] = self.z_coord.dat.data[:]


class MeshVelocitySolver(object):
    """Computes vertical mesh velocity for moving sigma mesh"""
    def __init__(self, solver, eta, uv, w, w_mesh, w_mesh_surf,
                 w_mesh_surf_2d, w_mesh_ddz_3d, bathymetry,
                 z_coord_ref, solver_parameters={}):
        solver_parameters.setdefault('ksp_atol', 1e-12)
        solver_parameters.setdefault('ksp_rtol', 1e-16)
        self.solver_obj = solver

        # compute w_mesh at the free surface (constant over vertical!)
        # w_mesh_surf = w - eta_grad[0]*uv[0] + eta_grad[1]*uv[1]
        fs = w_mesh.function_space()
        z = fs.mesh().coordinates[2]
        tri = TrialFunction(fs)
        test = TestFunction(fs)
        a = inner(tri, test)*dx
        eta_grad = nabla_grad(eta)
        l = (w[2] - eta_grad[0]*uv[0] - eta_grad[1]*uv[1])*test*dx
        self.prob_w_mesh_surf = LinearVariationalProblem(a, l, w_mesh_surf)
        self.solver_w_mesh_surf = LinearVariationalSolver(self.prob_w_mesh_surf, solver_parameters=solver_parameters)

        # compute w in the whole water column (0 at bed)
        # w_mesh = w_mesh_surf * (z+h)/(eta+h)
        fs = w_mesh.function_space()
        z = fs.mesh().coordinates[2]
        tri = TrialFunction(fs)
        test = TestFunction(fs)
        a = tri*test*dx
        tot_depth = eta + bathymetry
        l = (w_mesh_surf*(z+bathymetry)/tot_depth)*test*dx
        self.prob_w_mesh = LinearVariationalProblem(a, l, w_mesh)
        self.solver_w_mesh = LinearVariationalSolver(self.prob_w_mesh, solver_parameters=solver_parameters)

        # compute dw_mesh/dz in the whole water column
        fs = w_mesh.function_space()
        z = fs.mesh().coordinates[2]
        tri = TrialFunction(fs)
        test = TestFunction(fs)
        a = tri*test*dx
        l = (w_mesh_surf/tot_depth)*test*dx
        self.prob_dw_mesh_dz = LinearVariationalProblem(a, l, w_mesh_ddz_3d)
        self.solver_dw_mesh_dz = LinearVariationalSolver(self.prob_dw_mesh_dz, solver_parameters=solver_parameters)

    def solve(self):
        self.solver_w_mesh_surf.solve()
        self.solver_obj.extract_surf_w.solve()
        self.solver_obj.copy_surf_w_mesh_to_3d.solve()
        self.solver_w_mesh.solve()
        self.solver_dw_mesh_dz.solve()


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
    def __init__(self, uv, output, c_s, h_elem_size, solver_parameters={}):
        solver_parameters.setdefault('ksp_atol', 1e-12)
        solver_parameters.setdefault('ksp_rtol', 1e-16)
        self.min_val = 1e-10
        self.output = output

        fs = output.function_space()
        w = TestFunction(fs)
        tau = TrialFunction(fs)

        # rate of strain tensor
        d_t = Dx(uv[0], 0) - Dx(uv[1], 1)
        d_s = Dx(uv[0], 1) + Dx(uv[1], 0)
        nu = c_s**2*h_elem_size**2 * sqrt(d_t**2 + d_s**2)

        a = w*tau*dx
        l = w*nu*dx
        self.prob = LinearVariationalProblem(a, l, output)
        self.solver = LinearVariationalSolver(self.prob, solver_parameters=solver_parameters)

    def solve(self):
        self.solver.solve()

        # remove negative values
        ix = self.output.dat.data < self.min_val
        self.output.dat.data[ix] = self.min_val


class Projector(object):
    def __init__(self, input_func, output_func, solver_parameters={}):
        self.input = input_func
        self.output = output_func
        self.same_space = (self.input.function_space() ==
                           self.output.function_space())

        if not self.same_space:
            v = output_func.function_space()
            p = TestFunction(v)
            q = TrialFunction(v)
            a = inner(p, q) * dx
            l = inner(p, input_func) * dx

            solver_parameters.setdefault('ksp_type', 'cg')
            solver_parameters.setdefault('ksp_rtol', 1e-8)

            self.problem = LinearVariationalProblem(a, l, output_func)
            self.solver = LinearVariationalSolver(self.problem,
                                                  solver_parameters=solver_parameters)

    def project(self):
        if self.same_space:
            self.output.assign(self.input)
        else:
            try:
                self.solver.solve()
            except Exception as e:
                print 'projection failed for {:}'.format(self.input.name)
                raise e


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
        s0 = np.maximum(s, 0.0)  # ensure positive salinity
        a = self.a
        b = self.b
        pn = (a[0] + th*a[1] + th*th*a[2] + th*th*th*a[3] + s0*a[4] +
              th*s0*a[5] + s0*s0*a[6] + p*a[7] + p*th * th*a[8] + p*s0*a[9] +
              p*p*a[10] + p*p*th*th * a[11])
        pd = (b[0] + th*b[1] + th*th*b[2] + th*th*th*b[3] +
              th*th*th*th*b[4] + s0*b[5] + s0*th*b[6] + s0*th*th*th*b[7] +
              pow(s0, 1.5)*b[8] + pow(s0, 1.5)*th*th*b[9] + p*b[10] +
              p*p*th*th*th*b[11] + p*p*p*th*b[12])
        rho = pn/pd - rho0
        return rho
