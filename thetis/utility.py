"""
Utility functions and classes for 3D hydrostatic ocean model
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

# TODO move 3d model classes to separate module


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
    """
    Create a directory on disk

    Raises IOError if a file with the same name already exists.
    """
    if comm.rank == 0:
        if os.path.exists(path):
            if not os.path.isdir(path):
                raise IOError('file with same name exists', path)
        else:
            os.makedirs(path)
    comm.barrier()
    return path


def extrude_mesh_sigma(mesh2d, n_layers, bathymetry_2d):
    """
    Extrudes a 2d surface mesh with bathymetry data defined in a 2d field.

    Generates a uniform terrain following mesh.

    :arg mesh2d: 2D mesh
    :arg n_layers: number of vertical layers
    :arg bathymetry: 2D :class:`Function` of the bathymetry
        (the depth of the domain; positive downwards)
    """
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
    """Computes volume of the 2D domain as an integral of the elevation field"""
    val = assemble((eta+bath)*dx)
    return val


def comp_volume_3d(mesh):
    """Computes volume of the 3D domain as an integral"""
    one = Constant(1.0, domain=mesh.coordinates.ufl_domain())
    val = assemble(one*dx)
    return val


def comp_tracer_mass_3d(scalar_func):
    """
    Computes total tracer mass in the 3D domain

    :arg scalar_func: scalar :class:`Function` to integrate
    """
    val = assemble(scalar_func*dx)
    return val


def get_zcoord_from_mesh(zcoord, solver_parameters={}):
    """
    Evaluates z coordinates from the 3D mesh

    :arg zcoord: scalar :class:`Function` where coordinates will be stored
    """
    # TODO coordinates should probably be interpolated instead
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
    r"""
    Computes vertical velocity diagnostically from the continuity equation

    Vertical velocity is obtained from the continuity equation

    .. math::
        \frac{\partial w}{\partial z} = \nabla_h \cdot \textbf{u}

    and the bottom impermeability condition (:math:`h` denotes the bathymetry)

    .. math::
        \textbf{n}_h \cdot \textbf{u} + w n_z &= 0 \quad \forall \mathbf{x} \in \Gamma_{b} \\
        \Leftrightarrow w &= -\nabla_h h \cdot \mathbf{u} \quad \forall \mathbf{x} \in \Gamma_{b}

    :math:`w` can be solved with the weak form

    .. math::
        \int_{\Gamma_s} w n_z \varphi dS
        + \int_{\mathcal{I}_h} \text{avg}(w) \text{jump}(\varphi n_z) dS
        - \int_{\Omega} w \frac{\partial \varphi}{\partial z} dx
        = \\
        \int_{\Omega} \mathbf{u} \cdot \nabla_h \varphi dx
        - \int_{\mathcal{I}_h \cup \mathcal{I}_v} \text{avg}(\mathbf{u}) \cdot \text{jump}(\varphi \mathbf{n}_h) dS
        - \int_{\Gamma_s} \mathbf{u} \cdot \varphi \mathbf{n}_h dS

    where the :math:`\Gamma_b` terms vanish due to the bottom impermeability
    condition.
    """
    def __init__(self, solution, uv, bathymetry, boundary_funcs={},
                 solver_parameters={}):
        """
        :arg solution: w :class:`Function`
        :arg uv: horizontal velocity :class:`Function`
        :arg bathymetry: bathymetry :class:`Function`
        :kwarg dict boundary_funcs: boundary conditions used in the 3D momentum
            equation. Provides external values of uv (if any).
        :kwarg dict solver_parameters: PETSc solver options
        """
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
        """Compute w"""
        self.solver.solve()


class VerticalIntegrator(object):
    """
    Computes vertical integral (or average) of a field.

    """
    def __init__(self, input, output, bottom_to_top=True,
                 bnd_value=Constant(0.0), average=False,
                 bathymetry=None, elevation=None, solver_parameters={}):
        """
        :arg input: 3D field to integrate
        :arg output: 3D field where the integral is stored
        :kwarg bottom_to_top: Defines the integration direction: If True integration is performed along the z axis, from bottom surface to top surface.
        :kwarg bnd_value: Value of the integral at the bottom (top) boundary if bottom_to_top is True (False)
        :kwarg average: If True computes the vertical average instead. Requires bathymetry and elevation fields
        :kwarg bathymetry: 3D field defining the bathymetry
        :kwarg elevation: 3D field defining the free surface elevation
        :kwarg dict solver_parameters: PETSc solver options
        """
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
        self.prob = LinearVariationalProblem(self.a, self.l, output, constant_jacobian=average)
        self.solver = LinearVariationalSolver(self.prob, solver_parameters=solver_parameters)

    def solve(self):
        """
        Computes the integral and stores it in the output field.
        """
        self.solver.solve()


class DensitySolver(object):
    r"""
    Computes density from salinity and temperature using the equation of state.

    Water density is defined as

    .. math::
        \rho = \rho'(T, S, p) + \rho_0

    This method computes the density anomaly :math:`\rho'`.
    Density is computed point-wise assuming that temperature, salinity and
    density are in the same function space.
    """
    def __init__(self, salinity, temperature, density, eos_class):
        """
        :arg salinity: water salinity field
        :type salinity: :class:`Function`
        :arg temperature: water temperature field
        :type temperature: :class:`Function`
        :arg density: water density field
        :type density: :class:`Function`
        :arg eos_class: equation of state that defines water density
        :type eos_class: :class:`EquationOfState`
        """
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
        """Returns numpy data array from a :class:`Function`"""
        if isinstance(function, FiredrakeFunction):
            assert self.fs == function.function_space()
            return function.dat.data[:]
        if isinstance(function, FiredrakeConstant):
            return function.dat.data[0]
        # assume that function is a float
        return function

    def solve(self):
        """Compute density"""
        s = self._get_array(self.s)
        th = self._get_array(self.t)
        p = 0.0  # NOTE ignore pressure for now
        rho0 = self._get_array(physical_constants['rho0'])
        self.rho.dat.data[:] = self.eos.compute_rho(s, th, p, rho0)


def compute_baroclinic_head(solver):
    r"""
    Computes the baroclinic head :math:`r` from the density field

    .. math::
        r = \frac{1}{\rho_0} \int_{z}^\eta  \rho' d\zeta.
    """
    with timed_region('density_solve'):
        solver.density_solver.solve()
    with timed_region('rho_integral'):
        solver.rho_integrator.solve()
        solver.fields.baroc_head_3d *= -physical_constants['rho0_inv']
    with timed_region('extr_bot_bhead'):
        solver.extract_bot_baro_head.solve()
    with timed_region('average_bhead'):
        solver.baro_head_averager.solve()
    with timed_region('extr_top_bhead'):
        solver.extract_surf_baro_head.solve()
    with timed_region('copy_bhead_3d'):
        solver.copy_mean_baroc_head_to_3d.solve()


class VelocityMagnitudeSolver(object):
    """
    Computes magnitude of (u[0],u[1],w) and stores it in solution
    """
    def __init__(self, solution, u=None, w=None, min_val=1e-6,
                 solver_parameters={}):
        """
        :arg solution: scalar field for velocity magnitude scalar :class:`Function`
        :type solution: :class:`Function`
        :kwarg u: horizontal velocity
        :type u: :class:`Function`
        :kwarg w: vertical velocity
        :type w: :class:`Function`
        :kwarg float min_val: minimum value of magnitude. Minimum value of solution
            will be clipped to this value
        :kwarg dict solver_parameters: PETSc solver options


        If ``u`` is None computes magnitude of (0,0,w).

        If ``w`` is None computes magnitude of (u[0],u[1],0).
        """
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
        """Compute the magnitude"""
        self.solver.solve()
        np.maximum(self.solution.dat.data, self.min_val, self.solution.dat.data)


class ExpandFunctionTo3d(object):
    """
    Copy a 2D field to 3D

    Copies a field from 2D mesh to 3D mesh, assigning the same value over the
    vertical dimension. Horizontal function spaces must be the same.

    .. code-block:: python

        U = FunctionSpace(mesh, 'DG', 1)
        U_2d = FunctionSpace(mesh2d, 'DG', 1)
        func2d = Function(U_2d)
        func3d = Function(U)
        ex = ExpandFunctionTo3d(func2d, func3d)
        ex.solve()
    """
    def __init__(self, input_2d, output_3d, elem_height=None):
        """
        :arg input_2d: 2D source field
        :type input_2d: :class:`Function`
        :arg output_3d: 3D target field
        :type output_3d: :class:`Function`
        :kwarg elem_height: scalar :class:`Function` in 3D mesh that defines
            the vertical element size. Needed only in the case of HDiv function
            spaces.
        """
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
    Extract a 2D sub-function from a 3D function in an extruded mesh

    Given 2D and 3D functions,

    .. code-block:: python

        U = FunctionSpace(mesh, 'DG', 1)
        U_2d = FunctionSpace(mesh2d, 'DG', 1)
        func2d = Function(U_2d)
        func3d = Function(U)

    Get surface value:

    .. code-block:: python

        ex = SubFunctionExtractor(func3d, func2d,
            boundary='top', elem_facet='top')
        ex.solve()

    Get bottom value:

    .. code-block:: python

        ex = SubFunctionExtractor(func3d, func2d,
            boundary='bottom', elem_facet='bottom')
        ex.solve()

    Get value at the top of bottom element:

    .. code-block:: python

        ex = SubFunctionExtractor(func3d, func2d,
            boundary='bottom', elem_facet='top')
        ex.solve()
    """
    def __init__(self, input_3d, output_2d,
                 boundary='top', elem_facet=None,
                 elem_height=None):
        """
        :arg input_3d: 3D source field
        :type input_3d: :class:`Function`
        :arg output_2d: 2D target field
        :type output_2d: :class:`Function`
        :kwarg str boundary: 'top'|'bottom'
            Defines whether to extract from the surface or bottom 3D elements
        :kwarg str elem_facet: 'top'|'bottom'|'average'
            Defines which facet of the 3D element is extracted. The 'average'
            computes mean of the top and bottom facets of the 3D element.
        :kwarg elem_height: scalar :class:`Function` in 2D mesh that defines
            the vertical element size. Needed only in the case of HDiv function
            spaces.
        """
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
    Computes the element height on an extruded mesh.

    :arg zcoord: field that contains the z coordinates of the mesh
    :type zcoord: :class:`Function`
    :arg output: field where element height is stored
    :type output: :class:`Function`
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


def compute_bottom_drag(h_b, drag):
    r"""
    Computes bottom drag coefficient (Cd) from the law-of-the wall

    .. math::
        C_D = \left( \frac{\kappa}{\ln (h_b + z_0)/z_0} \right)^2

    :arg h_b: the height above bed where the bottom velocity is evaluated in
        the law-of-the-wall fit
    :type h_b: :class:`Function`
    :arg drag: field where C_D is stored
    :type drag: :class:`Function`
    """
    # FIXME z0 should be a field, i.e. an argument to this function
    von_karman = physical_constants['von_karman']
    z0_friction = physical_constants['z0_friction']
    drag.assign((von_karman / ln((h_b + z0_friction)/z0_friction))**2)
    return drag


def compute_bottom_friction(solver, uv_3d, uv_bottom_2d,
                            z_bottom_2d, bathymetry_2d,
                            bottom_drag_2d):
    """
    Updates bottom friction related fields for the 3D model

    :arg solver: :class:`FlowSolver` object
    :arg uv_3d: horizontal velocity
    :type uv_3d: 3D vector :class:`Function`
    :arg uv_bottom_2d: 2D bottom velocity field
    :type uv_bottom_2d: 2D vector :class:`Function`
    :arg z_bottom_2d: Bottom element z coordinate
    :type z_bottom_2d: 2D scalar :class:`Function`
    :arg bathymetry_2d: Bathymetry field
    :type bathymetry_2d: 2D scalar :class:`Function`
    :arg bottom_drag_2d: Bottom grad field
    :type bottom_drag_2d: 2D scalar :class:`Function`
    """
    # TODO all input fields could be just fetched from solver.fields ...
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
    Computes horizontal element size from the 2D mesh

    :arg sol2d: 2D :class:`Function` where result is stored
    """
    p1_2d = sol2d.function_space()
    mesh = p1_2d.mesh()
    test = TestFunction(p1_2d)
    tri = TrialFunction(p1_2d)
    a = inner(test, tri) * dx
    l = inner(test, sqrt(CellVolume(mesh))) * dx
    solve(a == l, sol2d)


def get_horizontal_elem_size_3d(sol2d, sol3d):
    """
    Computes horizontal element size from the 2D mesh, then copies it on a 3D
    field

    :arg sol2d: 2D :class:`Function` for the element size field
    :arg sol3d: 3D :class:`Function` for the element size field
    """
    get_horizontal_elem_size_2d(sol2d)
    ExpandFunctionTo3d(sol2d, sol3d).solve()


class ALEMeshUpdater(object):
    """
    Class that handles vertically moving ALE mesh

    Mesh geometry is updated to match the elevation field
    (``solver.fields.elev_2d``). First the discontinuous elevation field is
    projected to continuous space, and this field is used to update the mesh
    coordinates.

    This class stores the reference coordinate field and keeps track of the
    updated mesh coordinates. It also provides a method for computing the mesh
    velocity from two adjacent elevation fields.
    """
    def __init__(self, solver):
        """
        :arg solver: :class:`FlowSolver` object
        """
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
        """
        Computes mesh velocity from the elevation difference

        Stores the current 2D elevation state as the "new" field,
        and computes w_mesh using the given time step factor ``c``.
        """
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

        elev_2d is first projected to continous space
        """
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
    r"""
    Computes parabolic eddy viscosity profile assuming log layer flow

    .. math::
        \nu = \kappa u_{bf}  \frac{(-z)(h + z_0 + z)}{h + z_0}

    with

    .. math::
        u_{bf} = \sqrt{C_D} |\mathbf{u}_b|

    """
    def __init__(self, uv_bottom, bottom_drag, bathymetry, nu,
                 solver_parameters={}):
        """
        :arg uv_bottom: bottom velocity
        :type uv_bottom: 3D :class:`Function`
        :arg bottom_drag: bottom drag field
        :type bottom_drag: 3D :class:`Function`
        :arg bathymetry: bathymetry field
        :type bathymetry: 3D :class:`Function`
        :arg nu: eddy viscosity field
        :type nu: 3D :class:`Function`
        :kwarg dict solver_parameters: PETSc solver options
        """
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
        """
        Computes viscosity and stores it in nu field
        """
        self.solver.solve()
        # remove negative values
        ix = self.solution.dat.data[:] < self.min_val
        self.solution.dat.data[ix] = self.min_val


def beta_plane_coriolis_params(latitude):
    r"""
    Computes beta plane parameters :math:`f_0,\beta` based on latitude

    :arg float latitude: latitude in degrees
    :return: f_0, beta
    :rtype: float
    """
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


def beta_plane_coriolis_function(latitude, out_function, y_offset=0.0):
    """
    Interpolates beta plane Coriolis function to a field

    :arg float latitude: latitude in degrees
    :arg out_function: :class:`Function` where to interpolate
    :kwarg float y_offset: offset (y - y_0) used in Beta-plane approximation.
        A constant in mesh coordinates.
    """
    # NOTE assumes that mesh y coordinate spans [-L_y, L_y]
    f0, beta = beta_plane_coriolis_params(latitude)
    out_function.interpolate(
        Expression('f0+beta*(x[1]-y_0)', f0=f0, beta=beta, y_0=y_offset))


class SmagorinskyViscosity(object):
    r"""
    Computes Smagorinsky subgrid scale horizontal viscosity

    This formulation is according to Ilicak et al. (2012) and
    Griffies and Hallberg (2000).

    .. math::
        \nu = (C_s \Delta x)^2 |S|

    with the deformation rate

    .. math::
        |S| &= \sqrt{D_T^2 + D_S^2} \\
        D_T &= \frac{\partial u}{\partial x} - \frac{\partial v}{\partial y} \\
        D_S &= \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x}

    :math:`\Delta x` is the horizontal element size and :math:`C_s` is the
    Smagorinsky coefficient.

    To match a certain mesh Reynolds number :math:`Re_h` set
    :math:`C_s = 1/\sqrt{Re_h}`.

    Ilicak et al. (2012). Spurious dianeutral mixing and the role of
    momentum closure. Ocean Modelling, 45-46(0):37-58.
    http://dx.doi.org/10.1016/j.ocemod.2011.10.003

    Griffies and Hallberg (2000). Biharmonic friction with a
    Smagorinsky-like viscosity for use in large-scale eddy-permitting
    ocean models. Monthly Weather Review, 128(8):2935-2946.
    http://dx.doi.org/10.1175/1520-0493(2000)128%3C2935:BFWASL%3E2.0.CO;2
    """
    def __init__(self, uv, output, c_s, h_elem_size, max_val, min_val=1e-10,
                 weak_form=True, solver_parameters={}):
        """
        :arg uv_3d: horizontal velocity
        :type uv_3d: 3D vector :class:`Function`
        :arg output: Smagorinsky viscosity field
        :type output: 3D scalar :class:`Function`
        :arg c_s: Smagorinsky coefficient
        :type c_s: float or :class:`Constant`
        :arg h_elem_size: field that defines the horizontal element size
        :type h_elem_size: 3D scalar :class:`Function` or :class:`Constant`
        :arg float max_val: Maximum allowed viscosity. Viscosity will be clipped at
            this value.
        :kwarg float min_val: Minimum allowed viscosity. Viscosity will be clipped at
            this value.
        :kwarg bool weak_form: Compute velocity shear by integrating by parts.
            Necessary for some function spaces (e.g. P0).
        :kwarg dict solver_parameters: PETSc solver options
        """
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
        """Compute viscosity"""
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
    r"""
    Equation of State according of Jackett et al. (2006) for computing sea
    water density.

    .. math ::
        \rho = \rho'(T, S, p) + \rho_0

    Jackett et al. (2006). Algorithms for Density, Potential Temperature,
    Conservative Temperature, and the Freezing Temperature of Seawater.
    Journal of Atmospheric and Oceanic Technology, 23(12):1709-1728.
    http://dx.doi.org/10.1175/JTECH1946.1
    """
    # TODO make an abstract base class
    def __init__(self):
        """Sets coefficients"""
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
        r"""
        Computes sea water density.

        :arg s: Salinity expressed on the Practical Salinity Scale 1978
        :type s: float or numpy.array
        :arg th: Potential temperature in Celsius, referenced to pressure
            p_r = 0 dbar.
        :type th: float or numpy.array
        :arg p: Pressure in decibars (1 dbar = 1e4 Pa)
        :type p: float or numpy.array
        :kwarg float rho0: Optional reference density. If provided computes
            :math:`\rho' = \rho(S, Th, p) - \rho_0`
        :return: water density
        :rtype: float or numpy.array

        All pressures are gauge pressures: they are the absolute pressures minus standard atmosperic
        pressure 10.1325 dbar.
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
    r"""
    Linear Equation of State for computing sea water density

    .. math::
        \rho = \rho_{ref} - \alpha (T - T_{ref}) + \beta (S - S_{ref})
    """
    def __init__(self, rho_ref, alpha, beta, th_ref, s_ref):
        """
        :arg float rho_ref: reference density
        :arg float alpha: thermal expansion coefficient
        :arg float beta: haline contraction coefficient
        :arg float th_ref: reference temperature
        :arg float s_ref: reference salinity
        """
        self.rho_ref = rho_ref
        self.alpha = alpha
        self.beta = beta
        self.th_ref = th_ref
        self.S_ref = s_ref

    def compute_rho(self, s, th, p, rho0=0.0):
        r"""
        Computes sea water density.

        :arg s: Salinity expressed on the Practical Salinity Scale 1978
        :type s: float or numpy.array
        :arg th: Potential temperature in Celsius
        :type th: float or numpy.array
        :arg p: Pressure in decibars (1 dbar = 1e4 Pa)
        :type p: float or numpy.array
        :kwarg float rho0: Optional reference density. If provided computes
            :math:`\rho' = \rho(S, Th, p) - \rho_0`
        :return: water density
        :rtype: float or numpy.array

        Pressure is ingored in this equation of state.
        """
        rho = (self.rho_ref - rho0 -
               self.alpha*(th - self.th_ref) +
               self.beta*(s - self.S_ref))
        return rho


def tensor_jump(v, n):
    r"""
    Jump term for vector functions based on the tensor product

    .. math::
        \text{jump}(\mathbf{u}, \mathbf{n}) = (\mathbf{u}^+ \mathbf{n}^+) +
        (\mathbf{u}^- \mathbf{n}^-)

    This is the discrete equivalent of grad(u) as opposed to the
    vectorial UFL jump operator :meth:`ufl.jump` which represents div(u).
    """
    return outer(v('+'), n('+')) + outer(v('-'), n('-'))


def compute_boundary_length(mesh2d):
    """
    Computes the length of the boundary segments in given 2d mesh
    """
    p1 = FunctionSpace(mesh2d, 'CG', 1)
    boundary_markers = mesh2d.exterior_facets.unique_markers
    boundary_len = {}
    for i in boundary_markers:
        ds_restricted = ds(int(i))
        one_func = Function(p1).assign(1.0)
        boundary_len[i] = assemble(one_func * ds_restricted)
    return boundary_len
