"""
Utility functions and classes for 2D and 3D ocean models
"""
from __future__ import absolute_import
from firedrake import *
import os
import sys
from .physical_constants import physical_constants
from pyop2.profiling import timed_region, timed_function, timed_stage  # NOQA
from mpi4py import MPI  # NOQA
import ufl  # NOQA
import coffee.base as ast  # NOQA
from collections import OrderedDict, namedtuple  # NOQA
from .field_defs import field_metadata
from abc import ABCMeta, abstractmethod  # NOQA
from .log import *

ds_surf = ds_t
ds_bottom = ds_b


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
            if not isinstance(value, (Function, Constant)):
                raise TypeError('Value must be a Function or Constant object')
            fs = value.function_space()
            is_mixed = (isinstance(fs, MixedFunctionSpace)
                        or (isinstance(fs, WithGeometry)
                            and isinstance(fs.topological, MixedFunctionSpace)))
            if not is_mixed and key not in field_metadata:
                msg = 'Trying to add a field "{:}" that has no metadata. ' \
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


def get_functionspace(mesh, h_family, h_degree, v_family=None, v_degree=None,
                      vector=False, hdiv=False, variant=None, v_variant=None,
                      **kwargs):
    cell_dim = mesh.cell_dimension()
    assert cell_dim in [2, (2, 1)], 'Unsupported cell dimension'
    hdiv_families = [
        'RT', 'RTF', 'RTCF', 'RAVIART-THOMAS',
        'BDM', 'BDMF', 'BDMCF', 'BREZZI-DOUGLAS-MARINI',
    ]
    if variant is None:
        if h_family.upper() in hdiv_families:
            if h_family in ['RTCF', 'BDMCF']:
                variant = 'equispaced'
            else:
                variant = 'integral'
        else:
            variant = 'equispaced'
    if v_variant is None:
        v_variant = 'equispaced'
    if cell_dim == (2, 1):
        if v_family is None:
            v_family = h_family
        if v_degree is None:
            v_degree = h_degree
        h_cell, v_cell = mesh.ufl_cell().sub_cells()
        h_elt = FiniteElement(h_family, h_cell, h_degree, variant=variant)
        v_elt = FiniteElement(v_family, v_cell, v_degree, variant=v_variant)
        elt = TensorProductElement(h_elt, v_elt)
        if hdiv:
            elt = HDiv(elt)
    else:
        elt = FiniteElement(h_family, mesh.ufl_cell(), h_degree, variant=variant)

    constructor = VectorFunctionSpace if vector else FunctionSpace
    return constructor(mesh, elt, **kwargs)


def get_extruded_base_element(ufl_element):
    """
    Return UFL TensorProductElement of an extruded UFL element.

    In case of a non-extruded mesh, returns the element itself.
    """
    if isinstance(ufl_element, ufl.HDivElement):
        ufl_element = ufl_element._element
    if isinstance(ufl_element, ufl.MixedElement):
        ufl_element = ufl_element.sub_elements()[0]
    if isinstance(ufl_element, ufl.VectorElement):
        ufl_element = ufl_element.sub_elements()[0]  # take the first component
    if isinstance(ufl_element, ufl.EnrichedElement):
        ufl_element = ufl_element._elements[0]
    return ufl_element


ElementContinuity = namedtuple("ElementContinuity", ["horizontal", "vertical"])
"""
A named tuple describing the continuity of an element in the horizontal/vertical direction.

The field value is one of "cg", "hdiv", or "dg".
"""


def element_continuity(ufl_element):
    """Return an :class:`ElementContinuity` instance with the
    continuity of a given element.

    :arg ufl_element: The UFL element to determine the continuity
        of.
    :returns: A new :class:`ElementContinuity` instance.
    """
    elem = ufl_element
    elem_types = {
        'Discontinuous Lagrange': 'dg',
        'Lagrange': 'cg',
        'Raviart-Thomas': 'hdiv',
        'RTCF': 'hdiv',
        'Brezzi-Douglas-Marini': 'hdiv',
        'BDMCF': 'hdiv',
        'Q': 'cg',
        'DQ': 'dg',
    }

    base_element = get_extruded_base_element(ufl_element)
    if isinstance(elem, ufl.HDivElement):
        horiz_type = 'hdiv'
        vert_type = 'hdiv'
    elif isinstance(base_element, ufl.TensorProductElement):
        a, b = base_element.sub_elements()
        horiz_type = elem_types[a.family()]
        vert_type = elem_types[b.family()]
    else:
        horiz_type = elem_types[base_element.family()]
        vert_type = horiz_type
    return ElementContinuity(horiz_type, vert_type)


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


def get_facet_mask(function_space, facet='bottom'):
    """
    Returns the top/bottom nodes of extruded 3D elements.

    :arg function_space: Firedrake :class:`FunctionSpace` object
    :kwarg str facet: 'top' or 'bottom'

    .. note::
        The definition of top/bottom depends on the direction of the extrusion.
        Here we assume that the mesh has been extruded upwards (along positive
        z axis).
    """
    from tsfc.finatinterface import create_element as create_finat_element
    # get base element
    elem = get_extruded_base_element(function_space.ufl_element())
    assert isinstance(elem, TensorProductElement), \
        f'function space must be defined on an extruded 3D mesh: {elem}'
    # figure out number of nodes in sub elements
    h_elt, v_elt = elem.sub_elements()
    nb_nodes_h = create_finat_element(h_elt).space_dimension()
    nb_nodes_v = create_finat_element(v_elt).space_dimension()
    # compute top/bottom facet indices
    # extruded dimension is the inner loop in index
    # on interval elements, the end points are the first two dofs
    offset = 0 if facet == 'bottom' else 1
    indices = np.arange(nb_nodes_h)*nb_nodes_v + offset
    return indices


def extrude_mesh_sigma(mesh2d, n_layers, bathymetry_2d, z_stretch_fact=1.0,
                       min_depth=None):
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

    z_stretch_func = Function(fs_2d)
    if isinstance(z_stretch_fact, Function):
        assert z_stretch_fact.function_space() == fs_2d
        z_stretch_func = z_stretch_fact
    else:
        z_stretch_func.assign(z_stretch_fact)

    # number of nodes in vertical direction
    n_vert_nodes = fs_3d.finat_element.space_dimension() / fs_2d.finat_element.space_dimension()

    min_depth_arr = np.ones((n_layers+1, ))*1e22
    if min_depth is not None:
        for i, v in enumerate(min_depth):
            min_depth_arr[i] = v

    nodes = get_facet_mask(fs_3d, 'bottom')
    idx = op2.Global(len(nodes), nodes, dtype=np.int32, name='node_idx')
    min_depth_op2 = op2.Global(len(min_depth_arr), min_depth_arr, name='min_depth')
    kernel = op2.Kernel("""
        void my_kernel(double *new_coords, double *old_coords, double *bath2d, double *z_stretch, int *idx, double *min_depth) {
            for ( int d = 0; d < %(nodes)d; d++ ) {
                double s_fact = z_stretch[d];
                for ( int e = 0; e < %(v_nodes)d; e++ ) {
                    new_coords[3*(idx[d]+e) + 0] = old_coords[3*(idx[d]+e) + 0];
                    new_coords[3*(idx[d]+e) + 1] = old_coords[3*(idx[d]+e) + 1];
                    double sigma = 1.0 - old_coords[3*(idx[d]+e) + 2]; // top 0, bot 1
                    double new_z = -bath2d[d] * pow(sigma, s_fact) ;
                    int layer = fmin(fmax(round(sigma*(%(n_layers)d + 1) - 1.0), 0.0), %(n_layers)d);
                    double max_z = -min_depth[layer];
                    new_z = fmax(new_z, max_z);
                    new_coords[3*(idx[d]+e) + 2] = new_z;
                }
            }
        }""" % {'nodes': fs_2d.finat_element.space_dimension(),
                'v_nodes': n_vert_nodes,
                'n_layers': n_layers},
        'my_kernel')

    op2.par_loop(kernel, mesh.cell_set,
                 new_coordinates.dat(op2.WRITE, fs_3d.cell_node_map()),
                 coordinates.dat(op2.READ, fs_3d.cell_node_map()),
                 bathymetry_2d.dat(op2.READ, fs_2d.cell_node_map()),
                 z_stretch_func.dat(op2.READ, fs_2d.cell_node_map()),
                 idx(op2.READ),
                 min_depth_op2(op2.READ),
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


def comp_tracer_mass_2d(scalar_func, total_depth):
    """
    Computes total tracer mass in the 2D domain
    :arg scalar_func: depth-averaged scalar :class:`Function` to integrate
    :arg total_depth: scalar UFL expression (e.g. from get_total_depth())
    """
    val = assemble(scalar_func*total_depth*dx)
    return val


def comp_tracer_mass_3d(scalar_func):
    """
    Computes total tracer mass in the 3D domain

    :arg scalar_func: scalar :class:`Function` to integrate
    """
    val = assemble(scalar_func*dx)
    return val


def get_zcoord_from_mesh(zcoord, solver_parameters=None):
    """
    Evaluates z coordinates from the 3D mesh

    :arg zcoord: scalar :class:`Function` where coordinates will be stored
    """
    # TODO coordinates should probably be interpolated instead
    if solver_parameters is None:
        solver_parameters = {}
    solver_parameters.setdefault('ksp_atol', 1e-12)
    solver_parameters.setdefault('ksp_rtol', 1e-16)
    fs = zcoord.function_space()
    tri = TrialFunction(fs)
    test = TestFunction(fs)
    a = tri*test*dx
    l = fs.mesh().coordinates[2]*test*dx
    solve(a == l, zcoord, solver_parameters=solver_parameters)
    return zcoord


def compute_baroclinic_head(solver):
    r"""
    Computes the baroclinic head :math:`r` from the density field

    .. math::
        r = \frac{1}{\rho_0} \int_{z}^\eta  \rho' d\zeta.
    """
    with timed_stage('density_solve'):
        solver.density_solver.solve()
    with timed_stage('rho_integral'):
        solver.rho_integrator.solve()
        solver.fields.baroc_head_3d *= -physical_constants['rho0_inv']
    with timed_stage('int_pg_solve'):
        solver.int_pg_calculator.solve()


def extend_function_to_3d(func, mesh_extruded):
    """
    Returns a 3D view of a 2D :class:`Function` on the extruded domain.

    The 3D function resides in V x R function space, where V is the function
    space of the source function. The 3D function shares the data of the 2D
    function.
    """
    fs = func.function_space()
    assert fs.mesh().geometric_dimension() == 2, 'Function must be in 2D space'
    ufl_elem = fs.ufl_element()
    family = ufl_elem.family()
    degree = ufl_elem.degree()
    name = func.name()
    if isinstance(ufl_elem, ufl.VectorElement):
        # vector function space
        fs_extended = get_functionspace(mesh_extruded, family, degree, 'R', 0,
                                        dim=2, vector=True)
    else:
        fs_extended = get_functionspace(mesh_extruded, family, degree, 'R', 0)
    func_extended = Function(fs_extended, name=name, val=func.dat._data)
    func_extended.source = func
    return func_extended


class ExtrudedFunction(Function):
    """
    A 2D :class:`Function` that provides a 3D view on the extruded domain.

    The 3D function can be accessed as `ExtrudedFunction.view_3d`.
    The 3D function resides in V x R function space, where V is the function
    space of the source function. The 3D function shares the data of the 2D
    function.
    """
    def __init__(self, *args, mesh_3d=None, **kwargs):
        """
        Create a 2D :class:`Function` with a 3D view on extruded mesh.

        :arg mesh_3d: Extruded 3D mesh where the function will be extended to.
        """
        # create the 2d function
        super().__init__(*args, **kwargs)

        if mesh_3d is not None:
            self.view_3d = extend_function_to_3d(self, mesh_3d)


class SubdomainProjector(object):
    """
    Projector that projects the restriction of an expression to the specified subdomain.
    """
    def __init__(self, v, v_out, subdomain_id, solver_parameters=None, constant_jacobian=True):

        if isinstance(v, Expression) or \
           not isinstance(v, (ufl.core.expr.Expr, Function)):
            raise ValueError("Can only project UFL expression or Functions not '%s'" % type(v))

        self.v = v
        self.v_out = v_out

        V = v_out.function_space()

        p = TestFunction(V)
        q = TrialFunction(V)

        a = inner(p, q)*dx
        L = inner(p, v)*dx(subdomain_id)

        problem = LinearVariationalProblem(a, L, v_out,
                                           constant_jacobian=constant_jacobian)

        if solver_parameters is None:
            solver_parameters = {}

        solver_parameters.setdefault("ksp_type", "cg")

        self.solver = LinearVariationalSolver(problem,
                                              solver_parameters=solver_parameters)

    def project(self):
        """
        Apply the projection.
        """
        self.solver.solve()


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
        void my_kernel(double *func, double *zcoord) {
            for ( int d = 0; d < %(nodes)d/2; d++ ) {
                for ( int c = 0; c < %(func_dim)d; c++ ) {
                    double dz = fabs(zcoord[%(func_dim)d*(2*d+1) + c] - zcoord[%(func_dim)d*2*d + c]);
                    func[%(output_dim)d*2*d + c] = dz;
                    func[%(output_dim)d*(2*d+1) + c] = dz;
                }
            }
        }""" % {'nodes': zcoord.cell_node_map().arity,
                'func_dim': zcoord.function_space().value_size,
                'output_dim': output.function_space().value_size},
        'my_kernel')
    op2.par_loop(
        kernel, fs_out.mesh().cell_set,
        output.dat(op2.WRITE, fs_out.cell_node_map()),
        zcoord.dat(op2.READ, fs_in.cell_node_map()),
        iterate=iterate)

    return output


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


def get_facet_areas(mesh):
    """
    Compute area of each facet of `mesh`. The facet areas are stored as a HDiv trace field.

    NOTES:
      * In the 2D case, this gives edge lengths.
      * The plus sign is arbitrary and could equally well be chosen as minus.
    """
    HDivTrace = FunctionSpace(mesh, "HDiv Trace", 0)
    v, u = TestFunction(HDivTrace), TrialFunction(HDivTrace)
    facet_areas = Function(HDivTrace, name="Facet areas")
    mass_term = v('+')*u('+')*dS + v*u*ds
    rhs = v('+')*FacetArea(mesh)*dS + v*FacetArea(mesh)*ds
    solve(mass_term == rhs, facet_areas)
    return facet_areas


def get_minimum_angles_2d(mesh2d):
    """
    Compute the minimum angle in each element of a triangular mesh, `mesh2d`, using the
    cosine rule. The minimum angles are outputted as a P0 field.
    """
    try:
        assert mesh2d.topological_dimension() == 2
        assert mesh2d.ufl_cell() == ufl.triangle
    except AssertionError:
        raise NotImplementedError("Minimum angle only currently implemented for triangles.")
    edge_lengths = get_facet_areas(mesh2d)
    min_angles = Function(FunctionSpace(mesh2d, "DG", 0))
    par_loop("""for (int i=0; i<angle.dofs; i++) {

                  double min_edge = edges[0];
                  int min_index = 0;

                  for (int j=1; j<3; j++){
                    if (edges[j] < min_edge) {
                      min_edge = edges[j];
                      min_index = j;
                    }
                  }

                  double numerator = 0.0;
                  double denominator = 2.0;

                  for (int j=0; j<3; j++){
                    if (j == min_index) {
                      numerator -= edges[j]*edges[j];
                    } else {
                      numerator += edges[j]*edges[j];
                      denominator *= edges[j];
                    }
                  }
                  angle[0] = acos(numerator/denominator);
                }""", dx, {'edges': (edge_lengths, READ), 'angle': (min_angles, RW)})
    return min_angles


def get_cell_widths_2d(mesh2d):
    """
    Compute widths of mesh elements in each coordinate direction as the maximum distance
    between components of vertex coordinates.
    """
    try:
        assert mesh2d.topological_dimension() == 2
        assert mesh2d.ufl_cell() == ufl.triangle
    except AssertionError:
        raise NotImplementedError("Cell widths only currently implemented for triangles.")
    cell_widths = Function(VectorFunctionSpace(mesh2d, "DG", 0)).assign(np.finfo(0.0).min)
    par_loop("""for (int i=0; i<coords.dofs; i++) {
                  widths[0] = fmax(widths[0], fabs(coords[2*i] - coords[(2*i+2)%6]));
                  widths[1] = fmax(widths[1], fabs(coords[2*i+1] - coords[(2*i+3)%6]));
                }""", dx, {'coords': (mesh2d.coordinates, READ), 'widths': (cell_widths, RW)})
    return cell_widths


def anisotropic_cell_size(mesh):
    """
    Measure of cell size for anisotropic meshes, as described in
    Micheletti et al. (2003).

    This is used in the SUPG formulation for the 2D tracer model.

    Micheletti, Perotto and Picasso (2003). Stabilized finite
    elements on anisotropic meshes: a priori error estimates for
    the advection-diffusion and the Stokes problems. SIAM Journal
    on Numerical Analysis 41.3: 1131-1162.
    """
    try:
        from firedrake.slate.slac.compiler import PETSC_ARCH
    except ImportError:
        PETSC_ARCH = os.path.join(os.environ.get('PETSC_DIR'), os.environ.get('PETSC_ARCH'))
    include_dir = ["%s/include/eigen3" % PETSC_ARCH]

    # Compute cell Jacobian
    P0_ten = TensorFunctionSpace(mesh, "DG", 0)
    J = Function(P0_ten, name="Cell Jacobian")
    J.interpolate(Jacobian(mesh))

    # Compute minimum eigenvalue
    P0 = FunctionSpace(mesh, "DG", 0)
    min_evalue = Function(P0, name="Minimum eigenvalue")
    kernel_str = """
#include <Eigen/Dense>

using namespace Eigen;

void eigmin(double minEval[1], const double * J_) {
  Matrix<double, 2, 2, RowMajor> A;

  // Map input onto an Eigen object
  Map<Matrix<double, 2, 2, RowMajor> > J((double *)J_);

  // Compute singular value decomposition
  JacobiSVD<Matrix<double, 2, 2, RowMajor> > svd(J, ComputeFullV);

  // Get SPD part of polar decomposition
  A = svd.matrixV() * svd.singularValues().asDiagonal() * svd.matrixV().transpose();

  // Solve eigenvalue problem
  SelfAdjointEigenSolver<Matrix<double, 2, 2, RowMajor>> eigensolver(A);
  Vector2d D = eigensolver.eigenvalues();

  // Select minimum eigenvalue in modulus
  if (fabs(D(0)) < fabs(D(1))) minEval[0] = D(0);
  else minEval[0] = D(1);
}
"""
    kernel = op2.Kernel(kernel_str, 'eigmin', cpp=True, include_dirs=include_dir)
    op2.par_loop(kernel, P0_ten.node_set, min_evalue.dat(op2.RW), J.dat(op2.READ))
    return min_evalue


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
    coords = SpatialCoordinate(out_function.function_space().mesh())
    out_function.interpolate(f0 + beta * (coords[1] - y_offset))


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
    p1 = get_functionspace(mesh2d, 'CG', 1)
    boundary_markers = sorted(mesh2d.exterior_facets.unique_markers)
    boundary_len = OrderedDict()
    for i in boundary_markers:
        ds_restricted = ds(int(i))
        one_func = Function(p1).assign(1.0)
        boundary_len[i] = assemble(one_func * ds_restricted)
    return boundary_len


def select_and_move_detectors(mesh, detector_locations, detector_names=None,
                              maximum_distance=0.):
    """Select those detectors that are within the domain and/or move them to
    the nearest cell centre within the domain

    :arg mesh: Defines the domain in which detectors are to be located
    :arg detector_locations: List of x, y locations
    :arg detector_names: List of detector names (optional). If provided, a list
       of selected locations and a list of selected detector names are returned,
       otherwise only a list of selected locations is returned
    :arg maximum_distance: Detectors whose initial locations is outside the domain,
      but for which the nearest cell centre is within the specified distance, are
      moved to this location. By default a maximum distance of 0.0 is used, i.e
      no detectors are moved.
    """
    # auxilary function to test whether we can interpolate it in the given locations
    V = FunctionSpace(mesh, "CG", 1)
    v = Function(V)

    P0 = FunctionSpace(mesh, "DG", 0)
    VP0 = VectorFunctionSpace(mesh, "DG", 0)
    dist = Function(P0)
    loc_const = Constant(detector_locations[0])
    xy = SpatialCoordinate(mesh)
    p0xy = Function(VP0).interpolate(xy)

    # comparison operator that sorts on first entry first, etc.
    def min_lexsort(x, y, datatype):
        for xi, yi in zip(x, y):
            if xi < yi:
                return x
            elif yi < xi:
                return y
        # all entries the same:
        return x
    min_lexsort_op = MPI.Op.Create(min_lexsort, commute=False)

    def move_to_nearest_cell_center(location):
        loc_const.assign(location)
        dist.interpolate(dot(xy-loc_const, xy-loc_const))
        ind = dist.dat.data_ro.argmin()
        # smallest distance to a cell centre location on this process:
        local_loc = list(p0xy.dat.data_ro[ind])
        local_dist = np.sqrt(dist.dat.data_ro[ind])
        # select the smallest distance on all processes. If some distances are equal, pick a unique loc. based on lexsort
        global_dist_loc = mesh.comm.allreduce([local_dist]+local_loc, op=min_lexsort_op)
        return global_dist_loc[0], global_dist_loc[1:]

    accepted_locations = []
    accepted_names = []
    if detector_names is None:
        names = [None] * len(detector_locations)
    else:
        names = detector_names
    for location, name in zip(detector_locations, names):
        try:
            v(location)
        except PointNotInDomainError:
            moved_dist, location = move_to_nearest_cell_center(location)
            if moved_dist > maximum_distance:
                continue
        accepted_locations.append(location)
        accepted_names.append(name)

    min_lexsort_op.Free()

    if detector_names is None:
        return accepted_locations
    else:
        return accepted_locations, accepted_names


class DepthExpression:
    r"""
    Construct expression for depth depending on options

    If `not use_nonlinear_equations`, then the depth is simply the bathymetry:

  ..math::
        H = h

    Otherwise we include the free surface elevation:

  ..math::
        H = h + \eta

    and if `use_wetting_and_drying`, includes a bathymetry displacement term
    to ensure a positive depth (see Karna et al. 2011):

  ..math::
        H = h + f(h+\eta) + \eta

    where

  ..math::
        f(h+\eta) = (\sqrt{(h+\eta)^2 +\alpha^2} - (h+\eta))/2

    This introduces a wetting-drying parameter :math:`\alpha`, with dimensions
    of length. The value for :math:`\alpha` is specified by
    :attr:`.ModelOptions.wetting_and_drying_alpha`, in units of meters. The
    default value is 0.5, but the appropriate value is problem specific and
    should be set by the user.
    """

    def __init__(self, bathymetry_2d, use_nonlinear_equations=True,
                 use_wetting_and_drying=False, wetting_and_drying_alpha=0.5):
        self.bathymetry_2d = bathymetry_2d
        self.use_nonlinear_equations = use_nonlinear_equations
        self.use_wetting_and_drying = use_wetting_and_drying
        self.wetting_and_drying_alpha = wetting_and_drying_alpha

    def wd_bathymetry_displacement(self, eta):
        """
        Returns wetting and drying bathymetry displacement as described in:
        Karna et al.,  2011.
        :arg eta: current elevation as UFL expression
        """
        if self.use_wetting_and_drying:
            H = self.bathymetry_2d + eta
            return 0.5 * (sqrt(H ** 2 + self.wetting_and_drying_alpha ** 2) - H)
        else:
            return 0

    def get_total_depth(self, eta):
        """
        Returns total water column depth based on options
        :arg eta: current elevation as UFL expression
        """
        if self.use_nonlinear_equations:
            total_h = self.bathymetry_2d + eta + self.wd_bathymetry_displacement(eta)
        else:
            total_h = self.bathymetry_2d
        return total_h


class VorticityCalculator2D(object):
    r"""
    Linear solver for recovering fluid vorticity.

    It is recommended that the vorticity is sought
    in :math:`\mathbb P1` space.

    :arg vorticity_2d: :class:`Function` to hold calculated vorticity.
    :arg solver_obj: :class:`FlowSolver2d` object.
    :kwargs: to be passed to the :class:`LinearVariationalSolver`.
    """
    def __init__(self, vorticity_2d, solver_obj, **kwargs):
        fs = vorticity_2d.function_space()
        if element_continuity(fs.ufl_element()).horizontal != 'cg':
            raise NotImplementedError
        n = FacetNormal(fs.mesh())

        # Weak formulation
        test = TestFunction(fs)
        uv, elev = split(solver_obj.fields.solution_2d)
        a = TrialFunction(fs)*test*dx
        L = -inner(perp(uv), grad(test))*dx \
            + dot(perp(uv), n)*test*ds

        # Setup vorticity solver
        prob = LinearVariationalProblem(a, L, vorticity_2d)
        sp = {
            'ksp_type': 'gmres',
            'ksp_gmres_restart': 20,
            'ksp_rtol': 1.0e-05,
            'pc_type': 'sor',
        }
        kwargs.setdefault('solver_parameters', sp)
        self.solver = LinearVariationalSolver(prob, **kwargs)

        if 'vorticity_2d' not in solver_obj.options.fields_to_export:
            print_output("NOTE: 'vorticity_2d' will be calculated but not exported.")

    def solve(self):
        self.solver.solve()


class DepthIntegratedPoissonSolver(object):
    r"""
    Construct solvers for Poisson equation and updating velocities

    Poisson equation is related to 2d NH SWE system

    Non-hydrostatic pressure :math:`q` is obtained from the generic form of equation

    .. math::
        \nabla \cdot \nabla q^{n+1/2} + A \cdot \nabla q^{n+1/2} + B q^{n+1/2} + C = 0

    The parameter terms A, B and C are defined as

    .. math::
        A = \frac{\nabla (\eta^* - d)}{H^*}
        B = \nabla A - \frac{4}{(H^*)^2}
        C = -2 \frac{\rho_0}{\Delta t} ( \nabla \cdot \bar{\textbf{u}}^* + 2 \frac{\bar{w} - w_b}{H^*} )

    where the :math:`H = \eta + d` denotes the water depth
    and the superscript star symbol represents the intermediate level of terms.
    """
    def __init__(self, q_2d, uv_2d, w_2d, elev_2d, depth, dt, bnd_functions=None, solver_parameters=None):
        if solver_parameters is None:
            solver_parameters = {'snes_type': 'ksponly',
                                 'ksp_type': 'preonly',
                                 'mat_type': 'aij',
                                 'pc_type': 'lu'}
        rho_0 = physical_constants['rho0']
        self.q_2d = q_2d
        self.uv_2d = uv_2d
        self.w_2d = w_2d
        self.elev_2d = elev_2d
        self.depth = depth
        self.dt = dt
        self.bnd_functions = bnd_functions

        fs_q = self.q_2d.function_space()
        test_q = TestFunction(fs_q)
        normal = FacetNormal(fs_q.mesh())
        boundary_markers = fs_q.mesh().exterior_facets.unique_markers

        bath_2d = self.depth.bathymetry_2d
        h_star = self.depth.get_total_depth(self.elev_2d)
        w_b = -dot(self.uv_2d, grad(bath_2d))  # TODO account for bed movement

        # weak form of `div(grad(q^{n+1/2}))`
        f = -dot(grad(self.q_2d), grad(test_q))*dx
        # weak form of `dot(grad(self.elev_2d - bath_2d)/h_star, grad(q^{n+1/2}))`
        grad_hori = grad(self.elev_2d - bath_2d)
        f += dot(grad_hori/h_star, grad(self.q_2d))*test_q*dx
        # weak form of `q^{n+1/2}*div(grad(self.elev_2d - bath_2d))/h_star`
        f += -dot(grad(self.q_2d*test_q/h_star), grad_hori)*dx
        # weak form of `-q^{n+1/2}*(dot(grad(self.elev_2d - bath_2d), grad(h_star)) + 4)/h_star**2`
        f += -(dot(grad_hori, grad(h_star)) + 4.)/h_star**2*self.q_2d*test_q*dx
        # weak form of `-2.*rho_0/self.dt*(div(self.uv_2d) + (self.w_2d - w_b)/(0.5*h_star))`
        const = 2.*rho_0/self.dt
        f += const*(dot(grad(test_q), self.uv_2d)*dx - 2.*(self.w_2d - w_b)/h_star*test_q*dx)

        # boundary conditions
        bcs = []
        for bnd_marker in boundary_markers:
            func = self.bnd_functions['shallow_water'].get(bnd_marker)
            ds_bnd = ds(int(bnd_marker))
            if func is not None:  # e.g. inlet flow, TODO be more precise
                bc = DirichletBC(fs_q, 0., int(bnd_marker))
                bcs.append(bc)
                f += self.q_2d*test_q/h_star*dot(grad_hori, normal)*ds_bnd
                f += -const*dot(self.uv_2d, normal)*test_q*ds_bnd

        prob_q = NonlinearVariationalProblem(f, self.q_2d)
        self.solver_q = NonlinearVariationalSolver(
            prob_q,
            solver_parameters=solver_parameters,
            options_prefix='poisson_solver'
        )
        # horizontal velocity updater
        fs_u = self.uv_2d.function_space()
        tri_u = TrialFunction(fs_u)
        test_u = TestFunction(fs_u)
        a_u = inner(tri_u, test_u)*dx
        l_u = dot(self.uv_2d - 0.5*self.dt/rho_0*(grad(self.q_2d) + grad_hori/h_star*self.q_2d), test_u)*dx
        prob_u = LinearVariationalProblem(a_u, l_u, self.uv_2d)
        self.solver_u = LinearVariationalSolver(prob_u)
        # vertical velocity updater
        fs_w = self.w_2d.function_space()
        tri_w = TrialFunction(fs_w)
        test_w = TestFunction(fs_w)
        a_w = inner(tri_w, test_w)*dx
        l_w = dot(self.w_2d + self.dt/rho_0*(self.q_2d/h_star), test_w)*dx
        prob_w = LinearVariationalProblem(a_w, l_w, self.w_2d)
        self.solver_w = LinearVariationalSolver(prob_w)

    def solve(self, solve_w=True):
        # solve non-hydrostatic pressure q
        self.solver_q.solve()
        # update horizontal velocity uv_2d
        self.solver_u.solve()
        # update vertical velocity w_2d
        if solve_w:
            self.solver_w.solve()
