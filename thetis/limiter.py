"""
Slope limiter implementation.

Tuomas Karna 2015-08-26
"""
from __future__ import absolute_import
from .utility import *
from .firedrake import VertexBasedLimiter
import ufl
from pyop2.profiling import timed_region, timed_function, timed_stage  # NOQA


def assert_function_space(fs, family, degree):
    """
    Checks the family and degree of function space.

    Raises AssertionError if function space differs.
    If the function space lies on an extruded mesh, checks both spaces of the
    outer product.
    """
    ufl_elem = fs.ufl_element()
    if isinstance(ufl_elem, ufl.VectorElement):
        ufl_elem = ufl_elem.sub_elements()[0]

    if ufl_elem.family() == 'TensorProductElement':
        # extruded mesh
        A, B = ufl_elem.sub_elements()
        assert A.family() == family,\
            'horizontal space must be {0:s}'.format(family)
        assert B.family() == family,\
            'vertical space must be {0:s}'.format(family)
        assert A.degree() == degree,\
            'degree of horizontal space must be {0:d}'.format(degree)
        assert B.degree() == degree,\
            'degree of vertical space must be {0:d}'.format(degree)
    else:
        # assume 2D mesh
        assert ufl_elem.family() == family,\
            'function space must be {0:s}'.format(family)
        assert ufl_elem.degree() == degree,\
            'degree of function space must be {0:d}'.format(degree)


class VertexBasedP1DGLimiter(VertexBasedLimiter):
    """
    Vertex based limiter for P1DG tracer fields.

    Based on firedrake implementation by Andrew McRae.

    [1] Kuzmin Dmitri (2010). A vertex-based hierarchical slope limiter
    for p-adaptive discontinuous Galerkin methods. Journal of Computational
    and Applied Mathematics, 233(12):3077-3085.
    http://dx.doi.org/10.1016/j.cam.2009.05.028
    """
    def __init__(self, p1dg_space):
        """
        Initialize limiter.

        Parameters
        ----------

        p1dg_space : FunctionSpace instance
            P1DG function space where the scalar field belongs to
        p1cg_space : FunctionSpace instance
            Corresponding continuous function space (for min/max limits)
        p0_space : FunctionSpace instance
            Corresponding P0 function space (for centroids)

        """

        assert_function_space(p1dg_space, 'Discontinuous Lagrange', 1)
        super(VertexBasedP1DGLimiter, self).__init__(p1dg_space)
        self.mesh = self.P0.mesh()
        self.is_2d = self.mesh.geometric_dimension() == 2

    def _update_centroids(self, field):
        """
        Re-compute element centroid values
        """
        tri = TrialFunction(self.P0)
        test = TestFunction(self.P0)

        a = assemble(tri*test*dx)
        l = assemble(field*test*dx)
        solve(a, self.centroids, l)

    def compute_bounds(self, field):
        """
        Re-compute min/max values of all neighbouring centroids
        """
        # Call general-purpose bound computation.
        super(VertexBasedP1DGLimiter, self).compute_bounds(field)
        # NOTE This does not limit solution at lateral boundaries at all
        # NOTE Omit for now
        # # Add nodal values from lateral boundaries
        # par_loop("""
        #     for (int i=0; i<qmax.dofs; i++) {
        #         qmax[i][0] = fmax(qmax[i][0], field[i][0]);
        #         qmin[i][0] = fmin(qmin[i][0], field[i][0]);
        #     }""",
        #          ds,
        #          {'qmax': (self.max_field, RW),
        #           'qmin': (self.min_field, RW),
        #           'field': (field, READ)})

        if not self.is_2d:
            # Add nodal values from surface/bottom boundaries
            # NOTE calling firedrake par_loop with measure=ds_t raises an error
            bottom_nodes = self.P1CG.bt_masks['geometric'][0]
            top_nodes = self.P1CG.bt_masks['geometric'][1]
            bottom_idx = op2.Global(len(bottom_nodes), bottom_nodes, dtype=np.int32, name='node_idx')
            top_idx = op2.Global(len(top_nodes), top_nodes, dtype=np.int32, name='node_idx')
            code = """
                void my_kernel(double **qmax, double **qmin, double **centroids, int *idx) {
                    double face_mean = 0;
                    for (int i=0; i<%(nnodes)d; i++) {
                        face_mean += centroids[idx[i]][0];
                    }
                    face_mean /= %(nnodes)d;
                    for (int i=0; i<%(nnodes)d; i++) {
                        qmax[idx[i]][0] = fmax(qmax[idx[i]][0], face_mean);
                        qmin[idx[i]][0] = fmin(qmin[idx[i]][0], face_mean);
                    }
                }"""
            kernel = op2.Kernel(code % {'nnodes': len(bottom_nodes)}, 'my_kernel')

            op2.par_loop(kernel, self.mesh.cell_set,
                         self.max_field.dat(op2.WRITE, self.max_field.function_space().cell_node_map()),
                         self.min_field.dat(op2.WRITE, self.min_field.function_space().cell_node_map()),
                         field.dat(op2.READ, field.function_space().cell_node_map()),
                         bottom_idx(op2.READ),
                         iterate=op2.ON_BOTTOM)

            op2.par_loop(kernel, self.mesh.cell_set,
                         self.max_field.dat(op2.WRITE, self.max_field.function_space().cell_node_map()),
                         self.min_field.dat(op2.WRITE, self.min_field.function_space().cell_node_map()),
                         field.dat(op2.READ, field.function_space().cell_node_map()),
                         top_idx(op2.READ),
                         iterate=op2.ON_TOP)

    def apply(self, field):
        with timed_stage('limiter'):
            super(VertexBasedP1DGLimiter, self).apply(field)
