"""
Slope limiters for discontinuous fields
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

    :arg fs: function space
    :arg string family: name of element family
    :arg int degree: polynomial degree of the function space
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
    Vertex based limiter for P1DG tracer fields, see Kuzmin (2010)

    .. note::
        Currently only scalar fields are supported

    Kuzmin (2010). A vertex-based hierarchical slope limiter
    for p-adaptive discontinuous Galerkin methods. Journal of Computational
    and Applied Mathematics, 233(12):3077-3085.
    http://dx.doi.org/10.1016/j.cam.2009.05.028
    """
    def __init__(self, p1dg_space):
        """
        :arg p1dg_space: P1DG function space
        """

        assert_function_space(p1dg_space, 'Discontinuous Lagrange', 1)
        super(VertexBasedP1DGLimiter, self).__init__(p1dg_space)
        self.mesh = self.P0.mesh()
        self.is_2d = self.mesh.geometric_dimension() == 2

    def _update_centroids(self, field):
        """
        Re-compute element centroid values

        :arg field: :class:`Function` to limit
        """
        tri = TrialFunction(self.P0)
        test = TestFunction(self.P0)

        a = assemble(tri*test*dx)
        l = assemble(field*test*dx)
        solve(a, self.centroids, l)

    def compute_bounds(self, field):
        """
        Re-compute min/max values of all neighbouring centroids

        :arg field: :class:`Function` to limit
        """
        # Call general-purpose bound computation.
        super(VertexBasedP1DGLimiter, self).compute_bounds(field)

        # Add the average of lateral boundary facets to min/max fields
        # NOTE this just computes the arithmetic mean of nodal values on the facet,
        # which in general is not equivalent to the mean of the field over the bnd facet.
        # This is OK for P1DG triangles, but not exact for the extruded case (quad facets)
        from finat.finiteelementbase import entity_support_dofs

        if self.is_2d:
            entity_dim = 1  # get 1D facets
        else:
            entity_dim = (1, 1)  # get vertical facets
        boundary_dofs = entity_support_dofs(self.P1DG.finat_element, entity_dim)
        local_facet_nodes = np.array([boundary_dofs[e] for e in sorted(boundary_dofs.keys())])
        n_bnd_nodes = local_facet_nodes.shape[1]
        local_facet_idx = op2.Global(local_facet_nodes.shape, local_facet_nodes, dtype=np.int32, name='local_facet_idx')
        code = """
            void my_kernel(double **qmax, double **qmin, double **field, unsigned int *facet, unsigned int *local_facet_idx)
            {
                double face_mean = 0.0;
                for (int i = 0; i < %(nnodes)d; i++) {
                    unsigned int idx = local_facet_idx[facet[0]*%(nnodes)d + i];
                    face_mean += field[idx][0];
                }
                face_mean /= %(nnodes)d;
                for (int i = 0; i < %(nnodes)d; i++) {
                    unsigned int idx = local_facet_idx[facet[0]*%(nnodes)d + i];
                    qmax[idx][0] = fmax(qmax[idx][0], face_mean);
                    qmin[idx][0] = fmin(qmin[idx][0], face_mean);
                }
            }"""
        bnd_kernel = op2.Kernel(code % {'nnodes': n_bnd_nodes}, 'my_kernel')
        op2.par_loop(bnd_kernel,
                     self.P1DG.mesh().exterior_facets.set,
                     self.max_field.dat(op2.RW, self.max_field.exterior_facet_node_map()),
                     self.min_field.dat(op2.RW, self.min_field.exterior_facet_node_map()),
                     field.dat(op2.RW, field.exterior_facet_node_map()),
                     self.P1DG.mesh().exterior_facets.local_facet_dat(op2.READ),
                     local_facet_idx(op2.READ))
        if not self.is_2d:
            # Add nodal values from surface/bottom boundaries
            # NOTE calling firedrake par_loop with measure=ds_t raises an error
            bottom_nodes = get_facet_mask(self.P1CG, 'geometric', 'bottom')
            top_nodes = get_facet_mask(self.P1CG, 'geometric', 'top')
            bottom_idx = op2.Global(len(bottom_nodes), bottom_nodes, dtype=np.int32, name='node_idx')
            top_idx = op2.Global(len(top_nodes), top_nodes, dtype=np.int32, name='node_idx')
            code = """
                void my_kernel(double **qmax, double **qmin, double **field, int *idx) {
                    double face_mean = 0;
                    for (int i=0; i<%(nnodes)d; i++) {
                        face_mean += field[idx[i]][0];
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
        """
        Applies the limiter on the given field (in place)

        :arg field: :class:`Function` to limit
        """
        with timed_stage('limiter'):
            super(VertexBasedP1DGLimiter, self).apply(field)
