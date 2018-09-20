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
    fam_list = family
    if not isinstance(family, list):
        fam_list = [family]
    ufl_elem = fs.ufl_element()
    if isinstance(ufl_elem, ufl.VectorElement):
        ufl_elem = ufl_elem.sub_elements()[0]

    if ufl_elem.family() == 'TensorProductElement':
        # extruded mesh
        A, B = ufl_elem.sub_elements()
        assert A.family() in fam_list,\
            'horizontal space must be one of {0:s}'.format(fam_list)
        assert B.family() in fam_list,\
            'vertical space must be {0:s}'.format(fam_list)
        assert A.degree() == degree,\
            'degree of horizontal space must be {0:d}'.format(degree)
        assert B.degree() == degree,\
            'degree of vertical space must be {0:d}'.format(degree)
    else:
        # assume 2D mesh
        assert ufl_elem.family() in fam_list,\
            'function space must be one of {0:s}'.format(fam_list)
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
    def __init__(self, p1dg_space, elem_height=None, time_dependent_mesh=True):
        """
        :arg p1dg_space: P1DG function space
        """

        assert_function_space(p1dg_space, ['Discontinuous Lagrange', 'DQ'], 1)
        self.is_vector = p1dg_space.value_size > 1
        if self.is_vector:
            p1dg_scalar_space = FunctionSpace(p1dg_space.mesh(), 'DG', 1)
            super(VertexBasedP1DGLimiter, self).__init__(p1dg_scalar_space)
            self.detJ = Function(p1dg_scalar_space, name='detJ')
        else:
            super(VertexBasedP1DGLimiter, self).__init__(p1dg_space)
            self.detJ = Function(p1dg_space, name='detJ')
        self.mesh = self.P0.mesh()
        self.is_2d = self.mesh.geometric_dimension() == 2
        self.time_dependent_mesh = time_dependent_mesh
        self.elem_height = elem_height
        if not self.is_2d:
            assert self.elem_height is not None, 'Element height field must be provided'

    def _compute_detJ(self):
        """
        Computes detJ of the current mesh
        """
        J = Jacobian(self.mesh)
        f = abs(det(J))
        self.detJ.interpolate(f)

    def _construct_centroid_solver(self):
        """
        Constructs a linear problem for computing the centroids

        :return: LinearSolver instance
        """
        u = TrialFunction(self.P0)
        v = TestFunction(self.P0)
        self.a_form = u * v * dx
        a = assemble(self.a_form)
        return LinearSolver(a, solver_parameters={'ksp_type': 'preonly',
                                                  'pc_type': 'bjacobi',
                                                  'sub_pc_type': 'ilu'})

    def _update_centroids(self, field):
        """
        Update centroid values
        """
        b = assemble(TestFunction(self.P0) * field * dx)
        if self.time_dependent_mesh:
            assemble(self.a_form, self.centroid_solver.A)
            self.centroid_solver.A.force_evaluation()
        self.centroid_solver.solve(self.centroids, b)

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

        entity_dim = 1 if self.is_2d else (1, 1)  # get 1D or vertical facets
        boundary_dofs = entity_support_dofs(self.P1DG.finat_element, entity_dim)
        local_facet_nodes = np.array([boundary_dofs[e] for e in sorted(boundary_dofs.keys())])
        n_bnd_nodes = local_facet_nodes.shape[1]
        local_facet_idx = op2.Global(local_facet_nodes.shape, local_facet_nodes, dtype=np.int32, name='local_facet_idx')
        if self.is_2d:
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
        else:
            code = """
                void my_kernel(double **qmax, double **qmin, double **field, double **height, unsigned int *facet, unsigned int *local_facet_idx)
                {
                    double face_mean = 0.0;
                    double area = 0.0;
                    for (int i = 0; i < %(nnodes)d; i++) {
                        unsigned int idx = local_facet_idx[facet[0]*%(nnodes)d + i];
                        face_mean += field[idx][0]*height[idx][0];
                        area += height[idx][0];
                    }
                    face_mean /= area;
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
                         self.elem_height.dat(op2.RW, self.elem_height.exterior_facet_node_map()),
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

    def _apply(self, field):
        self.compute_bounds(field)
        self._compute_detJ()
        self._limit_kernel = """
double alpha = 1.0;
double qavg = qbar[0][0];
for (int i=0; i < q.dofs; i++) {
    if (q[i][0] > qavg)
        alpha = fmin(alpha, fmin(1, (qmax[i][0] - qavg)/(q[i][0] - qavg)));
    else if (q[i][0] < qavg)
        alpha = fmin(alpha, fmin(1, (qavg - qmin[i][0])/(qavg - q[i][0])));
}
for (int i=0; i<q.dofs; i++) {
    q[i][0] = qavg + alpha*(q[i][0] - qavg);
}
                             """
        self._optimal_kernel = """
double qavg = qbar[0][0];
bool fixed[q.dofs] = {0};
for (int iter=0; iter < 10; iter++) {
    bool no_violation = 1;
    double deviation = 0.0;
    // check violations
    for (int i=0; i < q.dofs; i++) {
        if (q[i][0] > qmax[i][0]) {
            double new_val = fmax(qmax[i][0], qavg);
            deviation += (q[i][0] - new_val)*detJ[i][0];
            q[i][0] = new_val;
            fixed[i] = 1;
            no_violation = 0;
        }
        else if (q[i][0] < qmin[i][0]) {
            double new_val = fmin(qmin[i][0], qavg);
            deviation += (q[i][0] - new_val)*detJ[i][0];
            q[i][0] = new_val;
            fixed[i] = 1;
            no_violation = 0;
        }
    }
    if (no_violation)
        break;
    // redistribute
    int nfree = q.dofs;
    for (int i=0; i < q.dofs; i++) {
        nfree -= (int)fixed[i];
    }
    for (int i=0; i < q.dofs; i++) {
        if (fixed[i])
            continue;
        q[i][0] += deviation/nfree/detJ[i][0];
    }
}
                             """
        par_loop(self._optimal_kernel, dx,
                 {"qbar": (self.centroids, READ),
                  "q": (field, RW),
                  "detJ": (self.detJ, READ),
                  "qmax": (self.max_field, READ),
                  "qmin": (self.min_field, READ)})

    def apply(self, field):
        """
        Applies the limiter on the given field (in place)

        :arg field: :class:`Function` to limit
        """
        with timed_stage('limiter'):
            if self.is_vector:
                tmp_func = self.P1DG.get_work_function()
                fs = field.function_space()
                for i in range(fs.value_size):
                    tmp_func.dat.data_with_halos[:] = field.dat.data_with_halos[:, i]
                    #super(VertexBasedP1DGLimiter, self).apply(tmp_func)
                    self._apply(tmp_func)
                    field.dat.data_with_halos[:, i] = tmp_func.dat.data_with_halos[:]
                self.P1DG.restore_work_function(tmp_func)
            else:
                #super(VertexBasedP1DGLimiter, self).apply(field)
                self._apply(field)
