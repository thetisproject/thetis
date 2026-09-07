"""
Slope limiters for discontinuous fields
"""
import pyop3 as op3
from .utility import *
from firedrake import VertexBasedLimiter
from pyop2.profiling import timed_stage
import numpy


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
    if isinstance(ufl_elem, firedrake.VectorElement):
        ufl_elem = ufl_elem.sub_elements[0]

    if isinstance(ufl_elem, firedrake.TensorProductElement):
        # extruded mesh
        A, B = ufl_elem.factor_elements
        assert A.family() in fam_list, \
            'horizontal space must be one of {0:s}'.format(fam_list)
        assert B.family() in fam_list, \
            'vertical space must be {0:s}'.format(fam_list)
        assert A.degree() == degree, \
            'degree of horizontal space must be {0:d}'.format(degree)
        assert B.degree() == degree, \
            'degree of vertical space must be {0:d}'.format(degree)
    else:
        # assume 2D mesh
        assert ufl_elem.family() in fam_list, \
            'function space must be one of {0:s}'.format(fam_list)
        assert ufl_elem.degree() == degree, \
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
    def __init__(self, p1dg_space, time_dependent_mesh=True):
        """
        :arg p1dg_space: P1DG function space
        """

        assert_function_space(p1dg_space, ['Discontinuous Lagrange', 'DQ'], 1)
        self.is_vector = p1dg_space.value_size > 1
        if self.is_vector:
            p1dg_scalar_space = FunctionSpace(p1dg_space.mesh(), 'DG', 1)
            super(VertexBasedP1DGLimiter, self).__init__(p1dg_scalar_space)
        else:
            super(VertexBasedP1DGLimiter, self).__init__(p1dg_space)
        self.mesh = self.P0.mesh()
        self.is_2d = self.mesh.geometric_dimension == 2
        self.time_dependent_mesh = time_dependent_mesh

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
            assemble(self.a_form, tensor=self.centroid_solver.A)
        self.centroid_solver.solve(self.centroids, b)

    @PETSc.Log.EventDecorator("thetis.VertexBasedP1DGLimiter.compute_bounds")
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
        local_facet_nodes = numpy.array([boundary_dofs[e] for e in sorted(boundary_dofs.keys())])
        n_bnd_nodes = local_facet_nodes.shape[1]
        local_facet_idx = op3.Dat.from_array(local_facet_nodes.ravel(), name='local_facet_idx', constant=True, rank_equal=True)
        code = """
                double face_mean = 0.0;
                for (int i = 0; i < %(nnodes)d; i++) {
                    unsigned int idx = local_facet_idx[facet[0]*%(nnodes)d + i];
                    face_mean += field[idx];
                }
                face_mean /= %(nnodes)d;
                for (int i = 0; i < %(nnodes)d; i++) {
                    unsigned int idx = local_facet_idx[facet[0]*%(nnodes)d + i];
                    qmax[idx] = fmax(qmax[idx], face_mean);
                    qmin[idx] = fmin(qmin[idx], face_mean);
                }"""
        bnd_kernel = op3.Function.from_c_string(
            "my_kernel",
            code % {'nnodes': n_bnd_nodes},
            [
                ("qmax", op3.ScalarType, op3.MAX_RW),
                ("qmin", op3.ScalarType, op3.MIN_RW),
                ("field", op3.ScalarType, op3.READ),
                ("facet", field.dat.dtype, op3.READ),
                ("local_facet_idx", local_facet_idx.dtype, op3.READ),
            ],
        )
        op3.loop(
            f := self.P1DG.mesh().iter("exterior_facet_vert"),
            bnd_kernel(
                pack(self.max_field, f),
                pack(self.min_field, f),
                pack(field, f),
                self.P1DG.mesh().exterior_facet_vert_local_facet_indices[f],
                local_facet_idx,
            ),
            eager=True,
        )
        if not self.is_2d:
            # Add nodal values from surface/bottom boundaries
            # NOTE calling firedrake par_loop with measure=ds_t raises an error
            bottom_nodes = get_facet_mask(self.P1CG, 'bottom')
            top_nodes = get_facet_mask(self.P1CG, 'top')
            bottom_idx = op3.Dat.from_array(bottom_nodes, name='node_idx', constant=True, rank_equal=True)
            top_idx = op3.Dat.from_array(top_nodes, name='node_idx', constant=True, rank_equal=True)
            code = """
                    double face_mean = 0;
                    for (int i=0; i<%(nnodes)d; i++) {
                        face_mean += field[idx[i]];
                    }
                    face_mean /= %(nnodes)d;
                    for (int i=0; i<%(nnodes)d; i++) {
                        qmax[idx[i]] = fmax(qmax[idx[i]], face_mean);
                        qmin[idx[i]] = fmin(qmin[idx[i]], face_mean);
                    }"""
            kernel = op3.Function.from_c_string(
                "my_kernel",
                code % {'nnodes': len(bottom_nodes)},
                [
                    ("qmax", op3.ScalarType, op3.MAX_RW),
                    ("qmin", op3.ScalarType, op3.MIN_RW),
                    ("field", op3.ScalarType, op3.READ),
                    ("idx", "int", op3.READ),
                ],
            )

            op3.loop(
                f := self.mesh.iter("exterior_facet_bottom"),
                kernel(
                    pack(self.max_field, f),
                    pack(self.min_field, f),
                    pack(field, f),
                    bottom_idx,
                ),
                eager=True,
            )

            op3.loop(
                f := self.mesh.iter("exterior_facet_top"),
                kernel(
                    pack(self.max_field, f),
                    pack(self.min_field, f),
                    pack(field, f),
                    top_idx,
                ),
                eager=True,
            )

    @PETSc.Log.EventDecorator("thetis.VertexBasedP1DGLimiter.apply")
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
                    super(VertexBasedP1DGLimiter, self).apply(tmp_func)
                    field.dat.data_with_halos[:, i] = tmp_func.dat.data_with_halos[:]
                self.P1DG.restore_work_function(tmp_func)
            else:
                super(VertexBasedP1DGLimiter, self).apply(field)
