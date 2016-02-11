"""
Slope limiter implementation.

Tuomas Karna 2015-08-26
"""
from utility import *


def assert_function_space(fs, family, degree):
    """
    Checks the family and degree of function space.

    Raises AssertionError if function space differs.
    If the function space lies on an extruded mesh, checks both spaces of the
    outer product.
    """
    ufl_elem = fs.ufl_element()
    if ufl_elem.family() == 'TensorProductElement':
        # extruded mesh
        assert ufl_elem._A.family() == family,\
            'horizontal space must be {0:s}'.format(family)
        assert ufl_elem._B.family() == family,\
            'vertical space must be {0:s}'.format(family)
        assert ufl_elem._A.degree() == degree,\
            'degree of horizontal space must be {0:d}'.format(degree)
        assert ufl_elem._B.degree() == degree,\
            'degree of vertical space must be {0:d}'.format(degree)
    else:
        # assume 2D mesh
        assert ufl_elem.family() == family,\
            'function space must be {0:s}'.format(family)
        assert ufl_elem.degree() == degree,\
            'degree of function space must be {0:d}'.format(degree)


class VertexBasedP1DGLimiter(object):
    """
    Vertex based limiter for P1DG tracer fields.

    Based on firedrake implementation by Andrew McRae.

    [1] Kuzmin Dmitri (2010). A vertex-based hierarchical slope limiter
    for p-adaptive discontinuous Galerkin methods. Journal of Computational
    and Applied Mathematics, 233(12):3077-3085.
    http://dx.doi.org/10.1016/j.cam.2009.05.028
    """
    def __init__(self, p1dg_space, p1cg_space, p0_space):
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
        assert_function_space(p0_space, 'Discontinuous Lagrange', 0)
        assert_function_space(p1cg_space, 'Lagrange', 1)
        self.P1DG = p1dg_space
        self.P0 = p0_space
        self.P1CG = p1cg_space
        # create auxiliary functions
        # P0 field containing the center (mean) values of elements
        self.centroids = Function(self.P0, name='limiter_p1_dg-centroid')
        # Allowed min/max values for each P1 node in the mesh
        self.max_field = Function(self.P1CG, name='limiter_p1_dg-maxvalue')
        self.min_field = Function(self.P1CG, name='limiter_p1_dg-minvalue')
        # store solvers for computing centroids
        self.centroid_solvers = {}

    def _construct_average_operator(self, field):
        """
        Constructs a linear problem for computing the centroids and
        adds it to the cache.
        """
        if field not in self.centroid_solvers:
            tri = TrialFunction(self.P0)
            test = TestFunction(self.P0)

            a = tri*test*dx
            l = field*test*dx

            params = {'ksp_type': 'preonly'}
            problem = LinearVariationalProblem(a, l, self.centroids)
            solver = LinearVariationalSolver(problem,
                                             solver_parameters=params)
            self.centroid_solvers[field] = solver
        return self.centroid_solvers[field]

    def _update_centroids(self, field):
        """
        Re-compute element centroid values
        """
        solver = self._construct_average_operator(field)
        solver.solve()

    def _update_min_max_values(self, field):
        """
        Re-compute min/max values of all neighbouring centroids
        """
        self.max_field.assign(-1e300)  # small number
        self.min_field.assign(1e300)  # big number

        # Set up fields containing max/min of neighbouring cell averages
        par_loop("""
    for (int i=0; i<qmax.dofs; i++) {
        qmax[i][0] = fmax(qmax[i][0], centroids[0][0]);
        qmin[i][0] = fmin(qmin[i][0], centroids[0][0]);
    }
    """,
                 dx,
                 {'qmax': (self.max_field, RW),
                  'qmin': (self.min_field, RW),
                  'centroids': (self.centroids, READ)})

    def _apply_limiter(self, field):
        """
        Applies the limiter to the given field.
        DG field values are limited to be within min/max of the neighbouring element centroids.
        """
        par_loop("""
    double alpha = 1.0;
    double cellavg = centroids[0][0];
    for (int i=0; i<q.dofs; i++) {
        if (q[i][0] > cellavg)
            alpha = fmin(alpha, fmin(1, (qmax[i][0] - cellavg)/(q[i][0] - cellavg)));
        else if (q[i][0] < cellavg)
            alpha = fmin(alpha, fmin(1, (cellavg - qmin[i][0])/(cellavg - q[i][0])));
    }
    for (int i=0; i<q.dofs; i++) {
        q[i][0] = cellavg + alpha*(q[i][0] - cellavg);
    }
    """,
                 dx,
                 {'q': (field, RW),
                  'qmax': (self.max_field, READ),
                  'qmin': (self.min_field, READ),
                  'centroids': (self.centroids, READ)})

    def apply(self, field):
        """
        Applies the limiter to the given field.
        """
        assert field.function_space() == self.P1DG,\
            'Given field belongs to wrong function space'
        self._update_centroids(field)
        self._update_min_max_values(field)
        self._apply_limiter(field)
