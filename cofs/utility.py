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
from cofs.fieldDefs import fieldMetadata

comm = op2.MPI.comm
commrank = op2.MPI.comm.rank

# NOTE some functions now depend on flowSolver object
# TODO move those functions in the solver class


class frozenClass(object):
    """A class where creating a new attribute will raise an exception if _isfrozen == True"""
    _isfrozen = False

    def __setattr__(self, key, value):
        if self._isfrozen and not hasattr(self, key):
            raise TypeError('Adding new attribute "{:}" to {:} class is forbidden'.format(key, self.__class__.__name__))
        super(frozenClass, self).__setattr__(key, value)


class sumFunction(object):
    """
    Helper class to keep track of sum of Coefficients.
    """
    def __init__(self):
        """
        Initialize empty sum.

        get operation returns Constant(0)
        """
        self.coeffList = []

    def add(self, coeff):
        """
        Adds a coefficient to self
        """
        if coeff is None:
            return
        # classes = (Function, Constant, ufl.algebra.Sum, ufl.algebra.Product)
        # assert not isinstance(coeff, classes), \
            # ('bad argument type: ' + str(type(coeff)))
        self.coeffList.append(coeff)

    def getSum(self):
        """
        Returns a sum of all added Coefficients
        """
        if len(self.coeffList) == 0:
            return None
        return sum(self.coeffList)


class equation(object):
    """Base class for all equations"""
    # TODO move to equation.py
    def mass_matrix(self, *args, **kwargs):
        """Returns weak form for left hand side."""
        raise NotImplementedError(('This method must be implemented '
                                   'in the derived class'))

    def RHS(self, *args, **kwargs):
        """Returns weak form for the right hand side."""
        raise NotImplementedError(('This method must be implemented '
                                   'in the derived class'))

    def RHS_implicit(self, *args, **kwargs):
        """Returns weak form for the right hand side of all implicit terms"""
        raise NotImplementedError(('This method must be implemented '
                                   'in the derived class'))

    def Source(self, *args, **kwargs):
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


class fieldDict(AttrDict):
    """
    AttrDict that checks that all added fields have proper meta data.

    Values can be either Function or Constant objects.
    """
    def _checkInputs(self, key, value):
        if key != '__dict__':
            if not isinstance(value, (Function, Constant)):
                raise TypeError('Value must be a Function or Constant object')
            fs = value.function_space()
            if not isinstance(fs, MixedFunctionSpace) and key not in fieldMetadata:
                msg = 'Trying to add a field "{:}" that has no meta data. ' \
                      'Add fieldMetadata entry to fieldDefs.py'.format(key)
                raise Exception(msg)

    def _setFunctionName(self, key, value):
        """Set function.name to key to ensure consistent naming"""
        if isinstance(value, Function):
            value.rename(name=key)

    def __setitem__(self, key, value):
        self._checkInputs(key, value)
        self._setFunctionName(key, value)
        super(fieldDict, self).__setitem__(key, value)

    def __setattr__(self, key, value):
        self._checkInputs(key, value)
        self._setFunctionName(key, value)
        super(fieldDict, self).__setattr__(key, value)


class temporaryFunctionCache(object):
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
        key = self._getKey(function_space)
        self.functions[key] = f

    def _getKey(self, function_space):
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

        def getElemStr(e):
            s = e.shortstr()
            d = e.degree()
            if not isinstance(d, tuple):
                s = s.replace('?', str(d))
            return s
        attrib = [getElemStr(ee) for ee in elems]
        key = '_'.join(attrib)
        return key

    def get(self, function_space):
        """
        Returns a tmp function for the given function space.

        If it doesn't exist, will allocate one.
        """
        if function_space not in self.functions:
            self.add(function_space)
        key = self._getKey(function_space)
        return self.functions[key]

    def clear(self):
        self.functions = {}


class problemCache(object):
    """Holds all variational problems that utility functions depend on."""
    # NOTE solvers are stored based on function names
    # NOTE for this to work all funcs must have unique names
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.solvers = {}

    def __getitem__(self, key):
        return self.solvers[key]

    def __setitem__(self, key, val):
        self.solvers[key] = val

    def __contains__(self, key):
        return key in self.solvers

    def add(self, key, val, msg=''):
        if self.verbose:
            print('adding solver {0} {1}'.format(msg, key))
        self.solvers[key] = val

    def clear(self):
        self.solvers = {}

tmpFunctionCache = temporaryFunctionCache()
linProblemCache = problemCache(verbose=False)


def printInfo(msg):
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


def createDirectory(path):
    if commrank == 0:
        if os.path.exists(path):
            if not os.path.isdir(path):
                raise Exception('file with same name exists', path)
        else:
            os.makedirs(path)
    return path


def extrudeMeshLinear(mesh2d, n_layers, xmin, xmax, dmin, dmax):
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
                    'base_coord_dim': base_coords.function_space().cdim,
                    'xmin': xmin, 'xmax': xmax, 'dmin': dmin, 'dmax': dmax},
        'uniform_extrusion_kernel')
    mesh = ExtrudedMesh(mesh2d, layers=n_layers,
                        layer_height=layer_height,
                        extrusion_type='custom',
                        kernel=extrusion_kernel, gdim=3)
    return mesh


def extrudeMeshSigma(mesh2d, n_layers, bathymetry_2d):
    """Extrudes 2d surface mesh with bathymetry data defined in a 2d field."""
    mesh = ExtrudedMesh(mesh2d, layers=n_layers, layer_height=1.0/n_layers)

    xyz = mesh.coordinates

    nNodes2d = bathymetry_2d.dat.data.shape[0]
    NVert = xyz.dat.data.shape[0]/nNodes2d
    # TODO can the loop be circumvented?
    for iNode in range(nNodes2d):
        xyz.dat.data[iNode*NVert:iNode*NVert+NVert, 2] *= -bathymetry_2d.dat.data[iNode]
    return mesh


def compVolume2d(eta, bath):
    mesh = bath.function_space().mesh()
    dx = mesh._dx
    val = assemble((eta+bath)*dx)
    return val


def compVolume3d(mesh):
    dx = mesh._dx
    one = Constant(1.0, domain=mesh.coordinates.domain())
    val = assemble(one*dx)
    return val


def compTracerMass3d(scalarFunc):
    mesh = scalarFunc.function_space().mesh()
    dx = mesh._dx
    val = assemble(scalarFunc*dx)
    return val


def getZCoordFromMesh(zcoord, solver_parameters={}):
    """Evaluates z coordinates from the 3D mesh"""
    solver_parameters.setdefault('ksp_atol', 1e-12)
    solver_parameters.setdefault('ksp_rtol', 1e-16)
    fs = zcoord.function_space()
    tri = TrialFunction(fs)
    test = TestFunction(fs)
    a = tri*test*dx
    L = fs.mesh().coordinates[2]*test*dx
    solve(a == L, zcoord, solver_parameters=solver_parameters)
    return zcoord


class verticalVelocitySolver(object):
    """Computes vertical velocity from continuity equation"""
    def __init__(self, solution, uv, bathymetry,
                 boundary_markers=[], boundary_funcs={},
                 solver_parameters={}):
        solver_parameters.setdefault('ksp_atol', 1e-12)
        solver_parameters.setdefault('ksp_rtol', 1e-16)
        H = solution.function_space()
        mesh = H.mesh()
        test = TestFunction(H)
        tri = TrialFunction(H)
        normal = FacetNormal(mesh)

        ds_surf = ds_b
        # NOTE weak dw/dz
        a = tri[2]*test[2]*normal[2]*ds_surf + \
            avg(tri[2])*jump(test[2], normal[2])*dS_h - Dx(test[2], 2)*tri[2]*dx
        # NOTE weak div(uv)
        uv_star = avg(uv)
        # NOTE in the case of mimetic uv the div must be taken over all components
        L = (inner(uv, nabla_grad(test[2]))*dx -
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
                L += -(uv[0]*normal[0] + uv[1]*normal[1])*test[2]*ds_bnd

        self.prob = LinearVariationalProblem(a, L, solution)
        self.solver = LinearVariationalSolver(self.prob, solver_parameters=solver_parameters)

    def solve(self):
        self.solver.solve()


class verticalIntegrator(object):
    """
    Computes vertical integral of the scalar field in the output
    function space.
    """
    def __init__(self, input, output, bottomToTop=True,
                 bndValue=Constant(0.0), average=False,
                 bathymetry=None, solver_parameters={}):
        solver_parameters.setdefault('ksp_atol', 1e-12)
        solver_parameters.setdefault('ksp_rtol', 1e-16)

        space = output.function_space()
        mesh = space.mesh()
        verticalIsDG = False
        if (hasattr(space.ufl_element(), '_B') and
                space.ufl_element()._B.family() != 'Lagrange'):
            # a normal outerproduct element
            verticalIsDG = True
        if 'HDiv' in space.ufl_element().shortstr():
            # Hdiv vector space, assume DG in vertical
            verticalIsDG = True
        tri = TrialFunction(space)
        phi = TestFunction(space)
        normal = FacetNormal(mesh)
        ds_surf = mesh._ds_b
        ds_bottom = mesh._ds_t
        if bottomToTop:
            bnd_term = normal[2]*inner(bndValue, phi)*ds_bottom
            mass_bnd_term = normal[2]*inner(tri, phi)*ds_surf
        else:
            bnd_term = normal[2]*inner(bndValue, phi)*ds_surf
            mass_bnd_term = normal[2]*inner(tri, phi)*ds_bottom

        a = -inner(Dx(phi, 2), tri)*mesh._dx + mass_bnd_term
        if verticalIsDG:
            if len(input.ufl_shape) > 0:
                dim = input.ufl_shape[0]
                for i in range(dim):
                    a += avg(tri[i])*jump(phi[i], normal[2])*mesh._dS_h
            else:
                a += avg(tri)*jump(phi, normal[2])*mesh._dS_h
        source = input
        if average:
            # FIXME this should be H not h
            source = input/bathymetry
        L = inner(source, phi)*mesh._dx + bnd_term
        self.prob = LinearVariationalProblem(a, L, output)
        self.solver = LinearVariationalSolver(self.prob, solver_parameters=solver_parameters)

    def solve(self):
        with timed_region('func_vert_int'):
            self.solver.solve()


def computeBaroclinicHead(solver, salt, baroc_head_3d, baroc_head_2d,
                          baroc_head_int_3d, bath3d):
    """
    Computes baroclinic head from density field

    r = 1/rho_0 int_{z=-h}^{\eta} rho' dz
    """
    solver.rhoIntegrator.solve()
    baroc_head_3d *= -physical_constants['rho0_inv']
    solver.baroHeadAverager.solve()
    solver.extractSurfBaroHead.solve()


class velocityMagnitudeSolver(object):
    """
    Computes magnitude of (u[0],u[1],w) and stores it in solution
    """
    def __init__(self, solution, u=None, w=None, minVal=1e-6,
                 solver_parameters={}):
        self.solution = solution
        self.min_val = minVal
        function_space = solution.function_space()
        test = TestFunction(function_space)
        tri = TrialFunction(function_space)

        a = test*tri*dx
        s = 0
        if u is not None:
            s += u[0]**2 + u[1]**2
        if w is not None:
            s += w**2
        L = test*sqrt(s)*dx
        self.prob = LinearVariationalProblem(a, L, solution)
        self.solver = LinearVariationalSolver(self.prob, solver_parameters=solver_parameters)

    def solve(self):
        self.solver.solve()
        self.solution.dat.data[:] = np.maximum(self.solution.dat.data[:], self.min_val)


class horizontalJumpDiffusivity(object):
    """Computes tracer jump diffusivity for horizontal advection."""
    def __init__(self, alpha, tracer, output, hElemSize, umag,
                 tracer_mag, maxval, minval=1e-6, solver_parameters={}):
        solver_parameters.setdefault('ksp_atol', 1e-6)
        solver_parameters.setdefault('ksp_rtol', 1e-8)
        if output.function_space() != maxval.function_space():
            raise Exception('output and maxval function spaces do not match')
        self.output = output
        self.minval = minval
        self.maxval = maxval

        fs = output.function_space()
        mesh = fs.mesh()
        test = TestFunction(fs)
        tri = TrialFunction(fs)
        a = inner(test, tri)*mesh._dx + jump(test, tri)*mesh._dS_v
        tracer_jump = jump(tracer)
        # TODO jump scalar must depend on the tracer value scale
        # TODO can this be estimated automatically e.g. global_max(abs(S))
        maxjump = Constant(0.05)*tracer_mag
        L = alpha*avg(umag*hElemSize)*(tracer_jump/maxjump)**2*avg(test)*mesh._dS_v
        self.prob = LinearVariationalProblem(a, L, output)
        self.solver = LinearVariationalSolver(self.prob, solver_parameters=solver_parameters)

    def solve(self):
        self.solver.solve()
        self.output.dat.data[:] = np.minimum(self.maxval.dat.data, self.output.dat.data)
        self.output.dat.data[self.output.dat.data[:] < self.minval] = self.minval


class expandFunctionTo3d(object):
    """
    Copies a field from 2d mesh to 3d mesh, assigning the same value over the
    vertical dimension. Horizontal function space must be the same.
    """
    def __init__(self, input2d, output3d, elemHeight=None):
        self.input2d = input2d
        self.output3d = output3d
        self.fs_2d = self.input2d.function_space()
        self.fs_3d = self.output3d.function_space()

        family_2d = self.fs_2d.ufl_element().family()
        if hasattr(self.fs_3d.ufl_element(), '_A'):
            # a normal outerproduct element
            family_3dh = self.fs_3d.ufl_element()._A.family()
            if family_2d != family_3dh:
                raise Exception('2D and 3D spaces do not match: {0:s} {1:s}'.format(family_2d, family_3dh))
        if family_2d == 'Raviart-Thomas' and elemHeight is None:
            raise Exception('elemHeight must be provided for Raviart-Thomas spaces')
        self.doRTScaling = family_2d == 'Raviart-Thomas'

        self.iter_domain = op2.ALL

        # number of nodes in vertical direction
        nVertNodes = len(self.fs_3d.fiat_element.B.entity_closure_dofs()[1][0])

        nodes = self.fs_3d.bt_masks['geometric'][0]
        self.idx = op2.Global(len(nodes), nodes, dtype=np.int32, name='nodeIdx')
        self.kernel = op2.Kernel("""
            void my_kernel(double **func, double **func2d, int *idx) {
                for ( int d = 0; d < %(nodes)d; d++ ) {
                    for ( int c = 0; c < %(func_dim)d; c++ ) {
                        for ( int e = 0; e < %(v_nodes)d; e++ ) {
                            func[idx[d]+e][c] = func2d[d][c];
                        }
                    }
                }
            }""" % {'nodes': self.input2d.cell_node_map().arity,
                    'func_dim': self.input2d.function_space().cdim,
                    'v_nodes': nVertNodes},
            'my_kernel')

        if self.doRTScaling:
            solver_parameters = {}
            solver_parameters.setdefault('ksp_atol', 1e-12)
            solver_parameters.setdefault('ksp_rtol', 1e-16)
            test = TestFunction(self.fs_3d)
            tri = TrialFunction(self.fs_3d)
            a = inner(tri, test)*dx
            L = inner(self.output3d, test)*elemHeight*dx
            prob = LinearVariationalProblem(a, L, self.output3d)
            self.RT_scale_solver = LinearVariationalSolver(
                prob, solver_parameters=solver_parameters)

    def solve(self):
        with timed_region('func_copy2dTo3d'):
            # execute par loop
            op2.par_loop(
                self.kernel, self.fs_3d.mesh().cell_set,
                self.output3d.dat(op2.WRITE, self.fs_3d.cell_node_map()),
                self.input2d.dat(op2.READ, self.fs_2d.cell_node_map()),
                self.idx(op2.READ),
                iterate=self.iter_domain)

            if self.doRTScaling:
                self.RT_scale_solver.solve()


class subFunctionExtractor(object):
    """
    Extract a 2D subfunction from a 3D extruded mesh.

    input3d: Function in 3d mesh
    output2d: Function in 2d mesh
    useBottomValue: bool
        Extract function at bottom elements
    elemBottomNodes: bool
        Extract function at bottom mask of the element
        Typically elemBottomNodes = useBottomValue to obtain values at
        surface/bottom faces of the extruded mesh
    elemHeight: Function in 2d mesh
        Function that defines the element heights
        (required for Raviart-Thomas spaces only)
    """
    def __init__(self, input3d, output2d, useBottomValue=True,
                 elemBottomNodes=None, elemHeight=None):
        self.input3d = input3d
        self.output2d = output2d
        self.fs_3d = self.input3d.function_space()
        self.fs_2d = self.output2d.function_space()

        # NOTE top/bottom are defined differently than in firedrake
        sub_domain = 'bottom' if useBottomValue else 'top'
        if elemBottomNodes is None:
            # extract surface/bottom face by default
            elemBottomNodes = useBottomValue

        family_2d = self.fs_2d.ufl_element().family()
        if hasattr(self.fs_3d.ufl_element(), '_A'):
            # a normal outerproduct element
            family_3dh = self.fs_3d.ufl_element()._A.family()
            if family_2d != family_3dh:
                raise Exception('2D and 3D spaces do not match: {0:s} {1:s}'.format(family_2d, family_3dh))
        if family_2d == 'Raviart-Thomas' and elemHeight is None:
            raise Exception('elemHeight must be provided for Raviart-Thomas spaces')
        self.doRTScaling = family_2d == 'Raviart-Thomas'

        if elemBottomNodes:
            nodes = self.fs_3d.bt_masks['geometric'][1]
        else:
            nodes = self.fs_3d.bt_masks['geometric'][0]
        if sub_domain == 'top':
            # 'top' means free surface, where extrusion started
            self.iter_domain = op2.ON_BOTTOM
        elif sub_domain == 'bottom':
            # 'bottom' means the bed, where extrusion ended
            self.iter_domain= op2.ON_TOP

        out_nodes = self.fs_2d.fiat_element.space_dimension()

        assert (len(nodes) == out_nodes)

        self.idx = op2.Global(len(nodes), nodes, dtype=np.int32, name='nodeIdx')
        self.kernel = op2.Kernel("""
            void my_kernel(double **func, double **func3d, int *idx) {
                for ( int d = 0; d < %(nodes)d; d++ ) {
                    for ( int c = 0; c < %(func_dim)d; c++ ) {
                        func[d][c] = func3d[idx[d]][c];
                        //func[d][c] = idx[d];
                    }
                }
            }""" % {'nodes': self.output2d.cell_node_map().arity,
                    'func_dim': self.output2d.function_space().cdim},
            'my_kernel')

        if self.doRTScaling:
            solver_parameters = {}
            solver_parameters.setdefault('ksp_atol', 1e-12)
            solver_parameters.setdefault('ksp_rtol', 1e-16)
            test = TestFunction(self.fs_2d)
            tri = TrialFunction(self.fs_2d)
            dx_2d = self.fs_2d.mesh()._dx
            a = inner(tri, test)*dx_2d
            L = inner(self.output2d, test)/elemHeight*dx_2d
            prob = LinearVariationalProblem(a, L, self.output2d)
            self.RT_scale_solver = LinearVariationalSolver(
                prob, solver_parameters=solver_parameters)

    def solve(self):
        with timed_region('func_copy3dTo2d'):
            # execute par loop
            op2.par_loop(self.kernel, self.fs_3d.mesh().cell_set,
                         self.output2d.dat(op2.WRITE, self.fs_2d.cell_node_map()),
                         self.input3d.dat(op2.READ, self.fs_3d.cell_node_map()),
                         self.idx(op2.READ),
                         iterate=self.iter_domain)

            if self.doRTScaling:
                self.RT_scale_solver.solve()


def computeElemHeight(zCoord, output):
    """
    Compute element heights on an extruded mesh.
    zCoord (P1CG) contains zcoordinates of the mesh
    element height is stored in output function (typ. P1DG).
    """
    fs_in = zCoord.function_space()
    fs_out = output.function_space()

    iterate = op2.ALL

    # NOTE height maybe <0 if mesh was extruded like that
    kernel = op2.Kernel("""
        void my_kernel(double **func, double **zCoord) {
            for ( int d = 0; d < %(nodes)d/2; d++ ) {
                for ( int c = 0; c < %(func_dim)d; c++ ) {
                    // NOTE is fabs required here??
                    double dz = zCoord[2*d+1][c]-zCoord[2*d][c];
                    func[2*d][c] = dz;
                    func[2*d+1][c] = dz;
                }
            }
        }""" % {'nodes': zCoord.cell_node_map().arity,
                'func_dim': zCoord.function_space().cdim},
        'my_kernel')
    op2.par_loop(
        kernel, fs_out.mesh().cell_set,
        output.dat(op2.WRITE, fs_out.cell_node_map()),
        zCoord.dat(op2.READ, fs_in.cell_node_map()),
        iterate=iterate)

    return output


def computeBottomDrag(uv_bottom, z_bottom, bathymetry, drag):
    """Computes bottom drag coefficient (Cd) from boundary log layer."""
    von_karman = physical_constants['von_karman']
    z0_friction = physical_constants['z0_friction']
    drag.assign((von_karman / ln((z_bottom)/z0_friction))**2)
    return drag


def computeBottomFriction(solver, uv_3d, uv_bottom_2d, uv_bottom_3d, z_coord_3d,
                          z_bottom_2d, bathymetry_2d,
                          bottom_drag_2d, bottom_drag_3d,
                          v_elem_size_2d=None, v_elem_size_3d=None):
    # compute velocity at middle of bottom element
    solver.extractUVBottom.solve()
    tmp = uv_bottom_2d.dat.data.copy()
    solver.extractUVBottom.solve()
    uv_bottom_2d.dat.data[:] = 0.5*(uv_bottom_2d.dat.data + tmp)
    solver.copyUVBottomTo3d.solve()
    solver.extractZBottom.solve()
    z_bottom_2d.dat.data[:] += bathymetry_2d.dat.data[:]
    z_bottom_2d.dat.data[:] *= 0.5
    computeBottomDrag(uv_bottom_2d, z_bottom_2d, bathymetry_2d, bottom_drag_2d)
    solver.copyBottomDragTo3d.solve()


def getHorizontalElemSize(sol2d, sol3d=None):
    """
    Computes horizontal element size from the 2D mesh, then copies it over a 3D
    field.
    """
    P1_2d = sol2d.function_space()
    mesh = P1_2d.mesh()
    cellsize = CellSize(mesh)
    test = TestFunction(P1_2d)
    tri = TrialFunction(P1_2d)
    sol2d = Function(P1_2d)
    dx_2d = mesh._dx
    a = test * tri * dx_2d
    L = test * cellsize * dx_2d
    solve(a == L, sol2d)
    if sol3d is None:
        return sol2d
    expandFunctionTo3d(sol2d, sol3d).solve()
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
        L = new_z*test*dx
        self.prob = LinearVariationalProblem(a, L, z_coord)
        self.solver = LinearVariationalSolver(self.prob, solver_parameters=solver_parameters)

    def solve(self):
        self.solver.solve()
        # assign to mesh
        self.coords.dat.data[:, 2] = self.z_coord.dat.data[:]


class meshVelocitySolver(object):
    """Computes vertical mesh velocity for moving sigma mesh"""
    def __init__(self, solver, eta, uv, w, w_mesh, w_mesh_surf,
                 w_mesh_surf_2d, w_mesh_ddz_3d, bathymetry,
                 z_coord_ref, solver_parameters={}):
        solver_parameters.setdefault('ksp_atol', 1e-12)
        solver_parameters.setdefault('ksp_rtol', 1e-16)
        self.solverObj = solver

        # compute w_mesh at the free surface (constant over vertical!)
        # w_mesh_surf = w - eta_grad[0]*uv[0] + eta_grad[1]*uv[1]
        fs = w_mesh.function_space()
        z = fs.mesh().coordinates[2]
        tri = TrialFunction(fs)
        test = TestFunction(fs)
        a = inner(tri, test)*dx
        eta_grad = nabla_grad(eta)
        L = (w[2] - eta_grad[0]*uv[0] - eta_grad[1]*uv[1])*test*dx
        self.probWMeshSurf = LinearVariationalProblem(a, L, w_mesh_surf)
        self.solverWMeshSurf = LinearVariationalSolver(self.probWMeshSurf, solver_parameters=solver_parameters)

        # compute w in the whole water column (0 at bed)
        # w_mesh = w_mesh_surf * (z+h)/(eta+h)
        fs = w_mesh.function_space()
        z = fs.mesh().coordinates[2]
        tri = TrialFunction(fs)
        test = TestFunction(fs)
        a = tri*test*dx
        H = eta + bathymetry
        L = (w_mesh_surf*(z+bathymetry)/H)*test*dx
        self.probWMesh = LinearVariationalProblem(a, L, w_mesh)
        self.solverWMesh = LinearVariationalSolver(self.probWMesh, solver_parameters=solver_parameters)

        # compute dw_mesh/dz in the whole water column
        fs = w_mesh.function_space()
        z = fs.mesh().coordinates[2]
        tri = TrialFunction(fs)
        test = TestFunction(fs)
        a = tri*test*dx
        H = eta + bathymetry
        L = (w_mesh_surf/H)*test*dx
        self.probDWMeshDz = LinearVariationalProblem(a, L, w_mesh_ddz_3d)
        self.solverDWMeshDz = LinearVariationalSolver(self.probDWMeshDz, solver_parameters=solver_parameters)

    def solve(self):
        self.solverWMeshSurf.solve()
        self.solverObj.extractSurfW.solve()
        self.solverObj.copySurfWMeshTo3d.solve()
        self.solverWMesh.solve()
        self.solverDWMeshDz.solve()


class parabolicViscosity(object):
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
        H = nu.function_space()
        x = H.mesh().coordinates
        test = TestFunction(H)
        tri = TrialFunction(H)
        a = tri*test*dx
        uv_mag = sqrt(uv_bottom[0]**2 + uv_bottom[1]**2)
        parabola = -x[2]*(bathymetry + z0 + x[2])/(bathymetry + z0)
        L = kappa*sqrt(bottom_drag)*uv_mag*parabola*test*dx
        self.prob = LinearVariationalProblem(a, L, nu)
        self.solver = LinearVariationalSolver(self.prob, solver_parameters=solver_parameters)

    def solve(self):
        self.solver.solve()
        # remove negative values
        ix = self.solution.dat.data[:] < self.min_val
        self.solution.dat.data[ix] = self.min_val


def betaPlaneCoriolisParams(latitude):
    """Computes beta plane parameters based on the latitude (given in degrees)."""
    Omega = 7.2921150e-5  # rad/s Earth rotation rate
    R = 6371.e3  # Earth radius
    # Coriolis parameter f = 2 Omega sin(alpha)
    # Beta plane approximation f_beta = f_0 + Beta y
    # f_0 = 2 Omega sin(alpha_0)
    # Beta = df/dy|_{alpha=alpha_0}
    #      = (df/dalpha*dalpha/dy)_{alpha=alpha_0}
    #      = 2 Omega cos(alpha_0) /R
    alpha_0 = 2*np.pi*latitude/360.0
    f_0 = 2*Omega*np.sin(alpha_0)
    beta = 2*Omega*np.cos(alpha_0)/R
    return f_0, beta


def betaPlaneCoriolisFunction(degrees, out_function, y_offset=0.0):
    """Interpolates beta plane Coriolis parameter to the given functions."""
    # NOTE assumes that mesh y coordinate spans [-L_y, L_y]
    f0, beta = betaPlaneCoriolisParams(45.0)
    out_function.interpolate(
        Expression('f0+beta*(x[1]-y_0)', f0=f0, beta=beta, y_0=y_offset))


class smagorinskyViscosity(object):
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
    def __init__(self, uv, output, C_s, hElemSize, solver_parameters={}):
        solver_parameters.setdefault('ksp_atol', 1e-12)
        solver_parameters.setdefault('ksp_rtol', 1e-16)
        self.min_val = 1e-10
        self.output = output

        fs = output.function_space()
        mesh = fs.mesh()
        w = TestFunction(fs)
        tau = TrialFunction(fs)

        # rate of strain tensor
        D_T = Dx(uv[0], 0) - Dx(uv[1], 1)
        D_S = Dx(uv[0], 1) + Dx(uv[1], 0)
        nu = C_s**2*hElemSize**2 * sqrt(D_T**2 + D_S**2)

        a = w*tau*mesh._dx
        L = w*nu*mesh._dx
        self.prob = LinearVariationalProblem(a, L, output)
        self.solver = LinearVariationalSolver(self.prob, solver_parameters=solver_parameters)

    def solve(self):
        self.solver.solve()

        # remove negative values
        ix = self.output.dat.data < self.min_val
        self.output.dat.data[ix] = self.min_val


class projector(object):
    def __init__(self, input_func, output_func, solver_parameters={}):
        self.input = input_func
        self.output = output_func
        self.same_space = (self.input.function_space() ==
                           self.output.function_space())

        if not self.same_space:
            V = output_func.function_space()
            p = TestFunction(V)
            q = TrialFunction(V)
            a = inner(p, q) * V.mesh()._dx
            L = inner(p, input_func) * V.mesh()._dx

            solver_parameters.setdefault('ksp_type', 'cg')
            solver_parameters.setdefault('ksp_rtol', 1e-8)

            self.problem = LinearVariationalProblem(a, L, output_func)
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

    def computeRho(self, S, Th, p, rho0=0.0):
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
        S0 = np.maximum(S, 0.0)  # ensure positive salinity
        a = self.a
        b = self.b
        Pn = (a[0] + Th*a[1] + Th*Th*a[2] + Th*Th*Th*a[3] + S0*a[4] +
              Th*S0*a[5] + S0*S0*a[6] + p*a[7] + p*Th * Th*a[8] + p*S0*a[9] +
              p*p*a[10] + p*p*Th*Th * a[11])
        Pd = (b[0] + Th*b[1] + Th*Th*b[2] + Th*Th*Th*b[3] +
              Th*Th*Th*Th*b[4] + S0*b[5] + S0*Th*b[6] + S0*Th*Th*Th*b[7] +
              pow(S0, 1.5)*b[8] + pow(S0, 1.5)*Th*Th*b[9] + p*b[10] +
              p*p*Th*Th*Th*b[11] + p*p*p*Th*b[12])
        rho = Pn/Pd - rho0
        return rho
