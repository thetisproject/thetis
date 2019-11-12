# Copyright (C) 2015-2019 Tormod Landet
# SPDX-License-Identifier: Apache-2.0

import numpy
import numpy.linalg
import dolfin
import contextlib
from .timer import timeit


# Default parameters when use_ksp is True
DEFAULT_ITR_CTRL = [3, 3]
DEFAULT_RTOL = [1e-6, 1e-8, 1e-10]
DEFAULT_ATOL = [1e-8, 1e-10, 1e-15]
DEFAULT_NITK = [10, 40, 100]


def linear_solver_from_input(
    simulation,
    path,
    default_solver='default',
    default_preconditioner='default',
    default_lu_method='default',
    default_parameters=None,
):
    """
    Create a linear equation solver from specifications in the input at the
    given path and the Ocellaris defaults for the given solver, passed as
    arguments to this function.

    The path (e.g "solver/u") must point to a dictionary in the input file that
    contains fields specifying the solver an preconditioner. If no such input is
    found the defaults are used.

    The solver definition can either be on the simplified dolfin format (with
    PETSc backend), or it can give the full PETSc configuration. The default
    depends on the Ocellaris solver, see the log file from a run without any
    specific configuration for info.

    Dolfin solver definition example::

        solver:
            u:
                solver: gmres
                preconditioner: additive_schwarz
            coupled:
                solver: lu
                lu_method: mumps
                parameters:
                    same_nonzero_pattern: True

    The PETSc solver definition is used when `use_ksp` is specified. The PETSc
    options are given prefixed by `petsc_`. This prefix is changed to `sol_VARNAME_`
    automatically, so in the below example the prefix is `sol_u_`. Normally this is
    an implementation detail, but if user code wants to change the PETSc option
    database for the solver then the prefix must be known. An example::

        solver:
            p:
                use_ksp: yes
                petsc_ksp_type: cg
                petsc_pc_type: gamg

    The log will contain the actual entries to the PETSc options database. For PETSc
    options that do not take a value, like ``-ksp_view`` and ``-help`` a special
    signal value 'ENABLED' can be specified on the input file. This is automatically
    translated to the correct syntax.
    """
    simulation.log.info('    Creating linear equation solver from input "%s"' % path)

    # Check if we are using the simplified DOLFIN solver wrappers or full PETSc solver setup
    use_ksp = default_parameters is not None and default_parameters.get('use_ksp', False)
    use_ksp = simulation.input.get_value('%s/use_ksp' % path, use_ksp, 'bool')
    simulation.log.info('        Advanced KSP configuration is %s' % ('ON' if use_ksp else 'OFF'))

    inp_data = simulation.input.get_value(path, {}, 'Input')

    if not use_ksp:
        # Use standard simplified DOLFIN solver wrappers
        # Get values from input dictionary
        solver_method = inp_data.get_value('solver', default_solver, 'string')
        preconditioner = inp_data.get_value('preconditioner', default_preconditioner, 'string')
        lu_method = inp_data.get_value('lu_method', default_lu_method, 'string')
        solver_parameters = inp_data.get_value('parameters', {}, 'dict(string:any)')

        if default_parameters:
            params = [default_parameters, solver_parameters]
        else:
            params = [solver_parameters]

        # Prevent confusion due to the two different ways of configuring the linear solvers
        for key in inp_data:
            if key.startswith('petsc_') or key.startswith('inner_iter_'):
                pth = '%s/%s' % (path, key)
                simulation.log.warning('Found input %s which is IGNORED since use_ksp=False' % pth)

        simulation.log.info('        Method:         %s' % solver_method)
        simulation.log.info('        Preconditioner: %s' % preconditioner)
        simulation.log.info('        LU-method:      %s' % lu_method)

        return LinearSolverWrapper(solver_method, preconditioner, lu_method, params)

    else:
        # Use more powerfull petsc4py interface
        params = default_parameters.copy()
        params.update(inp_data)

        # Prevent confusion due to the two different ways of configuring the linear solvers
        for unwanted in 'solver preconditioner lu_method parameters'.split():
            if unwanted in inp_data:
                pth = '%s/%s' % (path, unwanted)
                simulation.log.warning('Found input %s which is IGNORED since use_ksp=True' % pth)

        # Create the PETScKrylovSolver and show some info
        solver = KSPLinearSolverWrapper(simulation, path, params)
        simulation.log.info('        Method:         %s' % solver.ksp().getType())
        simulation.log.info('        Preconditioner: %s' % solver.ksp().pc.getType())
        simulation.log.info('        Options prefix: %s' % solver.ksp().getOptionsPrefix())
        return solver


class LinearSolverWrapper(object):
    def __init__(self, solver_method, preconditioner=None, lu_method=None, parameters=None):
        """
        Wrap a DOLFIN PETScKrylovSolver or PETScLUSolver

        You must either specify solver_method = 'lu' and give the name
        of the solver, e.g lu_solver='mumps' or give a valid Krylov
        solver name, eg. solver_method='minres' and give the name of a
        preconditioner, eg. preconditioner_name='hypre_amg'.

        The parameters argument is a *list* of dictionaries which are
        to be used as parameters to the Krylov solver. Settings in the
        first dictionary in this list will be (potentially) overwritten
        by settings in later dictionaries. The use case is to provide
        sane defaults as well as allow the user to override the defaults
        in the input file

        The reason for this wrapper is to provide easy querying of
        iterative/direct and not crash when set_reuse_preconditioner is
        run before the first solve. This simplifies usage
        """
        self.solver_method = solver_method
        self.preconditioner = preconditioner
        self.lu_method = lu_method
        self.input_parameters = parameters

        self.is_first_solve = True
        self.is_iterative = False
        self.is_direct = False

        if solver_method.lower() == 'lu':
            solver = dolfin.PETScLUSolver(lu_method)
            self.is_direct = True
        else:
            precon = dolfin.PETScPreconditioner(preconditioner)
            solver = dolfin.PETScKrylovSolver(solver_method, precon)
            self._pre_obj = precon  # Keep from going out of scope
            self.is_iterative = True

        for parameter_set in parameters:
            apply_settings(solver_method, solver.parameters, parameter_set)

        self._solver = solver

    def solve(self, *argv, **kwargs):
        ret = self._solver.solve(*argv, **kwargs)
        self.is_first_solve = False
        return ret

    def inner_solve(self, A, x, b, in_iter, co_iter):
        """
        This is not implemented for dolfin solvers, so just solve as usual
        """
        return self.solve(A, x, b)

    @property
    def parameters(self):
        return self._solver.parameters

    def set_parameter(self, name, value):
        params = self._solver.parameters
        keys = params.keys()
        # DOLFIN will SEGFAULT if using e.g 'same_nonzero_pattern' if that is not configurable
        assert name in keys, 'Solver parameter %s is not in %r' % (name, keys)
        params[name] = value

    def set_operator(self, A):
        return self._solver.set_operator(A)

    def set_reuse_preconditioner(self, *argv, **kwargs):
        if self.is_iterative and self.is_first_solve:
            return  # Nov 2016: this segfaults if running before the first solve
        else:
            return self._solver.set_reuse_preconditioner(*argv, **kwargs)

    def ksp(self):
        return self._solver.ksp()

    def __repr__(self):
        return (
            '<LinearSolverWrapper iterative=%r ' % self.is_iterative
            + 'direct=%r ' % self.is_direct
            + 'method=%r ' % self.solver_method
            + 'preconditioner=%r ' % self.preconditioner
            + 'LU-method=%r ' % self.lu_method
            + 'parameters=%r>' % self.input_parameters
        )


class KSPLinearSolverWrapper(object):
    def __init__(self, simulation, input_path, params):
        """
        Wrap a PETScKrylov solver that is configured through petsc4py
        """
        self.simulation = simulation
        self._input_path = input_path
        self._config_params = params
        self.is_first_solve = True
        self.is_iterative = True
        self.is_direct = False

        # Create the solver
        prefix = 'sol_%s_' % input_path.split('/')[-1]
        self._solver = dolfin.PETScKrylovSolver()

        # Help is treated specially, this is used to enquire about PETSc capabilities
        request_petsc_help = 'petsc_help' in params
        if request_petsc_help:
            params.pop('petsc_help')

        # Translate the petsc_* keys to the correct solver prefix
        # and insert them into the PETSc options database
        for param, value in sorted(params.items()):
            if not param.startswith('petsc_'):
                continue

            # Citations are treated specially, does not work with prefix
            if param == 'petsc_citations':
                option = 'citations'
            else:
                option = prefix + param[6:]
            simulation.log.info('        %-50s: %20r' % (option, value))

            # Some options do not have a value, but we must have one on the input
            # file, the below translation fixes that
            if value == 'ENABLED':
                dolfin.PETScOptions.set(option)
            elif value == 'DISABLED':
                pass
            else:
                # Normal option with value
                dolfin.PETScOptions.set(option, value)

        if request_petsc_help:
            simulation.log.warning('PETSc help coming up')
            simulation.log.warning('-' * 80)
            dolfin.PETScOptions.set('help')

        # Configure the solver
        self._solver.set_options_prefix(prefix)
        self._solver.ksp().setFromOptions()

        if request_petsc_help:
            simulation.log.warning('-' * 80)
            simulation.log.warning('Showing PETSc help done, exiting')
            exit()

        # Only used when calling the basic .solve() method
        self.reuse_precon = False

    @timeit.named('petsc4py solve')
    def solve(self, *argv, **kwargs):
        self._solver.set_from_options()

        inp = self.simulation.input.get_value(self._input_path, {}, 'Input')

        def get_updated(key, default, required_type):
            "The key may be changed by user code, lets get fresh info"
            prev = self._config_params.get(key, default)
            return inp.get_value(key, prev, required_type)

        # Use the setup for the final inner iterations (assumed to be strictest)
        rtol = get_updated('inner_iter_rtol', DEFAULT_RTOL, 'list(float)')[-1]
        atol = get_updated('inner_iter_atol', DEFAULT_ATOL, 'list(float)')[-1]
        nitk = get_updated('inner_iter_max_it', DEFAULT_NITK, 'list(int)')[-1]

        # Solver setup with petsc4py
        ksp = self._solver.ksp()
        pc = ksp.getPC()
        pc.setReusePreconditioner(self.reuse_precon)
        ksp.setTolerances(rtol=rtol, atol=atol, max_it=nitk)

        # Solve using the standard dolfin interface
        ret = self._solver.solve(*argv, **kwargs)
        self.is_first_solve = False
        return ret

    @timeit.named('petsc4py inner_solve')
    def inner_solve(self, A, x, b, in_iter, co_iter):
        """
        This solver method uses different convergence criteria depending
        on how far in into the inner iterations loop the solve is located

        When used in IPCS, SIMPLE etc then in_iter is the inner
        iteration in the splitting scheme and co_iter is the number
        of iterations left in the time step, i.e.,

            in_iter + co_iter == num_inner_iter

        This can be used to have more relaxed convergence criteria for the
        first iterations and more strict for the last iterations

        IMPORTANT: inner iteration here does NOT correspond to Krylov
        iterations. Outer iterations is time steps which can contain several
        inner iterations to perform iterative pressure corrections and in
        each of these inner iterations there are Krylov iterations to actually
        solve the resulting linear systems.
        """
        inp = self.simulation.input.get_value(self._input_path, {}, 'Input')

        def get_updated(key, default, required_type):
            "The key may be changed by user code, lets get fresh info"
            prev = self._config_params.get(key, default)
            return inp.get_value(key, prev, required_type)

        firstN, lastN = get_updated('inner_iter_control', DEFAULT_ITR_CTRL, 'list(int)')
        rtol_beg, rtol_mid, rtol_end = get_updated('inner_iter_rtol', DEFAULT_RTOL, 'list(float)')
        atol_beg, atol_mid, atol_end = get_updated('inner_iter_atol', DEFAULT_ATOL, 'list(float)')
        nitk_beg, nitk_mid, nitk_end = get_updated('inner_iter_max_it', DEFAULT_NITK, 'list(int)')

        # Solver setup with petsc4py
        ksp = self._solver.ksp()
        pc = ksp.getPC()

        # Special treatment of first inner iteration
        reuse_pc = True
        if in_iter == 1:
            reuse_pc = False
            ksp.setOperators(A.mat())

        if co_iter < lastN:
            # This is one of the last iterations
            rtol = rtol_end
            atol = atol_end
            max_it = nitk_end
        elif in_iter <= firstN:
            # This is one of the first iterations
            rtol = rtol_beg
            atol = atol_beg
            max_it = nitk_beg
        else:
            # This iteration is in the middle of the range
            rtol = rtol_mid
            atol = atol_mid
            max_it = nitk_mid

        pc.setReusePreconditioner(reuse_pc)
        ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_it)
        ksp.solve(b.vec(), x.vec())
        x.update_ghost_values()
        return ksp.getIterationNumber()

    @property
    def parameters(self):
        raise ValueError('Do not use dolfin parameters to configure KSP solver')

    def set_operator(self, A):
        self._ksp.setOperators(A.mat())

    def set_reuse_preconditioner(self, reuse_preconditioner):
        # Only used when calling the basic .solve() method
        self.reuse_precon = reuse_preconditioner

    def ksp(self):
        return self._solver.ksp()

    def __repr__(self):
        return '<KSPLinearSolverWrapper prefix=%r>' % self._ksp.getOptionsPrefix()


def apply_settings(solver_method, parameters, new_values):
    """
    This function does almost the same as::

        parameters.update(new_values)

    The difference is that subdictionaries are handled
    recursively and not replaced outright
    """
    skip = set()
    if solver_method == 'lu':
        skip.update(['nonzero_initial_guess', 'relative_tolerance', 'absolute_tolerance'])

    keys = parameters.keys()

    for key, value in new_values.items():
        if key in skip:
            continue
        elif key not in keys:
            raise KeyError(
                'Cannot set solver parameter %s, not one of the supported %r' % (key, keys)
            )
        elif isinstance(value, dict):
            apply_settings(solver_method, parameters[key], value)
        else:
            parameters[key] = value


@contextlib.contextmanager
def petsc_options(opts):
    """
    A context manager to set PETSc options for a limited amount of code.
    The parameter opts is a dictionary of PETSc/SLEPc options
    """
    from petsc4py import PETSc

    orig_opts = PETSc.Options().getAll()
    for key, val in opts.items():
        PETSc.Options().setValue(key, val)

    yield  # run the code

    for key in opts.keys():
        if key in orig_opts:
            PETSc.Options().setValue(key, orig_opts[key])
        else:
            PETSc.Options().delValue(key)


def create_block_matrix(V, blocks):
    """
    Create a sparse matrix to hold dense blocks that are larger than
    the normal DG block diagonal mass matrices (super-cell dense blocks)

    The argument ``blocks`` should be a list of lists/arrays containing
    the dofs in each block. The dofs are assumed to be the same for
    both rows and columns. If blocks == 'diag' then a diagonal matrix is
    returned
    """
    comm = V.mesh().mpi_comm()
    dm = V.dofmap()
    im = dm.index_map()

    # Create a tensor layout for the matrix
    ROW_MAJOR = 0
    tl = dolfin.TensorLayout(comm, ROW_MAJOR, dolfin.TensorLayout.Sparsity.SPARSE)
    tl.init([im, im], dolfin.TensorLayout.Ghosts.GHOSTED)

    # Setup the tensor layout's sparsity pattern
    sp = tl.sparsity_pattern()
    sp.init([im, im])
    if blocks == 'diag':
        Ndofs = im.size(im.MapSize.OWNED)
        entries = numpy.empty((2, 1), dtype=numpy.intc)
        for dof in range(Ndofs):
            entries[:] = dof
            sp.insert_local(entries)
    else:
        entries = None
        for block in blocks:
            N = len(block)
            if entries is None or entries.shape[1] != N:
                entries = numpy.empty((2, N), dtype=numpy.intc)
                entries[0, :] = block
                entries[1, :] = entries[0, :]
                sp.insert_local(entries)
    sp.apply()

    # Create a matrix with the newly created tensor layout
    A = dolfin.PETScMatrix(comm)
    A.init(tl)

    return A


def matmul(A, B, out=None):
    """
    A B (and potentially out) must be PETScMatrix
    The matrix out must be the result of a prior matmul
    call with the same sparsity patterns in A and B
    """
    assert A is not None and B is not None

    A = A.mat()
    B = B.mat()
    if out is not None:
        A.matMult(B, out.mat())
        C = out
    else:
        Cmat = A.matMult(B)
        C = dolfin.PETScMatrix(Cmat)
        C.apply('insert')

    return C


def invert_block_diagonal_matrix(V, M, Minv=None):
    """
    Given a block diagonal matrix (DG mass matrix or similar), use local
    dense inverses to compute the  inverse matrix and return it, optionally
    reusing the given Minv tensor
    """
    mesh = V.mesh()
    dm = V.dofmap()
    N = dm.cell_dofs(0).shape[0]
    Mlocal = numpy.zeros((N, N), float)

    if Minv is None:
        Minv = dolfin.as_backend_type(M.copy())

    # Loop over cells and get the block diagonal parts (should be moved to C++)
    istart = M.local_range(0)[0]
    for cell in dolfin.cells(mesh, 'regular'):
        # Get global dofs
        dofs = dm.cell_dofs(cell.index()) + istart

        # Get block diagonal part of approx_A, invert it and insert into M⁻¹
        M.get(Mlocal, dofs, dofs)
        Mlocal_inv = numpy.linalg.inv(Mlocal)
        Minv.set(Mlocal_inv, dofs, dofs)

    Minv.apply('insert')
    return Minv


def condition_number(A, method='simplified'):
    """
    Estimate the condition number of the matrix A
    """
    if method == 'simplified':
        # Calculate max(abs(A))/min(abs(A))
        amin, amax = 1e10, -1e10
        for irow in range(A.size(0)):
            _indices, values = A.getrow(irow)
            aa = abs(values)
            amax = max(amax, aa.max())
            aa[aa == 0] = amax
            amin = min(amin, aa.min())
        amin = dolfin.MPI.min(dolfin.MPI.comm_world, float(amin))
        amax = dolfin.MPI.max(dolfin.MPI.comm_world, float(amax))
        return amax / amin

    elif method == 'numpy':
        from numpy.linalg import cond

        A = mat_to_scipy_csr(A).todense()
        return cond(A)

    elif method == 'SLEPc':
        from petsc4py import PETSc
        from slepc4py import SLEPc

        # Get the petc4py matrix
        PA = dolfin.as_backend_type(A).mat()

        # Calculate the largest and smallest singular value
        opts = {
            'svd_type': 'cross',
            'svd_eps_type': 'gd',
            # 'help': 'svd_type'
        }
        with petsc_options(opts):
            S = SLEPc.SVD()
            S.create()
            S.setOperator(PA)
            S.setFromOptions()
            S.setDimensions(1, PETSc.DEFAULT, PETSc.DEFAULT)
            S.setWhichSingularTriplets(SLEPc.SVD.Which.LARGEST)
            S.solve()
            if S.getConverged() == 1:
                sigma_1 = S.getSingularTriplet(0)
            else:
                raise ValueError(
                    'Could not find the highest singular value (%d)' % S.getConvergedReason()
                )
            print('Highest singular value:', sigma_1)

            S.setWhichSingularTriplets(SLEPc.SVD.Which.SMALLEST)
            S.solve()
            if S.getConverged() == 1:
                sigma_n = S.getSingularTriplet(0)
            else:
                raise ValueError(
                    'Could not find the lowest singular value (%d)' % S.getConvergedReason()
                )
            print('Lowest singular value:', sigma_n)
            print(PETSc.Options().getAll())
        print(PETSc.Options().getAll())

        return sigma_1 / sigma_n


def mat_to_scipy_csr(dolfin_matrix):
    """
    Convert any dolfin.Matrix to csr matrix in scipy.
    Based on code by Miroslav Kuchta
    """
    assert dolfin.MPI.size(dolfin.MPI.comm_world) == 1, 'mat_to_csr assumes single process'
    import scipy.sparse

    rows = [0]
    cols = []
    values = []
    for irow in range(dolfin_matrix.size(0)):
        indices, values_ = dolfin_matrix.getrow(irow)
        rows.append(len(indices) + rows[-1])
        cols.extend(indices)
        values.extend(values_)

    shape = dolfin_matrix.size(0), dolfin_matrix.size(1)

    return scipy.sparse.csr_matrix(
        (
            numpy.array(values, dtype='float'),
            numpy.array(cols, dtype='int'),
            numpy.array(rows, dtype='int'),
        ),
        shape,
    )
