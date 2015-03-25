"""
3D momentum and tracer equations for hydrostatic Boussinesq flow.

Tuomas Karna 2015-02-23
"""
from utility import *
from physical_constants import *

g_grav = physical_constants['g_grav']
wd_alpha = physical_constants['wd_alpha']
mu_manning = physical_constants['mu_manning']
z0_friction = physical_constants['z0_friction']
von_karman = physical_constants['von_karman']


class ForwardEuler(timeIntegrator):
    """Standard forward Euler time integration scheme."""
    def __init__(self, equation, dt):
        """Creates forms for the time integrator"""
        timeIntegrator.__init__(self, equation)

        massTerm = self.equation.massTerm
        RHS = self.equation.RHS
        Source = self.equation.Source

        invdt = Constant(1.0/dt)

        self.solution_old = Function(self.equation.space)

        # dict of all input functions needed for the equation
        self.funcs = self.equation.kwargs
        # create functions to hold the values of previous time step
        self.funcs_old = {}
        for k in self.funcs:
            if self.funcs[k] is not None:
                self.funcs_old[k] = Function(self.funcs[k].function_space())

        u = self.equation.solution
        u_old = self.solution_old
        u_tri = self.equation.tri

        #self.F = (invdt*massTerm(u) - invdt*massTerm(u_old) -
                  #RHS(u_old, **self.funcs_old))

        a = (invdt*massTerm(u_tri))
        L = (invdt*massTerm(u_old) +
             RHS(u_old, **self.funcs_old) +
             Source(**self.funcs_old))
        prob = LinearVariationalProblem(a, L, self.equation.solution)
        self.solver = LinearVariationalSolver(prob)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        # assing values to old functions
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])
        self.solution_old.assign(solution)

    def advance(self, t, dt, solution, updateForcings):
        """Advances equations for one time step."""
        solver_parameters = {
            'snes_type': 'newtonls',
            'snes_monitor': True,
        }
        if updateForcings is not None:
            updateForcings(t+dt/2)
        #for k in self.funcs:
            #self.funcs_old[k].assign(0.5*(self.funcs[k]+self.funcs_old[k]))

        #solve(self.F == 0, solution, solver_parameters=solver_parameters)
        self.solver.solve()
        #solve(self.A == self.L, solution)
        # store old values
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])
        self.solution_old.assign(solution)


class LeapFrogAM3(timeIntegrator):
    """Leap frog - Adams Moulton 3 predictor-corrector scheme."""
    def __init__(self, equation, dt):
        """Creates forms for the time integrator"""
        timeIntegrator.__init__(self, equation)

        massTerm = self.equation.massTerm
        RHS = self.equation.RHS
        Source = self.equation.Source

        invdt = Constant(1.0/dt)

        self.solution_nminushalf = Function(self.equation.space)
        self.solution_nplushalf = Function(self.equation.space)
        self.solution_nminusone = Function(self.equation.space)
        self.solution_n = Function(self.equation.space)

        # dict of all input functions at t_{n}
        self.funcs = self.equation.kwargs

        u = self.equation.solution
        u_tri = self.equation.tri

        self.gamma = 1.0/12.0
        self.A = (invdt*massTerm(u_tri))
        self.L_predict = (invdt*massTerm(self.solution_nminushalf) +
                          RHS(self.solution_n, **self.funcs) +
                          Source(**self.funcs))
        self.L_correct = (invdt*massTerm(self.solution_n) +
                          RHS(self.solution_nplushalf, **self.funcs) +
                          Source(**self.funcs))

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_nminusone.assign(solution)
        self.solution_n.assign(solution)

    def predict(self, t, dt, solution, updateForcings):
        """Advances equations from t_{n-1/2} to t_{n+1/2}.
        RHS is evaluated at t_{n}."""
        solver_parameters = {
            'snes_type': 'newtonls',
            'snes_monitor': True,
        }
        if updateForcings is not None:
            updateForcings(t)
        # all self.funcs_old need to be at t_{n}
        self.solution_nminushalf.assign((0.5-2*self.gamma)*self.solution_nminusone +
                                        (0.5+2*self.gamma)*self.solution_n)
        solve(self.A == self.L_predict, solution)
        self.solution_nplushalf.assign(solution)

    def correct(self, t, dt, solution, updateForcings):
        """Advances equations from t_{n} to t_{n+1}
        RHS is evaluated at t_{n+1/2}."""
        solver_parameters = {
            'snes_type': 'newtonls',
            'snes_monitor': True,
        }
        if updateForcings is not None:
            updateForcings(t+dt/2)
        # all self.funcs_nplushalf need to be at t_{n+1/2}

        solve(self.A == self.L_correct, solution)
        # shift time
        self.solution_nminusone.assign(self.solution_n)
        self.solution_n.assign(solution)


class AdamsBashforth3(timeIntegrator):
    """Standard 3rd order Adams-Bashforth scheme."""
    def __init__(self, equation, dt):
        """Creates forms for the time integrator"""
        timeIntegrator.__init__(self, equation)

        massTerm = self.equation.massTerm
        RHS = self.equation.RHS
        Source = self.equation.Source

        self.solution_old = Function(self.equation.space)

        self.K1 = Function(self.equation.space)
        self.K2 = Function(self.equation.space)
        self.K3 = Function(self.equation.space)

        # dict of all input functions needed for the equation
        self.funcs = self.equation.kwargs
        # create functions to hold the values of previous time step
        self.funcs_old = {}
        for k in self.funcs:
            self.funcs_old[k] = Function(self.funcs[k].function_space())
        # assing values to old functions
        for k in self.funcs:
            self.funcs_old[k].assign(self.funcs[k])

        u = self.equation.solution
        u_old = self.solution_old
        u_tri = self.equation.tri

        # mass matrix for a linear equation
        self.A = massTerm(u_tri)
        self.RHS = RHS(u_old, **self.funcs_old) + Source(**self.funcs_old)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)

    def advance(self, t, dt, solution, updateForcings):
        """Advances equations for one time step."""
        if updateForcings is not None:
            updateForcings(t+dt/2)
        #for k in self.funcs:
            #self.funcs_old[k].assign(0.5*(self.funcs[k]+self.funcs_old[k]))

        solve(self.A == self.RHS, self.K1)
        K1_mix = 23.0/12.0
        K2_mix = -4.0/3.0
        K3_mix = 5.0/12.0
        solution.assign(self.solution_old +
                        dt*K1_mix*self.K1 +
                        dt*K2_mix*self.K2 +
                        dt*K3_mix*self.K3)
        # shift tendencies for next time step
        self.K3.assign(self.K2)
        self.K2.assign(self.K1)
        # store old values
        for k in self.funcs:
            self.funcs_old[k].assign(self.funcs[k])
        self.solution_old.assign(solution)


class SSPRK33(timeIntegrator):
    """
    3rd order Strong Stability Preserving Runge-Kutta scheme, SSP(3,3).

    This scheme has Butcher tableau
    0   |
    1   | 1
    1/2 | 1/4 1/4
    ---------------
        | 1/6 1/6 2/3

    CFL coefficient is 1.0
    """
    def __init__(self, equation, dt):
        """Creates forms for the time integrator"""
        timeIntegrator.__init__(self, equation)

        massTerm = self.equation.massTerm
        RHS = self.equation.RHS
        Source = self.equation.Source

        self.solution_old = Function(self.equation.space)

        self.K0 = Function(self.equation.space)
        self.K1 = Function(self.equation.space)
        self.K2 = Function(self.equation.space)

        # dict of all input functions needed for the equation
        self.funcs = self.equation.kwargs
        # create functions to hold the values of previous time step
        self.funcs_old = {}
        for k in self.funcs:
            if self.funcs[k] is not None:
                self.funcs_old[k] = Function(self.funcs[k].function_space())
        # values used in equations
        self.args = {}
        for k in self.funcs_old:
            self.args[k] = Function(self.funcs[k].function_space())

        u_old = self.solution_old
        u_tri = self.equation.tri

        dt_const = Constant(dt)
        a_RK = massTerm(u_tri)
        L_RK = dt_const*(RHS(u_old, **self.args) + Source(**self.args))

        probK0 = LinearVariationalProblem(a_RK, L_RK, self.K0)
        self.solverK0 = LinearVariationalSolver(probK0)
        probK1 = LinearVariationalProblem(a_RK, L_RK, self.K1)
        self.solverK1 = LinearVariationalSolver(probK1)
        probK2 = LinearVariationalProblem(a_RK, L_RK, self.K2)
        self.solverK2 = LinearVariationalSolver(probK2)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)
        # assing values to old functions
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])

    def advance(self, t, dt, solution, updateForcings):
        """Advances equations for one time step."""

        # stage 0
        for k in self.args:  # set args to t
            self.args[k].assign(self.funcs_old[k])
        if updateForcings is not None:
            updateForcings(t)
        self.solverK0.solve()
        # stage 1
        self.solution_old.assign(solution + self.K0)
        for k in self.args:  # set args to t+dt
            self.args[k].assign(self.funcs[k])
        if updateForcings is not None:
            updateForcings(t+dt)
        self.solverK1.solve()
        # stage 2
        self.solution_old.assign(solution + 0.25*self.K0 + 0.25*self.K1)
        for k in self.args:  # set args to t+dt/2
            self.args[k].assign(0.5*self.funcs[k] + 0.5*self.funcs_old[k])
        if updateForcings is not None:
            updateForcings(t+dt/2)
        self.solverK2.solve()
        # final solution
        solution.assign(solution + (1.0/6.0)*self.K0 + (1.0/6.0)*self.K1 +
                        (2.0/3.0)*self.K2)

        # store old values
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])
        self.solution_old.assign(solution)


class CrankNicolson(timeIntegrator):
    """Standard Crank-Nicolson time integration scheme."""
    def __init__(self, equation, dt, gamma=0.6):
        """Creates forms for the time integrator"""
        timeIntegrator.__init__(self, equation)

        massTerm = self.equation.massTerm
        RHS = self.equation.RHS
        Source = self.equation.Source

        invdt = Constant(1.0/dt)

        self.solution_old = Function(self.equation.space)

        # dict of all input functions needed for the equation
        self.funcs = self.equation.kwargs
        # create functions to hold the values of previous time step
        self.funcs_old = {}
        for k in self.funcs:
            if self.funcs[k] is not None:
                self.funcs_old[k] = Function(self.funcs[k].function_space())

        u = self.equation.solution
        u_old = self.solution_old
        u_tri = self.equation.tri
        #Crank-Nicolson
        gamma_const = Constant(gamma)
        self.F = (invdt*massTerm(u) - invdt*massTerm(u_old) -
                  gamma_const*RHS(u, **self.funcs) -
                  gamma_const*Source(**self.funcs) -
                  (1-gamma_const)*RHS(u_old, **self.funcs_old) -
                  (1-gamma_const)*Source(**self.funcs_old))

        self.A = (invdt*massTerm(u_tri) -
                  gamma_const*RHS(u_tri, **self.funcs))
        self.L = (invdt*massTerm(u_old) + gamma_const*Source(**self.funcs) +
                  (1-gamma_const)*RHS(u_old, **self.funcs_old) +
                  (1-gamma_const)*Source(**self.funcs_old))

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)
        # assing values to old functions
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])

    def advance(self, t, dt, solution, updateForcings):
        """Advances equations for one time step."""
        solver_parameters = {
            'snes_type': 'newtonls',
            'snes_monitor': True,
        }
        if updateForcings is not None:
            updateForcings(t+dt)
        self.solution_old.assign(solution)
        solve(self.F == 0, solution, solver_parameters=solver_parameters)
        # shift time
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])

    def advanceLinear(self, t, dt, solution, updateForcings):
        """Advances equations for one time step."""
        solver_parameters = {
            'snes_type': 'ksponly',
        }
        if updateForcings is not None:
            updateForcings(t+dt)
        self.solution_old.assign(solution)
        solve(self.A == self.L, solution, solver_parameters=solver_parameters)
        # shift time
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])


class momentumEquation(equation):
    """3D momentum equation for hydrostatic Boussinesq flow."""
    def __init__(self, mesh, space, space_scalar, bnd_markers, bnd_len,
                 solution, eta, bathymetry, w=None,
                 w_mesh=None, dw_mesh_dz=None,
                 uv_bottom=None, bottom_drag=None, viscosity_v=None,
                 nonlin=True):
        self.mesh = mesh
        self.space = space
        self.space_scalar = space_scalar
        self.nonlin = nonlin
        self.solution = solution
        # this dict holds all time dep. args to the equation
        self.kwargs = {'eta': eta,
                       'w': w,
                       'w_mesh': w_mesh,
                       'dw_mesh_dz': dw_mesh_dz,
                       'uv_bottom': uv_bottom,
                       'bottom_drag': bottom_drag,
                       'viscosity_v': viscosity_v,
                       }
        # time independent arg
        self.bathymetry = bathymetry

        # test and trial functions
        self.test = TestFunction(self.space)
        self.tri = TrialFunction(self.space)

        self.horizontal_DG = self.space.ufl_element()._A.family() != 'Lagrange'
        self.vertical_DG = self.space.ufl_element()._B.family() != 'Lagrange'

        # mesh dependent variables
        self.normal = FacetNormal(mesh)
        self.cellsize = CellSize(mesh)
        self.xyz = SpatialCoordinate(mesh)
        self.e_x, self.e_y, self.e_y = unit_vectors(3)

        # integral measures
        self.dx = Measure('dx', domain=self.mesh, subdomain_id='everywhere')
        self.dS_v = dS_v(domain=self.mesh)
        self.dS_h = dS_h(domain=self.mesh)

        # boundary definitions
        self.boundary_markers = bnd_markers
        self.boundary_len = bnd_len

        ## compute lenth of all boundaries
        #self.boundary_len = {}
        #for i in self.boundary_markers:
            ##ds_restricted = Measure('ds', subdomain_id=int(i))
            ##one_func = Function(self.space_scalar).interpolate(Expression(1.0))
            ##self.boundary_len[i] = assemble(one_func * ds_restricted)
            ##print 'bnd', i, self.boundary_len[i]

        # set boundary conditions
        # maps bnd_marker to dict of external functions e.g. {'elev':eta_ext}
        self.bnd_functions = {}

    def ds_v(self, bnd_marker):
        """Returns boundary measure for the appropriate mesh"""
        return ds_v(int(bnd_marker), domain=self.mesh)

    def getTimeStep(self, Umag=Constant(1.0)):
        csize = CellSize(self.mesh)
        H = FunctionSpace(self.mesh, 'CG', 1, vfamily='CG', vdegree=1)
        uu = TestFunction(H)
        grid_dt = TrialFunction(H)
        res = Function(H)
        a = uu * grid_dt * self.dx
        L = uu * csize / Umag * self.dx
        solve(a == L, res)
        return res

    def massTerm(self, solution):
        """All time derivative terms on the LHS, without the actual time
        derivative.

        Implements A(u) for  d(A(u_{n+1}) - A(u_{n}))/dt
        """
        return inner(solution, self.test) * self.dx
        #return (solution[0]*self.test[0] + solution[1]*self.test[1]) * self.dx

    def RHS(self, solution, eta, w=None, viscosity_v=None,
            uv_bottom=None, bottom_drag=None,
            w_mesh=None, dw_mesh_dz=None, **kwargs):
        """Returns the right hand side of the equations.
        RHS is all terms that depend on the solution (eta,uv)"""
        F = 0  # holds all dx volume integral terms
        G = 0  # holds all ds boundary interface terms

        # Advection term
        if self.nonlin:
            # in 3d: nabla_grad dot (u u)
            # weak form: nabla_grad(psi) : u u
            # NOTE not validated (term small in channel flow)
            Adv_h = -(Dx(self.test[0], 0)*solution[0]*solution[0] +
                      Dx(self.test[0], 1)*solution[0]*solution[1] +
                      Dx(self.test[1], 0)*solution[1]*solution[0] +
                      Dx(self.test[1], 1)*solution[1]*solution[1])
            F += Adv_h * self.dx
            if self.horizontal_DG:
                uv_rie = avg(solution)
                s = 0.5*(sign(uv_rie[0]*self.normal('-')[0] +
                              uv_rie[1]*self.normal('-')[1]) + 1.0)
                uv_up = solution('-')*s + solution('+')*(1-s)
                G += (uv_up[0]*jump(self.test[0], self.normal[0]*solution[0]) +
                      uv_up[0]*jump(self.test[0], self.normal[1]*solution[1]) +
                      uv_up[1]*jump(self.test[1], self.normal[0]*solution[0]) +
                      uv_up[1]*jump(self.test[1], self.normal[1]*solution[1]))*(self.dS_v)
                # Lax-Friedrichs stabilization
                gamma = abs(self.normal[0]('-')*avg(solution[0]) +
                            self.normal[1]('-')*avg(solution[1]))
                G += gamma*dot(jump(self.test), jump(solution))*self.dS_v
            if self.vertical_DG:
                # NOTE bottom bnd doesn't work for DG vertical mesh
                G += (uv_rie[0]*uv_rie[0]*jump(self.test[0], self.normal[0]) +
                      uv_rie[0]*uv_rie[1]*jump(self.test[0], self.normal[1]) +
                      uv_rie[1]*uv_rie[0]*jump(self.test[1], self.normal[0]) +
                      uv_rie[1]*uv_rie[1]*jump(self.test[1], self.normal[1]))*(self.dS_h)
            G += (solution[0]*solution[0]*self.test[0]*self.normal[0] +
                  solution[0]*solution[1]*self.test[0]*self.normal[1] +
                  solution[1]*solution[0]*self.test[1]*self.normal[0] +
                  solution[1]*solution[1]*self.test[1]*self.normal[1])*(ds_t + ds_b)
            # Vertical advection term
            vertvelo = w
            if w_mesh is not None:
                vertvelo = w-w_mesh
            #F += -solution*vertvelo*Dx(self.test, 2)*dx
            # Vertical advection
            Adv_v = -(Dx(self.test[0], 2)*solution[0]*vertvelo +
                      Dx(self.test[1], 2)*solution[1]*vertvelo)
            F += Adv_v * self.dx
            G += (solution[0]*vertvelo*self.test[0]*self.normal[2] +
                  solution[1]*vertvelo*self.test[1]*self.normal[2])*(ds_t + ds_b)
            #if self.horizontal_DG:
                #w_rie = avg(w)
                #uv_rie = avg(solution)
                #G += (uv_rie[0]*w_rie*jump(self.test[0], self.normal[2]) +
                      #uv_rie[1]*w_rie*jump(self.test[1], self.normal[2]))*self.dS_h

            # Non-conservative ALE source term
            if dw_mesh_dz is not None:
                F += solution*dw_mesh_dz*self.test*dx

        if self.nonlin:
            total_H = self.bathymetry + eta
        else:
            total_H = self.bathymetry

        # boundary conditions
        for bnd_marker in self.boundary_markers:
            funcs = self.bnd_functions.get(bnd_marker)
            ds_bnd = ds_v(int(bnd_marker), domain=self.mesh)
            if funcs is None:
                # assume land boundary
                #continue
                if self.nonlin:
                    G += (self.test[0]*solution[0]*solution[0]*self.normal[0] +
                          self.test[0]*solution[0]*solution[1]*self.normal[1] +
                          self.test[1]*solution[1]*solution[0]*self.normal[0] +
                          self.test[1]*solution[1]*solution[1]*self.normal[1])*(ds_bnd)

            elif 'elev' in funcs:
                # prescribe elevation only
                h_ext = funcs['elev']
                uv_ext = solution
                t = self.normal[1] * self.e_x - self.normal[0] * self.e_y
                ut_in = dot(solution, t)
                # ut_ext = -dot(uv_ext,t) # assume zero
                un_in = dot(solution, self.normal)
                un_ext = dot(uv_ext, self.normal)

                if self.nonlin:
                    H = self.bathymetry + (eta + h_ext) / 2
                else:
                    H = self.bathymetry
                c_roe = sqrt(g_grav * H)
                un_riemann = dot(solution, self.normal) + c_roe / H * (eta - h_ext)/2
                H_riemann = H
                ut_riemann = tanh(4 * un_riemann / 0.02) * (ut_in)
                uv_riemann = un_riemann * self.normal + ut_riemann * t

                if self.nonlin:
                    # NOTE just symmetric 3D flux with 2D eta correction
                    G += un_riemann * un_riemann * dot(self.normal, self.test) * ds_bnd

            elif 'un' in funcs:
                # prescribe normal volume flux
                un_in = dot(solution, self.normal)
                un_ext = funcs['un']
                if self.nonlin:
                    un_av = 0.5*(un_ext+un_in)
                    G += un_av*un_av*inner(self.normal, self.test)*ds_bnd
                    # Lax-Friedrichs stabilization
                    gamma = abs(self.normal[0]*(solution[0]) +
                                self.normal[1]*(solution[1]))
                    G += gamma*dot((self.test), (solution-self.normal*un_ext)/2)*ds_bnd

            elif 'flux' in funcs:
                # prescribe normal volume flux
                sect_len = Constant(self.boundary_len[bnd_marker])
                un_in = dot(solution, self.normal)
                un_ext = funcs['flux'] / total_H / sect_len
                if self.nonlin:
                    un_av = 0.5*(un_ext+un_in)
                    G += un_av*un_av*inner(self.normal, self.test)*ds_bnd
                    # Lax-Friedrichs stabilization
                    gamma = abs(self.normal[0]*(solution[0]) +
                                self.normal[1]*(solution[1]))
                    G += gamma*dot((self.test), (solution-self.normal*un_ext)/2)*ds_bnd

                #sect_len = Constant(self.boundary_len[bnd_marker])
                #un_in = dot(solution, self.normal)
                #un_ext = funcs['flux'] / total_H / sect_len
                #if self.nonlin:
                    ## NOTE symmetric normal flux -- forced in 2D
                    #G += un_in*un_in*inner(self.normal, self.test)*ds_bnd

        ## viscosity
        ## A double dot product of the stress tensor and grad(w).
        #K_momentum = -viscosity * (Dx(uv[0], 0) * Dx(self.U_test[0], 0) +
                                   #Dx(uv[0], 1) * Dx(self.U_test[0], 1) +
                                   #Dx(uv[1], 0) * Dx(self.U_test[1], 0) +
                                   #Dx(uv[1], 1) * Dx(self.U_test[1], 1))
        #K_momentum += viscosity/total_H*inner(dot(grad(total_H), grad(uv)),
                                              #self.U_test)
        #F -= K_momentum * self.dx

        # vertical viscosity
        if viscosity_v is not None:
            F += viscosity_v*(Dx(self.test[0], 2)*Dx(solution[0], 2) +
                              Dx(self.test[1], 2)*Dx(solution[1], 2))*dx
            if self.vertical_DG:
                raise NotImplementedError('Vertical diffusion has not been implemented for DG')
                # G += -viscosity_v * dot(psi, du/dz) * normal[2]
                # viscflux = viscosity_v*Dx(solution, 2)
                # G += -(avg(viscflux[0])*jump(self.test[0], normal[2]) +
                #        avg(viscflux[0])*jump(self.test[1], normal[2]))

        return -F - G

    def Source(self, eta, w=None, viscosity_v=None,
               uv_bottom=None, bottom_drag=None, **kwargs):
        """Returns the right hand side of the source terms.
        These terms do not depend on the solution."""
        F = 0  # holds all dx volume integral terms
        G = 0

        # external pressure gradient
        F += g_grav * inner(nabla_grad(eta), self.test) * self.dx

        if self.nonlin:
            total_H = self.bathymetry + eta
        else:
            total_H = self.bathymetry

        # boundary conditions
        for bnd_marker in self.boundary_markers:
            funcs = self.bnd_functions.get(bnd_marker)
            ds_bnd = ds_v(int(bnd_marker), domain=self.mesh)
            if funcs is None:
                # assume land boundary
                continue

            elif 'elev' in funcs:
                # prescribe elevation only
                h_ext = funcs['elev']
                G += g_grav * ((h_ext - eta) / 2) * \
                    inner(self.normal, self.test) * ds_bnd

            elif 'flux' in funcs:
                # prescribe normal volume flux
                sect_len = Constant(self.boundary_len[bnd_marker])
                un_ext = funcs['flux'] / total_H / sect_len
                if self.nonlin:
                    G += un_ext*un_ext*inner(self.normal, self.test)*ds_bnd

        if viscosity_v is not None:
            # bottom friction
            if bottom_drag is not None and uv_bottom is not None:
                stress = bottom_drag*sqrt(uv_bottom[0]**2 +
                                          uv_bottom[1]**2)*uv_bottom
                BotFriction = (stress[0]*self.test[0] +
                               stress[1]*self.test[1])*ds_t
                F += BotFriction

        return -F


class verticalMomentumEquation(equation):
    """Vertical advection and diffusion terms of 3D momentum equation for
    hydrostatic Boussinesq flow."""
    def __init__(self, mesh, space, space_scalar, solution, w=None,
                 viscosity_v=None, uv_bottom=None, bottom_drag=None):
        self.mesh = mesh
        self.space = space
        self.space_scalar = space_scalar
        self.solution = solution
        # this dict holds all time dep. args to the equation
        self.kwargs = {'w': w,
                       'viscosity_v': viscosity_v,
                       'uv_bottom': uv_bottom,
                       'bottom_drag': bottom_drag,
                       }

        # test and trial functions
        self.test = TestFunction(self.space)
        self.tri = TrialFunction(self.space)

        self.horizontal_DG = self.space.ufl_element()._A.family() != 'Lagrange'
        self.vertical_DG = self.space.ufl_element()._B.family() != 'Lagrange'

        # mesh dependent variables
        self.normal = FacetNormal(mesh)
        self.cellsize = CellSize(mesh)
        self.xyz = SpatialCoordinate(mesh)
        self.e_x, self.e_y, self.e_y = unit_vectors(3)

        # integral measures
        self.dx = Measure('dx', domain=self.mesh, subdomain_id='everywhere')
        self.dS_v = dS_v(domain=self.mesh)
        self.dS_h = dS_h(domain=self.mesh)

        # set boundary conditions
        # maps bnd_marker to dict of external functions e.g. {'elev':eta_ext}
        self.bnd_functions = {}

    def ds_v(self, bnd_marker):
        """Returns boundary measure for the appropriate mesh"""
        return ds_v(int(bnd_marker), domain=self.mesh)

    def getTimeStep(self, Umag=Constant(1.0)):
        raise NotImplementedError('getTimeStep not implemented')

    def massTerm(self, solution):
        """All time derivative terms on the LHS, without the actual time
        derivative.

        Implements A(u) for  d(A(u_{n+1}) - A(u_{n}))/dt
        """
        return inner(solution, self.test) * self.dx
        #return (solution[0]*self.test[0] + solution[1]*self.test[1]) * self.dx

    def RHS(self, solution, w=None, viscosity_v=None,
            uv_bottom=None, bottom_drag=None, **kwargs):
        """Returns the right hand side of the equations.
        Contains all terms that depend on the solution."""
        F = 0  # holds all dx volume integral terms
        G = 0  # holds all ds boundary interface terms

        # Advection term
        if w is not None:
            # Vertical advection
            Adv_v = -(Dx(self.test[0], 2)*solution[0]*w +
                      Dx(self.test[1], 2)*solution[1]*w)
            F += Adv_v * self.dx
            if self.horizontal_DG:
                raise NotImplementedError('Adv term not implemented for DG')
                #w_rie = avg(w)
                #uv_rie = avg(solution)
                #G += (uv_rie[0]*w_rie*jump(self.test[0], self.normal[2]) +
                      #uv_rie[1]*w_rie*jump(self.test[1], self.normal[2]))*self.dS_h

        # vertical viscosity
        if viscosity_v is not None:
            F += viscosity_v*(Dx(self.test[0], 2)*Dx(solution[0], 2) +
                              Dx(self.test[1], 2)*Dx(solution[1], 2))*dx
            if self.vertical_DG:
                raise NotImplementedError('Vertical diffusion has not been implemented for DG')
                # G += -viscosity_v * dot(psi, du/dz) * normal[2]
                # viscflux = viscosity_v*Dx(solution, 2)
                # G += -(avg(viscflux[0])*jump(self.test[0], normal[2]) +
                #        avg(viscflux[0])*jump(self.test[1], normal[2]))

        return -F - G

    def Source(self, w=None, viscosity_v=None,
               uv_bottom=None, bottom_drag=None, **kwargs):
        """Returns the right hand side of the source terms.
        These terms do not depend on the solution."""
        F = 0  # holds all dx volume integral terms

        if viscosity_v is not None:
            # bottom friction
            if bottom_drag is not None and uv_bottom is not None:
                stress = bottom_drag*sqrt(uv_bottom[0]**2 +
                                          uv_bottom[1]**2)*uv_bottom
                BotFriction = (stress[0]*self.test[0] +
                               stress[1]*self.test[1])*ds_t
                F += BotFriction

        return -F


class tracerEquation(equation):
    """3D tracer advection-diffusion equation"""
    def __init__(self, mesh, space, solution, eta, uv, w,
                 w_mesh=None, dw_mesh_dz=None,
                 diffusivity_h=None,
                 test_supg_h=None, test_supg_v=None, test_supg_mass=None,
                 nonlinStab_h=None, nonlinStab_v=None,
                 bnd_markers=None, bnd_len=None, nonlin=True):
        self.mesh = mesh
        self.space = space
        # this dict holds all args to the equation (at current time step)
        self.solution = solution
        self.kwargs = {'eta': eta,
                       'uv': uv,
                       'w': w,
                       'w_mesh': w_mesh,
                       'dw_mesh_dz': dw_mesh_dz,
                       'diffusivity_h': diffusivity_h,
                       'nonlinStab_h': nonlinStab_h,
                       'nonlinStab_v': nonlinStab_v}
        # SUPG terms (add to forms)
        self.test_supg_h = test_supg_h
        self.test_supg_v = test_supg_v
        self.test_supg_mass = test_supg_mass

        # trial and test functions
        self.test = TestFunction(self.space)
        self.tri = TrialFunction(self.space)

        self.horizontal_DG = self.space.ufl_element()._A.family() != 'Lagrange'
        self.vertical_DG = self.space.ufl_element()._B.family() != 'Lagrange'

        # mesh dependent variables
        self.normal = FacetNormal(mesh)
        self.cellsize = CellSize(mesh)
        self.xyz = SpatialCoordinate(mesh)
        self.e_x, self.e_y, self.e_y = unit_vectors(3)

        # integral measures
        self.dx = Measure('dx', domain=self.mesh, subdomain_id='everywhere')
        self.dS_v = dS_v(domain=self.mesh)
        self.dS_h = dS_h(domain=self.mesh)

        # boundary definitions
        self.boundary_markers = bnd_markers
        self.boundary_len = bnd_len

        # maps bnd_marker to dict of external functions e.g. {'elev':eta_ext}
        self.bnd_functions = {}

    def ds_v(self, bnd_marker):
        """Returns boundary measure for the appropriate mesh"""
        return ds_v(int(bnd_marker), domain=self.mesh)

    def massTerm(self, solution):
        """All time derivative terms on the LHS, without the actual time
        derivative.

        Implements A(u) for  d(A(u_{n+1}) - A(u_{n}))/dt
        """
        test = self.test
        if self.test_supg_mass is not None:
            test = self.test + self.test_supg_mass
        return inner(solution, test) * self.dx

    def RHS(self, solution, eta, uv, w, w_mesh=None, dw_mesh_dz=None,
            diffusivity_h=None, nonlinStab_h=None, nonlinStab_v=None, **kwargs):
        """Returns the right hand side of the equations.
        RHS is all terms that depend on the solution (eta,uv)"""
        F = 0  # holds all dx volume integral terms
        G = 0  # holds all ds boundary interface terms

        #test_h = self.test
        #if self.test_supg_h is not None:
            #test_h = self.test + self.test_supg_h

        # NOTE advection terms must be exactly as in 3d continuity equation
        # Horizontal advection term
        F += -solution*(uv[0]*Dx(self.test, 0) +
                        uv[1]*Dx(self.test, 1))*self.dx
        #F += self.test*(uv[0]*Dx(solution, 0) +
                        #uv[1]*Dx(solution, 1))*self.dx
        # Vertical advection term
        vertvelo = w
        if w_mesh is not None:
            vertvelo = w-w_mesh
        F += -solution*vertvelo*Dx(self.test, 2)*self.dx

        # Non-conservative ALE source term
        if dw_mesh_dz is not None:
            F += solution*dw_mesh_dz*self.test*self.dx

        # Bottom/top impermeability boundary conditions
        G += +solution*(uv[0]*self.normal[0] +
                        uv[1]*self.normal[1])*self.test*(ds_t + ds_b)
        # TODO what is the correct free surf bnd condition?
        G += +solution*vertvelo*self.normal[2]*self.test*(ds_t + ds_b)

        # boundary conditions
        for bnd_marker in self.boundary_markers:
            funcs = self.bnd_functions.get(bnd_marker)
            ds_bnd = ds_v(int(bnd_marker), domain=self.mesh)
            if funcs is None:
                # assume land boundary NOTE uv.n should be very close to 0
                #continue
                G += solution*(self.normal[0]*uv[0] +
                               self.normal[1]*uv[1])*self.test*ds_bnd

            elif 'value' in funcs:
                # prescribe external tracer value
                nudge = 1.00
                c_in = solution
                c_ext = nudge*funcs['value'] + (1-nudge)*c_in
                un = self.normal[0]*uv[0] + self.normal[1]*uv[1]
                alpha = 0.5*(tanh(4 * un / 0.02) + 1)
                c_up = alpha*c_in + (1-alpha)*c_ext  # for inv.part adv term
                #c_up = (1-alpha)*(c_ext-c_in)/4  # for direct adv term
                G += c_up*un*self.test*ds_bnd

            #if diffusivity_h is not None:
                #dflux = diffusivity_h*(Dx(solution, 0)*self.normal[0] +
                                       #Dx(solution, 1)*self.normal[1])/2
                #G += -dflux*self.test*ds_bnd


        # diffusion
        if diffusivity_h is not None:
            F += diffusivity_h*(Dx(solution, 0)*Dx(self.test, 0) +
                                Dx(solution, 1)*Dx(self.test, 1))*self.dx

        # SUPG stabilization
        if self.test_supg_h is not None:
            F += self.test_supg_h*(uv[0]*Dx(solution, 0) +
                                   uv[1]*Dx(solution, 1))*self.dx
            if diffusivity_h is not None:
                F += -diffusivity_h*self.test_supg_h*(Dx(Dx(solution, 0), 0) +
                                                      Dx(Dx(solution, 1), 1))*self.dx
        if self.test_supg_v is not None:
            F += self.test_supg_v*vertvelo*Dx(solution, 2)*self.dx

        # non-linear damping
        if nonlinStab_h is not None:
            F += nonlinStab_h*(Dx(solution, 0)*Dx(self.test, 0) +
                               Dx(solution, 1)*Dx(self.test, 1))*self.dx

        if nonlinStab_v is not None:
            F += nonlinStab_v*(Dx(solution, 2)*Dx(self.test, 2))*self.dx

        return -F - G

    def Source(self, eta, uv, w, **kwargs):
        """Returns the right hand side of the source terms.
        These terms do not depend on the solution."""
        F = 0  # holds all dx volume integral terms

        return -F


class freeSurfaceEquations3d(equation):
    """3D shallow water equations"""
    def __init__(self, mesh, space, bnd_markers, bnd_len,
                 solution, bathymetry, w=None,
                 uv_bottom=None, bottom_drag=None, viscosity_v=None,
                 nonlin=True):
        self.mesh = mesh
        self.space = space
        self.eta_space, self.U_space = self.space.split()
        self.nonlin = nonlin

        self.solution = solution
        # this dict holds all time dep. args to the equation
        self.kwargs = {'w': w,
                       'uv_bottom': uv_bottom,
                       'bottom_drag': bottom_drag,
                       'viscosity_v': viscosity_v,
                       }
        # time independent arg
        self.bathymetry = bathymetry

        # create mixed function space
        self.test = TestFunction(self.space)
        self.tri = TrialFunction(self.space)
        self.U_test, self.eta_test = TestFunctions(self.space)
        self.U_tri, self.eta_tri = TrialFunctions(self.space)

        self.uv_horizontal_DG = self.U_space.ufl_element()._A.family() != 'Lagrange'
        self.uv_vertical_DG = self.U_space.ufl_element()._B.family() != 'Lagrange'
        self.eta_horizontal_DG = self.eta_space.ufl_element()._A.family() != 'Lagrange'
        self.eta_vertical_DG = self.eta_space.ufl_element()._B.family() != 'Lagrange'

        # mesh dependent variables
        self.normal = FacetNormal(mesh)
        self.cellsize = CellSize(mesh)
        self.xyz = SpatialCoordinate(mesh)
        self.e_x, self.e_y = unit_vectors(2)

        # integral measures
        self.dx = Measure('dx', domain=self.mesh, subdomain_id='everywhere')
        self.dS = dS(domain=self.mesh)
        self.dS_v = dS_v(domain=self.mesh)
        self.dS_h = dS_h(domain=self.mesh)

        # boundary definitions
        self.boundary_markers = bnd_markers
        self.boundary_len = bnd_len

        # set boundary conditions
        # maps bnd_marker to dict of external functions e.g. {'elev':eta_ext}
        self.bnd_functions = {}

    def ds_v(self, bnd_marker):
        """Returns boundary measure for the appropriate mesh"""
        return ds_v(int(bnd_marker), domain=self.mesh)

    def massTerm(self, solution):
        """All time derivative terms on the LHS, without the actual time
        derivative.

        Implements A(u) for  d(A(u_{n+1}) - A(u_{n}))/dt
        """
        F = 0
        uv, eta = split(solution)

        # Mass term of momentum equation
        M_momentum = inner(uv, self.U_test)
        F += M_momentum

        # Mass term of free surface equation
        M_continuity = inner(eta, self.eta_test)
        F += M_continuity

        return F * self.dx

    def RHS(self, solution, w=None, viscosity_v=None,
            uv_bottom=None, bottom_drag=None, **kwargs):
        """Returns the right hand side of the equations.
        RHS is all terms that depend on the solution (eta,uv)"""
        F = 0  # holds all dx volume integral terms
        G = 0  # holds all ds boundary interface terms
        uv, eta = split(solution)

        # Advection term
        if self.nonlin:
            # in 3d: nabla_grad dot (u u)
            # weak form: nabla_grad(psi) : u u
            # NOTE not validated (term small in channel flow)
            Adv_h = -(Dx(self.U_test[0], 0)*uv[0]*uv[0] +
                      Dx(self.U_test[0], 1)*uv[0]*uv[1] +
                      Dx(self.U_test[1], 0)*uv[1]*uv[0] +
                      Dx(self.U_test[1], 1)*uv[1]*uv[1])
            F += Adv_h * self.dx
            if self.uv_horizontal_DG:
                uv_rie = avg(uv)
                s = 0.5*(sign(uv_rie[0]*self.normal('-')[0] +
                              uv_rie[1]*self.normal('-')[1]) + 1.0)
                uv_up = uv('-')*s + uv('+')*(1-s)
                G += (uv_up[0]*uv_rie[0]*jump(self.U_test[0], self.normal[0]) +
                      uv_up[0]*uv_rie[1]*jump(self.U_test[0], self.normal[1]) +
                      uv_up[1]*uv_rie[0]*jump(self.U_test[1], self.normal[0]) +
                      uv_up[1]*uv_rie[1]*jump(self.U_test[1], self.normal[1]))*(self.dS_v)
                # Lax-Friedrichs stabilization
                #gamma = Constant(1.0)
                #G += gamma*dot(jump(self.U_test), jump(uv))*self.dS_v
            if self.uv_vertical_DG:
                # NOTE bottom bnd doesn't work for DG vertical mesh
                G += (uv_rie[0]*uv_rie[0]*jump(self.U_test[0], self.normal[0]) +
                      uv_rie[0]*uv_rie[1]*jump(self.U_test[0], self.normal[1]) +
                      uv_rie[1]*uv_rie[0]*jump(self.U_test[1], self.normal[0]) +
                      uv_rie[1]*uv_rie[1]*jump(self.U_test[1], self.normal[1]))*(self.dS_h)
            G += (uv[0]*uv[0]*self.U_test[0]*self.normal[0] +
                  uv[0]*uv[1]*self.U_test[0]*self.normal[1] +
                  uv[1]*uv[0]*self.U_test[1]*self.normal[0] +
                  uv[1]*uv[1]*self.U_test[1]*self.normal[1])*(ds_t + ds_b)
            ## Vertical advection
            #Adv_v = -(Dx(self.U_test[0], 2)*uv[0]*w +
                      #Dx(self.U_test[1], 2)*uv[1]*w)
            #F += Adv_v * self.dx
            ##if self.horizontal_DG:
                ##w_rie = avg(w)
                ##uv_rie = avg(uv)
                ##G += (uv_rie[0]*w_rie*jump(self.U_test[0], self.normal[2]) +
                      ##uv_rie[1]*w_rie*jump(self.U_test[1], self.normal[2]))*self.dS_h

        # External pressure gradient
        F += g_grav * inner(nabla_grad(eta), self.U_test) * self.dx

        if self.nonlin:
            total_H = self.bathymetry + eta
        else:
            total_H = self.bathymetry

        # Divergence of depth-integrated velocity
        F += -total_H * inner(uv, nabla_grad(self.eta_test)) * self.dx
        if self.eta_horizontal_DG:
            Hu_star = avg(total_H*uv) +\
                sqrt(g_grav*avg(total_H))*jump(total_H, self.normal)
            G += inner(jump(self.eta_test, self.normal), Hu_star)*self.dS_v

        # boundary conditions
        for bnd_marker in self.boundary_markers:
            funcs = self.bnd_functions.get(bnd_marker)
            ds_bnd = ds(int(bnd_marker), domain=self.mesh)
            if funcs is None:
                # assume land boundary
                continue

        # vertical viscosity
        if viscosity_v is not None:
            F += viscosity_v*(Dx(self.U_test[0], 2)*Dx(uv[0], 2) +
                              Dx(self.U_test[1], 2)*Dx(uv[1], 2))*dx
            if self.vertical_DG:
                raise NotImplementedError('Vertical diffusion has not been implemented for DG')

        return -F - G

    def Source(self, w=None, viscosity_v=None,
               uv_bottom=None, bottom_drag=None, **kwargs):
        """Returns the right hand side of the source terms.
        These terms do not depend on the solution."""
        F = 0  # holds all dx volume integral terms
        G = 0

        if viscosity_v is not None:
            # bottom friction
            if bottom_drag is not None and uv_bottom is not None:
                stress = bottom_drag*sqrt(uv_bottom[0]**2 +
                                          uv_bottom[1]**2)*uv_bottom
                BotFriction = (stress[0]*self.U_test[0] +
                               stress[1]*self.U_test[1])*ds_t
                F += BotFriction

        return -F
