"""
Module for coupled 2D-3D flow solver.

Tuomas Karna 2015-04-01
"""
from __future__ import absolute_import
from .utility import *
from . import shallowwater_eq
from . import momentum_eq
from . import tracer_eq
from . import turbulence
from . import coupled_timeintegrator
import thetis.limiter as limiter
import time as time_mod
from mpi4py import MPI
from . import exporter
import weakref
from .field_defs import field_metadata
from .options import ModelOptions
from . import callback
from .log import *


class FlowSolver(FrozenClass):
    """Creates and solves coupled 2D-3D equations"""
    def __init__(self, mesh2d, bathymetry_2d, n_layers,
                 options=None):
        self._initialized = False

        self.bathymetry_cg_2d = bathymetry_2d

        # create 3D mesh
        self.mesh2d = mesh2d
        self.mesh = extrude_mesh_sigma(mesh2d, n_layers, bathymetry_2d)

        self.comm = mesh2d.comm
        # add boundary length info
        bnd_len = compute_boundary_length(self.mesh2d)
        self.mesh2d.boundary_len = bnd_len
        self.mesh.boundary_len = bnd_len

        # Time integrator setup
        self.dt = None
        self.dt_2d = None
        self.M_modesplit = None

        # override default options
        self.options = ModelOptions()
        if options is not None:
            self.options.update(options)

        # simulation time step bookkeeping
        self.simulation_time = 0
        self.iteration = 0
        self.i_export = 0
        self.next_export_t = self.simulation_time + self.options.t_export

        self.bnd_functions = {'shallow_water': {},
                              'momentum': {},
                              'salt': {},
                              'temp': {},
                              }

        self.callbacks = callback.CallbackManager()
        """Callback manager object"""

        self.fields = FieldDict()
        """Holds all functions needed by the solver object."""
        self.function_spaces = AttrDict()
        """Holds all function spaces needed by the solver object."""
        self.export_initial_state = True
        """Do export initial state. False if continuing a simulation"""
        self._isfrozen = True  # disallow creating new attributes

    def compute_dx_factor(self):
        """Computes a normalized distance between nodes in the horizontal mesh"""
        p = self.options.order
        if self.options.element_family == 'rt-dg':
            # velocity space is essentially p+1
            p = self.options.order + 1
        # assuming DG basis functions on triangles
        l_r = p**2/3.0 + 7.0/6.0*p + 1.0
        factor = 0.5*0.25/l_r
        return factor

    def compute_dz_factor(self):
        """Computes a normalized distance between nodes in the vertical mesh"""
        p = self.options.order
        # assuming DG basis functions in an interval
        l_r = 1.0/max(p, 1)
        factor = 0.5*0.25/l_r
        return factor

    def compute_dt_2d(self, u_mag):
        """
        Computes maximum explicit time step from CFL condition.

        dt = CellSize/U

        Assumes velocity scale U = sqrt(g*H) + u_mag
        where u_mag is estimated advective velocity
        """
        csize = self.fields.h_elem_size_2d
        bath = self.fields.bathymetry_2d
        fs = bath.function_space()
        bath_pos = Function(fs, name='bathymetry')
        bath_pos.assign(bath)
        min_depth = 0.05
        bath_pos.dat.data[bath_pos.dat.data < min_depth] = min_depth
        test = TestFunction(fs)
        trial = TrialFunction(fs)
        solution = Function(fs)
        g = physical_constants['g_grav']
        u = (sqrt(g * bath_pos) + u_mag)
        a = inner(test, trial) * dx
        l = inner(test, csize / u) * dx
        solve(a == l, solution)
        dt = float(solution.dat.data.min())
        dt = self.comm.allreduce(dt, op=MPI.MIN)
        dt *= self.compute_dx_factor()
        return dt

    def compute_dt_h_advection(self, u_scale):
        """
        Computes maximum explicit time step from CFL condition.

        dt = CellSize_h/u_scale

        Assumes advective horizontal velocity scale u_scale
        """
        u = u_scale
        if isinstance(u_scale, FiredrakeConstant):
            u = u_scale.dat.data[0]
        min_dx = self.fields.h_elem_size_2d.dat.data.min()
        # alpha = 0.5 if self.options.element_family == 'rt-dg' else 1.0
        # dt = alpha*1.0/10.0/(self.options.order + 1)*min_dx/u
        min_dx *= self.compute_dx_factor()
        dt = min_dx/u
        dt = self.comm.allreduce(dt, op=MPI.MIN)
        return dt

    def compute_dt_v_advection(self, w_scale):
        """
        Computes maximum explicit time step from CFL condition.

        dt = CellSize_v/w_scale

        Assumes advective vertical velocity scale w_scale
        """
        w = w_scale
        if isinstance(w_scale, FiredrakeConstant):
            w = w_scale.dat.data[0]
        min_dz = self.fields.v_elem_size_2d.dat.data.min()
        # alpha = 0.5 if self.options.element_family == 'rt-dg' else 1.0
        # dt = alpha*1.0/1.5/(self.options.order + 1)*min_dz/w
        min_dz *= self.compute_dz_factor()
        dt = min_dz/w
        dt = self.comm.allreduce(dt, op=MPI.MIN)
        return dt

    def compute_dt_diffusion(self, nu_scale):
        """
        Computes maximum explicit time step for horizontal diffusion.

        dt = alpha*CellSize**2/nu_scale

        where nu_scale is estimated diffusivity scale
        """
        nu = nu_scale
        if isinstance(nu_scale, FiredrakeConstant):
            nu = nu_scale.dat.data[0]
        min_dx = self.fields.h_elem_size_2d.dat.data.min()
        # alpha = 0.25 if self.options.element_family == 'rt-dg' else 1.0
        # dt = 0.75*alpha*1.0/60.0/(self.options.order + 1)*(min_dx)**2/nu
        min_dx *= 1.5*self.compute_dx_factor()
        dt = (min_dx)**2/nu
        dt = self.comm.allreduce(dt, op=MPI.MIN)
        return dt

    def set_time_step(self):
        cfl2d = self.timestepper.cfl_coeff_2d
        cfl3d = self.timestepper.cfl_coeff_3d
        max_dt_swe = self.compute_dt_2d(self.options.u_advection)
        max_dt_hadv = self.compute_dt_h_advection(self.options.u_advection)
        max_dt_vadv = self.compute_dt_v_advection(self.options.w_advection)
        max_dt_diff = self.compute_dt_diffusion(self.options.nu_viscosity)
        print_output('  - dt 2d swe: {:}'.format(max_dt_swe))
        print_output('  - dt h. advection: {:}'.format(max_dt_hadv))
        print_output('  - dt v. advection: {:}'.format(max_dt_vadv))
        print_output('  - dt viscosity: {:}'.format(max_dt_diff))
        max_dt_2d = cfl2d*max_dt_swe
        max_dt_3d = cfl3d*min(max_dt_hadv, max_dt_vadv, max_dt_diff)
        print_output('  - CFL adjusted dt: 2D: {:} 3D: {:}'.format(max_dt_2d, max_dt_3d))
        if round(max_dt_3d) > 0:
            max_dt_3d = np.floor(max_dt_3d)
        if self.options.dt_2d is not None or self.options.dt is not None:
            print_output('  - User defined dt: 2D: {:} 3D: {:}'.format(self.options.dt_2d, self.options.dt))
        self.dt = self.options.dt
        self.dt_2d = self.options.dt_2d

        if self.dt_mode == 'split':
            if self.dt is None:
                self.dt = max_dt_3d
            if self.dt_2d is None:
                self.dt_2d = max_dt_2d
            # compute mode split ratio and force it to be integer
            self.M_modesplit = int(np.ceil(self.dt/self.dt_2d))
            self.dt_2d = self.dt/self.M_modesplit
        elif self.dt_mode == '2d':
            if self.dt is None:
                self.dt = min(max_dt_2d, max_dt_3d)
            self.dt_2d = self.dt
            self.M_modesplit = 1
        elif self.dt_mode == '3d':
            if self.dt is None:
                self.dt = max_dt_3d
            self.dt_2d = self.dt
            self.M_modesplit = 1

        print_output('  - chosen dt: 2D: {:} 3D: {:}'.format(self.dt_2d, self.dt))

        # fit dt to export time
        m_exp = int(np.ceil(self.options.t_export/self.dt))
        self.dt = self.options.t_export/m_exp
        if self.dt_mode == 'split':
            self.M_modesplit = int(np.ceil(self.dt/self.dt_2d))
            self.dt_2d = self.dt/self.M_modesplit
        else:
            self.dt_2d = self.dt

        print_output('  - adjusted dt: 2D: {:} 3D: {:}'.format(self.dt_2d, self.dt))

        print_output('dt = {0:f}'.format(self.dt))
        if self.dt_mode == 'split':
            print_output('2D dt = {0:f} {1:d}'.format(self.dt_2d, self.M_modesplit))
        sys.stdout.flush()

    def create_function_spaces(self):
        """Creates function spaces"""
        self._isfrozen = False
        # ----- function spaces: elev in H, uv in U, mixed is W
        self.function_spaces.P0 = FunctionSpace(self.mesh, 'DG', 0, vfamily='DG', vdegree=0, name='P0')
        self.function_spaces.P1 = FunctionSpace(self.mesh, 'CG', 1, vfamily='CG', vdegree=1, name='P1')
        self.function_spaces.P1v = VectorFunctionSpace(self.mesh, 'CG', 1, vfamily='CG', vdegree=1, name='P1v')
        self.function_spaces.P1DG = FunctionSpace(self.mesh, 'DG', 1, vfamily='DG', vdegree=1, name='P1DG')
        self.function_spaces.P1DGv = VectorFunctionSpace(self.mesh, 'DG', 1, vfamily='DG', vdegree=1, name='P1DGv')

        # Construct HDiv TensorProductElements
        # for horizontal velocity component
        u_h_elt = FiniteElement('RT', triangle, self.options.order+1)
        u_v_elt = FiniteElement('DG', interval, self.options.order)
        u_elt = HDiv(TensorProductElement(u_h_elt, u_v_elt))
        # for vertical velocity component
        w_h_elt = FiniteElement('DG', triangle, self.options.order)
        w_v_elt = FiniteElement('CG', interval, self.options.order+1)
        w_elt = HDiv(TensorProductElement(w_h_elt, w_v_elt))
        # final spaces
        if self.options.element_family == 'rt-dg':
            # self.U = FunctionSpace(self.mesh, UW_elt)  # uv
            self.function_spaces.U = FunctionSpace(self.mesh, u_elt, name='U')  # uv
            self.function_spaces.W = FunctionSpace(self.mesh, w_elt, name='W')  # w
        elif self.options.element_family == 'dg-dg':
            self.function_spaces.U = VectorFunctionSpace(self.mesh, 'DG', self.options.order,
                                                         vfamily='DG', vdegree=self.options.order,
                                                         name='U')
            # NOTE for tracer consistency W should be equivalent to tracer space H
            self.function_spaces.W = VectorFunctionSpace(self.mesh, 'DG', self.options.order,
                                                         vfamily='DG', vdegree=self.options.order,
                                                         name='W')
        else:
            raise Exception('Unsupported finite element family {:}'.format(self.options.element_family))
        # auxiliary function space that will be used to transfer data between 2d/3d modes
        self.function_spaces.Uproj = self.function_spaces.U

        self.function_spaces.Uint = self.function_spaces.U  # vertical integral of uv
        # tracers
        self.function_spaces.H = FunctionSpace(self.mesh, 'DG', self.options.order, vfamily='DG', vdegree=max(0, self.options.order), name='H')
        # vertical integral of tracers
        self.function_spaces.Hint = FunctionSpace(self.mesh, 'DG', self.options.order, vfamily='CG', vdegree=self.options.order+1, name='Hint')
        self.function_spaces.turb_space = self.function_spaces.P0

        # 2D spaces
        self.function_spaces.P1_2d = FunctionSpace(self.mesh2d, 'CG', 1, name='P1_2d')
        self.function_spaces.P1v_2d = VectorFunctionSpace(self.mesh2d, 'CG', 1, name='P1v_2d')
        self.function_spaces.P1DG_2d = FunctionSpace(self.mesh2d, 'DG', 1, name='P1DG_2d')
        self.function_spaces.P1DGv_2d = VectorFunctionSpace(self.mesh2d, 'DG', 1, name='P1DGv_2d')
        # 2D velocity space
        if self.options.element_family == 'rt-dg':
            self.function_spaces.U_2d = FunctionSpace(self.mesh2d, 'RT', self.options.order+1)
        elif self.options.element_family == 'dg-dg':
            self.function_spaces.U_2d = VectorFunctionSpace(self.mesh2d, 'DG', self.options.order, name='U_2d')
        self.function_spaces.Uproj_2d = self.function_spaces.U_2d
        # TODO is this needed?
        # self.function_spaces.U_scalar_2d = FunctionSpace(self.mesh2d, 'DG', self.options.order, name='U_scalar_2d')
        self.function_spaces.H_2d = FunctionSpace(self.mesh2d, 'DG', self.options.order, name='H_2d')
        self.function_spaces.V_2d = MixedFunctionSpace([self.function_spaces.U_2d, self.function_spaces.H_2d], name='V_2d')
        self._isfrozen = True

    def create_equations(self):
        """Creates function spaces, functions, equations and time steppers."""
        if not hasattr(self, 'U_2d'):
            self.create_function_spaces()
        self._isfrozen = False

        if self.options.log_output and not self.options.no_exports:
            logfile = os.path.join(create_directory(self.options.outputdir), 'log')
            filehandler = logging.logging.FileHandler(logfile, mode='w')
            filehandler.setFormatter(logging.logging.Formatter('%(message)s'))
            output_logger.addHandler(filehandler)

        self.use_full_2d_mode = True  # 2d solution is (uv, eta) not (eta)

        # mesh velocity etc fields must be in the same space as 3D coordinates
        e = self.mesh2d.coordinates.function_space().fiat_element
        coord_is_dg = element_continuity(e).dg
        if coord_is_dg:
            coord_fs = FunctionSpace(self.mesh, 'DG', 1, vfamily='CG', vdegree=1)
            coord_fs_2d = self.function_spaces.P1DG_2d
        else:
            coord_fs = self.function_spaces.P1
            coord_fs_2d = self.function_spaces.P1_2d

        # ----- fields
        self.fields.solution_2d = Function(self.function_spaces.V_2d)
        # correct treatment of the split 2d functions
        uv_2d, eta2d = self.fields.solution_2d.split()
        self.fields.uv_2d = uv_2d
        self.fields.elev_2d = eta2d
        if self.options.use_bottom_friction:
            self.fields.uv_bottom_2d = Function(self.function_spaces.P1v_2d)
            self.fields.z_bottom_2d = Function(coord_fs_2d)
            self.fields.bottom_drag_2d = Function(coord_fs_2d)

        self.fields.elev_3d = Function(self.function_spaces.H)
        self.fields.elev_cg_3d = Function(coord_fs)
        self.fields.elev_cg_2d = Function(coord_fs_2d)
        self.fields.uv_3d = Function(self.function_spaces.U)
        if self.options.use_bottom_friction:
            self.fields.uv_bottom_3d = Function(self.function_spaces.P1v)
            self.fields.bottom_drag_3d = Function(self.function_spaces.P1)
        self.fields.bathymetry_2d = Function(coord_fs_2d)
        self.fields.bathymetry_3d = Function(coord_fs)
        # z coordinate in the strecthed mesh
        self.fields.z_coord_3d = Function(coord_fs)
        # z coordinate in the reference mesh (eta=0)
        self.fields.z_coord_ref_3d = Function(coord_fs)
        self.fields.uv_dav_3d = Function(self.function_spaces.Uproj)
        self.fields.uv_dav_2d = Function(self.function_spaces.Uproj_2d)
        self.fields.uv_mag_3d = Function(self.function_spaces.P0)
        self.fields.uv_p1_3d = Function(self.function_spaces.P1v)
        self.fields.w_3d = Function(self.function_spaces.W)
        if self.options.use_ale_moving_mesh:
            self.fields.w_mesh_3d = Function(coord_fs)
            self.fields.w_mesh_ddz_3d = Function(coord_fs)
            self.fields.w_mesh_surf_3d = Function(coord_fs)
            self.fields.w_mesh_surf_2d = Function(coord_fs_2d)
        if self.options.solve_salt:
            self.fields.salt_3d = Function(self.function_spaces.H, name='Salinity')
        if self.options.solve_temp:
            self.fields.temp_3d = Function(self.function_spaces.H, name='Temperature')
        if self.options.solve_vert_diffusion and self.options.use_parabolic_viscosity:
            self.fields.parab_visc_3d = Function(self.function_spaces.P1)
        if self.options.baroclinic:
            self.fields.density_3d = Function(self.function_spaces.H, name='Density')
            self.fields.baroc_head_3d = Function(self.function_spaces.Hint)
            # NOTE only used in 2D eqns no need to use higher order Hint space
            self.fields.baroc_head_int_3d = Function(self.function_spaces.H)
            self.fields.baroc_head_2d = Function(self.function_spaces.H_2d)
        if self.options.coriolis is not None:
            if isinstance(self.options.coriolis, FiredrakeConstant):
                self.fields.coriolis_3d = self.options.coriolis
            else:
                self.fields.coriolis_3d = Function(self.function_spaces.P1)
                ExpandFunctionTo3d(self.options.coriolis, self.fields.coriolis_3d).solve()
        if self.options.wind_stress is not None:
            if isinstance(self.options.wind_stress, FiredrakeFunction):
                # assume 2d function and expand to 3d
                self.fields.wind_stress_3d = Function(self.function_spaces.P1)
                ExpandFunctionTo3d(self.options.wind_stress, self.fields.wind_stress_3d).solve()
            elif isinstance(self.options.wind_stress, FiredrakeConstant):
                self.fields.wind_stress_3d = self.options.wind_stress
            else:
                raise Exception('Unsupported wind stress type: {:}'.format(type(self.options.wind_stress)))
        self.fields.v_elem_size_3d = Function(self.function_spaces.P1DG)
        self.fields.v_elem_size_2d = Function(self.function_spaces.P1DG_2d)
        self.fields.h_elem_size_3d = Function(self.function_spaces.P1)
        self.fields.h_elem_size_2d = Function(self.function_spaces.P1_2d)
        get_horizontal_elem_size_3d(self.fields.h_elem_size_2d, self.fields.h_elem_size_3d)
        self.fields.max_h_diff = Function(self.function_spaces.P1)
        if self.options.smagorinsky_factor is not None:
            self.fields.smag_visc_3d = Function(self.function_spaces.P1)
        if self.options.salt_jump_diff_factor is not None:
            self.fields.salt_jump_diff = Function(self.function_spaces.P1)
        if self.options.use_limiter_for_tracers and self.options.order > 0:
            self.tracer_limiter = limiter.VertexBasedP1DGLimiter(self.function_spaces.H)
        else:
            self.tracer_limiter = None
        if self.options.use_turbulence:
            # NOTE tke and psi should be in H as tracers ??
            self.fields.tke_3d = Function(self.function_spaces.turb_space)
            self.fields.psi_3d = Function(self.function_spaces.turb_space)
            # NOTE other turb. quantities should share the same nodes ??
            self.fields.eps_3d = Function(self.function_spaces.turb_space)
            self.fields.len_3d = Function(self.function_spaces.turb_space)
            if self.options.use_smooth_eddy_viscosity:
                self.fields.eddy_visc_3d = Function(self.function_spaces.P1)
                self.fields.eddy_diff_3d = Function(self.function_spaces.P1)
            else:
                self.fields.eddy_visc_3d = Function(self.function_spaces.turb_space)
                self.fields.eddy_diff_3d = Function(self.function_spaces.turb_space)
            # NOTE M2 and N2 depend on d(.)/dz -> use CG in vertical ?
            self.fields.shear_freq_3d = Function(self.function_spaces.turb_space)
            self.fields.buoy_freq_3d = Function(self.function_spaces.turb_space)
            self.gls_model = turbulence.GenericLengthScaleModel(weakref.proxy(self),
                                                                self.fields.tke_3d,
                                                                self.fields.psi_3d,
                                                                self.fields.uv_3d,
                                                                self.fields.get('density_3d'),
                                                                self.fields.len_3d,
                                                                self.fields.eps_3d,
                                                                self.fields.eddy_diff_3d,
                                                                self.fields.eddy_visc_3d,
                                                                self.fields.buoy_freq_3d,
                                                                self.fields.shear_freq_3d,
                                                                options=self.options.gls_options)
        else:
            self.gls_model = None
        # copute total viscosity/diffusivity
        self.tot_h_visc = SumFunction()
        self.tot_h_visc.add(self.options.get('h_viscosity'))
        self.tot_h_visc.add(self.fields.get('smag_visc_3d'))
        self.tot_v_visc = SumFunction()
        self.tot_v_visc.add(self.options.get('v_viscosity'))
        self.tot_v_visc.add(self.fields.get('eddy_visc_3d'))
        self.tot_v_visc.add(self.fields.get('parab_visc_3d'))
        self.tot_h_diff = SumFunction()
        self.tot_h_diff.add(self.options.get('h_diffusivity'))
        self.tot_v_diff = SumFunction()
        self.tot_v_diff.add(self.options.get('v_diffusivity'))
        self.tot_v_diff.add(self.fields.get('eddy_diff_3d'))

        # ----- Equations
        if self.use_full_2d_mode:
            # full 2D shallow water equations
            self.eq_sw = shallowwater_eq.ShallowWaterEquations(
                self.fields.solution_2d.function_space(),
                self.fields.bathymetry_2d,
                nonlin=self.options.nonlin,
                include_grad_div_viscosity_term=self.options.include_grad_div_viscosity_term,
                include_grad_depth_viscosity_term=self.options.include_grad_depth_viscosity_term
            )
        else:
            # solve elevation only: 2D free surface equation
            uv, eta = self.fields.solution_2d.split()
            eta_test = TestFunction(eta)
            self.eq_sw = shallowwater_eq.FreeSurfaceEquation(
                eta_test, eta.function_space(), uv.function_space(),
                self.fields.bathymetry_2d,
                nonlin=self.options.nonlin,
            )

        self.eq_momentum = momentum_eq.MomentumEquation(self.fields.uv_3d.function_space(),
                                                        bathymetry=self.fields.bathymetry_3d,
                                                        v_elem_size=self.fields.v_elem_size_3d,
                                                        h_elem_size=self.fields.h_elem_size_3d,
                                                        nonlin=self.options.nonlin,
                                                        use_bottom_friction=False,
                                                        use_elevation_gradient=not self.use_full_2d_mode)
        if self.options.solve_vert_diffusion:
            self.eq_vertmomentum = momentum_eq.MomentumEquation(self.fields.uv_3d.function_space(),
                                                                bathymetry=self.fields.bathymetry_3d,
                                                                v_elem_size=self.fields.v_elem_size_3d,
                                                                h_elem_size=self.fields.h_elem_size_3d,
                                                                nonlin=False,
                                                                use_bottom_friction=self.options.use_bottom_friction)
        if self.options.solve_salt:
            self.eq_salt = tracer_eq.TracerEquation(self.fields.salt_3d.function_space(),
                                                    bathymetry=self.fields.bathymetry_3d,
                                                    v_elem_size=self.fields.v_elem_size_3d,
                                                    h_elem_size=self.fields.h_elem_size_3d,
                                                    use_symmetric_surf_bnd=self.options.element_family == 'dg-dg')
            if self.options.solve_vert_diffusion:
                self.eq_salt_vdff = tracer_eq.TracerEquation(self.fields.salt_3d.function_space(),
                                                             bathymetry=self.fields.bathymetry_3d,
                                                             v_elem_size=self.fields.v_elem_size_3d,
                                                             h_elem_size=self.fields.h_elem_size_3d)

        if self.options.solve_temp:
            self.eq_temp = tracer_eq.TracerEquation(self.fields.temp_3d.function_space(),
                                                    bathymetry=self.fields.bathymetry_3d,
                                                    v_elem_size=self.fields.v_elem_size_3d,
                                                    h_elem_size=self.fields.h_elem_size_3d)
            if self.options.solve_vert_diffusion:
                self.eq_temp_vdff = tracer_eq.TracerEquation(self.fields.temp_3d.function_space(),
                                                             bathymetry=self.fields.bathymetry_3d,
                                                             v_elem_size=self.fields.v_elem_size_3d,
                                                             h_elem_size=self.fields.h_elem_size_3d)

        self.eq_sw.bnd_functions = self.bnd_functions['shallow_water']
        self.eq_momentum.bnd_functions = self.bnd_functions['momentum']
        if self.options.solve_salt:
            self.eq_salt.bnd_functions = self.bnd_functions['salt']
        if self.options.solve_temp:
            self.eq_temp.bnd_functions = self.bnd_functions['temp']
        if self.options.use_turbulence:
            if self.options.use_turbulence_advection:
                # explicit advection equations
                self.eq_tke_adv = tracer_eq.TracerEquation(self.fields.tke_3d.function_space(),
                                                           bathymetry=self.fields.bathymetry_3d,
                                                           v_elem_size=self.fields.v_elem_size_3d,
                                                           h_elem_size=self.fields.h_elem_size_3d)
                self.eq_psi_adv = tracer_eq.TracerEquation(self.fields.psi_3d.function_space(),
                                                           bathymetry=self.fields.bathymetry_3d,
                                                           v_elem_size=self.fields.v_elem_size_3d,
                                                           h_elem_size=self.fields.h_elem_size_3d)
            # implicit vertical diffusion eqn with production terms
            self.eq_tke_diff = turbulence.TKEEquation(self.fields.tke_3d.function_space(),
                                                      self.gls_model,
                                                      bathymetry=self.fields.bathymetry_3d,
                                                      v_elem_size=self.fields.v_elem_size_3d,
                                                      h_elem_size=self.fields.h_elem_size_3d)
            self.eq_psi_diff = turbulence.PsiEquation(self.fields.psi_3d.function_space(),
                                                      self.gls_model,
                                                      bathymetry=self.fields.bathymetry_3d,
                                                      v_elem_size=self.fields.v_elem_size_3d,
                                                      h_elem_size=self.fields.h_elem_size_3d)

        # ----- Time integrators
        self.dt_mode = '3d'  # 'split'|'2d'|'3d' use constant 2d/3d dt, or split
        if self.options.timestepper_type.lower() == 'ssprk33':
            self.timestepper = coupled_timeintegrator.CoupledSSPRKSemiImplicit(weakref.proxy(self))
        elif self.options.timestepper_type.lower() == 'leapfrog':
            assert self.options.use_ale_moving_mesh, '{:} time integrator requires ALE mesh'.format(self.options.timestepper_type)
            self.timestepper = coupled_timeintegrator.CoupledLeapFrogAM3(weakref.proxy(self))
        elif self.options.timestepper_type.lower() == 'imexale':
            assert self.options.use_ale_moving_mesh, '{:} time integrator requires ALE mesh'.format(self.options.timestepper_type)
            self.timestepper = coupled_timeintegrator.CoupledIMEXALE(weakref.proxy(self))
        elif self.options.timestepper_type.lower() == 'erkale':
            assert self.options.use_ale_moving_mesh, '{:} time integrator requires ALE mesh'.format(self.options.timestepper_type)
            self.timestepper = coupled_timeintegrator.CoupledERKALE(weakref.proxy(self))
            self.dt_mode = '2d'
        else:
            raise Exception('Unknown time integrator type: '+str(self.options.timestepper_type))

        # ----- File exporters
        # create export_managers and store in a list
        self.exporters = {}
        if not self.options.no_exports:
            e = exporter.ExportManager(self.options.outputdir,
                                       self.options.fields_to_export,
                                       self.fields,
                                       field_metadata,
                                       export_type='vtk',
                                       verbose=self.options.verbose > 0)
            self.exporters['vtk'] = e
            numpy_dir = os.path.join(self.options.outputdir, 'numpy')
            e = exporter.ExportManager(numpy_dir,
                                       self.options.fields_to_export_numpy,
                                       self.fields,
                                       field_metadata,
                                       export_type='numpy',
                                       verbose=self.options.verbose > 0)
            self.exporters['numpy'] = e
            hdf5_dir = os.path.join(self.options.outputdir, 'hdf5')
            e = exporter.ExportManager(hdf5_dir,
                                       self.options.fields_to_export_hdf5,
                                       self.fields,
                                       field_metadata,
                                       export_type='hdf5',
                                       verbose=self.options.verbose > 0)
            self.exporters['hdf5'] = e

        # ----- Operators
        self.w_solver = VerticalVelocitySolver(self.fields.w_3d,
                                               self.fields.uv_3d,
                                               self.fields.bathymetry_3d,
                                               self.eq_momentum.bnd_functions)
        self.uv_averager = VerticalIntegrator(self.fields.uv_3d,
                                              self.fields.uv_dav_3d,
                                              bottom_to_top=True,
                                              bnd_value=Constant((0.0, 0.0, 0.0)),
                                              average=True,
                                              bathymetry=self.fields.bathymetry_3d,
                                              elevation=self.fields.elev_cg_3d)
        if self.options.baroclinic:
            if self.options.solve_salt:
                s = self.fields.salt_3d
            else:
                s = self.options.constant_salt
            if self.options.solve_temp:
                t = self.fields.temp_3d
            else:
                t = self.options.constant_temp
            if self.options.equation_of_state == 'linear':
                eos_params = self.options.lin_equation_of_state_params
                self.equation_of_state = LinearEquationOfState(**eos_params)
            else:
                self.equation_of_state = EquationOfState()
            self.density_solver = DensitySolver(s, t, self.fields.density_3d,
                                                self.equation_of_state)
            self.rho_integrator = VerticalIntegrator(self.fields.density_3d,
                                                     self.fields.baroc_head_3d,
                                                     bottom_to_top=False)
            self.baro_head_averager = VerticalIntegrator(self.fields.baroc_head_3d,
                                                         self.fields.baroc_head_int_3d,
                                                         bottom_to_top=True,
                                                         average=True,
                                                         bathymetry=self.fields.bathymetry_3d,
                                                         elevation=self.fields.elev_cg_3d)
            self.extract_surf_baro_head = SubFunctionExtractor(self.fields.baroc_head_int_3d,
                                                               self.fields.baroc_head_2d,
                                                               boundary='top', elem_facet='top')
        self.extract_surf_dav_uv = SubFunctionExtractor(self.fields.uv_dav_3d,
                                                        self.fields.uv_dav_2d,
                                                        boundary='top', elem_facet='top',
                                                        elem_height=self.fields.v_elem_size_2d)
        self.copy_elev_to_3d = ExpandFunctionTo3d(self.fields.elev_2d, self.fields.elev_3d)
        self.copy_elev_cg_to_3d = ExpandFunctionTo3d(self.fields.elev_cg_2d, self.fields.elev_cg_3d)
        self.copy_uv_dav_to_uv_dav_3d = ExpandFunctionTo3d(self.fields.uv_dav_2d, self.fields.uv_dav_3d,
                                                           elem_height=self.fields.v_elem_size_3d)
        self.copy_uv_to_uv_dav_3d = ExpandFunctionTo3d(self.fields.uv_2d, self.fields.uv_dav_3d,
                                                       elem_height=self.fields.v_elem_size_3d)
        self.uv_mag_solver = VelocityMagnitudeSolver(self.fields.uv_mag_3d, u=self.fields.uv_3d)
        if self.options.use_bottom_friction:
            self.extract_uv_bottom = SubFunctionExtractor(self.fields.uv_p1_3d, self.fields.uv_bottom_2d,
                                                          boundary='bottom', elem_facet='average',
                                                          elem_height=self.fields.v_elem_size_2d)
            self.extract_z_bottom = SubFunctionExtractor(self.fields.z_coord_3d, self.fields.z_bottom_2d,
                                                         boundary='bottom', elem_facet='average',
                                                         elem_height=self.fields.v_elem_size_2d)
            if self.options.use_parabolic_viscosity:
                self.copy_uv_bottom_to_3d = ExpandFunctionTo3d(self.fields.uv_bottom_2d,
                                                               self.fields.uv_bottom_3d,
                                                               elem_height=self.fields.v_elem_size_3d)
                self.copy_bottom_drag_to_3d = ExpandFunctionTo3d(self.fields.bottom_drag_2d,
                                                                 self.fields.bottom_drag_3d,
                                                                 elem_height=self.fields.v_elem_size_3d)
        self.mesh_updater = ALEMeshUpdater(self)

        if self.options.salt_jump_diff_factor is not None:
            self.horiz_jump_diff_solver = HorizontalJumpDiffusivity(self.options.salt_jump_diff_factor, self.fields.salt_3d,
                                                                    self.fields.salt_jump_diff, self.fields.h_elem_size_3d,
                                                                    self.fields.uv_mag_3d, self.options.salt_range,
                                                                    self.fields.max_h_diff)
        if self.options.smagorinsky_factor is not None:
            self.smagorinsky_diff_solver = SmagorinskyViscosity(self.fields.uv_p1_3d, self.fields.smag_visc_3d,
                                                                self.options.smagorinsky_factor, self.fields.h_elem_size_3d,
                                                                self.fields.max_h_diff,
                                                                weak_form=self.options.order == 0)
        if self.options.use_parabolic_viscosity:
            self.parabolic_viscosity_solver = ParabolicViscosity(self.fields.uv_bottom_3d,
                                                                 self.fields.bottom_drag_3d,
                                                                 self.fields.bathymetry_3d,
                                                                 self.fields.parab_visc_3d)
        self.uv_p1_projector = Projector(self.fields.uv_3d, self.fields.uv_p1_3d)
        self.elev_3d_to_cg_projector = Projector(self.fields.elev_3d, self.fields.elev_cg_3d)
        self.elev_2d_to_cg_projector = Projector(self.fields.elev_2d, self.fields.elev_cg_2d)

        # ----- set initial values
        self.fields.bathymetry_2d.project(self.bathymetry_cg_2d)
        ExpandFunctionTo3d(self.fields.bathymetry_2d, self.fields.bathymetry_3d).solve()
        self.mesh_updater.initialize()
        self.set_time_step()
        self.timestepper.set_dt(self.dt, self.dt_2d)
        # compute maximal diffusivity for explicit schemes
        degree_h, degree_v = self.function_spaces.H.ufl_element().degree()
        max_diff_alpha = 1.0/60.0/max((degree_h*(degree_h + 1)), 1.0)  # FIXME depends on element type and order
        self.fields.max_h_diff.assign(max_diff_alpha/self.dt * self.fields.h_elem_size_3d**2)
        d = self.fields.max_h_diff.dat.data
        print_output('max h diff {:} - {:}'.format(d.min(), d.max()))

        self.next_export_t = self.simulation_time + self.options.t_export
        self._initialized = True
        self._isfrozen = True

    def assign_initial_conditions(self, elev=None, salt=None, temp=None,
                                  uv_2d=None, uv_3d=None, tke=None, psi=None):
        if not self._initialized:
            self.create_equations()
        if elev is not None:
            self.fields.elev_2d.project(elev)
        if uv_2d is not None:
            self.fields.uv_2d.project(uv_2d)
            ExpandFunctionTo3d(self.fields.uv_2d, self.fields.uv_3d,
                               elem_height=self.fields.v_elem_size_3d).solve()
        if uv_3d is not None:
            self.fields.uv_3d.project(uv_3d)
        if salt is not None and self.options.solve_salt:
            self.fields.salt_3d.project(salt)
        if temp is not None and self.options.solve_temp:
            self.fields.temp_3d.project(temp)
        if self.options.use_turbulence:
            if tke is not None:
                self.fields.tke_3d.project(tke)
            if psi is not None:
                self.fields.psi_3d.project(psi)
            self.gls_model.initialize()

        if self.options.use_ale_moving_mesh:
            self.timestepper._update_3d_elevation()
            self.timestepper._update_moving_mesh()
        self.timestepper.initialize()
        # update all diagnostic variables
        self.timestepper._update_all_dependencies(self.simulation_time,
                                                  do_2d_coupling=False,
                                                  do_vert_diffusion=False,
                                                  do_ale_update=True,
                                                  do_stab_params=True,
                                                  do_turbulence=False)
        if self.options.use_turbulence:
            self.gls_model.initialize()

    def add_callback(self, callback, eval_interval='export'):
        """Adds callback to solver object

        :arg callback: DiagnosticCallback instance
        "arg eval_interval: 'export'|'timestep' Determines when callback will be evaluated.
        """
        self.callbacks.add(callback, eval_interval)

    def export(self):
        self.callbacks.evaluate(mode='export')
        for key in self.exporters:
            self.exporters[key].export()

    def load_state(self, i_export, outputdir=None, t=None, iteration=None):
        """
        Loads simulation state from hdf5 outputs.

        This replaces assign_initial_conditions in model initilization.

        This assumes that model setup is kept the same (e.g. time step) and
        all pronostic state variables are exported in hdf5 format.  The required
        state variables are: elev_2d, uv_2d, uv_3d, salt_3d, temp_3d, tke_3d,
        psi_3d

        Currently hdf5 field import only works for the same number of MPI
        processes.
        """
        if not self._initialized:
            self.create_equations()
        if outputdir is None:
            outputdir = self.options.outputdir
        # create new ExportManager with desired outputdir
        state_fields = ['uv_2d', 'elev_2d', 'uv_3d',
                        'salt_3d', 'temp_3d', 'tke_3d', 'psi_3d']
        hdf5_dir = os.path.join(outputdir, 'hdf5')
        e = exporter.ExportManager(hdf5_dir,
                                   state_fields,
                                   self.fields,
                                   field_metadata,
                                   export_type='hdf5',
                                   verbose=self.options.verbose > 0)
        e.exporters['uv_2d'].load(i_export, self.fields.uv_2d)
        e.exporters['elev_2d'].load(i_export, self.fields.elev_2d)
        e.exporters['uv_3d'].load(i_export, self.fields.uv_3d)
        salt = temp = tke = psi = None
        if self.options.solve_salt:
            salt = self.fields.salt_3d
            e.exporters['salt_3d'].load(i_export, salt)
        if self.options.solve_temp:
            temp = self.fields.temp_3d
            e.exporters['temp_3d'].load(i_export, temp)
        if self.options.use_turbulence:
            tke = self.fields.tke_3d
            psi = self.fields.psi_3d
            e.exporters['tke_3d'].load(i_export, tke)
            e.exporters['psi_3d'].load(i_export, psi)
        self.assign_initial_conditions(elev=self.fields.elev_2d,
                                       uv_2d=self.fields.uv_2d,
                                       uv_3d=self.fields.uv_3d,
                                       salt=salt, temp=temp,
                                       tke=tke, psi=psi,
                                       )

        # time stepper bookkeeping for export time step
        self.i_export = i_export
        self.next_export_t = self.i_export*self.options.t_export
        if iteration is None:
            iteration = int(np.ceil(self.next_export_t/self.dt))
        if t is None:
            t = iteration*self.dt
        self.iteration = iteration
        self.simulation_time = t

        # for next export
        self.export_initial_state = outputdir != self.options.outputdir
        if self.export_initial_state:
            offset = 0
        else:
            offset = 1
        self.next_export_t += self.options.t_export
        for k in self.exporters:
            self.exporters[k].set_next_export_ix(self.i_export + offset)

    def print_state(self, cputime):
        norm_h = norm(self.fields.elev_2d)
        norm_u = norm(self.fields.uv_3d)

        line = ('{iexp:5d} {i:5d} T={t:10.2f} '
                'eta norm: {e:10.4f} u norm: {u:10.4f} {cpu:5.2f}')
        print_output(line.format(iexp=self.i_export, i=self.iteration,
                                 t=self.simulation_time, e=norm_h,
                                 u=norm_u, cpu=cputime))
        sys.stdout.flush()

    def iterate(self, update_forcings=None, update_forcings3d=None,
                export_func=None):
        if not self._initialized:
            self.create_equations()

        self.options.check_salt_conservation *= self.options.solve_salt
        self.options.check_salt_overshoot *= self.options.solve_salt
        self.options.check_temp_conservation *= self.options.solve_temp
        self.options.check_temp_overshoot *= self.options.solve_temp
        self.options.check_vol_conservation_3d *= self.options.use_ale_moving_mesh
        self.options.use_limiter_for_tracers *= self.options.order > 0

        t_epsilon = 1.0e-5
        cputimestamp = time_mod.clock()

        dump_hdf5 = self.options.export_diagnostics and not self.options.no_exports
        if self.options.check_vol_conservation_2d:
            c = callback.VolumeConservation2DCallback(self,
                                                      export_to_hdf5=dump_hdf5,
                                                      append_to_log=True)
            self.add_callback(c, eval_interval='export')
        if self.options.check_vol_conservation_3d:
            c = callback.VolumeConservation3DCallback(self,
                                                      export_to_hdf5=dump_hdf5,
                                                      append_to_log=True)
            self.add_callback(c, eval_interval='export')
        if self.options.check_salt_conservation:
            c = callback.TracerMassConservationCallback('salt_3d',
                                                        self,
                                                        export_to_hdf5=dump_hdf5,
                                                        append_to_log=True)
            self.add_callback(c, eval_interval='export')
        if self.options.check_salt_overshoot:
            c = callback.TracerOvershootCallBack('salt_3d',
                                                 self,
                                                 export_to_hdf5=dump_hdf5,
                                                 append_to_log=True)
            self.add_callback(c, eval_interval='export')
        if self.options.check_temp_conservation:
            c = callback.TracerMassConservationCallback('temp_3d',
                                                        self,
                                                        export_to_hdf5=dump_hdf5,
                                                        append_to_log=True)
            self.add_callback(c, eval_interval='export')
        if self.options.check_temp_overshoot:
            c = callback.TracerOvershootCallBack('temp_3d',
                                                 self,
                                                 export_to_hdf5=dump_hdf5,
                                                 append_to_log=True)
            self.add_callback(c, eval_interval='export')

        # initial export
        self.print_state(0.0)
        if self.export_initial_state:
            self.export()
            if export_func is not None:
                export_func()
            if 'vtk' in self.exporters:
                self.exporters['vtk'].export_bathymetry(self.fields.bathymetry_2d)

        while self.simulation_time <= self.options.t_end - t_epsilon:

            self.timestepper.advance(self.simulation_time, self.dt,
                                     update_forcings, update_forcings3d)

            # Move to next time step
            self.simulation_time += self.dt
            self.iteration += 1

            self.callbacks.evaluate(mode='timestep')

            # Write the solution to file
            if self.simulation_time >= self.next_export_t - t_epsilon:
                self.i_export += 1
                self.next_export_t += self.options.t_export

                cputime = time_mod.clock() - cputimestamp
                cputimestamp = time_mod.clock()
                self.print_state(cputime)

                self.export()
                if export_func is not None:
                    export_func()
