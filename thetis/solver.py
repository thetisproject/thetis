"""
Module for coupled 2D-3D flow solver.

Tuomas Karna 2015-04-01
"""
from utility import *
import shallowwater_eq
import momentum_eq
import tracer_eq
import turbulence
import coupled_timeintegrator
import limiter
import time as time_mod
from mpi4py import MPI
import exporter
import weakref
from thetis.field_defs import field_metadata
from thetis.options import ModelOptions
import thetis.callback as callback


class FlowSolver(FrozenClass):
    """Creates and solves coupled 2D-3D equations"""
    def __init__(self, mesh2d, bathymetry_2d, n_layers,
                 options=None):
        self._initialized = False

        # create 3D mesh
        self.mesh2d = mesh2d
        self.mesh = extrude_mesh_sigma(mesh2d, n_layers, bathymetry_2d)

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
        self.i_export = 1

        self.bnd_functions = {'shallow_water': {},
                              'momentum': {},
                              'salt': {}}

        self.callbacks = OrderedDict()
        """List of callback functions that will be called during exports"""

        self.fields = FieldDict()
        """Holds all functions needed by the solver object."""
        self.function_spaces = AttrDict()
        """Holds all function spaces needed by the solver object."""
        self.fields.bathymetry_2d = bathymetry_2d
        self._isfrozen = True  # disallow creating new attributes

    def set_time_step(self):
        if self.options.use_mode_split:
            self.dt = self.options.dt
            if self.dt is None:
                mesh_dt = self.eq_sw.get_time_step_advection(u_mag=self.options.u_advection)
                dt = self.options.cfl_3d*float(np.floor(mesh_dt.dat.data.min()/20.0))
                dt = comm.allreduce(dt, op=MPI.MIN)
                if round(dt) > 0:
                    dt = round(dt)
                self.dt = dt
            self.dt_2d = self.options.dt_2d
            if self.dt_2d is None:
                mesh2d_dt = self.eq_sw.get_time_step(u_mag=self.options.u_advection)
                dt_2d = self.options.cfl_2d*float(mesh2d_dt.dat.data.min()/20.0)
                dt_2d = comm.allreduce(dt_2d, op=MPI.MIN)
                self.dt_2d = dt_2d
            # compute mode split ratio and force it to be integer
            self.M_modesplit = int(np.ceil(self.dt/self.dt_2d))
            self.dt_2d = self.dt/self.M_modesplit
        else:
            mesh2d_dt = self.eq_sw.get_time_step(u_mag=self.options.u_advection)
            dt_2d = self.options.cfl_2d*float(mesh2d_dt.dat.data.min()/20.0)
            dt_2d = comm.allreduce(dt_2d, op=MPI.MIN)
            if self.dt is None:
                self.dt = dt_2d
            self.dt_2d = self.dt
            self.M_modesplit = 1

        print_info('dt = {0:f}'.format(self.dt))
        print_info('2D dt = {0:f} {1:d}'.format(self.dt_2d, self.M_modesplit))
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
        if self.options.mimetic:
            # self.U = FunctionSpace(self.mesh, UW_elt)  # uv
            self.function_spaces.U = FunctionSpace(self.mesh, u_elt, name='U')  # uv
            self.function_spaces.W = FunctionSpace(self.mesh, w_elt, name='W')  # w
        else:
            self.function_spaces.U = VectorFunctionSpace(self.mesh, 'DG', self.options.order,
                                                         vfamily='DG', vdegree=self.options.order,
                                                         name='U')
            # NOTE for tracer consistency W should be equivalent to tracer space H
            self.function_spaces.W = VectorFunctionSpace(self.mesh, 'DG', self.options.order,
                                                         vfamily='DG', vdegree=self.options.order,
                                                         name='W')
        # auxiliary function space that will be used to transfer data between 2d/3d modes
        self.function_spaces.Uproj = self.function_spaces.U

        self.function_spaces.Uint = self.function_spaces.U  # vertical integral of uv
        # tracers
        self.function_spaces.H = FunctionSpace(self.mesh, 'DG', self.options.order, vfamily='DG', vdegree=max(0, self.options.order), name='H')
        # vertical integral of tracers
        self.function_spaces.Hint = FunctionSpace(self.mesh, 'DG', self.options.order, vfamily='CG', vdegree=self.options.order+1, name='Hint')
        # for scalar fields to be used in momentum eq NOTE could be omitted ?
        # self.function_spaces.U_scalar = FunctionSpace(self.mesh, 'DG', self.options.order, vfamily='DG', vdegree=self.options.order, name='U_scalar')
        # for turbulence
        self.function_spaces.turb_space = self.function_spaces.P0

        # 2D spaces
        self.function_spaces.P1_2d = FunctionSpace(self.mesh2d, 'CG', 1, name='P1_2d')
        self.function_spaces.P1v_2d = VectorFunctionSpace(self.mesh2d, 'CG', 1, name='P1v_2d')
        self.function_spaces.P1DG_2d = FunctionSpace(self.mesh2d, 'DG', 1, name='P1DG_2d')
        self.function_spaces.P1DGv_2d = VectorFunctionSpace(self.mesh2d, 'DG', 1, name='P1DGv_2d')
        # 2D velocity space
        if self.options.mimetic:
            self.function_spaces.U_2d = FunctionSpace(self.mesh2d, 'RT', self.options.order+1)
        else:
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

        # ----- fields
        self.fields.solution_2d = Function(self.function_spaces.V_2d)
        # correct treatment of the split 2d functions
        uv_2d, eta2d = self.fields.solution_2d.split()
        self.fields.uv_2d = uv_2d
        self.fields.elev_2d = eta2d
        if self.options.use_bottom_friction:
            self.fields.uv_bottom_2d = Function(self.function_spaces.P1v_2d)
            self.fields.z_bottom_2d = Function(self.function_spaces.P1_2d)
            self.fields.bottom_drag_2d = Function(self.function_spaces.P1_2d)

        self.fields.elev_3d = Function(self.function_spaces.H)
        self.fields.elev_cg_3d = Function(self.function_spaces.P1)
        self.fields.bathymetry_3d = Function(self.function_spaces.P1)
        self.fields.uv_3d = Function(self.function_spaces.U)
        if self.options.use_bottom_friction:
            self.fields.uv_bottom_3d = Function(self.function_spaces.P1v)
            self.fields.bottom_drag_3d = Function(self.function_spaces.P1)
        # z coordinate in the strecthed mesh
        self.fields.z_coord_3d = Function(self.function_spaces.P1)
        # z coordinate in the reference mesh (eta=0)
        self.fields.z_coord_ref_3d = Function(self.function_spaces.P1)
        self.fields.uv_dav_3d = Function(self.function_spaces.Uproj)
        self.fields.uv_dav_2d = Function(self.function_spaces.Uproj_2d)
        self.fields.uv_mag_3d = Function(self.function_spaces.P0)
        self.fields.uv_p1_3d = Function(self.function_spaces.P1v)
        self.fields.w_3d = Function(self.function_spaces.W)
        if self.options.use_ale_moving_mesh:
            self.fields.w_mesh_3d = Function(self.function_spaces.H)
            self.fields.w_mesh_ddz_3d = Function(self.function_spaces.H)
            self.fields.w_mesh_surf_3d = Function(self.function_spaces.H)
            self.fields.w_mesh_surf_2d = Function(self.function_spaces.H_2d)
        if self.options.solve_salt:
            self.fields.salt_3d = Function(self.function_spaces.H, name='Salinity')
        if self.options.solve_vert_diffusion and self.options.use_parabolic_viscosity:
            # FIXME use_parabolic_viscosity is OBSOLETE
            self.fields.parab_visc_3d = Function(self.function_spaces.P1)
        if self.options.baroclinic:
            self.fields.baroc_head_3d = Function(self.function_spaces.Hint)
            self.fields.baroc_head_int_3d = Function(self.function_spaces.Hint)
            self.fields.baroc_head_2d = Function(self.function_spaces.H_2d)
        if self.options.coriolis is not None:
            if isinstance(self.options.coriolis, Constant):
                self.fields.coriolis_3d = self.options.coriolis
            else:
                self.fields.coriolis_3d = Function(self.function_spaces.P1)
                ExpandFunctionTo3d(self.options.coriolis, self.fields.coriolis_3d).solve()
        if self.options.wind_stress is not None:
            if isinstance(self.options.wind_stress, Function):
                # assume 2d function and expand to 3d
                self.fields.wind_stress_3d = Function(self.function_spaces.P1)
                ExpandFunctionTo3d(self.options.wind_stress, self.fields.wind_stress_3d).solve()
            elif isinstance(self.options.wind_stress, Constant):
                self.fields.wind_stress_3d = self.options.wind_stress
            else:
                raise Exception('Unsupported wind stress type: {:}'.format(type(self.options.wind_stress)))
        self.fields.v_elem_size_3d = Function(self.function_spaces.P1DG)
        self.fields.v_elem_size_2d = Function(self.function_spaces.P1DG_2d)
        self.fields.h_elem_size_3d = Function(self.function_spaces.P1)
        self.fields.h_elem_size_2d = Function(self.function_spaces.P1_2d)
        get_horizontal_elem_size(self.fields.h_elem_size_2d, self.fields.h_elem_size_3d)
        self.fields.max_h_diff = Function(self.function_spaces.P1)
        if self.options.smagorinsky_factor is not None:
            self.fields.smag_visc_3d = Function(self.function_spaces.P1)
        if self.options.salt_jump_diff_factor is not None:
            self.fields.salt_jump_diff = Function(self.function_spaces.P1)
        if self.options.use_limiter_for_tracers:
            self.tracer_limiter = limiter.VertexBasedP1DGLimiter(self.function_spaces.H,
                                                                 self.function_spaces.P1,
                                                                 self.function_spaces.P0)
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
                                                                self.fields.uv_p1_3d,
                                                                self.fields.get('salt_3d'),
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
        # assign viscosity/diffusivity to correct equations
        # if self.options.solve_vert_diffusion:
        #     implicit_v_visc = self.tot_v_visc.get_sum()
        #     explicit_v_visc = None
        #     implicit_v_diff = self.tot_v_diff.get_sum()
        #     explicit_v_diff = None
        # else:
        #     implicit_v_visc = None
        #     explicit_v_visc = self.tot_v_visc.get_sum()
        #     implicit_v_diff = None
        #     explicit_v_diff = self.tot_v_diff.get_sum()

        # ----- Equations
        if self.options.use_mode_split:
            # full 2D shallow water equations
            self.eq_sw = shallowwater_eq.ShallowWaterEquationsNew(
                self.fields.solution_2d.function_space(),
                self.fields.bathymetry_2d,
                nonlin=self.options.nonlin,
                include_grad_div_viscosity_term=self.options.include_grad_div_viscosity_term,
                include_grad_depth_viscosity_term=self.options.include_grad_depth_viscosity_term
            )
            # self.eq_sw = shallowwater_eq.ShallowWaterEquations(
            #     self.fields.solution_2d, self.fields.bathymetry_2d,
            #     self.fields.get('uv_bottom_2d'), self.fields.get('bottom_drag_2d'),
            #     baroc_head=self.fields.get('baroc_head_2d'),
            #     viscosity_h=self.options.get('h_viscosity'),  # FIXME add 2d smag
            #     uv_lax_friedrichs=self.options.uv_lax_friedrichs,
            #     coriolis=self.options.coriolis,
            #     wind_stress=self.options.wind_stress,
            #     uv_source=self.options.uv_source_2d,
            #     lin_drag=self.options.lin_drag,
            #     nonlin=self.options.nonlin)
        else:
            # solve elevation only: 2D free surface equation
            uv, eta = self.fields.solution_2d.split()
            self.eq_sw = shallowwater_eq.FreeSurfaceEquation(
                eta, uv, self.fields.bathymetry_2d,
                nonlin=self.options.nonlin)

        bnd_len = self.eq_sw.boundary_len
        bnd_markers = self.eq_sw.boundary_markers
        self.eq_momentum = momentum_eq.MomentumEquationNew(self.fields.uv_3d.function_space(),
                                                           bnd_markers, bnd_len,
                                                           bathymetry=self.fields.bathymetry_3d,
                                                           v_elem_size=self.fields.v_elem_size_3d,
                                                           h_elem_size=self.fields.h_elem_size_3d,
                                                           nonlin=self.options.nonlin,
                                                           use_bottom_friction=False)
        # self.eq_momentum = momentum_eq.MomentumEquation(
        #     bnd_markers,
        #     bnd_len, self.fields.uv_3d, self.fields.elev_3d,
        #     self.fields.bathymetry_3d, w=self.fields.w_3d,
        #     baroc_head=self.fields.get('baroc_head_3d'),
        #     w_mesh=self.fields.get('w_mesh_3d'),
        #     dw_mesh_dz=self.fields.get('w_mesh_ddz_3d'),
        #     viscosity_v=explicit_v_visc,
        #     viscosity_h=self.tot_h_visc.get_sum(),
        #     h_elem_size=self.fields.h_elem_size_3d,
        #     v_elem_size=self.fields.v_elem_size_3d,
        #     lax_friedrichs_factor=self.options.uv_lax_friedrichs,
        #     # uv_mag=self.uv_mag_3d,
        #     uv_p1=self.fields.get('uv_p1_3d'),
        #     coriolis=self.fields.get('coriolis_3d'),
        #     source=self.options.uv_source_3d,
        #     lin_drag=self.options.lin_drag,
        #     nonlin=self.options.nonlin)
        if self.options.solve_vert_diffusion:
            self.eq_vertmomentum = momentum_eq.MomentumEquationNew(self.fields.uv_3d.function_space(),
                                                                   bnd_markers, bnd_len,
                                                                   bathymetry=self.fields.bathymetry_3d,
                                                                   v_elem_size=self.fields.v_elem_size_3d,
                                                                   h_elem_size=self.fields.h_elem_size_3d,
                                                                   nonlin=False,
                                                                   use_bottom_friction=self.options.use_bottom_friction)
            # self.eq_vertmomentum = momentum_eq.VerticalMomentumEquation(
            #     self.fields.uv_3d, w=None,
            #     viscosity_v=implicit_v_visc,
            #     wind_stress=self.fields.get('wind_stress_3d'),
            #     v_elem_size=self.fields.v_elem_size_3d,
            #     h_elem_size=self.fields.h_elem_size_3d,
            #     use_bottom_friction=self.options.use_bottom_friction)
        if self.options.solve_salt:
            self.eq_salt = tracer_eq.TracerEquationNew(self.fields.salt_3d.function_space(),
                                                       bnd_markers, bnd_len,
                                                       bathymetry=self.fields.bathymetry_3d,
                                                       v_elem_size=self.fields.v_elem_size_3d,
                                                       h_elem_size=self.fields.h_elem_size_3d)
            # self.eq_salt = tracer_eq.TracerEquation(
            #     self.fields.salt_3d, self.fields.elev_3d, self.fields.uv_3d,
            #     w=self.fields.w_3d, w_mesh=self.fields.get('w_mesh_3d'),
            #     dw_mesh_dz=self.fields.get('w_mesh_ddz_3d'),
            #     diffusivity_h=self.tot_h_diff.get_sum(),
            #     diffusivity_v=explicit_v_diff,
            #     bathymetry=self.fields.bathymetry_3d,
            #     source=self.options.salt_source_3d,
            #     # uv_mag=self.uv_mag_3d,
            #     uv_p1=self.fields.get('uv_p1_3d'),
            #     lax_friedrichs_factor=self.options.tracer_lax_friedrichs,
            #     v_elem_size=self.fields.v_elem_size_3d,
            #     h_elem_size=self.fields.h_elem_size_3d,
            #     bnd_markers=bnd_markers,
            #     bnd_len=bnd_len)
            if self.options.solve_vert_diffusion:
                self.eq_salt_vdff = tracer_eq.TracerEquationNew(self.fields.salt_3d.function_space(),
                                                                bnd_markers, bnd_len,
                                                                bathymetry=self.fields.bathymetry_3d,
                                                                v_elem_size=self.fields.v_elem_size_3d,
                                                                h_elem_size=self.fields.h_elem_size_3d)
                # self.eq_salt_vdff = tracer_eq.TracerEquation(
                #     self.fields.salt_3d, self.fields.elev_3d,
                #     uv=None, w=None, w_mesh=None,
                #     diffusivity_v=implicit_v_diff,
                #     bathymetry=self.fields.bathymetry_3d,
                #     v_elem_size=self.fields.v_elem_size_3d,
                #     h_elem_size=self.fields.h_elem_size_3d,
                #     bnd_markers=bnd_markers,
                #     bnd_len=bnd_len)

        self.eq_sw.bnd_functions = self.bnd_functions['shallow_water']
        self.eq_momentum.bnd_functions = self.bnd_functions['momentum']
        if self.options.solve_salt:
            self.eq_salt.bnd_functions = self.bnd_functions['salt']
        if self.options.use_turbulence:
            if self.options.use_turbulence_advection:
                # explicit advection equations
                self.eq_tke_adv = tracer_eq.TracerEquationNew(self.fields.tke_3d.function_space(),
                                                              bnd_markers, bnd_len,
                                                              bathymetry=self.fields.bathymetry_3d,
                                                              v_elem_size=self.fields.v_elem_size_3d,
                                                              h_elem_size=self.fields.h_elem_size_3d)
                self.eq_psi_adv = tracer_eq.TracerEquationNew(self.fields.psi_3d.function_space(),
                                                              bnd_markers, bnd_len,
                                                              bathymetry=self.fields.bathymetry_3d,
                                                              v_elem_size=self.fields.v_elem_size_3d,
                                                              h_elem_size=self.fields.h_elem_size_3d)
                # self.eq_tke_adv = tracer_eq.TracerEquation(
                #     self.fields.tke_3d, self.fields.elev_3d, self.fields.uv_3d,
                #     w=self.fields.w_3d, w_mesh=self.fields.get('w_mesh_3d'),
                #     dw_mesh_dz=self.fields.get('w_mesh_ddz_3d'),
                #     diffusivity_h=None,
                #     diffusivity_v=None,
                #     uv_p1=self.fields.get('uv_p1_3d'),
                #     lax_friedrichs_factor=self.options.tracer_lax_friedrichs,
                #     v_elem_size=self.fields.v_elem_size_3d,
                #     h_elem_size=self.fields.h_elem_size_3d,
                #     bnd_markers=bnd_markers,
                #     bnd_len=bnd_len)
                # self.eq_psi_adv = tracer_eq.TracerEquation(
                #     self.fields.psi_3d, self.fields.elev_3d, self.fields.uv_3d,
                #     w=self.fields.w_3d, w_mesh=self.fields.get('w_mesh_3d'),
                #     dw_mesh_dz=self.fields.get('w_mesh_ddz_3d'),
                #     diffusivity_h=None,
                #     diffusivity_v=None,
                #     uv_p1=self.fields.get('uv_p1_3d'),
                #     lax_friedrichs_factor=self.options.tracer_lax_friedrichs,
                #     v_elem_size=self.fields.v_elem_size_3d,
                #     h_elem_size=self.fields.h_elem_size_3d,
                #     bnd_markers=bnd_markers,
                #     bnd_len=bnd_len)
            # implicit vertical diffusion eqn with production terms
            self.eq_tke_diff = turbulence.TKEEquationNew(self.fields.tke_3d.function_space(),
                                                         self.gls_model,
                                                         bnd_markers, bnd_len,
                                                         bathymetry=self.fields.bathymetry_3d,
                                                         v_elem_size=self.fields.v_elem_size_3d,
                                                         h_elem_size=self.fields.h_elem_size_3d)
            self.eq_psi_diff = turbulence.PsiEquationNew(self.fields.psi_3d.function_space(),
                                                         self.gls_model,
                                                         bnd_markers, bnd_len,
                                                         bathymetry=self.fields.bathymetry_3d,
                                                         v_elem_size=self.fields.v_elem_size_3d,
                                                         h_elem_size=self.fields.h_elem_size_3d)
            # self.eq_tke_diff = turbulence.TKEEquation(
            #     self.fields.tke_3d,
            #     self.fields.elev_3d, uv=None,
            #     w=None, w_mesh=None,
            #     dw_mesh_dz=None,
            #     diffusivity_h=None,
            #     diffusivity_v=implicit_v_diff,
            #     viscosity_v=implicit_v_visc,
            #     v_elem_size=self.fields.v_elem_size_3d,
            #     h_elem_size=self.fields.h_elem_size_3d,
            #     uv_mag=None, uv_p1=None, lax_friedrichs_factor=None,
            #     bnd_markers=bnd_markers, bnd_len=bnd_len,
            #     gls_model=self.gls_model)
            # self.eq_psi_diff = turbulence.PsiEquation(
            #     self.fields.psi_3d, self.fields.elev_3d, uv=None,
            #     w=None, w_mesh=None,
            #     dw_mesh_dz=None,
            #     diffusivity_h=None,
            #     diffusivity_v=implicit_v_diff,
            #     viscosity_v=implicit_v_visc,
            #     v_elem_size=self.fields.v_elem_size_3d,
            #     h_elem_size=self.fields.h_elem_size_3d,
            #     uv_mag=None, uv_p1=None, lax_friedrichs_factor=None,
            #     bnd_markers=bnd_markers, bnd_len=bnd_len,
            #     gls_model=self.gls_model)

        # ----- Time integrators
        self.set_time_step()
        if self.options.use_mode_split:
            if self.options.use_imex:
                self.timestepper = coupled_timeintegrator.CoupledSSPIMEX(weakref.proxy(self))
            elif self.options.use_semi_implicit_2d:
                self.timestepper = coupled_timeintegrator.CoupledSSPRKSemiImplicit(weakref.proxy(self))
            else:
                self.timestepper = coupled_timeintegrator.CoupledSSPRKSync(weakref.proxy(self))
        else:
            self.timestepper = coupled_timeintegrator.CoupledSSPRKSingleMode(weakref.proxy(self))
        print_info('using {:} time integrator'.format(self.timestepper.__class__.__name__))

        # compute maximal diffusivity for explicit schemes
        degree_h, degree_v = self.function_spaces.H.ufl_element().degree()
        max_diff_alpha = 1.0/360.0/max((degree_h*(degree_h + 1)), 1.0)  # FIXME depends on element type and order
        self.fields.max_h_diff.assign(max_diff_alpha/self.dt * self.fields.h_elem_size_3d**2)
        d = self.fields.max_h_diff.dat.data
        print_info('max h diff {:} - {:}'.format(d.min(), d.max()))

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
                                               self.eq_sw.boundary_markers,
                                               self.eq_momentum.bnd_functions)
        # NOTE averager is a word. now.
        self.uv_averager = VerticalIntegrator(self.fields.uv_3d,
                                              self.fields.uv_dav_3d,
                                              bottom_to_top=True,
                                              bnd_value=Constant((0.0, 0.0, 0.0)),
                                              average=True,
                                              bathymetry=self.fields.bathymetry_3d)
        if self.options.baroclinic:
            self.rho_integrator = VerticalIntegrator(self.fields.salt_3d,
                                                     self.fields.baroc_head_3d,
                                                     bottom_to_top=False)
            self.baro_head_averager = VerticalIntegrator(self.fields.baroc_head_3d,
                                                         self.fields.baroc_head_int_3d,
                                                         bottom_to_top=True,
                                                         average=True,
                                                         bathymetry=self.fields.bathymetry_3d)
            self.extract_surf_baro_head = SubFunctionExtractor(self.fields.baroc_head_int_3d,
                                                               self.fields.baroc_head_2d,
                                                               boundary='top', elem_facet='top')
        self.extract_surf_dav_uv = SubFunctionExtractor(self.fields.uv_dav_3d,
                                                        self.fields.uv_dav_2d,
                                                        boundary='top', elem_facet='top',
                                                        elem_height=self.fields.v_elem_size_2d)
        self.copy_v_elem_size_to_2d = SubFunctionExtractor(self.fields.v_elem_size_3d,
                                                           self.fields.v_elem_size_2d,
                                                           boundary='top', elem_facet='top')
        self.copy_elev_to_3d = ExpandFunctionTo3d(self.fields.elev_2d, self.fields.elev_3d)
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
        if self.options.use_ale_moving_mesh:
            self.mesh_coord_updater = ALEMeshCoordinateUpdater(self.mesh,
                                                               self.fields.elev_3d,
                                                               self.fields.bathymetry_3d,
                                                               self.fields.z_coord_3d,
                                                               self.fields.z_coord_ref_3d)
            self.extract_surf_w = SubFunctionExtractor(self.fields.w_mesh_surf_3d,
                                                       self.fields.w_mesh_surf_2d,
                                                       boundary='top', elem_facet='top')
            self.copy_surf_w_mesh_to_3d = ExpandFunctionTo3d(self.fields.w_mesh_surf_2d,
                                                             self.fields.w_mesh_surf_3d)
            self.w_mesh_solver = MeshVelocitySolver(self, self.fields.elev_3d,
                                                    self.fields.uv_3d,
                                                    self.fields.w_3d,
                                                    self.fields.w_mesh_3d,
                                                    self.fields.w_mesh_surf_3d,
                                                    self.fields.w_mesh_surf_2d,
                                                    self.fields.w_mesh_ddz_3d,
                                                    self.fields.bathymetry_3d,
                                                    self.fields.z_coord_ref_3d)

        if self.options.salt_jump_diff_factor is not None:
            self.horiz_jump_diff_solver = HorizontalJumpDiffusivity(self.options.salt_jump_diff_factor, self.fields.salt_3d,
                                                                    self.fields.salt_jump_diff, self.fields.h_elem_size_3d,
                                                                    self.fields.uv_mag_3d, self.options.salt_range,
                                                                    self.fields.max_h_diff)
        if self.options.smagorinsky_factor is not None:
            self.smagorinsky_diff_solver = SmagorinskyViscosity(self.fields.uv_p1_3d, self.fields.smag_visc_3d,
                                                                self.options.smagorinsky_factor, self.fields.h_elem_size_3d,
                                                                self.fields.max_h_diff)
        if self.options.use_parabolic_viscosity:
            self.parabolic_viscosity_solver = ParabolicViscosity(self.fields.uv_bottom_3d,
                                                                 self.fields.bottom_drag_3d,
                                                                 self.fields.bathymetry_3d,
                                                                 self.fields.parab_visc_3d)
        self.uv_p1_projector = Projector(self.fields.uv_3d, self.fields.uv_p1_3d)
        # self.uv_dav_to_tmp_projector = projector(self.uv_dav_3d, self.uv_3d_tmp)
        # self.uv_2d_to_dav_projector = projector(self.fields.solution_2d.split()[0],
        #                                         self.uv_dav_2d)
        # self.uv_2d_dav_to_uv_2d_projector = projector(self.uv_dav_2d,
        #                                              self.fields.solution_2d.split()[0])
        self.elev_3d_to_cg_projector = Projector(self.fields.elev_3d, self.fields.elev_cg_3d)

        # ----- set initial values
        ExpandFunctionTo3d(self.fields.bathymetry_2d, self.fields.bathymetry_3d).solve()
        get_zcoord_from_mesh(self.fields.z_coord_ref_3d)
        self.fields.z_coord_3d.assign(self.fields.z_coord_ref_3d)
        compute_elem_height(self.fields.z_coord_3d, self.fields.v_elem_size_3d)
        self.copy_v_elem_size_to_2d.solve()

        self._initialized = True
        self._isfrozen = True

    def assign_initial_conditions(self, elev=None, salt=None, uv_2d=None):
        if not self._initialized:
            self.create_equations()
        if elev is not None:
            self.fields.elev_2d.project(elev)
            self.copy_elev_to_3d.solve()
            self.fields.elev_cg_3d.project(self.fields.elev_3d)
            if self.options.use_ale_moving_mesh:
                self.mesh_coord_updater.solve()
                compute_elem_height(self.fields.z_coord_3d, self.fields.v_elem_size_3d)
                self.copy_v_elem_size_to_2d.solve()
        if uv_2d is not None:
            self.fields.uv_2d.project(uv_2d)
            ExpandFunctionTo3d(self.fields.uv_2d, self.fields.uv_3d,
                               elem_height=self.fields.v_elem_size_3d).solve()
        if salt is not None and self.options.solve_salt:
            self.fields.salt_3d.project(salt)
        self.w_solver.solve()
        if self.options.use_ale_moving_mesh:
            self.w_mesh_solver.solve()
        if self.options.baroclinic:
            compute_baroclinic_head(self, self.fields.salt_3d, self.fields.baroc_head_3d,
                                    self.fields.baroc_head_2d, self.fields.baroc_head_int_3d,
                                    self.fields.bathymetry_3d)
        if self.options.use_turbulence:
            self.gls_model.initialize()

        self.timestepper.initialize()

    def export(self):
        for key in self.exporters:
            self.exporters[key].export()

    def load_state(self, i_export, t, iteration):
        """Loads simulation state from hdf5 outputs."""
        # TODO use options to figure out which functions need to be loaded
        raise NotImplementedError('state loading is not yet implemented for 3d solver')

    def print_state(self, cputime):
        norm_h = norm(self.fields.solution_2d.split()[1])
        norm_u = norm(self.fields.solution_2d.split()[0])

        if commrank == 0:
            line = ('{iexp:5d} {i:5d} T={t:10.2f} '
                    'eta norm: {e:10.4f} u norm: {u:10.4f} {cpu:5.2f}')
            print(bold(line.format(iexp=self.i_export, i=self.iteration, t=self.simulation_time, e=norm_h,
                                   u=norm_u, cpu=cputime)))
            sys.stdout.flush()

    def iterate(self, update_forcings=None, update_forcings3d=None,
                export_func=None):
        if not self._initialized:
            self.create_equations()

        self.options.check_salt_conservation *= self.options.solve_salt
        self.options.check_salt_overshoot *= self.options.solve_salt
        self.options.check_vol_conservation_3d *= self.options.use_ale_moving_mesh

        t_epsilon = 1.0e-5
        cputimestamp = time_mod.clock()
        self.simulation_time = 0
        self.iteration = 0
        self.i_export = 1
        next_export_t = self.simulation_time + self.options.t_export

        if self.options.check_vol_conservation_2d:
            self.callbacks['vol2d'] = callback.VolumeConservation2DCallback()
        if self.options.check_vol_conservation_3d:
            self.callbacks['vol3d'] = callback.VolumeConservation3DCallback()
        if self.options.check_salt_conservation:
            self.callbacks['salt_3d_mass'] = callback.TracerMassConservationCallback('salt_3d')
        if self.options.check_salt_overshoot:
            self.callbacks['salt_3d_overshoot'] = callback.TracerOvershootCallBack('salt_3d')
        if self.options.check_salt_deviation:
            raise DeprecationWarning('check_salt_deviation option is deprecated use check_salt_overshoot instead')

        for key in self.callbacks:
            self.callbacks[key].initialize(self)

        # initial export
        self.export()
        if export_func is not None:
            export_func()
        if 'vtk' in self.exporters:
            self.exporters['vtk'].export_bathymetry(self.fields.bathymetry_2d)

        while self.simulation_time <= self.options.t_end + t_epsilon:

            self.timestepper.advance(self.simulation_time, self.dt,
                                     update_forcings, update_forcings3d)

            # Move to next time step
            self.simulation_time += self.dt
            self.iteration += 1

            # Write the solution to file
            if self.simulation_time >= next_export_t - t_epsilon:
                cputime = time_mod.clock() - cputimestamp
                cputimestamp = time_mod.clock()
                self.print_state(cputime)

                for key in self.callbacks:
                    self.callbacks[key].update(self)
                    self.callbacks[key].report()

                self.export()
                if export_func is not None:
                    export_func()

                next_export_t += self.options.t_export
                self.i_export += 1

                if commrank == 0 and len(self.options.timer_labels) > 0:
                    cost = {}
                    relcost = {}
                    totcost = 0
                    for label in self.options.timer_labels:
                        value = timing(label, reset=True)
                        cost[label] = value
                        totcost += value
                    for label in self.options.timer_labels:
                        c = cost[label]
                        relcost = c/max(totcost, 1e-6)
                        print '{0:25s} : {1:11.6f} {2:11.2f}'.format(
                            label, c, relcost)
                        sys.stdout.flush()
