
                self.bathymetry_cg_2d.project(self.bathymetry_dg)
                ExpandFunctionTo3d(self.bathymetry_cg_2d, self.fields.bathymetry_3d).solve()
                n_stages = 2
                advance_2d_first = True#False
                if advance_2d_first:
                    for i_stage in range(n_stages):
                        ## 2D advance
                        self.update_mid_uv(self.fields.uv_3d) # make sure elev_3d has been updated
                        self.fields.uv_2d.project(self.uv_dav_2d_mid) # elev_mid <= elev_2d (known)
                        self.timestepper.store_elevation(i_stage)
                        if i_stage == 1:
                            self.uv_3d_tmp.assign(self.fields.uv_3d)
                            timestepper_operator_splitting.advance(self.simulation_time, update_forcings)
                       # self.timestepper.timesteppers.swe2d.solve_stage(i_stage, self.simulation_time, update_forcings)
                        # compute mesh velocity
                        self.timestepper.compute_mesh_velocity(i_stage)

                        ## 3D advance in old mesh
                        # salt_eq
                        if self.options.solve_salinity:
                            self.timestepper.timesteppers.salt_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # tmp_eq
                        if self.options.solve_temperature:
                            self.timestepper.timesteppers.temp_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # turb_advection
                        if 'psi_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.psi_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        if 'tke_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.tke_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # momentum_eq
                        self.timestepper.timesteppers.mom_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)

                        ## update mesh
                        self.copy_elev_to_3d.solve()
                        if self.options.use_ale_moving_mesh:
                            self.mesh_updater.update_mesh_coordinates()

                        ## solve 3D
                        # salt_eq
                        if self.options.solve_salinity:
                            self.timestepper.timesteppers.salt_expl.solve_stage(i_stage)
                            if self.options.use_limiter_for_tracers:
                                self.tracer_limiter.apply(self.fields.salt_3d)
                        # temp_eq
                        if self.options.solve_temperature:
                            self.timestepper.timesteppers.temp_expl.solve_stage(i_stage)
                            if self.options.use_limiter_for_tracers:
                                self.tracer_limiter.apply(self.fields.temp_3d)
                        # turb_advection
                        if 'psi_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.psi_expl.solve_stage(i_stage)
                        if 'tke_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.tke_expl.solve_stage(i_stage)
                        # momentum_eq
                        self.timestepper.timesteppers.mom_expl.solve_stage(i_stage)
                        if self.options.use_limiter_for_velocity:
                            self.uv_limiter.apply(self.fields.uv_3d)

                        last_stage = i_stage == n_stages - 1

                        if last_stage:
                            ## compute final prognostic variables
                            # correct uv_3d
                            self.copy_uv_to_uv_dav_3d.solve()
                            self.fields.uv_3d.project(self.fields.uv_3d - (self.uv_dav_3d_mid - self.fields.uv_dav_3d))
                            if self.options.use_implicit_vertical_diffusion:
                                if self.options.solve_salinity:
                                    with timed_stage('impl_salt_vdiff'):
                                        self.timestepper.timesteppers.salt_impl.advance(self.simulation_time)
                                if self.options.solve_temperature:
                                    with timed_stage('impl_temp_vdiff'):
                                        self.timestepper.timesteppers.temp_impl.advance(self.simulation_time)
                                with timed_stage('impl_mom_vvisc'):
                                    self.timestepper.timesteppers.mom_impl.advance(self.simulation_time)
                            ## compute final diagnostic fields
                            # update baroclinicity
                            self.timestepper._update_baroclinicity()
                            # update w
                            #self.w_solver.solve()
                            # update parametrizations
                            self.timestepper._update_turbulence(self.simulation_time)
                            self.timestepper._update_bottom_friction()
                            self.timestepper._update_stabilization_params()
                        else:
                            ## update variables that explict solvers depend on
                            # correct uv_3d
                            self.copy_uv_to_uv_dav_3d.solve()
                            self.fields.uv_3d.project(self.fields.uv_3d - (self.uv_dav_3d_mid - self.fields.uv_dav_3d))
                            # update baroclinicity
                            self.timestepper._update_baroclinicity()
                            # update w
                            #self.w_solver.solve()

                else:
                    for i_stage in range(n_stages):
                        # 3D mom advance
                        self.timestepper.timesteppers.mom_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # update mesh
                        self.copy_elev_to_3d.solve()
                        if self.options.use_ale_moving_mesh:
                            self.mesh_updater.update_mesh_coordinates()
                        # solve 3D
                        self.timestepper.timesteppers.mom_expl.solve_stage(i_stage)
                        if self.options.use_limiter_for_velocity:
                            self.uv_limiter.apply(self.fields.uv_3d)
                        # 2D eta adavance
                        self.update_mid_uv(self.fields.uv_3d) # make sure elev_3d has been updated
                        self.fields.uv_2d.project(self.uv_dav_2d_mid) # elev_mid <= elev_2d (known)
                        self.timestepper.store_elevation(i_stage)
                        if i_stage == 1:
                            self.uv_3d_tmp.assign(self.fields.uv_3d)
                            timestepper_operator_splitting.advance(self.simulation_time, update_forcings)
                        #self.timestepper.timesteppers.swe2d.solve_stage(i_stage, self.simulation_time, update_forcings)
                        # compute mesh velocity
                        self.timestepper.compute_mesh_velocity(i_stage)
                        # correct uv_3d
                        self.copy_uv_to_uv_dav_3d.solve()
                        self.fields.uv_3d.project(self.fields.uv_3d - (self.uv_dav_3d_mid - self.fields.uv_dav_3d))

                self.poisson_solver(self.fields.q_3d, self.fields.uv_3d, self.fields.w_3d, A=None, B=None, C=1./self.dt, multi_layers=True)

                # update uv_3d
                uv_tri = TrialFunction(self.fields.uv_3d.function_space())
                uv_test = TestFunction(self.fields.uv_3d.function_space())
                self.fields.uv_3d.sub(2).assign(self.fields.w_3d.sub(2))
                a = dot(uv_tri, uv_test)*dx
                l = dot(self.fields.uv_3d - self.dt*grad(self.fields.q_3d), uv_test)*dx
                solve(a == l, self.fields.uv_3d)
                self.fields.w_3d.sub(2).assign(self.fields.uv_3d.sub(2))
                self.fields.uv_3d.sub(2).assign(0.)

             #   if self.options.vertical_2d:
             #       self.vertical_2d()
             #       self.fields.uv_3d.sub(1).assign(0.)

                # update final depth-averaged uv_2d
                self.update_mid_uv(self.fields.uv_3d) # make sure elev_3d has been updated
                self.fields.uv_2d.project(self.uv_dav_2d_mid)

                # not necessary already, due to w updated by non-hydrostatic pressure gradient
                # self.w_solver.solve()

                # update elev_2d: two ways
                solving_free_surface_eq = True
                if not solving_free_surface_eq:
                    # 1. based on 2D/3D difference in operator splitting
                    self.copy_elev_to_3d.solve()
                    self.update_mid_uv(self.fields.uv_3d - self.fields.uv_dav_3d)
                    elev_tri = TrialFunction(self.function_spaces.H_2d)
                    elev_test = TestFunction(self.function_spaces.H_2d)
                    a = elev_tri*elev_test*dx
                    l = (elev_2d - self.dt*div((elev_2d + self.bathymetry_dg)*self.uv_dav_2d_mid))*elev_test*dx
                    solve(a == l, self.fields.elev_2d)
                else:
                    # 2. based on solving free surface equation
                    timestepper_free_surface.advance(self.simulation_time, update_forcings)
                    self.fields.elev_2d.assign(self.elev_2d_old)



                # update eta by solve 2D equations with non-hydrostatic pressure
                self.fields.elev_cg_3d.project(self.elev_3d_old)
                if False:
                    q_integrator = VerticalIntegrator(self.fields.q_3d,
                                                  self.q_3d_mid,
                                                  bottom_to_top=False,
                                                  average=False,
                                                  bathymetry=self.fields.bathymetry_3d,
                                                  elevation=self.fields.elev_cg_3d)
                    extract_sum_q = SubFunctionExtractor(self.q_3d_mid,
                                                     self.q_2d_mid,
                                                     boundary='bottom', elem_facet='bottom',
                                                     elem_height=self.fields.v_elem_size_2d)
                    extract_bottom_q = SubFunctionExtractor(self.fields.q_3d,
                                                        self.fields.q_2d,
                                                        boundary='bottom', elem_facet='bottom',
                                                        elem_height=self.fields.v_elem_size_2d)
                #q_integrator.solve()
                #extract_sum_q.solve()
                #extract_bottom_q.solve()
                #q_sum = self.q_2d_mid #- self.fields.q_2d/2.
                #print(self.q_2d_mid.dat.data)
                # update elev => serious decay
               # self.fields.elev_cg_3d.project(self.elev_3d_old)
               # self.uv_averager.solve()
               # self.extract_surf_dav_uv.solve()
               # uv_vert_int = self.fields.uv_dav_2d*(self.bathymetry_dg + elev_2d_old)
               # elev_tri = TrialFunction(self.function_spaces.H_2d)
               # elev_test = TestFunction(self.function_spaces.H_2d)
               # a = elev_tri*elev_test*dx
               # l = (elev_2d_old - self.dt*div(uv_vert_int))*elev_test*dx
               # solve(a == l, self.fields.elev_2d)



                if self.simulation_time <= t_epsilon:
                    h_mid = eta + self.bathymetry_dg
                    timestepper_depth_integrated.F += 0#(self.dt*1./h_mid*inner(grad(q_sum) - self.fields.q_2d*grad(self.bathymetry_dg), uta_test)*dx)
                    prob_arbitrary_layer_int = NonlinearVariationalProblem(timestepper_depth_integrated.F, self.fields.solution_2d)
                    solver_arbitrary_layer_int = NonlinearVariationalSolver(prob_arbitrary_layer_int,
                                                                        solver_parameters=solver_parameters)


                #self.update_mid_uv(self.uv_3d_tmp)
                #timestepper_depth_integrated.solution_old.sub(0).project(self.uv_dav_2d_mid)
                #timestepper_depth_integrated.solution_old.sub(1).project(self.elev_2d_old)
                #solver_arbitrary_layer_int.solve()
                #self.copy_uv_to_uv_dav_3d.solve()
                #self.fields.uv_3d.project(self.fields.uv_3d - (self.uv_dav_3d_mid - self.fields.uv_dav_3d))













\\\\\\\\\\ for pressure correction

                self.bathymetry_cg_2d.project(self.bathymetry_dg)
                ExpandFunctionTo3d(self.bathymetry_cg_2d, self.fields.bathymetry_3d).solve()
                # update self.fields.ext_pg_3d
                self.external_pressure_gradient_calculator()
                # average self.fields.ext_pg_3d
                self.elev_3d_to_cg_projector.project()
                ext_pg_averager = VerticalIntegrator(self.fields.ext_pg_3d,
                                             self.pg_dav_3d,
                                             bottom_to_top=True,
                                             bnd_value=Constant((0.0, 0.0, 0.0)),
                                             average=True,
                                             bathymetry=self.fields.bathymetry_3d,
                                             elevation=self.fields.elev_cg_3d)
                extract_surf_dav_pg = SubFunctionExtractor(self.pg_dav_3d,
                                                       self.pg_dav_2d,
                                                       boundary='top', elem_facet='top',
                                                       elem_height=self.fields.v_elem_size_2d)
                copy_pg_dav_to_pg_dav_3d = ExpandFunctionTo3d(self.pg_dav_2d, self.pg_dav_3d_mid,
                                                          elem_height=self.fields.v_elem_size_3d)
                ext_pg_averager.solve()
                extract_surf_dav_pg.solve()
                copy_pg_dav_to_pg_dav_3d.solve()

                self.q_3d_old.assign(self.fields.q_3d)

                # update timestepper_operator_splitting
                if self.simulation_time <= t_epsilon:
                    timestepper_operator_splitting.F += self.dt*inner(self.pg_dav_2d, uta_test)*dx
                    timestepper_operator_splitting.update_solver()

                n_stages = 2
                advance_2d_first = True#False
                if advance_2d_first:
                    for i_stage in range(n_stages):
                        ## 2D advance
                        self.update_mid_uv(self.fields.uv_3d) # make sure elev_3d has been updated
                        self.fields.uv_2d.project(self.uv_dav_2d_mid) # elev_mid <= elev_2d (known)
                        self.timestepper.store_elevation(i_stage)
                        if i_stage == 1:
                            self.uv_3d_tmp.assign(self.fields.uv_3d)
                            timestepper_operator_splitting.advance(self.simulation_time, update_forcings)
                       # self.timestepper.timesteppers.swe2d.solve_stage(i_stage, self.simulation_time, update_forcings)
                        # compute mesh velocity
                        self.timestepper.compute_mesh_velocity(i_stage)

                        ## 3D advance in old mesh
                        # salt_eq
                        if self.options.solve_salinity:
                            self.timestepper.timesteppers.salt_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # tmp_eq
                        if self.options.solve_temperature:
                            self.timestepper.timesteppers.temp_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # turb_advection
                        if 'psi_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.psi_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        if 'tke_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.tke_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # momentum_eq
                        self.timestepper.timesteppers.mom_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)

                        ## update mesh
                        self.copy_elev_to_3d.solve()
                        if self.options.use_ale_moving_mesh:
                            self.mesh_updater.update_mesh_coordinates()

                        ## solve 3D
                        # salt_eq
                        if self.options.solve_salinity:
                            self.timestepper.timesteppers.salt_expl.solve_stage(i_stage)
                            if self.options.use_limiter_for_tracers:
                                self.tracer_limiter.apply(self.fields.salt_3d)
                        # temp_eq
                        if self.options.solve_temperature:
                            self.timestepper.timesteppers.temp_expl.solve_stage(i_stage)
                            if self.options.use_limiter_for_tracers:
                                self.tracer_limiter.apply(self.fields.temp_3d)
                        # turb_advection
                        if 'psi_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.psi_expl.solve_stage(i_stage)
                        if 'tke_expl' in self.timestepper.timesteppers:
                            self.timestepper.timesteppers.tke_expl.solve_stage(i_stage)
                        # momentum_eq
                        self.timestepper.timesteppers.mom_expl.solve_stage(i_stage)
                        if self.options.use_limiter_for_velocity:
                            self.uv_limiter.apply(self.fields.uv_3d)

                        last_stage = i_stage == n_stages - 1

                        if last_stage:
                            ## compute final prognostic variables
                            # correct uv_3d
                            self.copy_uv_to_uv_dav_3d.solve()
                            self.fields.uv_3d.project(self.fields.uv_3d + (self.fields.uv_dav_3d - self.uv_dav_3d_mid +
                                                      self.dt*(self.pg_dav_3d_mid - self.fields.ext_pg_3d)))
                            if self.options.use_implicit_vertical_diffusion:
                                if self.options.solve_salinity:
                                    with timed_stage('impl_salt_vdiff'):
                                        self.timestepper.timesteppers.salt_impl.advance(self.simulation_time)
                                if self.options.solve_temperature:
                                    with timed_stage('impl_temp_vdiff'):
                                        self.timestepper.timesteppers.temp_impl.advance(self.simulation_time)
                                with timed_stage('impl_mom_vvisc'):
                                    self.timestepper.timesteppers.mom_impl.advance(self.simulation_time)
                            ## compute final diagnostic fields
                            # update baroclinicity
                            self.timestepper._update_baroclinicity()
                            # update w
                            #self.w_solver.solve()
                            # update parametrizations
                            self.timestepper._update_turbulence(self.simulation_time)
                            self.timestepper._update_bottom_friction()
                            self.timestepper._update_stabilization_params()
                        else:
                            ## update variables that explict solvers depend on
                            # correct uv_3d
                            self.copy_uv_to_uv_dav_3d.solve()
                            self.fields.uv_3d.project(self.fields.uv_3d + (self.fields.uv_dav_3d - self.uv_dav_3d_mid +
                                                      self.dt*(self.pg_dav_3d_mid - self.fields.ext_pg_3d)))
                            # update baroclinicity
                            self.timestepper._update_baroclinicity()
                            # update w
                            #self.w_solver.solve()

                else:
                    for i_stage in range(n_stages):
                        # 3D mom advance
                        self.timestepper.timesteppers.mom_expl.prepare_stage(i_stage, self.simulation_time, update_forcings3d)
                        # update mesh
                        self.copy_elev_to_3d.solve()
                        if self.options.use_ale_moving_mesh:
                            self.mesh_updater.update_mesh_coordinates()
                        # solve 3D
                        self.timestepper.timesteppers.mom_expl.solve_stage(i_stage)
                        if self.options.use_limiter_for_velocity:
                            self.uv_limiter.apply(self.fields.uv_3d)
                        # 2D eta adavance
                        self.update_mid_uv(self.fields.uv_3d) # make sure elev_3d has been updated
                        self.fields.uv_2d.project(self.uv_dav_2d_mid) # elev_mid <= elev_2d (known)
                        self.timestepper.store_elevation(i_stage)
                        if i_stage == 1:
                            self.uv_3d_tmp.assign(self.fields.uv_3d)
                            timestepper_operator_splitting.advance(self.simulation_time, update_forcings)
                        #self.timestepper.timesteppers.swe2d.solve_stage(i_stage, self.simulation_time, update_forcings)
                        # compute mesh velocity
                        self.timestepper.compute_mesh_velocity(i_stage)
                        # correct uv_3d
                        self.copy_uv_to_uv_dav_3d.solve()
                        self.fields.uv_3d.project(self.fields.uv_3d - (self.uv_dav_3d_mid - self.fields.uv_dav_3d))

                self.poisson_solver(self.dq_3d, self.fields.uv_3d, self.fields.w_3d, A=None, B=None, C=1./self.dt, multi_layers=True)
                self.fields.q_3d.assign(self.q_3d_old + self.dq_3d)

                # update uv_3d
                uv_tri = TrialFunction(self.fields.uv_3d.function_space())
                uv_test = TestFunction(self.fields.uv_3d.function_space())
                #self.fields.uv_3d.sub(2).assign(self.fields.w_3d.sub(2))
                a = dot(uv_tri, uv_test)*dx
                l = dot(self.fields.uv_3d - self.dt*grad(self.dq_3d), uv_test)*dx
                solve(a == l, self.fields.uv_3d)
                #self.fields.w_3d.sub(2).assign(self.fields.uv_3d.sub(2))
                self.fields.uv_3d.sub(2).assign(0.)
                l = dot(self.fields.w_3d - self.dt*grad(self.fields.q_3d), uv_test)*dx
                solve(a == l, self.fields.w_3d)
                self.fields.w_3d.sub(0).assign(0.)
                self.fields.w_3d.sub(1).assign(0.)

             #   if self.options.vertical_2d:
             #       self.vertical_2d()
             #       self.fields.uv_3d.sub(1).assign(0.)

                # update final depth-averaged uv_2d
                self.update_mid_uv(self.fields.uv_3d) # make sure elev_3d has been updated
                self.fields.uv_2d.project(self.uv_dav_2d_mid)

                # not necessary already, due to w updated by non-hydrostatic pressure gradient
                # self.w_solver.solve()

                # update elev_2d: two ways
                solving_free_surface_eq = True
                if not solving_free_surface_eq:
                    # 1. based on 2D/3D difference in operator splitting
                    self.copy_elev_to_3d.solve()
                    self.update_mid_uv(self.fields.uv_3d - self.fields.uv_dav_3d)
                    elev_tri = TrialFunction(self.function_spaces.H_2d)
                    elev_test = TestFunction(self.function_spaces.H_2d)
                    a = elev_tri*elev_test*dx
                    l = (elev_2d - self.dt*div((elev_2d + self.bathymetry_dg)*self.uv_dav_2d_mid))*elev_test*dx
                    solve(a == l, self.fields.elev_2d)
                else:
                    # 2. based on solving free surface equation
                    timestepper_free_surface.advance(self.simulation_time, update_forcings)
                    self.fields.elev_2d.assign(self.elev_2d_old)



                # update eta by solve 2D equations with non-hydrostatic pressure
                self.fields.elev_cg_3d.project(self.elev_3d_old)
                if False:
                    q_integrator = VerticalIntegrator(self.fields.q_3d,
                                                  self.q_3d_mid,
                                                  bottom_to_top=False,
                                                  average=False,
                                                  bathymetry=self.fields.bathymetry_3d,
                                                  elevation=self.fields.elev_cg_3d)
                    extract_sum_q = SubFunctionExtractor(self.q_3d_mid,
                                                     self.q_2d_mid,
                                                     boundary='bottom', elem_facet='bottom',
                                                     elem_height=self.fields.v_elem_size_2d)
                    extract_bottom_q = SubFunctionExtractor(self.fields.q_3d,
                                                        self.fields.q_2d,
                                                        boundary='bottom', elem_facet='bottom',
                                                        elem_height=self.fields.v_elem_size_2d)
                #q_integrator.solve()
                #extract_sum_q.solve()
                #extract_bottom_q.solve()
                #q_sum = self.q_2d_mid #- self.fields.q_2d/2.
                #print(self.q_2d_mid.dat.data)
                # update elev => serious decay
               # self.fields.elev_cg_3d.project(self.elev_3d_old)
               # self.uv_averager.solve()
               # self.extract_surf_dav_uv.solve()
               # uv_vert_int = self.fields.uv_dav_2d*(self.bathymetry_dg + elev_2d_old)
               # elev_tri = TrialFunction(self.function_spaces.H_2d)
               # elev_test = TestFunction(self.function_spaces.H_2d)
               # a = elev_tri*elev_test*dx
               # l = (elev_2d_old - self.dt*div(uv_vert_int))*elev_test*dx
               # solve(a == l, self.fields.elev_2d)



                if self.simulation_time <= t_epsilon:
                    h_mid = eta + self.bathymetry_dg
                    timestepper_depth_integrated.F += 0#(self.dt*1./h_mid*inner(grad(q_sum) - self.fields.q_2d*grad(self.bathymetry_dg), uta_test)*dx)
                    prob_arbitrary_layer_int = NonlinearVariationalProblem(timestepper_depth_integrated.F, self.fields.solution_2d)
                    solver_arbitrary_layer_int = NonlinearVariationalSolver(prob_arbitrary_layer_int,
                                                                        solver_parameters=solver_parameters)


                #self.update_mid_uv(self.uv_3d_tmp)
                #timestepper_depth_integrated.solution_old.sub(0).project(self.uv_dav_2d_mid)
                #timestepper_depth_integrated.solution_old.sub(1).project(self.elev_2d_old)
                #solver_arbitrary_layer_int.solve()
                #self.copy_uv_to_uv_dav_3d.solve()
                #self.fields.uv_3d.project(self.fields.uv_3d - (self.uv_dav_3d_mid - self.fields.uv_dav_3d))



