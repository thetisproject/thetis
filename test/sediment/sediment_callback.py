from thetis.callback import DiagnosticCallback
from thetis.utility import *


class SedimentTotalMassConservation2DCallback(DiagnosticCallback):
    """
    Checks conservation of depth-averaged sediment mass for non-conservative and
    depth-integrated sediment mass for conservative, accounting for sediment leaving
    through boundary and source terms.

    Depth-averaged sediment mass is defined as the integral of 2D sediment
    multiplied by total depth, subtracting sediment leaving through boundary and through source term.

    Depth-integrated sediment mass is defined as the integral of 2D sediment
    subtracting sediment leaving through boundary and through source term.
    """
    name = 'sediment mass'
    variable_names = ['integral', 'relative_difference']

    def __init__(self, sediment_name, solver_obj, **kwargs):
        """
        :arg sediment_name: Name of the sediment. Use canonical field names as in
            :class:`.FieldDict`.
        :arg solver_obj: Thetis solver object
        :arg kwargs: any additional keyword arguments, see
            :class:`.DiagnosticCallback`.
        """

        self.name = sediment_name + ' total mass'  # override name for given sediment

        def mass():
            if not hasattr(self, 'initial_value'):
                self.initial_value = None

            if solver_obj.options.sediment_model_options.use_sediment_conservative_form:
                return self.comp_sediment_total_mass_2d_cons(sediment_name)
            else:
                return self.comp_sediment_total_mass_2d(sediment_name)

        super(SedimentTotalMassConservation2DCallback, self).__init__(solver_obj, **kwargs)
        self.scalar_callback = mass

        # printing all detector output to log is probably not a useful default:
        kwargs.setdefault('append_to_log', False)

    def __call__(self):
        value = self.scalar_callback()
        if self.initial_value is None:
            self.initial_value = value

        rel_diff = (value - self.update_value)/self.initial_value

        return value, rel_diff

    def message_str(self, *args):
        line = '{0:s} rel. error {1:11.4e}'.format(self.name, args[1])
        return line

    def comp_sediment_total_mass_2d(self, sediment_name):
        """
        Computes total sediment mass in the 2D domain accounting for sediment leaving through
        boundary conditions and source terms
        :arg sediment_name :class:`string` of function name of interest
        """
        # read in necessary variables from solver object
        eta = self.solver_obj.fields.elev_2d
        vel = self.solver_obj.fields.uv_2d

        scalar_func = self.solver_obj.fields[sediment_name]

        # calculate total depth
        term = self.solver_obj.equations.sediment.terms['SedimentAdvectionTerm']
        H = term.depth.get_total_depth(eta)

        # normal
        n = FacetNormal(self.solver_obj.mesh2d)

        # calculate contribution from sediment leaving boundary
        boundary_terms = 0

        for bnd_marker in term.boundary_markers:
            ds_bnd = ds(int(bnd_marker), degree=term.quad_degree)
            a = -(n[0]*vel[0] + n[1]*vel[1])*H*scalar_func*ds_bnd

            boundary_terms += assemble(a)

        # record the initial scalar value in the domain
        if self.initial_value is None:
            self.initial_value = assemble(H*scalar_func*dx)
            self.update_value = assemble(H*scalar_func*dx)
        else:
            # alter the initial value to record sediment transitioning through source term
            # and boundary terms
            self.update_value += self.solver_obj.options.timestep * boundary_terms

            self.update_value += self.solver_obj.options.timestep * \
                assemble(self.solver_obj.sediment_model.get_erosion_term()*dx)
            self.update_value -= self.solver_obj.options.timestep * \
                assemble(self.solver_obj.sediment_model.get_deposition_coefficient()*scalar_func*dx)

        # find the current scalar value in the domain
        val = assemble(H*scalar_func*dx)

        # initialise to first non-zero value to avoid division by 0
        if self.initial_value < 10**(-14):
            if self.update_value > 10**(-14):
                self.initial_value = self.update_value
        return val

    def comp_sediment_total_mass_2d_cons(self, sediment_name):
        """
        Computes total sediment mass in the 2D domain for the conservative form of the sediment
        equation accounting for sediment leaving through boundary conditions and source terms
        :arg sediment_name :class:`string` of function name of interest
        """

        # read in necessary variables from solver object
        eta = self.solver_obj.fields.elev_2d
        vel = self.solver_obj.fields.uv_2d

        scalar_func = self.solver_obj.fields[sediment_name]

        # calculate total depth
        term = self.solver_obj.equations.sediment.terms['ConservativeSedimentAdvectionTerm']
        H = term.depth.get_total_depth(eta)

        # normal
        n = FacetNormal(self.solver_obj.mesh2d)

        # calculate contribution from sediment leaving boundary
        boundary_terms = 0

        for bnd_marker in term.boundary_markers:
            ds_bnd = ds(int(bnd_marker), degree=term.quad_degree)
            a = -(n[0]*vel[0] + n[1]*vel[1])*scalar_func*ds_bnd

            boundary_terms += assemble(a)

        # record the initial scalar value in the domain
        if self.initial_value is None:
            self.initial_value = assemble(scalar_func*dx)
            self.update_value = assemble(scalar_func*dx)
        else:
            # alter the initial value to record sediment transitioning through source term
            # and boundary terms
            self.update_value += self.solver_obj.options.timestep * boundary_terms

            self.update_value += self.solver_obj.options.timestep * \
                assemble(self.solver_obj.sediment_model.get_erosion_term()*dx)
            self.update_value -= self.solver_obj.options.timestep * \
                assemble(self.solver_obj.sediment_model.get_deposition_coefficient()*scalar_func/H*dx)

        # find the current scalar value in the domain
        val = assemble(scalar_func*dx)

        # initialise to first non-zero value to avoid division by 0
        if self.initial_value < 10**(-14):
            if self.update_value > 10**(-14):
                self.initial_value = self.update_value
        return val
