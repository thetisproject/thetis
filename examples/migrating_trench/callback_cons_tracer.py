"""
Callback functions used to test if tracer is conserved in test cases with a 
changing morphology
"""

from firedrake import *
from thetis.callback import DiagnosticCallback


def comp_tracer_total_mass_2d(var, tracer_name):
    """
    Computes total tracer mass in the 2D domain accounting for tracer leaving through
    boundary conditions and source terms
    :arg var: class:`DiagnosticCallback`; callback object used to input values and
                                          record output values
    :arg tracer_name :class:`string` of function name of interest
    """

    # read in necessary variables from solver object
    eta = var.solver_obj.fields.elev_2d
    vel = var.solver_obj.fields.uv_2d

    scalar_func = var.solver_obj.fields[tracer_name]

    # calculate total depth
    term = term = var.solver_obj.eq_tracer.terms['HorizontalAdvectionTerm']
    H = term.get_total_depth(eta)

    # normal
    n = FacetNormal(var.solver_obj.mesh2d)

    # calculate contribution from tracer leaving boundary
    boundary_terms = 0

    for bnd_marker in term.boundary_markers:
        ds_bnd = ds(int(bnd_marker), degree=term.quad_degree)
        a = -(n[0]*vel[0] + n[1]*vel[1])*H*scalar_func*ds_bnd

        boundary_terms += assemble(a)

    # record the initial scalar value in the domain
    if var.initial_value is None:
        var.initial_value = assemble(H*scalar_func*dx)
        var.update_value = assemble(H*scalar_func*dx)
    else:
        # alter the initial value to record tracer transitioning through source term
        # and boundary terms
        var.update_value += var.solver_obj.options.timestep * boundary_terms

        if var.solver_obj.options.tracer_source_2d is not None:
            var.update_value += var.solver_obj.options.timestep * \
                assemble(var.solver_obj.options.tracer_source_2d*H*dx)

    # find the current scalar value in the domain
    val = assemble(H*scalar_func*dx)

    # initialise to first non-zero value to avoid division by 0
    if var.initial_value < 10**(-14):
        if var.update_value > 10**(-14):
            var.initial_value = var.update_value
    return val

def comp_tracer_total_mass_2d_cons(var, tracer_name):
    """
    Computes total tracer mass in the 2D domain for the conservative form of the tracer
    equation accounting for tracer leaving through boundary conditions and source terms
    :arg var: class:`DiagnosticCallback`; callback object used to input values and
                                          record output values
    :arg tracer_name :class:`string` of function name of interest
    """

    # read in necessary variables from solver object
    eta = var.solver_obj.fields.elev_2d
    vel = var.solver_obj.fields.uv_2d

    scalar_func = var.solver_obj.fields[tracer_name]
    
    # calculate total depth
    term = var.solver_obj.eq_tracer.terms['ConservativeHorizontalAdvectionTerm']
    H = term.get_total_depth(eta)

    # normal
    n = FacetNormal(var.solver_obj.mesh2d)

    # calculate contribution from tracer leaving boundary
    boundary_terms = 0

    for bnd_marker in term.boundary_markers:
        ds_bnd = ds(int(bnd_marker), degree=term.quad_degree)
        a = -(n[0]*vel[0] + n[1]*vel[1])*scalar_func*ds_bnd

        boundary_terms += assemble(a)

    # record the initial scalar value in the domain
    if var.initial_value is None:
        var.initial_value = assemble(scalar_func*dx)
        var.update_value = assemble(scalar_func*dx)
    else:
        # alter the initial value to record tracer transitioning through source term
        # and boundary terms
        var.update_value += var.solver_obj.options.timestep * boundary_terms

        if var.solver_obj.options.tracer_source_2d is not None:
            var.update_value += var.solver_obj.options.timestep * \
                assemble(var.solver_obj.options.tracer_source_2d*H*dx)

    # find the current scalar value in the domain
    val = assemble(scalar_func*dx)

    # initialise to first non-zero value to avoid division by 0
    if var.initial_value < 10**(-14):
        if var.update_value > 10**(-14):
            var.initial_value = var.update_value
    return val

class TracerTotalMassConservation2DCallback(DiagnosticCallback):
    """
    Checks conservation of depth-averaged tracer mass accounting for tracer leaving
    through boundary and source terms.

    Depth-averaged tracer mass is defined as the integral of 2D tracer
    multiplied by total depth, subtracting tracer leaving through boundary and through source term.
    """
    name = 'tracer mass'
    variable_names = ['integral', 'relative_difference']

    def __init__(self, tracer_name, solver_obj, **kwargs):
        """
        :arg tracer_name: Name of the tracer. Use canonical field names as in
            :class:`.FieldDict`.
        :arg solver_obj: Thetis solver object
        :arg kwargs: any additional keyword arguments, see
            :class:`.DiagnosticCallback`.
        """

        self.name = tracer_name + ' total mass'  # override name for given tracer

        def mass():
            if not hasattr(self, 'initial_value'):
                self.initial_value = None
            if solver_obj.options.use_tracer_conservative_form:
                return comp_tracer_total_mass_2d_cons(self, tracer_name)
            else:
                return comp_tracer_total_mass_2d(self, tracer_name)
        
        super(TracerTotalMassConservation2DCallback, self).__init__(solver_obj, **kwargs)
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
