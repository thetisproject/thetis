from thetis import *
from modules import input_barrages, lagoon_operation


class LagoonCallback(DiagnosticCallback):
    """Callback that sets fluxes through lagoons"""

    def __init__(self, solver_obj, marked_areas, lagoon_input, name="lagoon", thetis_boundaries=None,
                 number_timesteps=5, time=0., status=None, **kwargs):
        """
        transient field averaging for use in the simulation

        :arg marked_area: list of marked areas
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """

        kwargs.setdefault('append_to_log', False)

        # Assign a dummy function
        unity_function = Function(solver_obj.fields['elev_2d'].function_space()).assign(1.0)

        self.time_operation = time
        self.solver_obj = solver_obj

        self.area = {"inner": assemble(unity_function * marked_areas["inner"]),
                     "outer": assemble(unity_function * marked_areas["outer"])}

        self.elevation_value = {"inner": assemble(solver_obj.fields['elev_2d'] * marked_areas["inner"]) / self.area["inner"],
                                "outer": assemble(solver_obj.fields['elev_2d'] * marked_areas["outer"]) / self.area["outer"]}

        self.elevation_array = {"inner": numpy.empty(number_timesteps),
                                "outer": numpy.empty(number_timesteps)}

        self.elevation_array["inner"][:], self.elevation_array["outer"][:] =\
            self.elevation_value["inner"], self.elevation_value["outer"]

        self.elevation_mean = {"inner": self.elevation_value["inner"], "outer": self.elevation_value["outer"]}
        self.marked_areas = marked_areas

        if status is None:
            self.status = input_barrages.initialise_barrage(1, time=time)[0]
        else:
            self.status = status

        self.control, self.parameters = lagoon_input[0][0], lagoon_input[1][0]
        self.output = numpy.zeros(12)
        self.boundaries = thetis_boundaries

        field_dims = [12]
        attrs = {'field_names': numpy.array(["operation_output"], dtype='S'),
                 'field_dims': field_dims}

        super(LagoonCallback, self).__init__(solver_obj, array_dim=sum(field_dims), attrs=attrs, **kwargs)
        self._name = name
        self._variable_names = ["operation_output"]

    @property
    def name(self):
        return self._name

    @property
    def variable_names(self):
        return self._variable_names

    def __call__(self):
        self.time_operation += self.solver_obj.options.timestep
        for i in ["inner", "outer"]:
            self.elevation_value[i] = assemble(self.solver_obj.fields['elev_2d'] * self.marked_areas[i]) / self.area[i]
            self.elevation_array[i] = numpy.append(numpy.delete(self.elevation_array[i], [0]), self.elevation_value[i])
            self.elevation_mean[i] = numpy.average(self.elevation_array[i])

        self.output = lagoon_operation.lagoon(self.time_operation, self.solver_obj.options.timestep,
                                              self.elevation_mean["inner"], self.elevation_mean["outer"],
                                              self.status, self.control, self.parameters, self.boundaries)
        return [self.output]

    def message_str(self, *args):
        super().message_str(*args)
