"""
I/O utilities for loading turbine and farm configurations from YAML/JSON
into Thetis DiscreteTidalTurbineFarmOptions objects.
"""
from firedrake import *
from .options import DiscreteTidalTurbineFarmOptions
import json
import yaml


def load_turbine(path, mesh2d, include_support=True):
    """Load a single turbine definition into a DiscreteTidalTurbineFarmOptions."""
    with open(path) as f:
        if path.endswith(".json"):
            data = json.load(f)
        else:
            data = yaml.safe_load(f)

    opts = DiscreteTidalTurbineFarmOptions()
    opts.turbine_type = data.get("turbine_thrust_def", "constant")

    if opts.turbine_type == "table":
        opts.turbine_options.thrust_speeds = data["curves"]["speeds"]
        opts.turbine_options.thrust_coefficients = data["curves"]["thrust"]
        opts.turbine_options.power_coefficients = data["curves"]["power"]
    else:
        opts.turbine_options.thrust_coefficient = data["defaults"]["thrust_coefficient"]
        opts.turbine_options.power_coefficient = data["defaults"]["power_coefficient"]

    if include_support and "support_structure" in data:
        opts.turbine_options.C_support = data["support_structure"]["C_support"]
        opts.turbine_options.A_support = data["support_structure"]["A_support"]

    opts.turbine_options.diameter = data["diameter"]
    opts.upwind_correction = data.get("upwind_correction", True)

    opts.turbine_density = Function(FunctionSpace(mesh2d, "CG", 1),
                                    name=f"turbine_density_{data['name']}").assign(0.0)
    return opts
