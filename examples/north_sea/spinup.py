from thetis import *
import time as time_mod
from model_config import *


# Setup solver
solver_obj, start_time, update_forcings = construct_solver(
    output_directory="outputs_spinup",
    spinup=True,
    start_date=datetime.datetime(2022, 1, 1, tzinfo=sim_tz),
    end_date=datetime.datetime(2022, 1, 15, tzinfo=sim_tz),
    fields_to_export=[],
    fields_to_export_hdf5=["elev_2d", "uv_2d"],
    simulation_export_time=24 * 3600.0,
)
output_dir = solver_obj.options.output_directory
mesh2d = solver_obj.mesh2d
solver_obj.assign_initial_conditions()
update_forcings(0.0)

# Time integrate
tic = time_mod.perf_counter()
solver_obj.iterate(update_forcings=update_forcings)
toc = time_mod.perf_counter()
print_output(f"Total duration: {toc-tic:.2f} seconds")
