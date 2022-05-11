from thetis import *
import time as time_mod
from model_config import *


solver_obj, start_time, update_forcings = construct_solver(
    start_date=datetime.datetime(2022, 1, 15, tzinfo=sim_tz),
    end_date=datetime.datetime(2022, 1, 18, tzinfo=sim_tz),
)
solver_obj.load_state(
    14, outputdir="outputs_spinup", t=0, iteration=0, i_export=0
)
update_forcings(0.0)
tic = time_mod.perf_counter()
solver_obj.iterate(update_forcings=update_forcings)
toc = time_mod.perf_counter()
print_output(f"Total duration: {toc-tic:.2f} seconds")
