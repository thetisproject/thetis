# this script is only run as part of the CI suite
from thetis import *
from model_config import *

# Run 6 hour spin up run with hourly checkpoints
pwd = os.path.abspath(os.path.dirname(__file__))
mesh2d = read_mesh_from_checkpoint(f'{pwd}/north_sea_bathymetry.h5')
solver_obj, start_time, update_forcings = construct_solver(
    mesh2d,
    output_directory="outputs_spinup",
    spinup=False,  # we don't want to ramp up for this test
    start_date=datetime.datetime(2022, 1, 1, tzinfo=sim_tz),
    end_date=datetime.datetime(2022, 1, 1, 6, tzinfo=sim_tz),
    fields_to_export=[],
    fields_to_export_hdf5=["elev_2d", "uv_2d"],
    simulation_export_time=3600.0,
)
solver_obj.assign_initial_conditions()
update_forcings(0.0)
solver_obj.iterate(update_forcings=update_forcings)

# Pick up from third hour
solver_obj2, start_time, update_forcings = construct_solver(
    mesh2d,
    start_date=datetime.datetime(2022, 1, 1, 3, tzinfo=sim_tz),
    end_date=datetime.datetime(2022, 1, 1, 6, tzinfo=sim_tz),
)
solver_obj2.load_state(
    3, outputdir="outputs_spinup", t=0, iteration=0, i_export=0
)
update_forcings(0.0)
solver_obj2.iterate(update_forcings=update_forcings)

# should end up with same end state at hour 6
assert errornorm(solver_obj.fields.uv_2d, solver_obj2.fields.uv_2d) < 1e-6 * norm(solver_obj.fields.uv_2d)
assert errornorm(solver_obj.fields.elev_2d, solver_obj2.fields.elev_2d) < 1e-6 * norm(solver_obj.fields.elev_2d)

# finally, check that the committed checkpoint of the demo is still functional
solver_obj, start_time, update_forcings = construct_solver(
    mesh2d,
    start_date=datetime.datetime(2022, 1, 15, tzinfo=sim_tz),
    end_date=datetime.datetime(2022, 1, 15, 3, tzinfo=sim_tz),
)
solver_obj.load_state(14, outputdir=f"{pwd}/../../demos/outputs_spinup", t=0, iteration=0)
update_forcings(0.0)
solver_obj.iterate(update_forcings=update_forcings)

# just check things haven't gone completely haywire
# (they will be a bit random as we don't test with the correct forcing)
assert solver_obj.fields.elev_2d.dat.data[:].min() > -4
assert solver_obj.fields.elev_2d.dat.data[:].max() < 4
assert solver_obj.fields.uv_2d.dat.data[:].min() > -4
assert solver_obj.fields.uv_2d.dat.data[:].max() < 4
