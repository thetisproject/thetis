from thetis import *
import thetis.coordsys as coordsys
import thetis.forcing as forcing
import csv
import netCDF4
import os
import scipy.interpolate as si
import numpy

# Setup zones
sim_tz = timezone.pytz.utc
coord_system = coordsys.UTMCoordinateSystem(utm_zone=30)


def read_station_data():
    """
    Load tide gauge metadata from the CSV file
    `stations_elev.csv`.

    :return: a dictionary containing gauge locations
        in latitude-longitude coordinates and the
        corresponding region code used in the CMEMS
        database
    """
    with open("stations_elev.csv", "r") as csvfile:
        stations = {
            d["name"]: {
                "latlon": (float(d["latitude"]), float(d["longitude"])),
                "region": d["region"],
            }
            for d in csv.DictReader(csvfile, delimiter=",", skipinitialspace=True)
        }
    return stations


def interpolate_bathymetry(bathymetry_2d, dataset="etopo1", cap=10.0):
    """
    Interpolate a bathymetry field from some data set.

    :arg bathymetry_2d: :class:`Function` to store the data in
    :kwarg dataset: the data set name, which defines the NetCDF file name
    :kwarg cap: minimum value to cap the bathymetry at in the shallows
    """
    if cap <= 0.0:
        raise NotImplementedError(
            "Bathymetry cap must be positive because"
            " wetting and drying is not enabled in this example"
        )
    mesh = bathymetry_2d.function_space().mesh()

    # Read data from file
    with netCDF4.Dataset(f"{dataset}.nc", "r") as nc:
        interp = si.RectBivariateSpline(
            nc.variables["lat"][:],  # latitude
            nc.variables["lon"][:],  # longitude
            nc.variables["Band1"][:, :],  # elevation
        )

    # Interpolate at mesh vertices
    lonlat_func = coord_system.get_mesh_lonlat_function(mesh)
    lon, lat = lonlat_func.dat.data_ro.T
    bathymetry_2d.dat.data[:] = numpy.maximum(-interp(lat, lon), cap)


def construct_solver(spinup=False, store_station_time_series=True, **model_options):
    """
    Construct a :class:`FlowSolver2d` instance for inverse modelling
    in the North Sea.

    :kwarg spinup: is this a spin-up run, or a subsequent simulation?
    :kwarg store_station_time_series: should gauge measurements be
        stored to disk?
    :return: :class:`FlowSolver2d` instance, the start date for the
        simulation and a function for updating forcings
    """

    # Setup mesh and lonlat coords
    mesh2d = Mesh("north_sea.msh")
    lonlat = coord_system.get_mesh_lonlat_function(mesh2d)
    lon, lat = lonlat

    # Setup bathymetry
    P1_2d = get_functionspace(mesh2d, "CG", 1)
    bathymetry_2d = Function(P1_2d, name="Bathymetry")
    with CheckpointFile("north_sea_bathymetry.h5", "r") as f:
        m = f.load_mesh("firedrake_default")
        g = f.load_function(m, "Bathymetry")
        bathymetry_2d.assign(g)

    # Setup Manning friction
    manning_2d = Function(P1_2d, name="Manning coefficient")
    manning_2d.assign(3.0e-02)

    # Setup Coriolis forcing
    omega = 7.292e-05
    coriolis_2d = Function(P1_2d, name="Coriolis forcing")
    coriolis_2d.interpolate(2 * omega * sin(lat * pi / 180.0))

    # Setup temporal discretisation
    default_start_date = datetime.datetime(2022, 1, 1, tzinfo=sim_tz)
    default_end_date = datetime.datetime(2022, 1, 2, tzinfo=sim_tz)
    start_date = model_options.pop("start_date", default_start_date)
    end_date = model_options.pop("end_date", default_end_date)
    dt = 3600.0
    t_export = 3600.0
    t_end = (end_date - start_date).total_seconds()
    if os.getenv("THETIS_REGRESSION_TEST") is not None:
        t_end = 5 * t_export

    # Create solver
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    options = solver_obj.options
    options.element_family = "dg-dg"
    options.polynomial_degree = 1
    options.coriolis_frequency = coriolis_2d
    options.manning_drag_coefficient = manning_2d
    options.horizontal_velocity_scale = Constant(1.5)
    options.use_lax_friedrichs_velocity = True
    options.simulation_initial_date = start_date
    options.simulation_end_date = end_date
    options.simulation_export_time = t_export
    options.swe_timestepper_type = "DIRK22"
    options.swe_timestepper_options.use_semi_implicit_linearization = True
    options.timestep = dt
    options.fields_to_export = ["elev_2d", "uv_2d"]
    options.fields_to_export_hdf5 = []

    # The mesh is quite coarse, so it is reasonable to solve the underlying
    # linear systems by applying a full LU decomposition as a preconditioner
    options.swe_timestepper_options.solver_parameters = {
        "snes_type": "newtonls",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    options.update(model_options)
    print_output(f"Exporting to {options.output_directory}")
    solver_obj.create_equations()

    # Set up gauges
    if store_station_time_series:
        for name, data in read_station_data().items():
            sta_lat, sta_lon = data["latlon"]
            sta_x, sta_y = coord_system.to_xy(sta_lon, sta_lat)
            cb = TimeSeriesCallback2D(
                solver_obj,
                ["elev_2d"],
                sta_x,
                sta_y,
                name,
                append_to_log=False,
            )
            solver_obj.add_callback(cb)

    # Setup forcings
    data_dir = os.path.join(os.environ.get("DATA", "./data"), "tpxo")
    if not os.path.exists(data_dir):
        raise IOError(f"Data directory {data_dir} does not exist")
    forcing_constituents = ["Q1", "O1", "P1", "K1", "N2", "M2", "S2", "K2"]
    elev_tide_2d = Function(solver_obj.function_spaces.P1_2d, name="Tidal elevation")
    tbnd = forcing.TPXOTidalBoundaryForcing(
        elev_tide_2d,
        start_date,
        coord_system,
        data_dir=data_dir,
        constituents=forcing_constituents,
        boundary_ids=[100],
    )

    # Set time to zero for the tidal forcings
    tbnd.set_tidal_field(0.0)

    # Account for spinup
    bnd_time = Constant(0.0)
    if spinup:
        ramp_t = t_end
        elev_ramp = conditional(bnd_time < ramp_t, bnd_time / ramp_t, 1.0)
    else:
        elev_ramp = Constant(1.0)
    tide_elev_expr_2d = elev_ramp * elev_tide_2d

    # Setup boundary conditions for open ocean segments
    solver_obj.bnd_functions["shallow_water"] = {
        100: {"elev": tide_elev_expr_2d, "uv": Constant(as_vector([0, 0]))},
    }

    def update_forcings(t):
        bnd_time.assign(t)
        tbnd.set_tidal_field(t)

    return solver_obj, start_date, update_forcings
