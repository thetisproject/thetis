from thetis import *
import thetis.coordsys as coordsys
from okada import *
from sources import *
import csv
import netCDF4
import numpy
import os
import scipy.interpolate as si


# Setup UTM zone
coord_system = coordsys.UTMCoordinateSystem(utm_zone=54)

# Earthquake epicentre in longitude-latitude coordinates
epicentre = (142.369, 38.322)


def read_station_data():
    """
    Load tide gauge metadata from the CSV file
    `stations_elev.csv`.

    :return: a dictionary containing gauge locations
        in latitude-longitude coordinates and the
        corresponding start and end times for the
        time windows of interest.
    """
    pwd = os.path.abspath(os.path.dirname(__file__))
    with open(f"{pwd}/stations_elev.csv", "r") as csvfile:
        stations = {
            d["name"]: {
                "lat": float(d["latitude"]),
                "lon": float(d["longitude"]),
                "start": float(d["start"]),
                "end": float(d["end"]),
            }
            for d in csv.DictReader(csvfile, delimiter=",", skipinitialspace=True)
        }
    return stations


def get_source(mesh2d, source_model, initial_guess=None):
    """
    Construct a :class:`TsunamiSource` object for the chosen
    source model.

    Choices:

      - 'CGp':    Piece-wise polynomial (order p) and continuous.

      - 'DGp':    Piece-wise polynomial (order p) and discontinuous.

      - 'radial': Rectangular array of radial basis functions,
                  represented in P1 space.

      - 'box':    Rectangular array of piece-wise constant functions,
                  represented in P1 space.

      - 'okada':  Rectangular array of Okada functions, represented
                  in P1 space.

    :arg mesh2d: the underlying mesh
    :kwarg source_model: method for approximating the tsunami source
    :kwarg initial_guess: list of :class:`Function` s for setting the
        initial condition
    """
    if source_model == "box":
        return BoxArrayTsunamiSource(mesh2d, coord_system, initial_guess=initial_guess)
    elif source_model == "radial":
        return RadialArrayTsunamiSource(mesh2d, coord_system, initial_guess=initial_guess)
    elif source_model == "okada":
        return OkadaArrayTsunamiSource(mesh2d, coord_system, initial_guess=initial_guess)
    else:
        family = source_model[:2]
        if family not in ("CG", "DG"):
            raise ValueError(f"Element family {family} not supported for source inversion")
        degree = int(source_model[2:])
        element = FiniteElement(family, mesh2d.ufl_cell(), degree)
        return FiniteElementTsunamiSource(mesh2d, coord_system, element, initial_guess=initial_guess)


def interpolate_bathymetry(bathymetry_2d, dataset="etopo1", cap=30.0):
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
    pwd = os.path.abspath(os.path.dirname(__file__))
    with netCDF4.Dataset(f"{pwd}/{dataset}.nc", "r") as nc:
        interp = si.RectBivariateSpline(
            nc.variables["lat"][:],  # latitude
            nc.variables["lon"][:],  # longitude
            nc.variables["Band1"][:, :],  # elevation
        )

    # Interpolate at mesh vertices
    lonlat_func = coord_system.get_mesh_lonlat_function(mesh)
    lon, lat = lonlat_func.dat.data_ro.T
    bathymetry_2d.dat.data[:] = numpy.maximum(-interp(lat, lon, grid=False), cap)


def construct_solver(elev_init, store_station_time_series=True, **model_options):
    """
    Construct a *linear* shallow water equation solver for tsunami
    propagation modelling.
    """
    mesh2d = elev_init.function_space().mesh()

    t_end = 2 * 3600.0
    u_mag = Constant(5.0)
    t_export = 60.0
    dt = 60.0
    if os.getenv("THETIS_REGRESSION_TEST") is not None:
        t_end = 5 * t_export

    # Bathymetry
    P1_2d = get_functionspace(mesh2d, "CG", 1)
    bathymetry_2d = Function(P1_2d, name="Bathymetry")
    pwd = os.path.abspath(os.path.dirname(__file__))
    with CheckpointFile(f"{pwd}/japan_sea_bathymetry.h5", "r") as f:
        m = f.load_mesh("firedrake_default")
        g = f.load_function(m, "Bathymetry")
        bathymetry_2d.assign(g)
    bathymetry_2d.interpolate(bathymetry_2d - elev_init)

    # Create solver
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    options = solver_obj.options
    options.use_nonlinear_equations = False
    options.element_family = "dg-dg"
    options.simulation_export_time = t_export
    options.fields_to_export = ["elev_2d"]
    options.simulation_end_time = t_end
    options.horizontal_velocity_scale = u_mag
    options.swe_timestepper_type = "CrankNicolson"
    if not hasattr(options.swe_timestepper_options, "use_automatic_timestep"):
        options.timestep = dt
    options.swe_timestepper_options.solver_parameters = {
        "snes_type": "newtonls",
        "ksp_type": "gmres",
        "pc_type": "bjacobi",
        "sub_pc_type": "ilu",
    }
    options.update(model_options)
    solver_obj.create_equations()

    # Set up gauges
    if store_station_time_series:
        for name, data in read_station_data().items():
            sta_x, sta_y = coord_system.to_xy(data["lon"], data["lat"])
            cb = TimeSeriesCallback2D(
                solver_obj,
                ["elev_2d"],
                sta_x,
                sta_y,
                name,
                append_to_log=False,
                start_time=data["start"],
                end_time=data["end"],
            )
            solver_obj.add_callback(cb)

    # Set boundary conditions
    solver_obj.bnd_functions["shallow_water"] = {
        100: {"un": Constant(0.0), "elev": Constant(0.0)},
    }

    # Set initial condition
    solver_obj.assign_initial_conditions(elev=elev_init)
    return solver_obj
