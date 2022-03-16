from thetis import *
import thetis.coordsys as coordsys
from okada import *
import csv
import netCDF4
import numpy
import os
import scipy.interpolate as si


# Setup UTM zone
coord_system = coordsys.UTMCoordinateSystem(utm_zone=54)

# Default Okada parameter values to set in the absence of an initial guess
# NOTE: Angles are in degrees, distances are in metres
okada_defaults = {
    "depth": 20000.0,
    "dip": 10.0,
    "slip": 1.0e-03,  # arbitrary small value
    "rake": 90.0,
}

# Bounds on each Okada parameter to pass to L-BFGS-B
okada_bounds = {
    "depth": (0.0, numpy.inf),
    "dip": (0.0, 90.0),
    "slip": (0.0, numpy.inf),
    "rake": (-numpy.inf, numpy.inf),
}

# Subfault array parameters
epicentre = (142.369, 38.322)  # Earthquake epicentre in longitude-latitude coordinates
array_centre = (730.0e03, 4200.0e03)  # Centre of subfault array
strike_angle = 75 * numpy.pi / 180  # Angle of subfault array from North
nx, ny = 13, 10  # Number of subfaults parallel/perpendicular to the fault
Dx = 560.0e03  # Length of fault plane
Dy = 240.0e03  # Width of fault plane


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


def initial_condition(mesh2d, initial_guess=None, okada_parameters=None):
    """
    Construct an initial condition :class:`Function` using the
    Okada model.

    The 'okada' model, on the other hand, involves a nonlinear
    combination of nine input parameters on each subfault. Some of these
    parameters are fixed by the choice of subfault array. These are the
    focal length, width and strike, as well as the latitude and longitude
    of a point on the subfault (chosen here to be its centroid). Here,
    strike is the angle of the subfault from North. This leaves
    four parameters which can be modified during the inversion:

      * 'depth': depth of the fault below the sea bed;
      * 'dip': angle of the fault plane in the vertical;
      * 'slip': magnitude of the displacement in the fault plane;
      * 'rake': angle of the displacement in the fault plane.

    :arg mesh2d: the underlying mesh
    :kwarg initial_guess: list of :class:`Function` s for setting the
        initial condition
    :kwarg okada_parameters: specifies which control parameters are to
        be optimised for in the case of the Okada model
    """
    xy = SpatialCoordinate(mesh2d)
    lonlat = coord_system.get_mesh_lonlat_function(mesh2d)

    # Define subfault array coordinate system
    x0, y0 = array_centre
    xy0 = Constant(as_vector([x0, y0]))
    theta = strike_angle
    R = as_matrix([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

    # Setup the subfault array
    l = Dx / nx  # length of one subfault
    w = Dy / ny  # width of one subfault
    X = numpy.linspace(-0.5 * Dx, 0.5 * Dx, nx)
    Y = numpy.linspace(-0.5 * Dy, 0.5 * Dy, ny)

    # Define basis functions on the array
    nb = nx * ny
    P1 = get_functionspace(mesh2d, "CG", 1)
    elev_init = Function(P1, name="Elevation")
    basis_functions = [Function(P1, name=f"basis function {i}") for i in range(nb)]

    # Setup a Function for each control parameter, on each subfault
    R0 = get_functionspace(mesh2d, "R", 0)
    variables = ["depth", "dip", "slip", "rake"]  # Default control parameters
    active_controls = okada_parameters or variables
    controls_dict = {
        control: [Function(R0, name=f"{control} {i}") for i in range(nb)]
        for control in active_controls
    }
    controls = sum(list(controls_dict.values()), start=[])

    # Assign initial guess
    if initial_guess is None:
        for j, control in enumerate(active_controls):
            for i in range(nb):
                controls[j * nb + i].assign(okada_defaults[control])
    else:
        assert len(initial_guess) == len(controls)
        for cin, cout in zip(initial_guess, controls):
            cout.assign(cin)

    # Loop over each subfault
    xyij = Constant(as_vector([0, 0]))  # NOTE: This exists to avoid re-compiling code
    for j, y in enumerate(Y):
        for i, x in enumerate(X):
            k = i + j * nx
            phi = basis_functions[k]
            xyij.assign(numpy.array([x, y]))

            # Get coordinates of the centre of subfault k
            centre = Constant(xy0 + dot(R, xyij))

            # Create a dictionary of Okada parameters for subfault k
            P = {key: val[k] for key, val in controls_dict.items()}
            P["lon"], P["lat"] = coord_system.to_lonlat(*centre)
            P["length"], P["width"] = l, w
            P["strike"] = 198.0
            for c in variables:
                if c not in active_controls:
                    P[c] = okada_defaults[c]
            subfault = OkadaParameters(P)

            # Run the Okada model and interpolate the values into the basis function
            # for subfault k
            phi.interpolate(okada(subfault, *lonlat))

    # Sum the contributions from all subfaults
    for bf in basis_functions:
        elev_init += bf

    return elev_init, controls


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
    bathymetry_2d.dat.data[:] = numpy.maximum(-interp(lat, lon), cap)


def construct_solver(elev_init, store_station_time_series=True, **model_options):
    """
    Construct a *linear* shallow water equation solver for tsunami
    propagation modelling.
    """
    mesh2d = elev_init.function_space().mesh()

    # Timestepping parameters
    t_end = 2 * 3600.0
    u_mag = Constant(5.0)
    t_export = 60.0
    dt = 60.0
    if os.getenv("THETIS_REGRESSION_TEST") is not None:
        t_end = 5 * t_export

    # Set up bathymetry
    P1_2d = get_functionspace(mesh2d, "CG", 1)
    bathymetry_2d = Function(P1_2d, name="Bathymetry")
    pwd = os.path.abspath(os.path.dirname(__file__))
    with DumbCheckpoint(f"{pwd}/japan_sea_bathymetry", mode=FILE_READ) as h5:
        h5.load(bathymetry_2d)
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

    # Set [non]linear solver parameters and preconditioner
    options.swe_timestepper_options.solver_parameters = {
        "snes_type": "newtonls",  # nonlinear solver: Newton with line search
        "ksp_type": "gmres",  # linear solver: GMRES
        "pc_type": "bjacobi",  # preconditioner: block Jacobi
        "sub_pc_type": "ilu",  # preconditioner on the blocks: incomplete LU decomposition
    }

    # Set any other parameters that were passed to the function
    options.update(model_options)

    # Create the shallow water equation object
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
    zero = Constant(0.0)
    solver_obj.bnd_functions["shallow_water"] = {
        100: {"un": zero, "elev": zero},  # Weakly reflective open ocean boundary
        200: {"un": zero},  # No-slip condition along the coast of Miyagi Prefecture
        300: {"un": zero},  # No-slip condition along the rest of the coast
    }

    # Set initial condition
    solver_obj.assign_initial_conditions(elev=elev_init)
    return solver_obj
