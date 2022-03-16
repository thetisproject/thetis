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


def rect_gaussian(coords, extents):
    """
    UFL expression for a rectangular Gaussian.

    :args coords: spatial coordinates
    :args extents: extents in the coordinate directions
    """
    assert len(coords) == len(extents)
    return exp(-(sum([(x / w) ** 2 for (x, w) in zip(coords, extents)])))


def box(coords, extents):
    """
    UFL expression for a rectangular indicator.

    :args coords: spatial coordinates
    :args extents: extents in the coordinate directions
    """
    x, w = coords[0], extents[0]
    expr = And(x > -w / 2, x < w / 2)
    for x, w in zip(list(coords)[1:], list(extents)[1:]):
        expr = And(expr, And(x > -w / 2, x < w / 2))
    return conditional(expr, 1, 0)


def basis_function(source_model, *args):
    """
    Get a UFL expression for a given source model.

    :arg source_model: choose from 'radial', 'box'
    :args coords: spatial coordinates
    :args extents: extents in the coordinate directions
    """
    try:
        return {"radial": rect_gaussian, "box": box}[source_model](*args)
    except KeyError:
        raise ValueError(f"Source model '{source_model}' not supported.")


def initial_condition(mesh2d, source_model="CG1", initial_guess=None, okada_parameters=None, **mask_kw):
    """
    Construct an initial condition :class:`Function` for the chosen
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

    For the 'radial', 'box' and 'okada' source models, the rectangular
    array can be viewed as subfaults which together comprise a larger
    earthquake fault. The 'radial' and 'box' models have just one
    degree of freedom per subfault - the scalar coefficient which
    multiplies the basis function, as part of a linear combination.

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
    :kwarg source_model: method for approximating the tsunami source
    :kwarg initial_guess: list of :class:`Function` s for setting the
        initial condition
    :kwarg okada_parameters: specifies which control parameters are to
        be optimised for in the case of the Okada model
    :kwarg mask_kw: kwargs for the mask function
    """
    xy = SpatialCoordinate(mesh2d)
    eps = 1.0e-03  # Small non-zero default

    # Define subfault array coordinate system
    x0, y0 = array_centre
    xy0 = Constant(as_vector([x0, y0]))
    X = xy - xy0
    theta = strike_angle
    R = as_matrix([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

    # Create the control Functions and the elevation Function
    if source_model[:2] in ("CG", "DG"):
        coord = dot(R, X)
        family = source_model[:2]
        degree = int(source_model[2:])
        Pk = get_functionspace(mesh2d, family, degree)
        elev_init = Function(Pk, name="Elevation")
        mask2d = mask(mesh2d, **mask_kw)

        # Setup controls
        if initial_guess is None:
            controls = [Function(Pk, name="Elevation")]
            controls[0].interpolate(eps * mask2d)
        else:
            controls = initial_guess
            assert len(controls) == 1
            elev_init.assign(controls[0])

        # Apply mask
        elev_init.project(mask2d * controls[0])
    else:
        nb = nx * ny
        P1 = get_functionspace(mesh2d, "CG", 1)
        elev_init = Function(P1, name="Elevation")
        basis_functions = [Function(P1, name=f"basis function {i}") for i in range(nb)]

        # Get lonlat Function
        lonlat = coord_system.get_mesh_lonlat_function(mesh)

        # Setup controls
        R0 = get_functionspace(mesh2d, "R", 0)
        if source_model == "okada":
            variables = ["depth", "dip", "slip", "rake"]
            active_controls = okada_parameters or variables
            controls_dict = {
                control: [Function(R0, name=f"{control} {i}") for i in range(nb)]
                for control in active_controls
            }
            controls = sum(list(controls_dict.values()), start=[])
            if initial_guess is None:
                for j, control in enumerate(active_controls):
                    for i in range(nb):
                        controls[j * nb + i].assign(okada_defaults[control])
            else:
                assert len(initial_guess) == len(controls)
                for cin, cout in zip(initial_guess, controls):
                    cout.assign(cin)
        else:
            controls = [Function(R0, name=f"control {i}") for i in range(nb)]
            if initial_guess is None:
                for c in controls:
                    c.assign(eps)
            else:
                assert len(initial_guess) == len(controls)
                for cin, cout in zip(initial_guess, controls):
                    cout.assign(cin)

        # Setup the subfault array
        l = Dx / nx  # Length of one subfault
        w = Dy / ny  # Width of one subfault
        X = numpy.linspace(-0.5 * Dx, 0.5 * Dx, nx)
        Y = numpy.linspace(-0.5 * Dy, 0.5 * Dy, ny)

        # Interpolate basis functions for each subfault
        xyij = Constant(as_vector([0, 0]))
        for j, y in enumerate(Y):
            for i, x in enumerate(X):
                phi = basis_functions[i + j * nx]
                xyij.assign(numpy.array([x, y]))
                centre = Constant(xy0 + dot(R, xyij))
                if source_model == "okada":
                    P = {key: val[i + j * nx] for key, val in controls_dict.items()}
                    P["lon"], P["lat"] = coord_system.to_lonlat(*centre)
                    P["length"], P["width"] = l, w
                    P["strike"] = 198.0
                    for c in variables:
                        if c not in active_controls:
                            P[c] = okada_defaults[c]
                    subfault = OkadaParameters(P)
                    phi.interpolate(okada(subfault, *lonlat))
                else:
                    coord = dot(transpose(R), xy - centre)
                    phi.interpolate(basis_function(source_model, coord, (l, w)))

        # Sum the contributions from all subfaults
        if source_model == "okada":
            for bf in basis_functions:
                elev_init += bf
        else:
            elev_init.interpolate(
                sum(c * bf for (c, bf) in zip(controls, basis_functions))
            )
    return elev_init, controls


def mask(mesh2d, shape="rectangle"):
    """
    Mask to apply to the initial surface so that is constrained
    to only be non-zero within a particular region.

    :arg mesh2d: mesh defining the coordinate field
    :kwarg shape: choose from None, 'rectangle' and 'circle'
    """
    if shape is None:
        return Constant(1.0)
    xy = SpatialCoordinate(mesh2d)
    x0, y0 = 700.0e03, 4200.0e03
    xy0 = Constant(as_vector([x0, y0]))
    X = xy - xy0

    if shape == "rectangle":
        l, w = 560.0e03, 240.0e03  # Width and length of rectangular region
        theta = strike_angle
        R = as_matrix([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
        xx, yy = dot(R, X)
        cond = And(And(xx > -l / 2, xx < l / 2), And(yy > -w / 2, yy < w / 2))
        return conditional(cond, 1, 0)
    elif shape == "circle":
        r = 200.0e03
        xx, yy = X
        return conditional(xx**2 + yy**2 < r**2, 1, 0)
    else:
        raise ValueError(f"Mask shape '{shape}' not supported.")


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
    zero = Constant(0.0)
    solver_obj.bnd_functions["shallow_water"] = {
        100: {"un": zero, "elev": zero},
        200: {"un": zero},
        300: {"un": zero},
    }

    # Set initial condition
    solver_obj.assign_initial_conditions(elev=elev_init)
    return solver_obj
