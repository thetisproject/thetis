from thetis import *
import netCDF4
import numpy
import os
import pyproj
import scipy.interpolate as si


__all__ = ["stations", "rect_gaussian", "initial_condition", "construct_solver"]


stations = {
    "801": {"latlon": (38.2325, 141.6856), "interval": (0.0, 3600.0)},
    "802": {"latlon": (39.2586, 142.0969), "interval": (0.0, 3600.0)},
    "803": {"latlon": (38.8578, 141.8944), "interval": (0.0, 3600.0)},
    "804": {"latlon": (39.6272, 142.1867), "interval": (0.0, 3600.0)},
    "806": {"latlon": (36.9714, 141.1856), "interval": (0.0, 3600.0)},
    "807": {"latlon": (40.1167, 142.0667), "interval": (0.0, 3600.0)},
    "P02": {"latlon": (38.5002, 142.5016), "interval": (0.0, 3600.0)},
    "P06": {"latlon": (38.6340, 142.5838), "interval": (0.0, 3600.0)},
    "KPG1": {"latlon": (41.7040, 144.4375), "interval": (0.0, 3600.0)},
    "KPG2": {"latlon": (42.2365, 144.8485), "interval": (0.0, 3600.0)},
    "MPG1": {"latlon": (32.3907, 134.4753), "interval": (4800.0, 7200.0)},
    "MPG2": {"latlon": (32.6431, 134.3712), "interval": (4800.0, 7200.0)},
    "21401": {"latlon": (42.617, 152.583), "interval": (3000.0, 7200.0)},
    "21413": {"latlon": (30.533, 152.132), "interval": (3000.0, 7200.0)},
    "21418": {"latlon": (38.735, 148.655), "interval": (0.0, 3600.0)},
    "21419": {"latlon": (44.435, 155.717), "interval": (3000.0, 7200.0)},
}

UTM_ZONE54 = pyproj.Proj(proj="utm", zone=54, datum="WGS84", units="m", errcheck=True)
LL = pyproj.Proj(proj="latlong", errcheck=True)


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
        return {"radial": rect_gaussian, "box": box}[source_model](
            *args
        )
    except KeyError:
        raise ValueError(f"Source model '{source_model}' not supported.")


def initial_condition(mesh2d, source_model="CG1", initial_guess=None, **mask_kw):
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

    :arg mesh2d: the underlying mesh
    :kwarg source_model: method for approximating the tsunami source
    :kwarg initial_guess: list of :class:`Function` s for setting the
        initial condition
    :kwarg mask_kw: kwargs for the mask function
    """
    xy = SpatialCoordinate(mesh2d)
    x0, y0 = 700.0e03, 4200.0e03
    xy0 = Constant(as_vector([x0, y0]))
    X = xy - xy0
    theta = 7 * pi / 12
    R = as_matrix([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

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
            eps = 1.0e-03  # Small non-zero default
            controls[0].interpolate(eps * mask2d)
        else:
            controls = initial_guess
            assert len(controls) == 1
            elev_init.assign(controls[0])

        # Apply mask
        elev_init.project(mask2d * controls[0])
    else:
        nx, ny = 13, 10
        nb = nx * ny
        P1 = get_functionspace(mesh2d, "CG", 1)
        elev_init = Function(P1, name="Elevation")
        basis_functions = [Function(P1, name=f"basis function {i}") for i in range(nb)]

        # Setup controls
        R0 = get_functionspace(mesh2d, "R", 0)
        controls = [Function(R0, name=f"control {i}") for i in range(nb)]
        if initial_guess is None:
            eps = 1.0e-03  # Small non-zero default
            for c in controls:
                c.assign(eps)
        else:
            assert len(initial_guess) == len(controls)
            for cin, cout in zip(initial_guess, controls):
                cout.assign(cin)

        # Interpolate basis functions
        xyij = Constant(as_vector([0, 0]))
        dx, dy = 560.0e03, 240.0e03
        extent = w, l = 48.0e03, 24.0e03
        X = numpy.linspace(-0.5 * dx, 0.5 * dx, nx)
        Y = numpy.linspace(-0.5 * dy, 0.5 * dy, ny)
        for j, y in enumerate(Y):
            for i, x in enumerate(X):
                phi = basis_functions[i + j * nx]
                xyij.assign(numpy.array([x, y]))
                coord = dot(R, xy - (xy0 + dot(transpose(R), xyij)))
                phi.interpolate(basis_function(source_model, coord, extent))

        # Interpolate initial condition
        elev_init.interpolate(sum(c * bf for (c, bf) in zip(controls, basis_functions)))
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
    x, y = SpatialCoordinate(mesh2d)
    x0, y0 = 700.0e03, 4200.0e03  # Earthquake epicentre in UTM coordinates
    X = as_vector([x - x0, y - y0])
    if shape == "rectangle":
        w, l = 560.0e03, 240.0e03  # Width and length of rectangular region
        theta = 7 * pi / 12  # Angle of rotation
        R = as_matrix([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        xx, yy = dot(R, X)
        cond = And(And(xx > -w / 2, xx < w / 2), And(yy > -l / 2, yy < l / 2))
        return conditional(cond, 1, 0)
    elif shape == "circle":
        r = 200.0e03
        xx, yy = X
        return conditional(xx**2 + yy**2 < r**2, 1, 0)
    else:
        raise ValueError(f"Mask shape '{shape}' not supported.")


def interpolate_bathymetry(bathymetry_2d, cap=30.0):
    """
    Interpolate a bathymetry field from the ETOPO1 data set.

    :arg bathymetry_2d: :class:`Function` to store the data in
    :kwarg cap: minimum value to cap the bathymetry at in the shallows
    """
    if cap <= 0.0:
        raise NotImplementedError("Wetting and drying is not enabled in this example")
    mesh = bathymetry_2d.function_space().mesh()

    # Read data from file
    pwd = os.path.abspath(os.path.dirname(__file__))
    with netCDF4.Dataset(f"{pwd}/etopo1.nc", "r") as nc:
        lon = nc.variables["lon"][:]
        lat = nc.variables["lat"][:]
        elev = nc.variables["Band1"][:, :]

    # Interpolate at mesh vertices
    trans = pyproj.Transformer.from_crs(UTM_ZONE54.srs, LL.srs)
    interp = si.RectBivariateSpline(lat, lon, elev)
    for i, xy in enumerate(mesh.coordinates.dat.data_ro):
        lon, lat = trans.transform(*xy)
        bathymetry_2d.dat.data[i] -= min(interp(lat, lon), -30)


def construct_solver(store_station_time_series=True, **model_options):
    """
    Construct a *linear* shallow water equation solver for tsunami
    propagation modelling.
    """
    mesh2d = Mesh(f"{os.path.abspath(os.path.dirname(__file__))}/japan_sea.msh")

    t_end = 2 * 3600.0
    u_mag = Constant(5.0)
    t_export = 60.0
    dt = 60.0
    if os.getenv("THETIS_REGRESSION_TEST") is not None:
        t_end = 5 * t_export

    # Bathymetry
    P1_2d = get_functionspace(mesh2d, "CG", 1)
    bathymetry_2d = Function(P1_2d, name="Bathymetry")
    interpolate_bathymetry(bathymetry_2d)

    # Create solver
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    options = solver_obj.options
    options.use_nonlinear_equations = False
    options.element_family = "dg-dg"
    options.simulation_export_time = t_export
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
    trans = pyproj.Transformer.from_crs(LL.srs, UTM_ZONE54.srs)
    if store_station_time_series:
        for name, data in stations.items():
            sta_lat, sta_lon = data["latlon"]
            tstart, tend = data["interval"]
            sta_x, sta_y = trans.transform(sta_lon, sta_lat)
            cb = TimeSeriesCallback2D(
                solver_obj,
                ["elev_2d"],
                sta_x,
                sta_y,
                name,
                append_to_log=False,
                start_time=tstart,
                end_time=tend,
            )
            solver_obj.add_callback(cb)

    # Set boundary conditions
    zero = Constant(0.0)
    solver_obj.bnd_functions["shallow_water"] = {
        100: {"un": zero, "elev": zero},
        200: {"un": zero},
        300: {"un": zero},
    }
    return solver_obj
