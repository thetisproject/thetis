import firedrake as fd
from thetis.configuration import *
import thetis.utility as utility
import ufl
from okada import OkadaParameters, okada
import abc
import numpy


__all__ = ["FiniteElementTsunamiSource", "RadialArrayTsunamiSource",
           "BoxArrayTsunamiSource", "OkadaArrayTsunamiSource"]


class TsunamiSource(FrozenConfigurable):
    """
    Base class for generating tsunami source conditions.
    """

    __metaclass__ = abc.ABCMeta

    fault_length = PositiveFloat(560e3, help="Length of fault plane").tag(config=True)
    fault_width = PositiveFloat(240e3, help="Width of fault plane").tag(config=True)
    strike_angle = PositiveFloat(198, help="Angle of fault plane from North in degrees").tag(config=True)
    fault_centroid_lon = BoundedFloat(143.6, bounds=(-180, 180), help="Longitude of the fault centroid").tag(config=True)
    fault_centroid_lat = BoundedFloat(37.9, bounds=(-90, 90), help="Latitude of the fault centroid").tag(config=True)

    @utility.unfrozen
    def __init__(self, mesh2d, coord_system, element=None):
        """
        :arg mesh2d: mesh upon which to define the source
        :arg coord_system: :class:`CoordinateSystem` instance
        :kwarg element: :class:`FiniteElement` with which to define the source
        """
        self.mesh2d = mesh2d
        self.coord_system = coord_system
        if element is None:
            element = ufl.FiniteElement("Lagrange", mesh2d.ufl_cell(), 1)
        self.function_space = utility.get_functionspace(mesh2d, element.family(), element.degree())
        self._elev_init = fd.Function(self.function_space, name="Elevation")
        self.xy = ufl.SpatialCoordinate(mesh2d)
        fault_centroid = (self.fault_centroid_lon, self.fault_centroid_lat)
        self.xy0 = fd.Constant(ufl.as_vector(coord_system.to_xy(*fault_centroid)))
        self.generated = False

    @abc.abstractmethod
    def generate(self):
        """
        Run the source model to compute an initial ocean elevation field.
        """
        pass

    def rotation_matrix(self, backend=ufl):
        """
        Construct a roration matrix to transform from UTM coordinates
        to a coordinate system whose first component is parallel to the
        earthquake fault.

        :kwarg backend: specify whether to obtain a UFL expression or a
            NumPy array
        """
        if backend not in (ufl, numpy):
            raise ValueError(f"Backend {backend} not supported")
        theta = (270 - self.strike_angle) * numpy.pi / 180
        cs = backend.cos(theta)
        sn = backend.sin(theta)
        R = [[cs, -sn], [sn, cs]]
        return ufl.as_matrix(R) if backend == ufl else numpy.array(R)

    @property
    def elev_init(self):
        """
        The initial ocean elevation according to the source model.
        """
        if not self.generated:
            self.generate()
        return self._elev_init

    @property
    def controls(self):
        """
        The control parameter values used by the source model.
        """
        if not self.generated:
            self.generate()
        return self._controls

    @property
    @abc.abstractmethod
    def control_bounds(self):
        """
        A list of bounds for the control parameters associated with
        the source model.
        """
        pass


class FiniteElementTsunamiSource(TsunamiSource):
    """
    Generate a tsunami source condition in a finite element space.
    """

    mask_shape = Enum(
        ["rectangle", "circle"],
        default_value="rectangle", allow_none=True,
        help="Shape of mask function").tag(config=True)

    @utility.unfrozen
    def __init__(self, mesh2d, coord_system, element=None, initial_guess=None):
        """
        :arg mesh2d: mesh upon which to define the source
        :arg coord_system: :class:`CoordinateSystem` instance
        :kwarg element: :class:`FiniteElement` with which to define the source
        :kwarg initial_guess: :class:`Function` or filename to use for the
            initial guess
        """
        super().__init__(mesh2d, coord_system, element=element)
        self.initial_guess = initial_guess

    @utility.unfrozen
    def generate(self):
        mask = self.mask
        self._controls = [fd.Function(self.function_space, name="Elevation")]
        if self.initial_guess is None:
            eps = 1.0e-03
            self._controls[0].interpolate(eps * mask)
        elif isinstance(self.initial_guess, str):
            with CheckpointFile(self.initial_guess, "r") as chk:
                mesh_name = self.mesh2d.name
                if mesh_name is None:
                    mesh_name = "firedrake_default"
                mesh = chk.load_mesh(mesh_name)
                f = chk.load_function(mesh, self._controls[0].name())
                self._controls[0].assign(f)
        else:
            self._controls[0].assign(self.initial_guess)
        assert len(self._controls) == 1
        self._elev_init.project(mask * self._controls[0])
        self.generated = True

    @property
    def mask(self):
        """
        Construct a UFL expression for masking the initial ocean elevation.

        Note that the mask depends upon :attr:`strike_angle`,
        :attr:`fault_length` and :attr:`fault_width`.
        """
        if self.mask_shape is None:
            return fd.Constant(1.0)
        X = self.xy - self.xy0
        if self.mask_shape == "rectangle":
            R = self.rotation_matrix()
            x, y = ufl.dot(R, X)
            L, W = self.fault_length, self.fault_width
            cond = ufl.And(ufl.And(x > -L / 2, x < L / 2), ufl.And(y > -W / 2, y > W / 2))
            return ufl.conditional(cond, 1, 0)
        elif self.mask_shape == "circle":
            x, y = X
            r = self.fault_width
            return ufl.conditional(x ** 2 + y ** 2 < r ** 2, 1, 0)
        else:
            raise ValueError(f"Mask shape '{shape}' not supported")

    @property
    def control_bounds(self):
        return [-numpy.inf, numpy.inf]


class ArrayTsunamiSource(TsunamiSource):
    r"""
    Base class for generating tsunami source conditions which assume
    a subfault array structure.

    The array is assumed to be rectangular. Its length and width are
    described by :attr:`fault_length` and :attr:`fault_width` and its
    angle from North is described by :attr:`strike_angle`.

    The contribution from each subfault is approximated in
    :math:`\mathbb P1` space.
    """

    __metaclass__ = abc.ABCMeta

    num_subfaults_par = PositiveInteger(13, help="Number of subfaults parallel to fault").tag(config=True)
    num_subfaults_perp = PositiveInteger(10, help="Number of subfaults parpendicular to fault").tag(config=True)

    @property
    def num_controls(self):
        return len(self.controls)

    @property
    def num_subfaults(self):
        return self.num_subfaults_par * self.num_subfaults_perp

    @property
    def subfault_length(self):
        return self.fault_length / self.num_subfaults_par

    @property
    def subfault_width(self):
        return self.fault_width / self.num_subfaults_perp

    @utility.unfrozen
    def __init__(self, mesh2d, coord_system, element=None, initial_guess=None):
        """
        :arg mesh2d: mesh upon which to define the source
        :arg coord_system: :class:`CoordinateSystem` instance
        :kwarg element: :class:`FiniteElement` with which to define the source
        :kwarg initial_guess: array or filename to use for the initial guess
        """
        super().__init__(mesh2d, coord_system, element=element)
        if isinstance(initial_guess, str):
            initial_guess = numpy.load(initial_guess)[-1]
        self.initial_guess = initial_guess
        self.lonlat = coord_system.get_mesh_lonlat_function(mesh2d)
        self.R0 = utility.get_functionspace(mesh2d, "R", 0)
        self._controls = None

    @utility.unfrozen
    def create_subfaults(self):
        r"""
        Setup the arrays of coordinates for the subfault centroids and
        :class:`Function`\s for holding the corresponding contributions to the
        initial ocean elevation.
        """
        L, W = self.fault_length, self.fault_width
        l, w = self.subfault_length, self.subfault_width
        self.X = numpy.linspace(-0.5 * (L - l), 0.5 * (L - l), self.num_subfaults_par)
        self.Y = numpy.linspace(-0.5 * (W - w), 0.5 * (W - w), self.num_subfaults_perp)
        self.contributions = [fd.Function(self.function_space, name=f"func {i}") for i in range(self.num_subfaults)]

    @abc.abstractmethod
    def calculate_contribution(self, k, centroid):
        """
        Calculate the contribution of the source model on a single subfault.

        :arg k: the subfault index
        :arg centroid: UTM coordinates for the subfault centroid
        """
        pass

    def generate(self):
        if not hasattr(self, "contributions"):
            self.create_subfaults()
        xyij = fd.Constant(ufl.as_vector([0, 0]))
        R = self.rotation_matrix()
        for j, y in enumerate(self.Y):
            for i, x in enumerate(self.X):
                k = i + j * self.num_subfaults_par
                xyij.assign(numpy.array([x, y]))
                centroid = fd.Constant(self.xy0 + ufl.dot(R, xyij))
                self.calculate_contribution(k, centroid)


class LinearArrayTsunamiSource(ArrayTsunamiSource):
    """
    Base class for generating tsunami source conditions which take a
    linear combination of basis functions over a subfault array structure.
    """

    __metaclass__ = abc.ABCMeta

    def create_subfaults(self):
        super().create_subfaults()
        if self.initial_guess is None:
            self.initial_guess = 1.0e-03 * numpy.ones(self.num_subfaults)
        self._controls = [fd.Function(self.R0, name=f"ctrl {i}") for i in range(self.num_subfaults)]
        for ig, c in zip(self.initial_guess, self._controls):
            c.assign(ig)

    @abc.abstractmethod
    def basis_function(self, centroid):
        """
        Implementation of a basis function defined on a single subfault.

        :arg centroid: UTM coordinates for the subfault centroid
        """
        pass

    def calculate_contribution(self, k, centroid):
        R = self.rotation_matrix()
        coord = ufl.dot(ufl.transpose(R), self.xy - centroid)
        self.contributions[k].interpolate(self.basis_function(coord))

    def generate(self):
        super().generate()
        self._elev_init.interpolate(sum([c * phi for c, phi in zip(self._controls, self.contributions)]))
        self.generated = True

    @property
    def control_bounds(self):
        nc = self.num_controls
        ii = [-numpy.inf, numpy.inf]
        if nc == 1:
            return ii
        ones = numpy.ones(nc)
        return numpy.kron(ii, ones).reshape((2, nc))


class RadialArrayTsunamiSource(LinearArrayTsunamiSource):
    """
    Generate a tsunami source condition which takes a linear
    combination of radial basis functions over a subfault array
    structure.
    """

    def basis_function(self, centroid):
        extents = self.subfault_length, self.subfault_width
        return ufl.exp(-(sum([(x / w) ** 2 for x, w in zip(centroid, extents)])))


class BoxArrayTsunamiSource(LinearArrayTsunamiSource):
    """
    Generate a tsunami source condition which takes a linear
    combination of piece-wise constant basis functions over a
    subfault array structure.
    """

    def basis_function(self, centroid):
        x, y = centroid
        l, w = self.subfault_length, self.subfault_width
        cond = ufl.And(ufl.And(x > -l / 2, x < l / 2), ufl.And(y > -w / 2, y < w / 2))
        return ufl.conditional(cond, 1, 0)


class OkadaArrayTsunamiSource(ArrayTsunamiSource):
    """
    Generate a tsunami source condition which takes a linear
    combination of Okada functions over a subfault array structure.

    Okada parameters which are assumed fixed:
      * latitude: the latitude of the centroid of the subfault;
      * longitude: the longitude of the centroid of the subfault;
      * length: the length of the subfault (parallel to the fault);
      * width: the width of the subfault (perpendicular to the fault);
      * strike: the angle of the fault clockwise from North.

    Okada parameters which may vary:
      * depth: depth of the fault beneath the seabed;
      * dip: angle below the vertical in the fault plane;
      * slip: magnitude of the displacement vector;
      * rake: angle of the displacement vector.
    """

    fault_variables = []
    subfault_variables = ["depth", "dip", "slip", "rake"]

    defaults = {
        "depth": 20000.0,
        "dip": 10.0,
        "slip": 1.0e-03,
        "rake": 90.0,
    }

    bounds = {
        "depth": (0.0, numpy.inf),
        "dip": (0.0, 90.0),
        "slip": (0.0, numpy.inf),
        "rake": (-numpy.inf, numpy.inf),
    }

    @utility.unfrozen
    def create_subfaults(self):
        super().create_subfaults()
        sv = self.subfault_variables
        nb = self.num_subfaults
        if not set(sv).issubset({"depth", "dip", "slip", "rake"}):
            raise ValueError(f"subfault_variables should be a subset of {'depth', 'dip', 'slip', 'rake'}")
        self.control_dict = {c: [fd.Function(self.R0, name=f"{c} {i}") for i in range(nb)] for c in sv}
        self._controls = sum(list(self.control_dict.values()), start=[])
        if self.initial_guess is None:
            self.initial_guess = sum([[self.defaults[c]] * nb for c in sv], start=[])
        for ig, c in zip(self.initial_guess, self._controls):
            c.assign(ig)

    def calculate_contribution(self, k, centroid):
        sv = self.subfault_variables
        P = self.defaults.copy()
        P.update({c: self.control_dict[c][k] for c in sv})
        P["strike"] = self.strike_angle
        P["length"] = self.subfault_length
        P["width"] = self.subfault_width
        P["lon"], P["lat"] = self.coord_system.to_lonlat(*centroid)
        subfault = OkadaParameters(P)
        self.contributions[k].interpolate(okada(subfault, *self.lonlat))

    def generate(self):
        super().generate()
        for func in self.contributions:
            self._elev_init += func
        self.generated = True

    @property
    def control_bounds(self):
        nc = self.num_controls
        sv = self.subfault_variables
        if nc == 1:
            return self.bounds[list(sv)[0]]
        ones = numpy.ones(nc // len(sv))
        bounds = numpy.transpose([self.bounds[c] for c in sv]).flatten()
        return numpy.kron(bounds, ones).reshape((2, nc))
