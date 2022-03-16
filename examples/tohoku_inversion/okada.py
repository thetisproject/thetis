from thetis.utility import AttrDict
from ufl import cos, sin, sqrt, ln, atan, pi


__all__ = ["OkadaParameters", "okada"]


deg2rad = lambda deg: pi * deg / 180.0
rad2deg = lambda rad: 180.0 * rad / pi
deg2cos = lambda deg: cos(deg2rad(deg))
deg2sin = lambda deg: sin(deg2rad(deg))
earth_radius = 6367.5e3
lat2meter = lambda lat: earth_radius * deg2rad(lat)
meter2lat = lambda m: rad2deg(m / earth_radius)


class OkadaParameters(AttrDict):
    """
    Attribute dictionary class for holding parameters
    associated with the Okada earthquake source model.
    """

    _expected = {
        "depth",  # focal depth
        "length",  # fault length
        "width",  # fault width
        "strike",  # strike direction
        "dip",  # dip angle
        "rake",  # rake angle
        "slip",  # strike angle
        "lat",  # epicentre latitude
        "lon",  # epicentre longitude
    }

    def __init__(self, params):
        """
        :arg params: a dictionary containing values for all nine
            Okada parameters
        """
        keys = set(params.keys())
        if not self._expected.issubset(keys):
            diff = self._expected.difference(keys)
            raise AttributeError(f"Missing Okada parameters {diff}")
        if not keys.issubset(self._expected):
            diff = keys.difference(self._expected)
            raise AttributeError(f"Unexpected Okada parameters {diff}")
        super().__init__(params)

    def __repr__(self):
        """
        Convert any :class:`Constant` instances to `float` for printing.
        """
        return str({key: float(value) for key, value in self.items()})


def okada(parameters, x, y):
    """
    Run the Okada source model.

    :arg parameters: :class:`OkadaParameters` instance
    :arg x, y: coordinates at which to evaluate the model

    Yoshimitsu Okada, "Surface deformation due to shear
      and tensile faults in a half-space", Bulletin of the
      Seismological Society of America, Vol. 75, No. 4,
      pp.1135--1154, (1985).
    """
    P = OkadaParameters(parameters)
    half_length = 0.5 * P.length
    poisson = 0.25  # Poisson ratio

    # Get centres
    x_bot = P.lon - 0.5 * meter2lat(-P.width * deg2cos(P.dip) * deg2cos(P.strike) / deg2cos(P.lat))
    y_bot = P.lat - 0.5 * meter2lat(P.width * deg2cos(P.dip) * deg2sin(P.strike))
    z_bot = P.depth + 0.5 * P.width * deg2sin(P.dip)

    # Convert from degrees to meters
    xx = lat2meter(deg2cos(y) * (x - x_bot))
    yy = lat2meter(y - y_bot)

    # Convert to distance in strike-dip plane
    sn = deg2sin(P.strike)
    cs = deg2cos(P.strike)
    x1 = xx * sn + yy * cs
    x2 = xx * cs - yy * sn
    x2 = -x2
    sn = deg2sin(P.dip)
    cs = deg2cos(P.dip)
    p = x2 * cs + z_bot * sn
    q = x2 * sn - z_bot * cs

    def strike_slip(y1, y2):
        d_bar = y2 * sn - q * cs
        r = sqrt(y1**2 + y2**2 + q**2)
        a4 = 2 * poisson * (ln(r + d_bar) - sn * ln(r + y2)) / cs
        return -0.5 * (d_bar * q / (r * (r + y2)) + q * sn / (r + y2) + a4 * sn) / pi

    f1 = strike_slip(x1 + half_length, p)
    f2 = strike_slip(x1 + half_length, p - P.width)
    f3 = strike_slip(x1 - half_length, p)
    f4 = strike_slip(x1 - half_length, p - P.width)

    def dip_slip(y1, y2):
        d_bar = y2 * sn - q * cs
        r = sqrt(y1**2 + y2**2 + q**2)
        xx = sqrt(y1**2 + q**2)
        a5 = 4 * poisson * atan((y2 * (xx + q * cs) + xx * (r + xx) * sn) / (y1 * (r + xx) * cs)) / cs
        return -0.5 * (d_bar * q / (r * (r + y1)) + sn * atan(y1 * y2 / (q * r)) - a5 * sn * cs) / pi

    g1 = dip_slip(x1 + half_length, p)
    g2 = dip_slip(x1 + half_length, p - P.width)
    g3 = dip_slip(x1 - half_length, p)
    g4 = dip_slip(x1 - half_length, p - P.width)

    ds = P.slip * deg2cos(P.rake)
    dd = P.slip * deg2sin(P.rake)

    us = (f1 - f2 - f3 + f4) * ds
    ud = (g1 - g2 - g3 + g4) * dd

    return us + ud
