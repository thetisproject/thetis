"""
gen_ww3_bc_spectra_simple.py
========================
Generate N identical WW3 BOUNC spectral boundary-condition files plus the
matching ww3_bounc.inp, using Hasselmann et al., 1973 JONSWAP directional spectra

Note:
    set N_POINTS to the number of spectral files to generate.
    Every file carries the same JONSWAP spectrum, so WW3 (ww3_bounc) maps the open-boundary points defined
    in ww3_grid.inp onto these files by nearest-neighbour.
    BC nodes in ww3_grid.inp are set manually based on the mesh boundary nodes selected
    WW3 v6071 is run on .msh v2.2 (v4.1 must be converted to v2.2 for WW3 compatibility, e.g. gmsh -> export -> v2.2)

Usage:
    python gen_ww3_bc_spectra_simple.py
"""

import os
import glob
import numpy as np
import pandas
import netCDF4 as nc
from datetime import datetime


# ---------------
# JONSWAP directional spectrum
# ---------------

class JONSWAP:
    """
    JONSWAP directional wave spectrum (Hasselmann et al., 1973).

    Parameters
    ----------
    hs        : significant wave height [m]
    tp        : peak period [s]
    direction : mean wave direction [deg, meteorological convention]
    ndir      : number of directional bins
    freqs     : 1-D array of frequency bins [Hz]
    gamma     : peak-enhancement factor (default 3.3)
    """

    def __init__(self, hs, tp, direction, ndir, freqs, gamma=3.3):
        self.hs, self.tp, self.dir = hs, tp, direction
        self.ndir, self.freqs, self.gamma = ndir, freqs, gamma

    def _energy(self):
        """1-D JONSWAP energy density spectrum [m² s]."""
        fp = 1.0 / self.tp
        fp4 = fp ** 4
        alpha = (self.hs**2 * fp4) / \
            ((0.06533 * (self.gamma**0.8015 + 0.13467)) * 16)
        spec = []
        for f in self.freqs:
            cpshap = 1.25 * fp4 / f**4
            ra = 0.0 if cpshap > 10 else (alpha / f**5) * np.exp(-cpshap)
            sigma = 0.07 if f <= fp else 0.09
            apshap = 0.5 * ((f - fp) / (sigma * fp)) ** 2
            syf = 1.0 if apshap > 10 else self.hs ** np.exp(-apshap)
            spec.append(syf * ra / (2 * np.pi * f))
        return np.array(spec)

    def _ln_gamma(self, val):
        """Log-Gamma via Lanczos approximation."""
        coefs = [76.18009173, -86.50532033, 24.01409822,
                 -1.231739516, 0.120858003e-2, -0.536382e-5]
        fpf = 4.5
        tmp = (val + 0.5 + fpf) * np.log(val + fpf) - (val + fpf)
        ser = 1.0
        for c in coefs:
            ser += c / val
            val += 1
        return tmp + np.log(2.50662827465 * ser)

    def _gamma(self, val):
        return np.exp(np.clip(self._ln_gamma(val), -30, 30))

    def compute(self):
        """
        Full 2-D directional JONSWAP spectrum.

        Returns
        -------
        energy  : (nfreq,)       1-D energy density [m² s]
        dirspec : (ndir, nfreq)  directional spectrum [m² s rad⁻¹]
        """
        energy = self._energy()
        dirs = [np.pi * i / 180 for i in np.arange(0, 360, 360.0 / self.ndir)]
        rad = np.pi * self.dir / 180
        dir_spread = 360.0 / self.ndir

        if dir_spread < 12:
            ctot = (2.0**dir_spread) * self._gamma(0.5 * dir_spread + 1.0)**2 \
                / (np.pi * self._gamma(dir_spread + 1.0))
        else:
            ctot = np.sqrt(0.5 * dir_spread / np.pi) / \
                (1.0 - 0.25 / dir_spread)

        dirspec = []
        for d in dirs:
            acos = np.cos(d - rad)
            cdir = ctot * np.clip(acos**dir_spread, 1.e-10,
                                  None) if acos > 0 else 0.0
            dirspec.extend(cdir * e for e in energy)

        return energy, np.array(dirspec).reshape(self.ndir, len(self.freqs))


# ---------------
# Frequency / direction grid helpers
# ---------------

def get_freqs(f_min, n_freq, ratio):
    """Geometric frequency sequence of length n_freq starting at f_min."""
    freqs, f = [], f_min
    for _ in range(n_freq):
        freqs.append(f)
        f *= ratio
    return np.array(freqs)


def get_dirs(n_dir):
    """Uniformly-spaced directional bin centres [deg, 0–360)."""
    return np.arange(0, 360, 360.0 / n_dir)


# ---------------
# WW3 BOUNC netCDF I/O
# ---------------

def _make_bc_dataset(path, times, station_id, freqs, dirs, spherical=True):
    """
    Create an open netCDF4 Dataset pre-configured for WW3 BOUNC format.
    Caller fills per-timestep data fields and then calls ds.close().
    """
    ds = nc.Dataset(path, 'w', format='NETCDF4')
    for dim, size in [('time', len(times)), ('station', 1),
                      ('frequency', len(freqs)), ('direction', len(dirs)),
                      ('string16', 16)]:
        ds.createDimension(dim, size)

    # time
    epoch = datetime.fromtimestamp(0)
    tv = ds.createVariable('time', 'f8', ('time',))
    tv[:] = [(datetime.strptime(str(t)[:19], '%Y-%m-%dT%H:%M:%S') - epoch).total_seconds()
             for t in times.values]
    tv.setncatts({'units': 'seconds since 1970-01-01 00:00:00.0', 'calendar': 'gregorian',
                  'long_name': 'time', 'standard_name': 'time', 'axis': 'T'})

    # frequency
    fv = ds.createVariable('frequency', 'f4', ('frequency',))
    fv[:] = freqs
    fv.setncatts({'units': 's-1', 'long_name': 'frequency of center band',
                  'standard_name': 'sea_surface_wave_frequency', 'globwave_name': 'frequency',
                  'valid_min': 0., 'valid_max': 10., 'axis': 'Y'})

    # direction
    dv = ds.createVariable('direction', 'f4', ('direction',))
    dv[:] = dirs
    dv.setncatts({'units': 'degree', 'long_name': 'sea surface wave to direction',
                  'standard_name': 'sea surface wave to direction', 'globwave_name': 'direction',
                  'valid_min': 0., 'valid_max': 360., 'axis': 'Z'})

    # station
    sv = ds.createVariable('station', 'i4', ('station',))
    sv[:] = [int(station_id)]
    sv.setncatts({'long_name': 'station id', 'axis': 'X'})

    # spatial position
    xy = ('time', 'station')
    if spherical:
        xv = ds.createVariable('longitude', 'f4', xy)
        yv = ds.createVariable('latitude', 'f4', xy)
        xv.setncatts({'units': 'degree_east', 'long_name': 'longitude',
                      'standard_name': 'longitude', 'valid_min': -180., 'valid_max': 180.,
                      'content': 'TX', 'associates': 'time'})
        yv.setncatts({'units': 'degree_north', 'long_name': 'latitude',
                      'standard_name': 'latitude', 'valid_min': -90., 'valid_max': 90.,
                      'content': 'TY', 'associates': 'time'})
    else:
        xv = ds.createVariable('x', 'f4', xy)
        yv = ds.createVariable('y', 'f4', xy)
        xv.setncatts({'units': 'meters', 'long_name': 'x', 'standard_name': 'x',
                      'valid_min': 0., 'valid_max': 1000000., 'content': 'TX',
                      'associates': 'time'})
        yv.setncatts({'units': 'meters', 'long_name': 'y', 'standard_name': 'y',
                      'valid_min': 0., 'valid_max': 1000000., 'content': 'TY',
                      'associates': 'time'})

    # auxiliary forcing
    aux = [
        ('depth', 'm', 'depth', 'depth', 0., 10000.),
        ('u10m', 'm s-1', 'wind speed at 10m', 'wind speed ', 0., 100.),
        ('udir', 'degree', 'wind direction', 'wind_from_direction', 0., 360.),
        ('curr', 'm s-1', 'sea water speed', 'sea_water_speed', 0., 100.),
        ('currdir', 'degree', 'direction from of sea water velocity',
         'direction_of_sea_water_velocity', 0., 360.),
    ]
    for vname, units, lname, sname, vmin, vmax in aux:
        v = ds.createVariable(vname, 'f4', xy)
        v.setncatts({'units': units, 'long_name': lname, 'standard_name': sname,
                     'globwave_name': sname, 'valid_min': vmin, 'valid_max': vmax,
                     'scale_factor': 1., 'add_offset': 0., 'content': 'TX',
                     'associates': 'time station'})
    ds['curr'][:] = 0.0
    ds['currdir'][:] = 0.0

    # directional variance spectral density
    ef = ds.createVariable(
        'efth',
        'f8',
        ('time',
         'station',
         'frequency',
         'direction'))
    ef.setncatts({'units': 'm2 s rad-1',
                  'long_name': 'sea surface wave directional variance spectral density',
                  'standard_name': 'sea surface wave directional variance spectral density',
                  'globwave_name': 'directional_variance_spectral_density',
                  'valid_min': 0., 'valid_max': 1.e+20, 'scale_factor': 1., 'add_offset': 0.,
                  'content': '', 'associates': 'time frequency direction', 'axis': 'TXYZ'})

    # WW3 string metadata
    sn = ds.createVariable('station_name', 'str', ('station', 'string16'))
    sn.setncatts({'long_name': 'station name', 'content': 'XW',
                 'associates': 'station string16'})
    st = ds.createVariable('string16', 'i4', ('string16',))
    st.setncatts(
        {'long_name': 'station_name number of characters', 'axis': 'W'})
    st[:] = str(station_id)

    return ds, xv, yv


def write_point_bc(x, y, depth, times, hs, tp, direction,
                   w_speed, w_dir, n_dir, freqs, point_name, outpath, spherical=True):
    """
    Generate and write a single boundary-point JONSWAP spectral netCDF file.

    The spectrum is computed per timestep, normalised so 4·√m0 = Hs exactly,
    and written in WW3 BOUNC format (efth shape: time × station × freq × dir).
    """
    outfile = os.path.join(outpath, f'id_{point_name}_spec.nc')
    dirs = get_dirs(n_dir)
    ds, xv, yv = _make_bc_dataset(
        outfile, times, point_name, freqs, dirs, spherical)

    hs_arr = np.full(
        len(times),
        hs,
        dtype=float) if np.isscalar(hs) else np.asarray(
        hs,
        dtype=float)
    tp_arr = np.full(
        len(times),
        tp,
        dtype=float) if np.isscalar(tp) else np.asarray(
        tp,
        dtype=float)
    if len(hs_arr) != len(times) or len(tp_arr) != len(times):
        raise ValueError("hs/tp array length must match len(times)")

    df = np.empty_like(freqs, dtype=float)
    df[0] = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    if len(freqs) > 1:
        df[1:] = np.diff(freqs)
    dtheta = 2.0 * np.pi / float(n_dir)

    for i in range(len(times)):
        hs_t = max(0.0, hs_arr[i])
        tp_t = max(0.01, tp_arr[i])

        _, dimSpec = JONSWAP(hs_t, tp_t, direction, n_dir, freqs).compute()

        m0 = np.nansum(dimSpec * df[np.newaxis, :] * dtheta)
        if m0 > 0:
            dimSpec *= (hs_t / 4.0)**2 / m0

        xv[i, 0] = x
        yv[i, 0] = y
        ds['depth'][i, 0] = depth
        ds['u10m'][i, 0] = w_speed
        ds['udir'][i, 0] = w_dir
        ds['efth'][i, 0] = dimSpec.T     # (nfreq, ndir)

    ds.close()


def verify_bc_files(pattern):
    """Read back output spectral files and print min/max Hs per file."""
    for path in sorted(glob.glob(pattern)):
        ds = nc.Dataset(path)
        ef = ds['efth'][:]
        frq = ds['frequency'][:]
        drs = ds['direction'][:]
        df = np.diff(frq, prepend=frq[0])
        dtheta = np.deg2rad(np.diff(np.hstack((drs, drs[0] + 360.0))))
        m0 = np.nansum(
            ef * df[np.newaxis, None, :, None]
            * dtheta[np.newaxis, None, None, :],
            axis=(2, 3))
        Hs = 4.0 * np.sqrt(m0)
        print(
            f'  {
                os.path.basename(path)}  Hs={
                float(
                    np.nanmin(Hs)):.3f}–{
                    float(
                        np.nanmax(Hs)):.3f} m')
        ds.close()


# ---------------
# Main
# ---------------

if __name__ == '__main__':

    # CONFIGURATION

    #  paths
    OUTPATH = './harmonics'       # spectral files + ww3_bounc.inp written here
    BOUNCDIR = './harmonics/'    # path to the spectra as written in ww3_bounc.inp
    os.makedirs(OUTPATH, exist_ok=True)

    #  number of spectral files
    N_POINTS = 10        # number of (identical) boundary spectral files

    #  wave configuration
    DIRECTION = 180      # deg, meteorological convention (waves to south)
    W_SPEED = 0.1        # nominal wind speed [m/s]
    N_DIR = 24           # number of directional bins (e.g. 24 -> 15 deg bins)
    FREQS = get_freqs(f_min=0.052, n_freq=32, ratio=1.1)  # 32 freq bins
    # water depth at boundary nodes [m] (all identical for inlet domain)
    DEPTH = 15.0

    #  wave height / period
    HS_FIXED = 1.0          # significant wave height [m]
    TP_FIXED = 10.0         # peak period [s]
    TIDAL_SIGNAL = False    # True -> modulate Hs/Tp with a tidal harmonic

    # Tidal signal params (used if Tidal_signal = True)
    HS_TIDAL_AMP = 0.6      # tidal amplitude [m]
    TP_TIDAL_AMP = 1.0      # tidal amplitude [s]
    TIDAL_PERIOD_H = 12.0   # tidal period [hours]

    #  time axis
    times = pandas.date_range('2018-08-01T00:00:00.000Z',
                              '2018-08-03T05:00:00.000Z', freq='0.5h')

    #  build Hs / Tp series
    if TIDAL_SIGNAL:
        t_sec = (times - times[0]).total_seconds().astype(float)
        omega = 2.0 * np.pi / (TIDAL_PERIOD_H * 3600.0)
        hs_series = HS_FIXED + HS_TIDAL_AMP * np.cos(omega * t_sec)
        tp_series = TP_FIXED + TP_TIDAL_AMP * np.cos(omega * t_sec)
    else:
        hs_series = np.full(len(times), HS_FIXED)
        tp_series = np.full(len(times), TP_FIXED)

    #  placeholder station coordinates (no mesh read)
    # All N_POINTS files carry the same spectrum, so the station positions are
    # physically irrelevant.  They are spread along a line only to give WW3
    # distinct nearest-neighbour anchors;
    # adjust if domain dimensions change or leave as-is for tidal inlet (see
    # the .msh)
    xs = np.linspace(0.0, 15000.0, N_POINTS)   # x of each station [m]
    Y_POS = 14000.0                            # constant y [m]

    #  write spectral files
    print(f'Writing {N_POINTS} spectral files -> {OUTPATH}/')
    bc_files = []
    for i in range(N_POINTS):
        name = str(i + 1)
        write_point_bc(
            xs[i], Y_POS, DEPTH, times,
            hs_series, tp_series, DIRECTION,
            W_SPEED, DIRECTION, N_DIR, FREQS,
            name, OUTPATH, spherical=False)
        bc_files.append(os.path.join(BOUNCDIR, f'id_{name}_spec.nc'))
        print(f'  [{i + 1}/{N_POINTS}] id_{name}_spec.nc')

    #  ww3_bounc.inp
    bounc_inp = os.path.join(OUTPATH, 'ww3_bounc.inp')
    with open(bounc_inp, 'w') as fh:
        fh.write(
            '$ boundary option: READ or WRITE\n'
            '  WRITE\n'
            '$ Interpolation method. 1: nearest, 2: linear interpolation\n'
            '  1\n0\n'
            '$ list of spectra files.\n'
            + ''.join(f + '\n' for f in bc_files)
            + "'STOPSTRING'\n$\n"
        )
    print(f'\nBounc input written: {bounc_inp}')

    #  verify
    print('\n--- Output verification ---')
    verify_bc_files(os.path.join(OUTPATH, 'id_*_spec.nc'))
