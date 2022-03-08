from model_config import *
import matplotlib.pyplot as plt
import numpy


# Plot mesh
fig, axes = plt.subplots(figsize=(5.4, 4.8))
mesh2d = Mesh("north_sea.msh")
triplot(
    mesh2d,
    axes=axes,
    interior_kw={"linewidth": 0.5, "edgecolor": "gray"},
    boundary_kw={"linewidth": 0.8},
)
axes.axis(False)
axes.legend()
plt.tight_layout()
plt.savefig("north_sea_mesh.png", dpi=300)

# Plot bathymetry in a logarithmic colourmap
P1_2d = get_functionspace(mesh2d, "CG", 1)
bathymetry_2d = Function(P1_2d, name="Bathymetry")
with DumbCheckpoint("north_sea_bathymetry", mode=FILE_READ) as h5:
    h5.load(bathymetry_2d)
fig, axes = plt.subplots(figsize=(6.4, 4.8))
triplot(mesh2d, axes=axes, boundary_kw={"linewidth": 0.5, "edgecolor": "k"})
bathymetry_2d.dat.data[:] = numpy.log10(bathymetry_2d.dat.data[:])
tc = tricontourf(bathymetry_2d, axes=axes, levels=50)
cb = fig.colorbar(tc, ax=axes)
ticks = cb.get_ticks()
cb.set_ticklabels([f"{10 ** t:.0f}" for t in ticks])
cb.set_label("Bathymetry (m)")

# Mark tide gauges
trans = pyproj.Transformer.from_crs(coordsys.LL_WGS84.srs, UTM_ZONE30.srs)
for name, data in read_station_data().items():
    sta_lat, sta_lon = data["latlon"]
    sta_x, sta_y = trans.transform(sta_lon, sta_lat)
    off = -10000 if data["region"] == "NO" else 10000
    axes.plot(sta_x, sta_y, "x", color="C1")
axes.axis(False)
plt.tight_layout()
plt.savefig("north_sea_bathymetry.png", dpi=300)

# Plot initial elevation
P1DG_2d = get_functionspace(mesh2d, "DG", 1)
elev_2d = Function(P1DG_2d, name="elev_2d")
with DumbCheckpoint("outputs_spinup/hdf5/Elevation2d_00014", mode=FILE_READ) as chk:
    chk.load(elev_2d)
fig, axes = plt.subplots(figsize=(6.4, 4.8))
triplot(mesh2d, axes=axes, boundary_kw={"linewidth": 0.5, "edgecolor": "k"})
levels = numpy.linspace(-2, 2, 51)
tc = tricontourf(elev_2d, axes=axes, levels=levels, cmap="RdBu_r")
cb = fig.colorbar(tc, ax=axes)
ticks = numpy.linspace(-2, 2, 9)
cb.set_ticks(ticks)
cb.set_label("Initial elevation (m)")
axes.axis(False)
plt.tight_layout()
plt.savefig("north_sea_init.png", dpi=300)
