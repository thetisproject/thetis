from model_config import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load mesh and bathymetry
with CheckpointFile("north_sea_bathymetry.h5", "r") as f:
    mesh2d = f.load_mesh("firedrake_default")
    bathymetry_2d = f.load_function(mesh2d, "Bathymetry")

# Plot mesh
fig, axes = plt.subplots(figsize=(5.4, 4.8))
triplot(
    mesh2d,
    axes=axes,
    interior_kw={"linewidth": 0.5, "edgecolor": "gray"},
    boundary_kw={"linewidth": 0.8},
)
axes.axis(False)
axes.legend()
plt.tight_layout()
imgfile = "north_sea_mesh.png"
print_output(f'Saving {imgfile}')
plt.savefig(imgfile, dpi=300)

# Plot bathymetry in a logarithmic colourmap
fig, axes = plt.subplots(figsize=(6.4, 4.8))
triplot(mesh2d, axes=axes, boundary_kw={"linewidth": 0.5, "edgecolor": "k"})
norm = mcolors.LogNorm(vmin=10, vmax=3200)
tc = tripcolor(bathymetry_2d, axes=axes, norm=norm)
ticks = [10, 30, 100, 300, 1000, 3000]
cb = fig.colorbar(tc, ax=axes, ticks=ticks, format='%4.0f')
cb.set_label("Bathymetry (m)")

# Mark tide gauges
for name, data in read_station_data().items():
    sta_lat, sta_lon = data["latlon"]
    sta_x, sta_y = coord_system.to_xy(sta_lon, sta_lat)
    off = -10000 if data["region"] == "NO" else 10000
    axes.plot(sta_x, sta_y, "x", color="C1")
axes.axis(False)
plt.tight_layout()
imgfile = "north_sea_bathymetry.png"
print_output(f'Saving {imgfile}')
plt.savefig(imgfile, dpi=300)

# Plot initial elevation
with CheckpointFile("outputs_spinup/hdf5/Elevation2d_00014.h5", "r") as f:
    m = f.load_mesh("firedrake_default")
    elev_2d = f.load_function(m, "elev_2d")
fig, axes = plt.subplots(figsize=(6.4, 4.8))
triplot(mesh2d, axes=axes, boundary_kw={"linewidth": 0.5, "edgecolor": "k"})
norm = mcolors.CenteredNorm()
tc = tripcolor(elev_2d, axes=axes, cmap="RdBu_r", norm=norm)
cb = fig.colorbar(tc, ax=axes)
cb.set_label("Initial elevation (m)")
axes.axis(False)
plt.tight_layout()
imgfile = "north_sea_init.png"
print_output(f'Saving {imgfile}')
plt.savefig(imgfile, dpi=300)
