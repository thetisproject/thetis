# Setting up a tidal simulation in the North Sea

There are four necessary components for a tidal simulation in Thetis, as
well as a number of optional extras. At the very least, you will require:

* The domain of interest.
* A mesh of the domain.
* The bathymetry field.
* Tidal forcing data.

This example shows you how to obtain all of those things.

### Domain

In order to generate a domain, we need coastline data. The approach taken
in this example uses vector coastline data extracted from the NOAA
[GSHHS data set](https://www.ngdc.noaa.gov/mgg/shorelines/).
A number of different resolutions are available, from very coarse to very
fine. We use the 'intermediate' option, with suffix `i`.

In most cases, we need to post-process the coastline data to some extent,
as well as include artificial boundary segments at the limit of where we
would like to model. The data in this example were generated using the
[QGIS](https://qgis.org) package. First, the shape files (with `.shp`
extension) were added to a new QGIS project. Any islands that are so small
that they would only appear on a very fine mesh were deleted from the
shape files. Next, a new layer was added in the QGIS project with an
artificial boundary - in this case a rectangle in latitude-longitude
coordinates. The coastline data were trimmed so that all points and edges
outside of the rectangle were deleted and then saved with file name
`Coastline`. Finally, another layer was added to combine the coastlines
inside with segments of the rectangle such that all of the edges in the
resulting `Boundary` shape file are parts of loops. For the boundary
file, we add an attribute with name `PhysID` to tag different parts of
the boundary, so that different boundary conditions can be enforced on
different segments. We use tag `100` for artificial (open ocean) segments
and `200` for coasts.

### Mesh

Given the shape files for the boundary and coastline data, saved in the
`data` subdirectory, we are able to run the `generate_mesh.py` script.
The output is the mesh `north_sea.msh`, which has been included here.
Note that this script requires the geoscientific mesh generator package
[qmesh](https://www.qmesh.org/).

### Bathymetry

There are many bathymetry data sets that are freely available on the web.
For the purposes of this demo, we use the NOAA
[ETOPO1 data set](https://www.ngdc.noaa.gov/mgg/global/).
From the ETOPO1 webpage, you can select a latitude-longitude region of
interest and download the bathymetry data as a NetCDF file. We have
included an example here as `etopo1.nc`. For a different example, see the
[Tohoku tsunami](https://github.com/thetisproject/thetis/tree/master/examples/tohoku_inversion)
example.

### Tidal forcings

Thetis has inbuilt support for the
[TPXO tidal forcing data set](https://www.tpxo.net/). To get access to these
data for academic or non-commerical purposes, you will need to follow the
instructions on the
[TPXO access page](https://www.tpxo.net/tpxo-products-and-registration).
Request the `tpxo9v5a` data set in NetCDF format and save it in some directory,
which can be referenced by the `$DATA` environment variable, such as `./data`.
The file has not been included here for copyright reasons.

### Optional extra: tide gauge data

With the above resources, we are able to create and run a tidal model in Thetis.
Given coordinates of tide gauges positioned within the domain, Thetis is able to
extract timeseries and save them in HDF5 format. It is often useful to compare
these simulated timeseries against observations.

In this example, we specify a number of different tide gauges and their
latitude-longitude coordinates in `station_elev.csv`. (Note that these are not the
exact coordinates - some have been modified slightly to ensure the gauges fall
within the meshed domain.) We also include the relevant region codes used in the
[CMEMS catalogue](http://www.marineinsitu.eu/access-data/), e.g. `IR`,
`GL`, `NO`. Tide gauge data for the time period of interest were downloaded
from the CMEMS dashboard and saved in the `data` subdirectory, so that they can
be loaded by `plot_elevation.py`. These files have not been included here for
copyright reasons. Note that you will need to register before you are able to
download data from CMEMS.
