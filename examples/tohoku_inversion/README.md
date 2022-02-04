# Source inversion for the 2011 Tohoku tsunami

Given point-wise timeseries data from sixteen gauges in the Japan Sea and
Pacific Ocean, this example seeks to determine an initial sea surface
elevation due to the Great Japan Earthquake. The method for doing so is
gradient-based PDE-constrained optimisation with Pyadjoint.

The example contains a number of externally gathered data, as follows.

* [ETOPO1 bathymetry](https://www.ngdc.noaa.gov/mgg/global/).
* Mesh generated using [qmesh](https://www.qmesh.org/).
* GPS gauge data due to the Port and Airport Research Institute of Japan [(PARI)](https://www.pari.go.jp/en/).
* Near-field pressure gauge data, provided by the authors of [1].
* Mid-field pressure gauge data due to the Japanese Agency for Marine-Earth Science and Technology [(JAMSTEC)](http://www.jamstec.go.jp/scdc/top_e.html).
* Far-field pressure gauge data due to the National Oceanic and Atmospheric Administration [(NOAA)](https://www.ngdc.noaa.gov/mgg/global).

Gauge data have been post-processed to remove tidal signals, interpolate
away NaNs and smooth out noise. The post-processed timeseries are stored
in HDF5 format.

[1] T. Saito, Y. Ito, D. Inazu, & R. Hino, "Tsunami source of the 2011
    Tohoku‐Oki earthquake, Japan: Inversion analysis based on dispersive
    tsunami simulations" (2011), Geophysical Research Letters, 38(7).
