# Source inversion for the 2011 Tohoku tsunami

Given point-wise timeseries data from sixteen gauges in the Japan Sea and
Pacific Ocean, this example seeks to determine an initial sea surface
elevation due to the Great Japan Earthquake. The method for doing so is
gradient-based PDE-constrained optimisation with Pyadjoint.

The example contains a number of externally gathered data, as follows.

* Bathymetry field generated using the [ETOPO1 data set](https://www.ngdc.noaa.gov/mgg/global/) [1,2].
* Mesh generated using [qmesh](https://www.qmesh.org/) [3] and the [GSHHS coastline data set](https://www.ngdc.noaa.gov/mgg/shorelines/) [4].
* GPS gauge data due to the Port and Airport Research Institute of Japan [(PARI)](https://www.pari.go.jp/en/).
* Near-field pressure gauge data, provided by the authors of [5].
* Mid-field pressure gauge data due to the Japanese Agency for Marine-Earth Science and Technology [(JAMSTEC)](http://www.jamstec.go.jp/scdc/top_e.html).
* Far-field pressure gauge data due to the Deep-ocean Assessment and Reporting of Tsunamis [(DART)](https://nctr.pmel.noaa.gov/Dart/) programme of the National Oceanic and Atmospheric Administration [(NOAA)](https://www.ngdc.noaa.gov/mgg/global).

Gauge data have been post-processed to remove tidal signals, interpolate
away NaNs and smooth out noise. The post-processed timeseries are stored
in HDF5 format.

### References

[1] NOAA National Geophysical Data Center. 2009: ETOPO1 1 Arc-Minute Global
    Relief Model. NOAA National Centers for Environmental Information.
    Accessed 2022/03/14.

[2] Amante, C. and B.W. Eakins, 2009. ETOPO1 1 Arc-Minute Global Relief Model:
    Procedures, Data Sources and Analysis. NOAA Technical Memorandum NESDIS
    NGDC-24. National Geophysical Data Center, NOAA. doi:10.7289/V5C8276M
    [2022/03/14].

[3] Avdis, A. & Candy A. S. & Hill J. & Kramer C. S. & Piggott M. D.,
    “Efficient unstructured mesh generation for marine renewable energy
    applications”, Renewable Energy, Volume 116, Part A, February 2018,
    Pages 842-856, https://doi.org/10.1016/j.renene.2017.09.058.

[4] Wessel, P., and W. H. F. Smith (1996), A global, self-consistent,
    hierarchical, high-resolution shoreline database, J. Geophys. Res.,
    101(B4), 8741–8743, doi:10.1029/96JB00104.

[5] T. Saito, Y. Ito, D. Inazu, & R. Hino, "Tsunami source of the 2011
    Tohoku‐Oki earthquake, Japan: Inversion analysis based on dispersive
    tsunami simulations" (2011), Geophysical Research Letters, 38(7).
