This directory contains .json files associated with `test/swe2d/test_rossby_wave.py`.

The files `FVCOM.json` and `ROMS.json` contain error metrics computed from model outputs of
Finite Element Coastal Ocean Model (FVCOM) and Regional Ocean Modeling System (ROMS) applied to the
same test case. The test case is described on p.3-6 of [1], where the error metrics are presented
in Table 1.

Exectuting `python3 test_rossby_wave.py` will run the test case using Thetis for the same timesteps
and meshes considered in [1], outputting a file `Thetis_dg-cg_SSPRK33.json`, where `'dg-cg'` and
`'SSPRK33'` indicate the element pair and time integration scheme used.

[1] H. Huang, C. Chen, G.W. Cowles, C.D. Winant, R.C. Beardsley, K.S. Hedstrom and D.B. Haidvogel,
"FVCOM validation experiments: Comparisons with ROMS for three idealized barotropic test problems"
(2008), Journal of Geophysical Research: Oceans, 113(C7).
