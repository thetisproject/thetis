"""
Definitions and meta data of fields
"""

field_metadata = {}
"""
Dictionary that contains the meta data of each field.

Required meta data entries are:

- **name**: human readable description
- **shortname**: description used in visualization etc
- **unit**: SI unit of the field
- **filename**: filename for output files

The naming convention for field keys is snake_case: ``field_name_3d``
"""

field_metadata['bathymetry_2d'] = {
    'name': 'Bathymetry',
    'shortname': 'Bathymetry',
    'unit': 'm',
    'filename': 'bathymetry2d',
}
field_metadata['bathymetry_3d'] = {
    'name': 'Bathymetry',
    'shortname': 'Bathymetry',
    'unit': 'm',
    'filename': 'bathymetry3d',
}
field_metadata['z_coord_3d'] = {
    'name': 'Mesh z coordinates',
    'shortname': 'Z coordinates',
    'unit': 'm',
    'filename': 'ZCoord3d',
}
field_metadata['z_bottom_2d'] = {
    'name': 'Bottom cell z coordinates',
    'shortname': 'Bottom cell z coordinates',
    'unit': 'm',
    'filename': 'ZBottom2d',
}
field_metadata['z_coord_ref_3d'] = {
    'name': 'Static mesh z coordinates',
    'shortname': 'Z coordinates',
    'unit': 'm',
    'filename': 'ZCoordRef3d',
}
field_metadata['bottom_drag_2d'] = {
    'name': 'Bottom drag coefficient',
    'shortname': 'Bottom drag coefficient',
    'unit': '',
    'filename': 'BottomDrag2d',
}
field_metadata['bottom_drag_3d'] = {
    'name': 'Bottom drag coefficient',
    'shortname': 'Bottom drag coefficient',
    'unit': '',
    'filename': 'BottomDrag3d',
}
field_metadata['uv_2d'] = {
    'name': 'Depth averaged velocity',
    'shortname': 'Depth averaged velocity',
    'unit': 'm s-1',
    'filename': 'Velocity2d',
}
field_metadata['tracer_2d'] = {
    'name': 'Depth averaged tracer',
    'shortname': 'Tracer',
    'unit': '',
    'filename': 'Tracer2d',
}
field_metadata['uv_dav_2d'] = {
    'name': 'Depth averaged velocity',
    'shortname': 'Depth averaged velocity',
    'unit': 'm s-1',
    'filename': 'DAVelocity2d',
}
field_metadata['split_residual_2d'] = {
    'name': 'Momentum eq. residual for mode splitting',
    'shortname': 'Momentum residual',
    'unit': 'm s-2',
    'filename': 'SplitResidual2d',
}
field_metadata['uv_bottom_2d'] = {
    'name': 'Bottom velocity',
    'shortname': 'Bottom velocity',
    'unit': 'm s-1',
    'filename': 'BottomVelo2d',
}
field_metadata['uv_bottom_3d'] = {
    'name': 'Bottom velocity',
    'shortname': 'Bottom velocity',
    'unit': 'm s-1',
    'filename': 'BottomVelo3d',
}
field_metadata['uv_dav_3d'] = {
    'name': 'Depth averaged velocity',
    'shortname': 'Depth averaged velocity',
    'unit': 'm s-1',
    'filename': 'DAVelocity3d',
}
field_metadata['uv_mag_3d'] = {
    'name': 'Magnitude of horizontal velocity',
    'shortname': 'Velocity magnitude',
    'unit': 'm s-1',
    'filename': 'VeloMag3d',
}
field_metadata['uv_p1_3d'] = {
    'name': 'P1 projection of horizontal velocity',
    'shortname': 'P1 Velocity',
    'unit': 'm s-1',
    'filename': 'VeloCG3d',
}
field_metadata['elev_2d'] = {
    'name': 'Water elevation',
    'shortname': 'Elevation',
    'unit': 'm',
    'filename': 'Elevation2d',
}
field_metadata['elev_3d'] = {
    'name': 'Water elevation',
    'shortname': 'Elevation',
    'unit': 'm',
    'filename': 'Elevation3d',
}
field_metadata['elev_cg_3d'] = {
    'name': 'Water elevation CG',
    'shortname': 'Elevation',
    'unit': 'm',
    'filename': 'ElevationCG3d',
}
field_metadata['elev_cg_2d'] = {
    'name': 'Water elevation CG',
    'shortname': 'Elevation',
    'unit': 'm',
    'filename': 'ElevationCG2d',
}
field_metadata['uv_3d'] = {
    'name': 'Horizontal velocity',
    'shortname': 'Horizontal velocity',
    'unit': 'm s-1',
    'filename': 'Velocity3d',
}
field_metadata['w_3d'] = {
    'name': 'Vertical velocity',
    'shortname': 'Vertical velocity',
    'unit': 'm s-1',
    'filename': 'VertVelo3d',
}
field_metadata['w_mesh_3d'] = {
    'name': 'Mesh velocity',
    'shortname': 'Mesh velocity',
    'unit': 'm s-1',
    'filename': 'MeshVelo3d',
}
field_metadata['w_mesh_surf_3d'] = {
    'name': 'Surface mesh velocity',
    'shortname': 'Surface mesh velocity',
    'unit': 'm s-1',
    'filename': 'SurfMeshVelo3d',
}
field_metadata['w_mesh_surf_2d'] = {
    'name': 'Surface mesh velocity',
    'shortname': 'Surface mesh velocity',
    'unit': 'm s-1',
    'filename': 'SurfMeshVelo3d',
}
field_metadata['salt_3d'] = {
    'name': 'Water salinity',
    'shortname': 'Salinity',
    'unit': 'psu',
    'filename': 'Salinity3d',
}
field_metadata['temp_3d'] = {
    'name': 'Water temperature',
    'shortname': 'Temperature',
    'unit': 'C',
    'filename': 'Temperature3d',
}
field_metadata['density_3d'] = {
    'name': 'Water density',
    'shortname': 'Density',
    'unit': 'kg m-3',
    'filename': 'Density3d',
}
field_metadata['parab_visc_3d'] = {
    'name': 'Parabolic Viscosity',
    'shortname': 'Parabolic Viscosity',
    'unit': 'm2 s-1',
    'filename': 'ParabVisc3d',
}
field_metadata['eddy_visc_3d'] = {
    'name': 'Eddy Viscosity',
    'shortname': 'Eddy Viscosity',
    'unit': 'm2 s-1',
    'filename': 'EddyVisc3d',
}
field_metadata['eddy_diff_3d'] = {
    'name': 'Eddy diffusivity',
    'shortname': 'Eddy diffusivity',
    'unit': 'm2 s-1',
    'filename': 'EddyDiff3d',
}
field_metadata['shear_freq_3d'] = {
    'name': 'Vertical shear frequency squared',
    'shortname': 'Vertical shear frequency squared',
    'unit': 's-2',
    'filename': 'ShearFreq3d',
}
field_metadata['buoy_freq_3d'] = {
    'name': 'Buoyancy frequency squared',
    'shortname': 'Buoyancy frequency squared',
    'unit': 's-2',
    'filename': 'BuoyFreq3d',
}
field_metadata['tke_3d'] = {
    'name': 'Turbulent Kinetic Energy',
    'shortname': 'Turbulent Kinetic Energy',
    'unit': 'm2 s-2',
    'filename': 'TurbKEnergy3d',
}
field_metadata['psi_3d'] = {
    'name': 'Turbulence psi variable',
    'shortname': 'Turbulence psi variable',
    'unit': '',
    'filename': 'TurbPsi3d',
}
field_metadata['eps_3d'] = {
    'name': 'TKE dissipation rate',
    'shortname': 'TKE dissipation rate',
    'unit': 'm2 s-3',
    'filename': 'TurbEps3d',
}
field_metadata['len_3d'] = {
    'name': 'Turbulent length scale',
    'shortname': 'Turbulent length scale',
    'unit': 'm',
    'filename': 'TurbLen3d',
}
field_metadata['baroc_head_3d'] = {
    'name': 'Baroclinic head',
    'shortname': 'Baroclinic head',
    'unit': 'm',
    'filename': 'BaroHead3d',
}
field_metadata['int_pg_3d'] = {
    'name': 'Internal pressure gradient',
    'shortname': 'Int. Pressure gradient',
    'unit': 'm s-2',
    'filename': 'IntPG3d',
}
field_metadata['smag_visc_3d'] = {
    'name': 'Smagorinsky viscosity',
    'shortname': 'Smagorinsky viscosity',
    'unit': 'm2 s-1',
    'filename': 'SmagViscosity3d',
}
field_metadata['max_h_diff'] = {
    'name': 'Maximum stable horizontal diffusivity',
    'shortname': 'Maximum horizontal diffusivity',
    'unit': 'm2 s-1',
    'filename': 'MaxHDiffusivity3d',
}
field_metadata['v_elem_size_3d'] = {
    'name': 'Element size in vertical direction',
    'shortname': 'Vertical element size',
    'unit': 'm',
    'filename': 'VElemSize3d',
}
field_metadata['v_elem_size_2d'] = {
    'name': 'Element size in vertical direction',
    'shortname': 'Vertical element size',
    'unit': 'm',
    'filename': 'VElemSize2d',
}
field_metadata['h_elem_size_3d'] = {
    'name': 'Element size in horizontal direction',
    'shortname': 'Horizontal element size',
    'unit': 'm',
    'filename': 'h_elem_size_3d',
}
field_metadata['h_elem_size_2d'] = {
    'name': 'Element size in horizontal direction',
    'shortname': 'Horizontal element size',
    'unit': 'm',
    'filename': 'h_elem_size_2d',
}
field_metadata['coriolis_2d'] = {
    'name': 'Coriolis parameter',
    'shortname': 'Coriolis parameter',
    'unit': 's-1',
    'filename': 'coriolis_2d',
}
field_metadata['coriolis_3d'] = {
    'name': 'Coriolis parameter',
    'shortname': 'Coriolis parameter',
    'unit': 's-1',
    'filename': 'coriolis_3d',
}
field_metadata['wind_stress_3d'] = {
    'name': 'Wind stress',
    'shortname': 'Wind stress',
    'unit': 'Pa',
    'filename': 'wind_stress_3d',
}
field_metadata['hcc_metric_3d'] = {
    'name': 'HCC mesh quality',
    'shortname': 'HCC metric',
    'unit': '-',
    'filename': 'HCCMetric3d',
}

# Below is for non-hydrostatic (nh) extension that WPan is adding
field_metadata['q_3d'] = {
    'name': 'Non-hydrostatic pressure',
    'shortname': 'NH pressure',
    'unit': 'Pa',
    'filename': 'q_NH_3d',
}
field_metadata['q_2d'] = {
    'name': 'Non-hydrostatic pressure',
    'shortname': 'NH pressure',
    'unit': 'Pa',
    'filename': 'q_NH_2d',
}
field_metadata['w_nh'] = {
    'name': 'Vertical velocity for Non-hydrostatic solver',
    'shortname': 'Vertical velocity for NH',
    'unit': 'm s-1',
    'filename': 'VertVelo_NH',
}
field_metadata['uv_nh'] = {
    'name': 'Horizontal velocity for Non-hydrostatic solver',
    'shortname': 'Horizontal velocity for NH',
    'unit': 'm s-1',
    'filename': 'HoriVelo_NH',
}
field_metadata['elev_nh'] = {
    'name': 'Water elevation for Non-hydrostatic solver',
    'shortname': 'Elevation for NH',
    'unit': 'm',
    'filename': 'Elevation_NH',
}
field_metadata['ext_pg_3d'] = {
    'name': 'External pressure gradient for Non-hydrostatic solver',
    'shortname': 'Ext. Pressure gradient for NH',
    'unit': 'm s-2',
    'filename': 'ExtPG_NH',
}
field_metadata['uv_delta'] = {
    'name': 'Layer velocity difference',
    'shortname': 'Layer velocity difference',
    'unit': 'm s-1',
    'filename': 'VelocityDiff',
}
field_metadata['uv_delta_2'] = {
    'name': 'Layer velocity difference 2',
    'shortname': 'Layer velocity difference 2',
    'unit': 'm s-1',
    'filename': 'VelocityDiff_2',
}
field_metadata['uv_delta_3'] = {
    'name': 'Layer velocity difference 3',
    'shortname': 'Layer velocity difference 3',
    'unit': 'm s-1',
    'filename': 'VelocityDiff_3',
}
field_metadata['uv_ls'] = {
    'name': 'Velocity of landslide',
    'shortname': 'Velocity of landslide',
    'unit': 'm s-1',
    'filename': 'Velocity2d_ls',
}
field_metadata['elev_ls'] = {
    'name': 'Elevation of landslide',
    'shortname': 'Upper surface of slide',
    'unit': 'm',
    'filename': 'Elevation2d_ls',
}
field_metadata['slide_source'] = {
    'name': '2D slide source for displacement',
    'shortname': '2D slide source for displacement',
    'unit': 'm s-1',
    'filename': 'SlideSource_2d',
}
field_metadata['slide_source_3d'] = {
    'name': '3D slide source for displacement',
    'shortname': '3D slide source for displacement',
    'unit': 'm s-1',
    'filename': 'SlideSource_3d',
}
field_metadata['bed_slope'] = {
    'name': 'Bed slope for granular slide modelling',
    'shortname': 'Bed slope for granular slide modelling',
    'unit': '-',
    'filename': 'BedSlope_ls',
}
field_metadata['sigma_dt'] = {
    'name': 'Time derivative of z in sigma mesh',
    'shortname': 'Time derivative of z in sigma mesh',
    'unit': '-',
    'filename': 'Sigma_dt',
}
field_metadata['sigma_dx'] = {
    'name': 'Hori x-derivative of z in sigma mesh',
    'shortname': 'Hori x-derivative of z in sigma mesh',
    'unit': '-',
    'filename': 'Sigma_dx',
}
field_metadata['sigma_dy'] = {
    'name': 'Hori y-derivative of z in sigma mesh',
    'shortname': 'Hori y-derivative of z in sigma mesh',
    'unit': '-',
    'filename': 'Sigma_dy',
}
field_metadata['omega'] = {
    'name': 'Vertical velocity in sigma mesh',
    'shortname': 'Vertical velocity in sigma mesh',
    'unit': '-',
    'filename': 'Omega',
}
field_metadata['c_3d'] = {
    'name': 'Sediment concentration',
    'shortname': 'Sediment',
    'unit': '-',
    'filename': 'Concentration3d',
}

