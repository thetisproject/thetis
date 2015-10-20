"""
Definitions of fields.

Tuomas Karna 2015-10-17
"""

# TODO come up with consistent field naming scheme
# name_3d_version?

fieldMetadata = {}
"""
Holds description, units and output file information for each field.

name      - human readable description
shortname - description used in visualization etc
unit      - SI unit of the field
filename  - filename for output files
"""

fieldMetadata['bathymetry2d'] = {
    'name': 'Bathymetry',
    'shortname': 'Bathymetry',
    'unit': 'm',
    'filename': 'bathymetry2d',
    }
fieldMetadata['bathymetry3d'] = {
    'name': 'Bathymetry',
    'shortname': 'Bathymetry',
    'unit': 'm',
    'filename': 'bathymetry3d',
    }
fieldMetadata['z_coord3d'] = {
    'name': 'Mesh z coordinates',
    'shortname': 'Z coordinates',
    'unit': 'm',
    'filename': 'ZCoord3d',
    }
fieldMetadata['z_bottom2d'] = {
    'name': 'Bottom cell z coordinates',
    'shortname': 'Bottom cell z coordinates',
    'unit': 'm',
    'filename': 'ZBottom2d',
    }
fieldMetadata['z_coord_ref3d'] = {
    'name': 'Static mesh z coordinates',
    'shortname': 'Z coordinates',
    'unit': 'm',
    'filename': 'ZCoordRef3d',
    }
fieldMetadata['bottom_drag2d'] = {
    'name': 'Bottom drag coefficient',
    'shortname': 'Bottom drag coefficient',
    'unit': '',
    'filename': 'BottomDrag2d',
    }
fieldMetadata['bottom_drag3d'] = {
    'name': 'Bottom drag coefficient',
    'shortname': 'Bottom drag coefficient',
    'unit': '',
    'filename': 'BottomDrag3d',
    }
fieldMetadata['uv2d'] = {
    'name': 'Depth averaged velocity',
    'shortname': 'Depth averaged velocity',
    'unit': 'm s-1',
    'filename': 'Velocity2d',
    }
fieldMetadata['uvDav2d'] = {
    'name': 'Depth averaged velocity',
    'shortname': 'Depth averaged velocity',
    'unit': 'm s-1',
    'filename': 'DAVelocity2d',
    }
fieldMetadata['uv_bottom2d'] = {
    'name': 'Bottom velocity',
    'shortname': 'Bottom velocity',
    'unit': 'm s-1',
    'filename': 'BottomVelo2d',
    }
fieldMetadata['uv_bottom3d'] = {
    'name': 'Bottom velocity',
    'shortname': 'Bottom velocity',
    'unit': 'm s-1',
    'filename': 'BottomVelo3d',
    }
fieldMetadata['uvDav3d'] = {
    'name': 'Depth averaged velocity',
    'shortname': 'Depth averaged velocity',
    'unit': 'm s-1',
    'filename': 'DAVelocity3d',
    }
fieldMetadata['uv3d_mag'] = {
    'name': 'Magnitude of horizontal velocity',
    'shortname': 'Velocity magnitude',
    'unit': 'm s-1',
    'filename': 'VeloMag3d',
    }
fieldMetadata['uv3d_P1'] = {
    'name': 'P1 projection of horizontal velocity',
    'shortname': 'P1 Velocity',
    'unit': 'm s-1',
    'filename': 'VeloCG3d',
    }
fieldMetadata['uvBot2d'] = {
    'name': 'Bottom velocity',
    'shortname': 'Bottom velocity',
    'unit': 'm s-1',
    'filename': 'BotVelocity2d',
    }
fieldMetadata['elev2d'] = {
    'name': 'Water elevation',
    'shortname': 'Elevation',
    'unit': 'm',
    'filename': 'Elevation2d',
    }
fieldMetadata['elev3d'] = {
    'name': 'Water elevation',
    'shortname': 'Elevation',
    'unit': 'm',
    'filename': 'Elevation3d',
    }
fieldMetadata['elev3dCG'] = {
    'name': 'Water elevation CG',
    'shortname': 'Elevation',
    'unit': 'm',
    'filename': 'ElevationCG3d',
    }
fieldMetadata['uv3d'] = {
    'name': 'Horizontal velocity',
    'shortname': 'Horizontal velocity',
    'unit': 'm s-1',
    'filename': 'Velocity3d',
    }
fieldMetadata['w3d'] = {
    'name': 'Vertical velocity',
    'shortname': 'Vertical velocity',
    'unit': 'm s-1',
    'filename': 'VertVelo3d',
    }
fieldMetadata['wMesh3d'] = {
    'name': 'Mesh velocity',
    'shortname': 'Mesh velocity',
    'unit': 'm s-1',
    'filename': 'MeshVelo3d',
    }
fieldMetadata['wMeshSurf3d'] = {
    'name': 'Surface mesh velocity',
    'shortname': 'Surface mesh velocity',
    'unit': 'm s-1',
    'filename': 'SurfMeshVelo3d',
    }
fieldMetadata['wMeshSurf2d'] = {
    'name': 'Surface mesh velocity',
    'shortname': 'Surface mesh velocity',
    'unit': 'm s-1',
    'filename': 'SurfMeshVelo3d',
    }
fieldMetadata['dwMeshDz3d'] = {
    'name': 'Vertical grad of mesh velocity',
    'shortname': 'Vertical grad of mesh velocity',
    'unit': 's-1',
    'filename': 'dMeshVeloDz3d',
    }
fieldMetadata['salt3d'] = {
    'name': 'Water salinity',
    'shortname': 'Salinity',
    'unit': 'psu',
    'filename': 'Salinity3d',
    }
fieldMetadata['parabVisc3d'] = {
    'name': 'Parabolic Viscosity',
    'shortname': 'Parabolic Viscosity',
    'unit': 'm2 s-1',
    'filename': 'ParabVisc3d',
    }
fieldMetadata['eddyVisc3d'] = {
    'name': 'Eddy Viscosity',
    'shortname': 'Eddy Viscosity',
    'unit': 'm2 s-1',
    'filename': 'EddyVisc3d',
    }
fieldMetadata['eddyDiff3d'] = {
    'name': 'Eddy diffusivity',
    'shortname': 'Eddy diffusivity',
    'unit': 'm2 s-1',
    'filename': 'EddyDiff3d',
    }
fieldMetadata['shearFreq3d'] = {
    'name': 'Vertical shear frequency squared',
    'shortname': 'Vertical shear frequency squared',
    'unit': 's-2',
    'filename': 'ShearFreq3d',
    }
fieldMetadata['buoyFreq3d'] = {
    'name': 'Buoyancy frequency squared',
    'shortname': 'Buoyancy shear frequency squared',
    'unit': 's-2',
    'filename': 'BuoyFreq3d',
    }
fieldMetadata['tke3d'] = {
    'name': 'Turbulent Kinetic Energy',
    'shortname': 'Turbulent Kinetic Energy',
    'unit': 'm2 s-2',
    'filename': 'TurbKEnergy3d',
    }
fieldMetadata['psi3d'] = {
    'name': 'Turbulence psi variable',
    'shortname': 'Turbulence psi variable',
    'unit': '',
    'filename': 'TurbPsi3d',
    }
fieldMetadata['eps3d'] = {
    'name': 'TKE dissipation rate',
    'shortname': 'TKE dissipation rate',
    'unit': 'm2 s-2',
    'filename': 'TurbEps3d',
    }
fieldMetadata['len3d'] = {
    'name': 'Turbulent lenght scale',
    'shortname': 'Turbulent lenght scale',
    'unit': 'm',
    'filename': 'TurbLen3d',
    }
fieldMetadata['baroHead3d'] = {
    'name': 'Baroclinic head',
    'shortname': 'Baroclinic head',
    'unit': 'm',
    'filename': 'BaroHead3d',
    }
fieldMetadata['baroHeadInt3d'] = {
    'name': 'Vertical integral of baroclinic head',
    'shortname': 'Vertically integrated baroclinic head',
    'unit': 'm2',
    'filename': 'BaroHeadInt3d',
    }
fieldMetadata['baroHead2d'] = {
    'name': 'Dav baroclinic head',
    'shortname': 'Dav baroclinic head',
    'unit': 'm',
    'filename': 'BaroHead2d',
    }
fieldMetadata['gjvAlphaH3d'] = {
    'name': 'GJV Parameter h',
    'shortname': 'GJV Parameter h',
    'unit': '',
    'filename': 'GJVParamH',
    }
fieldMetadata['gjvAlphaV3d'] = {
    'name': 'GJV Parameter v',
    'shortname': 'GJV Parameter v',
    'unit': '',
    'filename': 'GJVParamV',
    }
fieldMetadata['smagViscosity'] = {
    'name': 'Smagorinsky viscosity',
    'shortname': 'Smagorinsky viscosity',
    'unit': 'm2 s-1',
    'filename': 'SmagViscosity3d',
    }
fieldMetadata['saltJumpDiff'] = {
    'name': 'Salt Jump Diffusivity',
    'shortname': 'Salt Jump Diffusivity',
    'unit': 'm2 s-1',
    'filename': 'SaltJumpDiff3d',
    }
fieldMetadata['maxHDiffusivity'] = {
    'name': 'Maximum stable horizontal diffusivity',
    'shortname': 'Maximum horizontal diffusivity',
    'unit': 'm2 s-1',
    'filename': 'MaxHDiffusivity3d',
    }
fieldMetadata['vElemSize3d'] = {
    'name': 'Element size in vertical direction',
    'shortname': 'Vertical element size',
    'unit': 'm',
    'filename': 'VElemSize3d',
    }
fieldMetadata['vElemSize2d'] = {
    'name': 'Element size in vertical direction',
    'shortname': 'Vertical element size',
    'unit': 'm',
    'filename': 'VElemSize2d',
    }
fieldMetadata['hElemSize3d'] = {
    'name': 'Element size in horizontal direction',
    'shortname': 'Horizontal element size',
    'unit': 'm',
    'filename': 'hElemSize3d',
    }
fieldMetadata['hElemSize2d'] = {
    'name': 'Element size in horizontal direction',
    'shortname': 'Horizontal element size',
    'unit': 'm',
    'filename': 'hElemSize2d',
    }
fieldMetadata['coriolis2d'] = {
    'name': 'Coriolis parameter',
    'shortname': 'Coriolis parameter',
    'unit': 's-1',
    'filename': 'coriolis2d',
    }
fieldMetadata['coriolis3d'] = {
    'name': 'Coriolis parameter',
    'shortname': 'Coriolis parameter',
    'unit': 's-1',
    'filename': 'coriolis3d',
    }
fieldMetadata['windStress3d'] = {
    'name': 'Wind stress',
    'shortname': 'Wind stress',
    'unit': 'Pa',
    'filename': 'windStress3d',
    }
