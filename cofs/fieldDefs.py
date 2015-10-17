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
fieldname - description used in visualization etc
filename  - filename for output files
unit      - SI unit of the field
"""

fieldMetadata['bathymetry2d'] = {
    'name': 'Bathymetry',
    'fieldname': 'Bathymetry',
    'filename': 'bathymetry2d',
    'unit': 'm',
    }
fieldMetadata['bathymetry3d'] = {
    'name': 'Bathymetry',
    'fieldname': 'Bathymetry',
    'filename': 'bathymetry3d',
    'unit': 'm',
    }
fieldMetadata['z_coord3d'] = {
    'name': 'Mesh z coordinates',
    'fieldname': 'Z coordinates',
    'filename': 'ZCoord3d',
    'unit': 'm',
    }
fieldMetadata['z_bottom2d'] = {
    'name': 'Bottom cell z coordinates',
    'fieldname': 'Bottom cell z coordinates',
    'filename': 'ZBottom2d',
    'unit': 'm',
    }
fieldMetadata['z_coord_ref3d'] = {
    'name': 'Static mesh z coordinates',
    'fieldname': 'Z coordinates',
    'filename': 'ZCoordRef3d',
    'unit': 'm',
    }
fieldMetadata['bottom_drag2d'] = {
    'name': 'Bottom drag coefficient',
    'fieldname': 'Bottom drag coefficient',
    'filename': 'BottomDrag2d',
    'unit': '',
    }
fieldMetadata['bottom_drag3d'] = {
    'name': 'Bottom drag coefficient',
    'fieldname': 'Bottom drag coefficient',
    'filename': 'BottomDrag3d',
    'unit': '',
    }
fieldMetadata['uv2d'] = {
    'name': 'Depth averaged velocity',
    'fieldname': 'Depth averaged velocity',
    'filename': 'Velocity2d',
    'unit': 'm s-1',
    }
fieldMetadata['uvDav2d'] = {
    'name': 'Depth averaged velocity',
    'fieldname': 'Depth averaged velocity',
    'filename': 'DAVelocity2d',
    'unit': 'm s-1',
    }
fieldMetadata['uv_bottom2d'] = {
    'name': 'Bottom velocity',
    'fieldname': 'Bottom velocity',
    'filename': 'BottomVelo2d',
    'unit': 'm s-1',
    }
fieldMetadata['uv_bottom3d'] = {
    'name': 'Bottom velocity',
    'fieldname': 'Bottom velocity',
    'filename': 'BottomVelo3d',
    'unit': 'm s-1',
    }
fieldMetadata['uvDav3d'] = {
    'name': 'Depth averaged velocity',
    'fieldname': 'Depth averaged velocity',
    'filename': 'DAVelocity3d',
    'unit': 'm s-1',
    }
fieldMetadata['uv3d_mag'] = {
    'name': 'Magnitude of horizontal velocity',
    'fieldname': 'Velocity magnitude',
    'filename': 'VeloMag3d',
    'unit': 'm s-1',
    }
fieldMetadata['uv3d_P1'] = {
    'name': 'P1 projection of horizontal velocity',
    'fieldname': 'P1 Velocity',
    'filename': 'VeloCG3d',
    'unit': 'm s-1',
    }
fieldMetadata['uvBot2d'] = {
    'name': 'Bottom velocity',
    'fieldname': 'Bottom velocity',
    'filename': 'BotVelocity2d',
    'unit': 'm s-1',
    }
fieldMetadata['elev2d'] = {
    'name': 'Water elevation',
    'fieldname': 'Elevation',
    'filename': 'Elevation2d',
    'unit': 'm',
    }
fieldMetadata['elev3d'] = {
    'name': 'Water elevation',
    'fieldname': 'Elevation',
    'filename': 'Elevation3d',
    'unit': 'm',
    }
fieldMetadata['elev3dCG'] = {
    'name': 'Water elevation CG',
    'fieldname': 'Elevation',
    'filename': 'ElevationCG3d',
    'unit': 'm',
    }
fieldMetadata['uv3d'] = {
    'name': 'Horizontal velocity',
    'fieldname': 'Horizontal velocity',
    'filename': 'Velocity3d',
    'unit': 'm s-1',
    }
fieldMetadata['w3d'] = {
    'name': 'Vertical velocity',
    'fieldname': 'Vertical velocity',
    'filename': 'VertVelo3d',
    'unit': 'm s-1',
    }
fieldMetadata['wMesh3d'] = {
    'name': 'Mesh velocity',
    'fieldname': 'Mesh velocity',
    'filename': 'MeshVelo3d',
    'unit': 'm s-1',
    }
fieldMetadata['wMeshSurf3d'] = {
    'name': 'Surface mesh velocity',
    'fieldname': 'Surface mesh velocity',
    'filename': 'SurfMeshVelo3d',
    'unit': 'm s-1',
    }
fieldMetadata['wMeshSurf2d'] = {
    'name': 'Surface mesh velocity',
    'fieldname': 'Surface mesh velocity',
    'filename': 'SurfMeshVelo3d',
    'unit': 'm s-1',
    }
fieldMetadata['dwMeshDz3d'] = {
    'name': 'Vertical grad of mesh velocity',
    'fieldname': 'Vertical grad of mesh velocity',
    'filename': 'dMeshVeloDz3d',
    'unit': 's-1',
    }
fieldMetadata['salt3d'] = {
    'name': 'Water salinity',
    'fieldname': 'Salinity',
    'filename': 'Salinity3d',
    'unit': 'psu',
    }
fieldMetadata['parabNuv3d'] = {
    'name': 'Parabolic Viscosity',
    'fieldname': 'Parabolic Viscosity',
    'filename': 'ParabVisc3d',
    'unit': 'm2 s-1',
    }

fieldMetadata['eddyNuv3d'] = {
    'name': 'Eddy Viscosity',
    'fieldname': 'Eddy Viscosity',
    'filename': 'EddyVisc3d',
    'unit': 'm2 s-1',
    }
fieldMetadata['shearFreq3d'] = {
    'name': 'Vertical shear frequency squared',
    'fieldname': 'Vertical shear frequency squared',
    'filename': 'ShearFreq3d',
    'unit': 's-2',
    }
fieldMetadata['tke3d'] = {
    'name': 'Turbulent Kinetic Energy',
    'fieldname': 'Turbulent Kinetic Energy',
    'filename': 'TurbKEnergy3d',
    'unit': 'm2 s-2',
    }
fieldMetadata['psi3d'] = {
    'name': 'Turbulence psi variable',
    'fieldname': 'Turbulence psi variable',
    'filename': 'TurbPsi3d',
    'unit': '',
    }
fieldMetadata['eps3d'] = {
    'name': 'TKE dissipation rate',
    'fieldname': 'TKE dissipation rate',
    'filename': 'TurbEps3d',
    'unit': 'm2 s-2',
    }
fieldMetadata['len3d'] = {
    'name': 'Turbulent lenght scale',
    'fieldname': 'Turbulent lenght scale',
    'filename': 'TurbLen3d',
    'unit': 'm',
    }
fieldMetadata['barohead3d'] = {
    'name': 'Baroclinic head',
    'fieldname': 'Baroclinic head',
    'filename': 'Barohead3d',
    'unit': 'm',
    }
fieldMetadata['barohead2d'] = {
    'name': 'Dav baroclinic head',
    'fieldname': 'Dav baroclinic head',
    'filename': 'Barohead2d',
    'unit': 'm',
    }
fieldMetadata['gjvAlphaH3d'] = {
    'name': 'GJV Parameter h',
    'fieldname': 'GJV Parameter h',
    'filename': 'GJVParamH',
    'unit': '',
    }
fieldMetadata['gjvAlphaV3d'] = {
    'name': 'GJV Parameter v',
    'fieldname': 'GJV Parameter v',
    'filename': 'GJVParamV',
    'unit': '',
    }
fieldMetadata['smagViscosity'] = {
    'name': 'Smagorinsky viscosity',
    'fieldname': 'Smagorinsky viscosity',
    'filename': 'SmagViscosity3d',
    'unit': 'm2 s-1',
    }
fieldMetadata['saltJumpDiff'] = {
    'name': 'Salt Jump Diffusivity',
    'fieldname': 'Salt Jump Diffusivity',
    'filename': 'SaltJumpDiff3d',
    'unit': 'm2 s-1',
    }
fieldMetadata['maxHDiffusivity'] = {
    'name': 'Maximum stable horizontal diffusivity',
    'fieldname': 'Maximum horizontal diffusivity',
    'filename': 'MaxHDiffusivity3d',
    'unit': 'm2 s-1',
    }
fieldMetadata['vElemSize3d'] = {
    'name': 'Element size in vertical direction',
    'fieldname': 'Vertical element size',
    'filename': 'VElemSize3d',
    'unit': 'm',
    }
fieldMetadata['vElemSize2d'] = {
    'name': 'Element size in vertical direction',
    'fieldname': 'Vertical element size',
    'filename': 'VElemSize2d',
    'unit': 'm',
    }
fieldMetadata['hElemSize3d'] = {
    'name': 'Element size in horizontal direction',
    'fieldname': 'Horizontal element size',
    'filename': 'hElemSize3d',
    'unit': 'm',
    }
fieldMetadata['hElemSize2d'] = {
    'name': 'Element size in horizontal direction',
    'fieldname': 'Horizontal element size',
    'filename': 'hElemSize2d',
    'unit': 'm',
    }
