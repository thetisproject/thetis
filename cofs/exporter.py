"""
Routines for handling file exports.

Tuomas Karna 2015-07-06
"""
from utility import *


class exporter(object):
    """Class that handles Paraview file exports."""
    def __init__(self, fs_visu, func_name, outputDir, filename):
        """Creates exporter object.
        fs_visu:  function space where data will be projected before exporting
        func_name: name of the function
        outputDir: output directory
        filename: name of the pvd file
        """
        self.fs_visu = fs_visu
        self.outfunc = Function(self.fs_visu, name=func_name)
        self.outfile = File(os.path.join(outputDir, filename))
        self.P = {}

    def export(self, function):
        """Exports given function to disk."""
        if function not in self.P:
            self.P[function] = projector(function, self.outfunc)
        self.P[function].project()
        # self.outfunc.project(function)  # NOTE this allocates a function
        self.outfile << self.outfunc


class exportManager(object):
    """Handles a list of file exporter objects"""
    # TODO remove name from the list, use the name of the funct
    # maps each fieldname to long name and filename
    exportRules = {
        'uv2d': {'name': 'Depth averaged velocity',
                 'file': 'Velocity2d.pvd'},
        'elev2d': {'name': 'Elevation',
                   'file': 'Elevation2d.pvd'},
        'elev3d': {'name': 'Elevation',
                   'file': 'Elevation3d.pvd'},
        'uv3d': {'name': 'Velocity',
                 'file': 'Velocity3d.pvd'},
        'w3d': {'name': 'V.Velocity',
                'file': 'VertVelo3d.pvd'},
        'w3d_mesh': {'name': 'Mesh Velocity',
                     'file': 'MeshVelo3d.pvd'},
        'salt3d': {'name': 'Salinity',
                   'file': 'Salinity3d.pvd'},
        'uv2d_dav': {'name': 'Depth Averaged Velocity',
                     'file': 'DAVelocity2d.pvd'},
        'uv3d_dav': {'name': 'Depth Averaged Velocity',
                     'file': 'DAVelocity3d.pvd'},
        'uv2d_bot': {'name': 'Bottom Velocity',
                     'file': 'BotVelocity2d.pvd'},
        'nuv3d': {'name': 'Vertical Viscosity',
                  'file': 'Viscosity3d.pvd'},
        'barohead3d': {'name': 'Baroclinic head',
                       'file': 'Barohead3d.pvd'},
        'barohead2d': {'name': 'Dav baroclinic head',
                       'file': 'Barohead2d.pvd'},
        'gjvAlphaH3d': {'name': 'GJV Parameter h',
                        'file': 'GJVParamH.pvd'},
        'gjvAlphaV3d': {'name': 'GJV Parameter v',
                        'file': 'GJVParamV.pvd'},
        'smagViscosity': {'name': 'Smagorinsky viscosity',
                          'file': 'SmagViscosity3d.pvd'},
        'saltJumpDiff': {'name': 'Salt Jump Diffusivity',
                         'file': 'SaltJumpDiff3d.pvd'},
        }

    def __init__(self, outputDir, fieldsToExport, exportFunctions,
                 verbose=False):
        self.outputDir = outputDir
        self.fieldsToExport = fieldsToExport
        self.exportFunctions = exportFunctions
        self.verbose = verbose
        # for each field create an exporter
        self.exporters = {}
        for key in fieldsToExport:
            name = self.exportRules[key]['name']
            fn = self.exportRules[key]['file']
            space = self.exportFunctions[key][1]
            self.exporters[key] = exporter(space, name, outputDir, fn)

    def export(self):
        if self.verbose and commrank == 0:
            sys.stdout.write('Exporting: ')
        for key in self.exporters:
            field = self.exportFunctions[key][0]
            if field is not None:
                if self.verbose and commrank == 0:
                    sys.stdout.write(key+' ')
                    sys.stdout.flush()
                self.exporters[key].export(field)
        if self.verbose and commrank == 0:
            sys.stdout.write('\n')
            sys.stdout.flush()

    def exportBathymetry(self, bathymetry2d):
        bathfile = File(os.path.join(self.outputDir, 'bath.pvd'))
        bathfile << bathymetry2d
