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


class naiveFieldExporter(object):
    """
    Exports function nodal values to disk in numpy binary format.

    Works for simple Pn and PnDG fields.
    """
    def __init__(self, function, outputDir, filename_prefix):
        """
        Create exporter object for given function.

        Parameters
        ----------
        func : Function
            function to export
        outputDir : string
            directory where outputs will be stored
        filename : string
            prefix of output filename. Filename is prefix_nnnnn.npy
            where nnnn is the export number.
        """
        self.function = function
        self.outputDir = createDirectory(outputDir)
        self.filename = filename_prefix
        self.function_space = function.function_space()

        # create mappings between local/global node indices
        # construct global node ordering based on (x,y) coords
        dim = self.function_space.dim
        fs = self.function_space
        x_func = Function(fs).interpolate(Expression(['x[0]']*dim))
        y_func = Function(fs).interpolate(Expression(['x[1]']*dim))
        z_func = Function(fs).interpolate(Expression(['x[2]']*dim))
        if dim > 1:
            rankNodeX = comm.gather(x_func.dat.data[:, 0], root=0)
            rankNodeY = comm.gather(y_func.dat.data[:, 0], root=0)
            rankNodeZ = comm.gather(y_func.dat.data[:, 0], root=0)
        else:
            rankNodeX = comm.gather(x_func.dat.data, root=0)
            rankNodeY = comm.gather(y_func.dat.data, root=0)
            rankNodeZ = comm.gather(y_func.dat.data, root=0)

        # mapping of local dof to global array
        self.localToGlobal = []
        self.globalToLocal = []
        if commrank == 0:
            # construct a single array for all the nodes
            x = np.concatenate(tuple(rankNodeX), axis=0)
            y = np.concatenate(tuple(rankNodeY), axis=0)
            z = np.concatenate(tuple(rankNodeY), axis=0)
            self.nGlobalNodes = len(x)
            # round nodes coords to 1 m resolution
            # FIXME why is this needed?
            x = np.round(x, 1)
            y = np.round(y, 1)
            z = np.round(z, 1)
            # construct global invariant node ordering
            # nodes are sorted first by z then y then x
            sorted_ix = np.lexsort((x, y, z))
            # construct inverse map global_ix -> sorted_ix
            sorted_ix_inv = np.ones_like(sorted_ix)
            sorted_ix_inv[sorted_ix] = np.arange(len(sorted_ix))
            # construct maps between local node numbering and global
            # invariant numbering
            # globalToLocal[iRank][globalIx] - returns local node index
            #                                  for process iRank
            # localToGlobal[iRank[localIx]   - returns global node index
            #                                  for process iRank
            offset = 0
            for i in xrange(comm.size):
                nNodes = len(rankNodeX[i])
                ix = sorted_ix[offset:offset+nNodes]
                self.globalToLocal.append(ix)
                ix_inv = sorted_ix_inv[offset:offset+nNodes]
                self.localToGlobal.append(ix_inv)
                offset += nNodes

    def genFilename(self, iExport):
        filename = '{0:s}_{1:05d}.npy'.format(self.filename, iExport)
        return os.path.join(self.outputDir, filename)

    def export(self, iExport, function):
        """
        Exports the given function to disk.
        """
        assert function.function_space() == self.function_space,\
            'Function space does not match'
        dim = self.function_space.dim
        localData = comm.gather(function.dat.data, root=0)
        if commrank == 0:
            globalData = np.zeros((self.nGlobalNodes, dim))
            for i in xrange(comm.size):
                vals = localData[i] if dim > 1 else localData[i][:, None]
                globalData[self.localToGlobal[i], :] = vals
            filename = self.genFilename(iExport)
            print 'saving state to', filename
            np.save(filename, globalData)

    def load(self, iExport, function):
        """
        Loads nodal values from disk and assigns to the given function.
        """
        assert function.function_space() == self.function_space,\
            'Function space does not match'
        dim = self.function_space.dim
        if commrank == 0:
            filename = self.genFilename(iExport)
            print 'loading state from', filename
            globalData = np.load(filename)
            assert globalData.shape[0] == self.nGlobalNodes,\
                'Number of nodes does not match: {0:d} != {1:d}'.format(
                    self.nGlobalNodes, globalData.shape[0])
            localData = []
            for i in xrange(comm.size):
                localData.append(globalData[self.localToGlobal[i], :])
        else:
            localData = None
        data = comm.scatter(localData, root=0)
        if dim == 1:
            data = data.ravel()
        function.dat.data[:] = data
