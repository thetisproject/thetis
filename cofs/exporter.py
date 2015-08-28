"""
Routines for handling file exports.

Tuomas Karna 2015-07-06
"""
from utility import *


class exporterBase(object):
    """
    Base class for exporter objects.
    """
    def export(self, function):
        raise NotImplementedError('This method must be implemented in the derived class')


class exporter(exporterBase):
    """Class that handles Paraview file exports."""
    def __init__(self, fs_visu, func_name, outputDir, filename):
        """Creates exporter object.
        fs_visu:  function space where data will be projected before exporting
        func_name: name of the function
        outputDir: output directory
        filename: name of the pvd file
        """
        self.fs_visu = fs_visu
        self.filename = filename
        suffix = '.pvd'
        # append suffix if missing
        if (len(filename) < len(suffix)+1 or filename[:len(suffix)] != suffix):
            self.filename += suffix
        self.outfunc = Function(self.fs_visu, name=func_name)
        self.outfile = File(os.path.join(outputDir, self.filename))
        self.P = {}

    def export(self, function):
        """Exports given function to disk."""
        if function not in self.P:
            self.P[function] = projector(function, self.outfunc)
        self.P[function].project()
        # self.outfunc.project(function)  # NOTE this allocates a function
        self.outfile << self.outfunc


class naiveFieldExporter(exporterBase):
    """
    Exports function nodal values to disk in numpy binary format.

    Works for simple Pn and PnDG fields.
    """
    def __init__(self, function_space, outputDir, filename_prefix,
                 verbose=False):
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
        self.outputDir = createDirectory(outputDir)
        self.filename = filename_prefix
        self.function_space = function_space
        self.verbose = verbose
        # keeps track of export numbers
        self.nextExportIx = 0

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
            rankNodeZ = comm.gather(z_func.dat.data[:, 0], root=0)
        else:
            rankNodeX = comm.gather(x_func.dat.data, root=0)
            rankNodeY = comm.gather(y_func.dat.data, root=0)
            rankNodeZ = comm.gather(z_func.dat.data, root=0)

        # mapping of local dof to global array
        self.localToGlobal = []
        self.globalToLocal = []
        if commrank == 0:
            # construct a single array for all the nodes
            x = np.concatenate(tuple(rankNodeX), axis=0)
            y = np.concatenate(tuple(rankNodeY), axis=0)
            z = np.concatenate(tuple(rankNodeZ), axis=0)
            # round coordinates to avoid noise affecting sort
            x = np.round(x, decimals=1)
            y = np.round(y, decimals=1)
            z = np.round(z, decimals=5)
            self.nGlobalNodes = len(x)
            # construct global invariant node ordering
            # nodes are sorted first by z then y then x
            sorted_ix = np.lexsort((x, y, z))
            # construct inverse map global_ix -> sorted_ix
            sorted_ix_inv = np.ones_like(sorted_ix)
            sorted_ix_inv[sorted_ix] = np.arange(len(sorted_ix))
            # store global coordinates
            self.xyz = np.vstack((x, y, z))[:, sorted_ix].T
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

        # construct local element connectivity array
        if self.function_space.extruded:
            ufl_elem = self.function_space.ufl_element()
            if ufl_elem.family() != 'OuterProductElement':
                raise NotImplementedError('Only OuterProductElement is supported')
            # extruded mesh generate connectivity for all layers
            nLayers = self.function_space.mesh().layers - 1  # element layers
            # connectivity for first layer
            surfConn = self.function_space.cell_node_map().values
            nSurfElem, nElemNode = surfConn.shape
            if ufl_elem._B.family() == 'Lagrange':
                layer_node_offset = 1
            elif ufl_elem._B.family() == 'Discontinuous Lagrange':
                layer_node_offset = nElemNode
            else:
                raise NotImplementedError('Unsupported vertical space')
            # construct element table for all layers
            conn = np.zeros((nLayers*nSurfElem, nElemNode), dtype=np.int32)
            for i in range(nLayers):
                o = i*layer_node_offset
                conn[i*nSurfElem:(i+1)*nSurfElem, :] = surfConn + o
        else:
            # 2D mesh
            conn = self.function_space.cell_node_map().values
        # construct global connectivity array
        # NOTE connectivity table is not unique
        self.connectivity = []
        rankConn = comm.gather(conn, root=0)
        if commrank == 0:
            for i in xrange(comm.size):
                # convert each connectivity array to global index
                rankConn[i] = self.localToGlobal[i][rankConn[i]]
            # concatenate to single array
            self.connectivity = np.concatenate(tuple(rankConn), axis=0)

    def genFilename(self, iExport):
        filename = '{0:s}_{1:05d}.npz'.format(self.filename, iExport)
        return os.path.join(self.outputDir, filename)

    def exportAsIndex(self, iExport, function):
        """
        Exports the given function to disk using the specified export
        index number.
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
            if self.verbose:
                print 'saving state to', filename
            np.savez(filename, xyz=self.xyz, connectivity=self.connectivity,
                     data=globalData)
        self.nextExportIx = iExport+1

    def export(self, function):
        """
        Exports the given function to disk.
        Increments previous export index by 1.
        """
        self.exportAsIndex(self.nextExportIx, function)

    def load(self, iExport, function):
        """
        Loads nodal values from disk and assigns to the given function.
        """
        assert function.function_space() == self.function_space,\
            'Function space does not match'
        dim = self.function_space.dim
        if commrank == 0:
            filename = self.genFilename(iExport)
            if self.verbose:
                print 'loading state from', filename
            npzFile = np.load(filename)
            globalData = npzFile['data']
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


class exportManager(object):
    """Handles a list of file exporter objects"""
    # TODO remove name from the list, use the name of the funct
    # maps each fieldname to long name and filename
    exportRules = {
        'uv2d': {'name': 'Depth averaged velocity',
                 'file': 'Velocity2d'},
        'elev2d': {'name': 'Elevation',
                   'file': 'Elevation2d'},
        'elev3d': {'name': 'Elevation',
                   'file': 'Elevation3d'},
        'uv3d': {'name': 'Velocity',
                 'file': 'Velocity3d'},
        'w3d': {'name': 'V.Velocity',
                'file': 'VertVelo3d'},
        'w3d_mesh': {'name': 'Mesh Velocity',
                     'file': 'MeshVelo3d'},
        'salt3d': {'name': 'Salinity',
                   'file': 'Salinity3d'},
        'uv2d_dav': {'name': 'Depth Averaged Velocity',
                     'file': 'DAVelocity2d'},
        'uv3d_dav': {'name': 'Depth Averaged Velocity',
                     'file': 'DAVelocity3d'},
        'uv2d_bot': {'name': 'Bottom Velocity',
                     'file': 'BotVelocity2d'},
        'nuv3d': {'name': 'Vertical Viscosity',
                  'file': 'Viscosity3d'},
        'barohead3d': {'name': 'Baroclinic head',
                       'file': 'Barohead3d'},
        'barohead2d': {'name': 'Dav baroclinic head',
                       'file': 'Barohead2d'},
        'gjvAlphaH3d': {'name': 'GJV Parameter h',
                        'file': 'GJVParamH'},
        'gjvAlphaV3d': {'name': 'GJV Parameter v',
                        'file': 'GJVParamV'},
        'smagViscosity': {'name': 'Smagorinsky viscosity',
                          'file': 'SmagViscosity3d'},
        'saltJumpDiff': {'name': 'Salt Jump Diffusivity',
                         'file': 'SaltJumpDiff3d'},
        }

    def __init__(self, outputDir, fieldsToExport, exportFunctions,
                 exportType='vtk', verbose=False):
        self.outputDir = outputDir
        self.fieldsToExport = fieldsToExport
        self.exportFunctions = exportFunctions
        self.verbose = verbose
        # for each field create an exporter
        self.exporters = {}
        for key in fieldsToExport:
            name = self.exportRules[key]['name']
            fn = self.exportRules[key]['file']
            field = self.exportFunctions[key][0]
            if field is not None:
                visu_space = self.exportFunctions[key][1]
                native_space = field.function_space()
                if exportType == 'vtk':
                    self.exporters[key] = exporter(visu_space, name,
                                                   outputDir, fn)
                elif exportType == 'numpy':
                    self.exporters[key] = naiveFieldExporter(native_space,
                                                             outputDir, fn)

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
