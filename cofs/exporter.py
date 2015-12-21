"""
Routines for handling file exports.

Tuomas Karna 2015-07-06
"""
from utility import *


class exporterBase(object):
    """
    Base class for exporter objects.
    """
    def __init__(self, filename, outputDir, nextExportIx=0, verbose=False):
        self.filename = filename
        self.outputDir = createDirectory(outputDir)
        self.verbose = verbose
        # keeps track of export numbers
        self.nextExportIx = nextExportIx

    def setNextExportIx(self, nextExportIx):
        """Sets the index of next export"""
        self.nextExportIx = nextExportIx

    def export(self, function):
        raise NotImplementedError('This method must be implemented in the derived class')


class vtkExporter(exporterBase):
    """Class that handles Paraview file exports."""
    def __init__(self, fs_visu, func_name, outputDir, filename,
                 nextExportIx=0, verbose=False):
        """Creates exporter object.
        fs_visu:  function space where data will be projected before exporting
        func_name: name of the function
        outputDir: output directory
        filename: name of the pvd file
        """
        super(vtkExporter, self).__init__(filename, outputDir, nextExportIx,
                                          verbose)
        self.fs_visu = fs_visu
        self.func_name = func_name
        suffix = '.pvd'
        # append suffix if missing
        if (len(filename) < len(suffix)+1 or filename[:len(suffix)] != suffix):
            self.filename += suffix
        self.proj_func = tmpFunctionCache.get(self.fs_visu)
        self.outfile = File(os.path.join(outputDir, self.filename))
        self.P = {}

    def setNextExportIx(self, nextExportIx):
        """Sets the index of next export"""
        # NOTE vtk io objects store current export index not next
        super(vtkExporter, self).setNextExportIx(nextExportIx - 1)

    def export(self, function):
        """Exports given function to disk."""
        if function not in self.P:
            self.P[function] = projector(function, self.proj_func)
        self.P[function].project()
        # ensure correct output function name
        old_name = self.proj_func.name()
        self.proj_func.rename(name=self.func_name)
        # self.proj_func.project(function)  # NOTE this allocates a function
        self.outfile << (self.proj_func, self.nextExportIx)
        self.nextExportIx += 1
        # restore old name
        self.proj_func.rename(name=old_name)


class naiveFieldExporter(exporterBase):
    """
    Exports function nodal values to disk in numpy binary format.

    Works for simple Pn and PnDG fields.
    """
    def __init__(self, function_space, outputDir, filename_prefix,
                 nextExportIx=0, verbose=False):
        """
        Create exporter object for given function.

        Parameters
        ----------
        function_space : FunctionSpace
            function space where the exported functions belong
        outputDir : string
            directory where outputs will be stored
        filename : string
            prefix of output filename. Filename is prefix_nnnnn.npy
            where nnnn is the export number.
        """
        super(naiveFieldExporter, self).__init__(filename_prefix, outputDir,
                                                 nextExportIx, verbose)
        self.function_space = function_space

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
                if dim > 1:
                    globalData[self.localToGlobal[i], :] = localData[i]
                else:
                    globalData[self.localToGlobal[i], 0] = localData[i]

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


class hdf5Exporter(exporterBase):
    """Stores fields in disk in native discretization using HDF5 containers"""
    def __init__(self, function_space, outputDir, filename_prefix,
                 nextExportIx=0, verbose=False):
        """
        Create exporter object for given function.

        Parameters
        ----------
        function_space : FunctionSpace
            function space where the exported functions belong
        outputDir : string
            directory where outputs will be stored
        filename : string
            prefix of output filename. Filename is prefix_nnnnn.h5
            where nnnnn is the export number.
        """
        super(hdf5Exporter, self).__init__(filename_prefix, outputDir,
                                           nextExportIx, verbose)
        self.function_space = function_space

    def setNextExportIx(self, nextExportIx):
        """Sets the index of next export"""
        self.nextExportIx = nextExportIx

    def genFilename(self, iExport):
        filename = '{0:s}_{1:05d}.h5'.format(self.filename, iExport)
        return os.path.join(self.outputDir, filename)

    def exportAsIndex(self, iExport, function):
        """
        Exports the given function to disk using the specified export
        index number.
        """
        assert function.function_space() == self.function_space,\
            'Function space does not match'
        filename = self.genFilename(iExport)
        if self.verbose:
            print('saving {:} state to {:}'.format(function.name, filename))
        with DumbCheckpoint(filename, mode=FILE_CREATE) as f:
            f.store(function)
        self.nextExportIx = iExport + 1

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
        filename = self.genFilename(iExport)
        if self.verbose:
            print('loading {:} state from {:}'.format(function.name, filename))
        with DumbCheckpoint(filename, mode=FILE_READ) as f:
            f.load(function)


class exportManager(object):
    """Handles a list of file exporter objects"""

    def __init__(self, outputDir, fieldsToExport, functions,
                 visualizationSpaces, fieldMetadata,
                 exportType='vtk', nextExportIx=0,
                 verbose=False):
        self.outputDir = outputDir
        self.fieldsToExport = fieldsToExport
        self.functions = functions
        self.fieldMetadata = fieldMetadata
        self.visualizationSpaces = visualizationSpaces
        self.verbose = verbose
        # for each field create an exporter
        self.exporters = {}
        for key in fieldsToExport:
            shortname = self.fieldMetadata[key]['shortname']
            fn = self.fieldMetadata[key]['filename']
            field = self.functions.get(key)
            if field is not None and isinstance(field, Function):
                native_space = field.function_space()
                visu_space = self.visualizationSpaces.get(native_space)
                if visu_space is None:
                    raise Exception('missing visualization space for: '+key)
                if exportType.lower() == 'vtk':
                    self.exporters[key] = vtkExporter(visu_space, shortname,
                                                      outputDir, fn,
                                                      nextExportIx=nextExportIx)
                elif exportType.lower() == 'numpy':
                    self.exporters[key] = naiveFieldExporter(native_space,
                                                             outputDir, fn,
                                                             nextExportIx=nextExportIx)
                elif exportType.lower() == 'hdf5':
                    self.exporters[key] = hdf5Exporter(native_space,
                                                       outputDir, fn,
                                                       nextExportIx=nextExportIx)

    def setNextExportIx(self, nextExportIx):
        """Sets the correct export index to all child exporters"""
        for k in self.exporters:
            self.exporters[k].setNextExportIx(nextExportIx)

    def export(self):
        if self.verbose and commrank == 0:
            sys.stdout.write('Exporting: ')
        for key in self.exporters:
            field = self.functions[key]
            if field is not None:
                if self.verbose and commrank == 0:
                    sys.stdout.write(key+' ')
                    sys.stdout.flush()
                self.exporters[key].export(field)
        if self.verbose and commrank == 0:
            sys.stdout.write('\n')
            sys.stdout.flush()

    def exportBathymetry(self, bathymetry_2d):
        bathfile = File(os.path.join(self.outputDir, 'bath.pvd'))
        bathfile << bathymetry_2d
