"""
Routines for handling file exports.

Tuomas Karna 2015-07-06
"""
from utility import *
import ufl


class ExporterBase(object):
    """
    Base class for exporter objects.
    """
    def __init__(self, filename, outputdir, next_export_ix=0, verbose=False):
        self.filename = filename
        self.outputdir = create_directory(outputdir)
        self.verbose = verbose
        # keeps track of export numbers
        self.next_export_ix = next_export_ix

    def set_next_export_ix(self, next_export_ix):
        """Sets the index of next export"""
        self.next_export_ix = next_export_ix

    def export(self, function):
        raise NotImplementedError('This method must be implemented in the derived class')


class VTKExporter(ExporterBase):
    """Class that handles Paraview file exports."""
    def __init__(self, fs_visu, func_name, outputdir, filename,
                 next_export_ix=0, verbose=False):
        """Creates exporter object.
        fs_visu:  function space where data will be projected before exporting
        func_name: name of the function
        outputdir: output directory
        filename: name of the pvd file
        """
        super(VTKExporter, self).__init__(filename, outputdir, next_export_ix,
                                          verbose)
        self.fs_visu = fs_visu
        self.func_name = func_name
        suffix = '.pvd'
        # append suffix if missing
        if (len(filename) < len(suffix)+1 or filename[:len(suffix)] != suffix):
            self.filename += suffix
        self.outfile = File(os.path.join(outputdir, self.filename))
        self.P = {}

    def set_next_export_ix(self, next_export_ix):
        """Sets the index of next export"""
        # NOTE vtk io objects store current export index not next
        super(VTKExporter, self).set_next_export_ix(next_export_ix - 1)

    def export(self, function):
        """Exports given function to disk."""
        assert self.fs_visu.max_work_functions == 1
        tmp_proj_func = self.fs_visu.get_work_function()
        if function not in self.P:
            # NOTE tmp function must be invariant as the projector is built only once
            self.P[function] = Projector(function, tmp_proj_func)
        self.P[function].project()
        # ensure correct output function name
        old_name = tmp_proj_func.name()
        tmp_proj_func.rename(name=self.func_name)
        self.outfile.write(tmp_proj_func, time=self.next_export_ix)
        self.next_export_ix += 1
        # restore old name
        tmp_proj_func.rename(name=old_name)
        self.fs_visu.restore_work_function(tmp_proj_func)


class NaiveFieldExporter(ExporterBase):
    """
    Exports function nodal values to disk in numpy binary format.

    Works for simple Pn and PnDG fields.
    """
    def __init__(self, function_space, outputdir, filename_prefix,
                 next_export_ix=0, verbose=False):
        """
        Create exporter object for given function.

        Parameters
        ----------
        function_space : FunctionSpace
            function space where the exported functions belong
        outputdir : string
            directory where outputs will be stored
        filename : string
            prefix of output filename. Filename is prefix_nnnnn.npy
            where nnnn is the export number.
        """
        super(NaiveFieldExporter, self).__init__(filename_prefix, outputdir,
                                                 next_export_ix, verbose)
        self.function_space = function_space

        # create mappings between local/global node indices
        # construct global node ordering based on (x,y) coords
        dim = self.function_space.dim
        fs = self.function_space
        x_func = Function(fs).interpolate(Expression(['x[0]']*dim))
        y_func = Function(fs).interpolate(Expression(['x[1]']*dim))
        z_func = Function(fs).interpolate(Expression(['x[2]']*dim))
        if dim > 1:
            rank_node_x = comm.gather(x_func.dat.data[:, 0], root=0)
            rank_node_y = comm.gather(y_func.dat.data[:, 0], root=0)
            rank_node_z = comm.gather(z_func.dat.data[:, 0], root=0)
        else:
            rank_node_x = comm.gather(x_func.dat.data, root=0)
            rank_node_y = comm.gather(y_func.dat.data, root=0)
            rank_node_z = comm.gather(z_func.dat.data, root=0)

        # mapping of local dof to global array
        self.local_to_global = []
        self.global_to_local = []
        if commrank == 0:
            # construct a single array for all the nodes
            x = np.concatenate(tuple(rank_node_x), axis=0)
            y = np.concatenate(tuple(rank_node_y), axis=0)
            z = np.concatenate(tuple(rank_node_z), axis=0)
            # round coordinates to avoid noise affecting sort
            x = np.round(x, decimals=1)
            y = np.round(y, decimals=1)
            z = np.round(z, decimals=5)
            self.n_global_nodes = len(x)
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
            # global_to_local[i_rank][global_ix] - returns local node index
            #                                  for process i_rank
            # local_to_global[i_rank[local_ix]   - returns global node index
            #                                  for process i_rank
            offset = 0
            for i in xrange(comm.size):
                n_nodes = len(rank_node_x[i])
                ix = sorted_ix[offset:offset+n_nodes]
                self.global_to_local.append(ix)
                ix_inv = sorted_ix_inv[offset:offset+n_nodes]
                self.local_to_global.append(ix_inv)
                offset += n_nodes

        # construct local element connectivity array
        if self.function_space.extruded:
            ufl_elem = self.function_space.ufl_element()
            if ufl_elem.family() != 'TensorProductElement':
                raise NotImplementedError('Only TensorProductElement is supported')
            # extruded mesh generate connectivity for all layers
            n_layers = self.function_space.mesh().layers - 1  # element layers
            # connectivity for first layer
            surf_conn = self.function_space.cell_node_map().values
            n_surf_elem, n_elem_node = surf_conn.shape

            if ufl_elem.num_sub_elements() > 0:
                # VectorElement case
                assert isinstance(ufl_elem, ufl.VectorElement)
                ufl_elem = ufl_elem.sub_elements()[0]
            if ufl_elem._B.family() == 'Lagrange':
                layer_node_offset = 1
            elif ufl_elem._B.family() == 'Discontinuous Lagrange':
                layer_node_offset = n_elem_node
            else:
                raise NotImplementedError('Unsupported vertical space')
            # construct element table for all layers
            conn = np.zeros((n_layers*n_surf_elem, n_elem_node), dtype=np.int32)
            for i in range(n_layers):
                o = i*layer_node_offset
                conn[i*n_surf_elem:(i+1)*n_surf_elem, :] = surf_conn + o
        else:
            # 2D mesh
            conn = self.function_space.cell_node_map().values
        # construct global connectivity array
        # NOTE connectivity table is not unique
        self.connectivity = []
        rank_conn = comm.gather(conn, root=0)
        if commrank == 0:
            for i in xrange(comm.size):
                # convert each connectivity array to global index
                rank_conn[i] = self.local_to_global[i][rank_conn[i]]
            # concatenate to single array
            self.connectivity = np.concatenate(tuple(rank_conn), axis=0)

    def gen_filename(self, iexport):
        filename = '{0:s}_{1:05d}.npz'.format(self.filename, iexport)
        return os.path.join(self.outputdir, filename)

    def export_as_indexx(self, iexport, function):
        """
        Exports the given function to disk using the specified export
        index number.
        """
        assert function.function_space() == self.function_space,\
            'Function space does not match'
        dim = self.function_space.dim
        local_data = comm.gather(function.dat.data, root=0)
        if commrank == 0:
            global_data = np.zeros((self.n_global_nodes, dim))
            for i in xrange(comm.size):
                if dim > 1:
                    global_data[self.local_to_global[i], :] = local_data[i]
                else:
                    global_data[self.local_to_global[i], 0] = local_data[i]

            filename = self.gen_filename(iexport)
            if self.verbose:
                print 'saving state to', filename
            np.savez(filename, xyz=self.xyz, connectivity=self.connectivity,
                     data=global_data)
        self.next_export_ix = iexport+1

    def export(self, function):
        """
        Exports the given function to disk.
        Increments previous export index by 1.
        """
        self.export_as_indexx(self.next_export_ix, function)

    def load(self, iexport, function):
        """
        Loads nodal values from disk and assigns to the given function.
        """
        assert function.function_space() == self.function_space,\
            'Function space does not match'
        dim = self.function_space.dim
        if commrank == 0:
            filename = self.gen_filename(iexport)
            if self.verbose:
                print 'loading state from', filename
            npzfile = np.load(filename)
            global_data = npzfile['data']
            assert global_data.shape[0] == self.n_global_nodes,\
                'Number of nodes does not match: {0:d} != {1:d}'.format(
                    self.n_global_nodes, global_data.shape[0])
            local_data = []
            for i in xrange(comm.size):
                local_data.append(global_data[self.local_to_global[i], :])
        else:
            local_data = None
        data = comm.scatter(local_data, root=0)
        if dim == 1:
            data = data.ravel()
        function.dat.data[:] = data


class HDF5Exporter(ExporterBase):
    """Stores fields in disk in native discretization using HDF5 containers"""
    def __init__(self, function_space, outputdir, filename_prefix,
                 next_export_ix=0, verbose=False):
        """
        Create exporter object for given function.

        Parameters
        ----------
        function_space : FunctionSpace
            function space where the exported functions belong
        outputdir : string
            directory where outputs will be stored
        filename : string
            prefix of output filename. Filename is prefix_nnnnn.h5
            where nnnnn is the export number.
        """
        super(HDF5Exporter, self).__init__(filename_prefix, outputdir,
                                           next_export_ix, verbose)
        self.function_space = function_space

    def set_next_export_ix(self, next_export_ix):
        """Sets the index of next export"""
        self.next_export_ix = next_export_ix

    def gen_filename(self, iexport):
        filename = '{0:s}_{1:05d}.h5'.format(self.filename, iexport)
        return os.path.join(self.outputdir, filename)

    def export_as_indexx(self, iexport, function):
        """
        Exports the given function to disk using the specified export
        index number.
        """
        assert function.function_space() == self.function_space,\
            'Function space does not match'
        filename = self.gen_filename(iexport)
        if self.verbose:
            print('saving {:} state to {:}'.format(function.name, filename))
        with DumbCheckpoint(filename, mode=FILE_CREATE) as f:
            f.store(function)
        self.next_export_ix = iexport + 1

    def export(self, function):
        """
        Exports the given function to disk.
        Increments previous export index by 1.
        """
        self.export_as_indexx(self.next_export_ix, function)

    def load(self, iexport, function):
        """
        Loads nodal values from disk and assigns to the given function.
        """
        assert function.function_space() == self.function_space,\
            'Function space does not match'
        filename = self.gen_filename(iexport)
        if self.verbose:
            print('loading {:} state from {:}'.format(function.name, filename))
        with DumbCheckpoint(filename, mode=FILE_READ) as f:
            f.load(function)


class ExportManager(object):
    """Handles a list of file exporter objects"""

    def __init__(self, outputdir, fields_to_export, functions,
                 visualization_spaces, field_metadata,
                 export_type='vtk', next_export_ix=0,
                 verbose=False):
        self.outputdir = outputdir
        self.fields_to_export = fields_to_export
        self.functions = functions
        self.field_metadata = field_metadata
        self.visualization_spaces = visualization_spaces
        self.verbose = verbose
        # for each field create an exporter
        self.exporters = {}
        for key in fields_to_export:
            shortname = self.field_metadata[key]['shortname']
            fn = self.field_metadata[key]['filename']
            field = self.functions.get(key)
            if field is not None and isinstance(field, Function):
                native_space = field.function_space()
                visu_space = self.visualization_spaces.get(native_space)
                if visu_space is None:
                    raise Exception('missing visualization space for: '+key)
                if export_type.lower() == 'vtk':
                    self.exporters[key] = VTKExporter(visu_space, shortname,
                                                      outputdir, fn,
                                                      next_export_ix=next_export_ix)
                elif export_type.lower() == 'numpy':
                    self.exporters[key] = NaiveFieldExporter(native_space,
                                                             outputdir, fn,
                                                             next_export_ix=next_export_ix)
                elif export_type.lower() == 'hdf5':
                    self.exporters[key] = HDF5Exporter(native_space,
                                                       outputdir, fn,
                                                       next_export_ix=next_export_ix)

    def set_next_export_ix(self, next_export_ix):
        """Sets the correct export index to all child exporters"""
        for k in self.exporters:
            self.exporters[k].set_next_export_ix(next_export_ix)

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

    def export_bathymetry(self, bathymetry_2d):
        bathfile = File(os.path.join(self.outputdir, 'bath.pvd'))
        bathfile.write(bathymetry_2d)
