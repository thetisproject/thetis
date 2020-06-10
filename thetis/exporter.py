"""
Routines for handling file exports.
"""
from __future__ import absolute_import
from .utility import *
from firedrake.output import is_cg
from collections import OrderedDict
import itertools


def is_2d(fs):
    """Tests wether a function space is 2D or 3D"""
    return fs.mesh().geometric_dimension() == 2


def get_visu_space(fs):
    """
    Returns an appropriate VTK visualization space for a function space

    :arg fs: function space
    :return: function space for VTK visualization
    """
    is_vector = len(fs.ufl_element().value_shape()) == 1
    mesh = fs.mesh()
    family = 'Lagrange' if is_cg(fs) else 'Discontinuous Lagrange'
    if is_vector:
        dim = fs.ufl_element().value_shape()[0]
        visu_fs = get_functionspace(mesh, family, 1, family, 1,
                                    vector=True, dim=dim)
    else:
        visu_fs = get_functionspace(mesh, family, 1, family, 1)
    # make sure that you always get the same temp work function
    visu_fs.max_work_functions = 1
    return visu_fs


class ExporterBase(object):
    """
    Base class for exporter objects.
    """
    def __init__(self, filename, outputdir, next_export_ix=0, verbose=False):
        """
        :arg string filename: output file name (without directory)
        :arg string outputdir: directory where file is stored
        :kwarg int next_export_ix: set the index for next output
        :kwarg bool verbose: print debug info to stdout
        """
        self.filename = filename
        self.outputdir = create_directory(outputdir)
        self.verbose = verbose
        # keeps track of export numbers
        self.next_export_ix = next_export_ix

    def set_next_export_ix(self, next_export_ix):
        """Sets the index of next export"""
        self.next_export_ix = next_export_ix

    def export(self, function):
        """Exports given function to disk"""
        raise NotImplementedError('This method must be implemented in the derived class')


class VTKExporter(ExporterBase):
    """Class that handles Paraview VTK file exports"""
    def __init__(self, fs_visu, func_name, outputdir, filename,
                 next_export_ix=0, project_output=False,
                 add_subdir=True,
                 coord_func=None, verbose=False):
        """
        :arg fs_visu: function space where input function will be cast
            before exporting
        :arg func_name: name of the function
        :arg outputdir: output directory
        :arg filename: name of the pvd file
        :kwarg int next_export_ix: index for next export (default 0)
        :kwarg bool project_output: project function to output space instead of
            interpolating
        :kwarg bool add_subdir: Store each field in separate sub-directory
        :kwarg bool coord_func: Custom coordinate field. Needed to avoid
            allocating new coordinate field in the File object in case the mesh
            coordinate function is incompatible.
        :kwarg bool verbose: print debug info to stdout
        """
        super(VTKExporter, self).__init__(filename, outputdir, next_export_ix,
                                          verbose)
        self.fs_visu = fs_visu
        self.func_name = func_name
        self.project_output = project_output
        self.coord_func = coord_func
        suffix = '.pvd'
        if add_subdir:
            path = os.path.join(outputdir, filename)
        else:
            path = outputdir
        # append suffix if missing
        if os.path.splitext(filename)[1] != suffix:
            self.filename += suffix
        path = os.path.join(path, self.filename)
        self.outfile = File(path)
        self.cast_operators = {}

    def set_next_export_ix(self, next_export_ix):
        """Sets the index of next export"""
        # NOTE vtk io objects store current export index not next
        super(VTKExporter, self).set_next_export_ix(next_export_ix)
        # FIXME hack to change correct output file count
        self.outfile.counter = itertools.count(start=self.next_export_ix)

    def export(self, function):
        """Exports given function to disk"""
        assert self.fs_visu.max_work_functions == 1
        tmp_proj_func = self.fs_visu.get_work_function()
        # NOTE tmp function must be invariant as the projector is built only once
        op = self.cast_operators.get(function)
        if self.project_output:
            if op is None:
                op = Projector(function, tmp_proj_func)
                self.cast_operators[function] = op
            op.project()
        else:
            if op is None:
                op = Interpolator(function, tmp_proj_func)
                self.cast_operators[function] = op
            op.interpolate()
        coordfunc = function.function_space().mesh().coordinates
        if coordfunc not in self.outfile._output_functions and self.coord_func is not None:
            # workaround to avoid allocating coordinate function in File object
            self.outfile._output_functions[coordfunc] = self.coord_func
        # ensure correct output function name
        old_name = tmp_proj_func.name()
        tmp_proj_func.rename(name=self.func_name)
        self.outfile.write(tmp_proj_func, time=self.next_export_ix)
        self.next_export_ix += 1
        # restore old name
        tmp_proj_func.rename(name=old_name)
        self.fs_visu.restore_work_function(tmp_proj_func)


class HDF5Exporter(ExporterBase):
    """
    Stores fields in disk in native discretization using HDF5 containers
    """
    def __init__(self, function_space, outputdir, filename_prefix,
                 next_export_ix=0, verbose=False):
        """
        Create exporter object for given function.

        :arg function_space: space where the exported functions belong
        :type function_space: :class:`FunctionSpace`
        :arg string outputdir: directory where outputs will be stored
        :arg string filename_prefix: prefix of output filename. Filename is
            prefix_nnnnn.h5 where nnnnn is the export number.
        :kwarg int next_export_ix: index for next export (default 0)
        :kwarg bool verbose: print debug info to stdout
        """
        super(HDF5Exporter, self).__init__(filename_prefix, outputdir,
                                           next_export_ix, verbose)
        self.function_space = function_space

    def gen_filename(self, iexport):
        """
        Generate file name 'prefix_nnnnn.h5' for i-th export

        :arg int iexport: export index >= 0
        """
        filename = '{0:s}_{1:05d}'.format(self.filename, iexport)
        return os.path.join(self.outputdir, filename)

    def export_as_index(self, iexport, function):
        """
        Export function to disk using the specified export index number

        :arg int iexport: export index >= 0
        :arg function: :class:`Function` to export
        """
        assert function.function_space() == self.function_space,\
            'Function space does not match'
        filename = self.gen_filename(iexport)
        if self.verbose:
            print_output('saving {:} state to {:}'.format(function.name(), filename))
        with DumbCheckpoint(filename, mode=FILE_CREATE, comm=function.comm) as f:
            f.store(function)
        self.next_export_ix = iexport + 1

    def export(self, function):
        """
        Export function to disk.

        Increments export index by 1.

        :arg function: :class:`Function` to export
        """
        self.export_as_index(self.next_export_ix, function)

    def load(self, iexport, function):
        """
        Loads nodal values from disk and assigns to the given function

        :arg int iexport: export index >= 0
        :arg function: target :class:`Function`
        """
        assert function.function_space() == self.function_space,\
            'Function space does not match'
        filename = self.gen_filename(iexport)
        if self.verbose:
            print_output('loading {:} state from {:}'.format(function.name(), filename))
        with DumbCheckpoint(filename, mode=FILE_READ, comm=function.comm) as f:
            f.load(function)


class CoordinateFunctionCache:
    """
    Stores custom coordinate functions in for different function spaces.
    """
    def __init__(self):
        self.functions = {}
        self.interpolators = {}

    def _fs_kind(self, function_space):
        """
        Return function space identifier: (is_2d, is_cg)
        """
        return is_2d(function_space), is_cg(function_space)

    def get(self, function_space):
        """
        Return compatible coordinate function.

        Returns mesh coordinate function if it is in the right function space.
        Otherwise, returns a separate coordinate function in the appropriate
        function space. If it does not exits, it will be created.
        """
        fs_type = self._fs_kind(function_space)

        mesh = function_space.mesh()
        mesh_coordinates = mesh.coordinates
        mesh_coord_fs_type = self._fs_kind(mesh_coordinates.function_space())
        if fs_type == mesh_coord_fs_type:
            return mesh_coordinates

        coord_func = self.functions.get(fs_type)
        if coord_func is None:
            family = 'CG' if fs_type[1] else 'DG'
            coord_fs = get_functionspace(mesh, family, 1, vector=True)
            coord_func = Function(coord_fs, name=f'Coordinates {family}')
            interpolator = Interpolator(mesh_coordinates, coord_func)
            interpolator.interpolate()
            self.functions[fs_type] = coord_func
            self.interpolators[fs_type] = interpolator
        return coord_func

    def update_coordinates(self, only_3d=False):
        """
        Re-interpolates mesh coordinates on custom coordinate functions.

        This is needed if mesh is moving in time.

        :kwarg bool only_3d: If true, updates only 3D coordinate functions.
            Useful when 2D mesh is static.
        """
        for fs_type in self.interpolators:
            fs_is_3d = not fs_type[0]
            if not only_3d or fs_is_3d:
                self.interpolators[fs_type].interpolate()


class ExportManager(object):
    """
    Helper object for exporting multiple fields simultaneously

    .. code-block:: python

        from .field_defs import field_metadata
        field_dict = {'elev_2d': Function(...), 'uv_3d': Function(...), ...}
        e = exporter.ExportManager('mydirectory',
                                   ['elev_2d', 'uv_3d', salt_3d'],
                                   field_dict,
                                   field_metadata,
                                   export_type='vtk')
        e.export()

    """
    def __init__(self, outputdir, fields_to_export, functions, field_metadata,
                 export_type='vtk', next_export_ix=0, verbose=False):
        """
        :arg string outputdir: directory where files are stored
        :arg fields_to_export: list of fields to export
        :type fields_to_export: list of strings
        :arg functions: dict that contains all existing :class:`Function` s
        :arg field_metadata: dict of all field metadata.
            See :mod:`.field_defs`
        :kwarg str export_type: export format, either 'vtk' or 'hdf5'
        :kwarg int next_export_ix: index for next export (default 0)
        :kwarg bool verbose: print debug info to stdout
        """
        self.outputdir = outputdir
        self.fields_to_export = fields_to_export
        # functions dict must be mutable for custom exports
        self.functions = {}
        self.functions.update(functions)
        self.field_metadata = field_metadata
        self.verbose = verbose
        self.coord_func_cache = CoordinateFunctionCache()
        self.preproc_callbacks = {}
        # for each field create an exporter
        self.exporters = OrderedDict()
        for key in fields_to_export:
            field = self.functions.get(key)
            if field is not None and isinstance(field, Function):
                self.add_export(key, field, export_type,
                                next_export_ix=next_export_ix)

    def add_export(self, fieldname, function,
                   export_type='vtk', next_export_ix=0, outputdir=None,
                   shortname=None, filename=None, preproc_func=None):
        """
        Adds a new field exporter in the manager.

        This method allows exporting both default Thetis fields and user
        defined fields. In the latter case the user must provide sufficient
        metadata, i.e. fieldname, shortname and filename.

        :arg string fieldname: canonical field name
        :arg function: Firedrake function to export
        :kwarg str export_type: export format, either 'vtk' or 'hdf5'
        :kwarg int next_export_ix: index for next export (default 0)
        :kwarg string outputdir: optional directory where files are stored
        :kwarg string shortname: override shortname defined in field_metadata
        :kwarg string filename: override filename defined in field_metadata
        :kwarg preproc_func: optional funtion that will be called prior to
            exporting. E.g. for computing diagnostic fields.
        """
        if outputdir is None:
            outputdir = self.outputdir
        self.functions[fieldname] = function
        if shortname is None or filename is None:
            assert fieldname in self.field_metadata, \
                'Unknown field "{:}". For custom fields shortname and filename must be defined.'.format(fieldname)
        if shortname is None:
            shortname = self.field_metadata[fieldname]['shortname']
        if filename is None:
            filename = self.field_metadata[fieldname]['filename']
        field = self.functions.get(fieldname)
        if preproc_func is not None:
            self.preproc_callbacks[fieldname] = preproc_func
        if field is not None and isinstance(field, Function):
            native_space = field.function_space()
            if export_type.lower() == 'vtk':
                visu_space = get_visu_space(native_space)
                coord_func = self.coord_func_cache.get(visu_space)
                self.exporters[fieldname] = VTKExporter(visu_space, shortname,
                                                        outputdir, filename,
                                                        coord_func=coord_func,
                                                        next_export_ix=next_export_ix)
            elif export_type.lower() == 'hdf5':
                self.exporters[fieldname] = HDF5Exporter(native_space,
                                                         outputdir, filename,
                                                         next_export_ix=next_export_ix)

    def set_next_export_ix(self, next_export_ix):
        """Set export index to all child exporters"""
        for k in self.exporters:
            self.exporters[k].set_next_export_ix(next_export_ix)

    def export(self):
        """
        Export all designated functions to disk

        Increments export index by 1.
        """
        self.coord_func_cache.update_coordinates(only_3d=True)
        if self.verbose and COMM_WORLD.rank == 0:
            sys.stdout.write('Exporting: ')
        for key in self.exporters:
            field = self.functions[key]
            if field is not None:
                if self.verbose and COMM_WORLD.rank == 0:
                    sys.stdout.write(key+' ')
                    sys.stdout.flush()
                if key in self.preproc_callbacks:
                    self.preproc_callbacks[key]()
                self.exporters[key].export(field)
        if self.verbose and COMM_WORLD.rank == 0:
            sys.stdout.write('\n')
            sys.stdout.flush()

    def export_bathymetry(self, bathymetry_2d):
        """
        Special function to export 2D bathymetry data to disk

        Bathymetry does not vary in time so this only needs to be called once.

        :arg bathymetry_2d: 2D bathymetry :class:`Function`
        """
        native_space = bathymetry_2d.function_space()
        visu_space = get_visu_space(native_space)
        coord_func = self.coord_func_cache.get(visu_space)
        e = VTKExporter(visu_space, 'Bathymetry', self.outputdir, 'bath.pvd',
                        coord_func=coord_func, next_export_ix=0,
                        add_subdir=False)
        e.export(bathymetry_2d)
