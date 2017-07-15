Model outputs and visualization
===============================

VTK outputs
-----------

By default Thetis stores outputs in `VTK <http://www.vtk.org/>`__
format, suitable for visualization with `ParaView <http://www.paraview.org/>`__.

By default results are stored to ``outputs`` sub-directory.
Users can define a custom output directory with :ref:`ModelOptions3d<model_options_3d>`.\ :py:attr:`.output_directory`
option. The fields to be exported are defined with
:ref:`ModelOptions3d<model_options_3d>`.\ :py:attr:`.fields_to_export` list.

For example to store only 2D water elevation and 3D temperature fields to ``mydir``
directory one would set::

    solver_obj = FlowSolver(...)
    options = solver_obj.options
    options.outputdir = 'mydir'
    options.fields_to_export = ['elev_2d', 'temp_3d']

For a full list of available fields, and their ``*.pvd`` file names, refer to
:doc:`physical fields <field_documentation>` page.
Note that fields are exported only if they are defined; e.g. exporting a field
``'temp_3d'`` is ignored if you are running a 2D model.

In some cases it is useful to suppress all output to disk (e.g. with some test
cases). This can be achieved by setting :ref:`ModelOptions3d<model_options_3d>`.\ :py:attr:`.no_exports`
option to ``True``.

See Firedrake's
`documentation <http://firedrakeproject.org/visualisation.html>`__
for more information on visualization cababilities.

Visualizing stored ParaView state files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating a complex visualization is a time consuming task.
ParaView allows users to store the active visualization state into a file
(File → Save State), and restore it later (File → Load State).

To quickly launch a previously saved visualization you can use
the ``visualize_output.py`` utility.
This script opens a ParaView state file using model outputs in any user-defined
directory::

    visualize_output.py <outputdir> <statefile.pvsm>

The script replaces all ``*.pvd`` file paths in ``statefile.pvsm`` to point
to ``outputdir``, and launches ParaView.
You'll need to have ``paraview`` in your seach path for this to work.
Run ``visualize-output.py -h`` to see full usage instructions.

.. note::

    If you add ParaView's ``bin`` directory to your ``PATH``
    environment variable, be careful:
    ParaView has its own ``python`` and ``mpiexec`` binaries which may
    break your Firedrake installation. It's advisable to modify ``PATH``
    before activating the Firedrake virtual environment, or appending
    (rather than prepending) the directory to ``PATH``.


HDF5 outputs
------------

Thetis also stores model state variables in lossless HDF5 format that allows
loading a previous model state from disk.

.. note::

    Currently HDF5 checkpointing file can only be
    loaded on same number of MPI processes as were used to create the file.
    See Firedrake's `documentation <http://firedrakeproject.org/checkpointing.html>`__
    for more information.

The fields to be exported in HDF5 format are defined in
:ref:`ModelOptions3d<model_options_3d>`.\ :py:attr:`.fields_to_export_hdf5` list, that is empty by default.
The files are stored in ``outputdir/hdf5`` directory in format
``Elevation2d_00001.h5`` where the prefix is the output file name of the field,
followed by the export index. HDF5 files are stored at the same time intervals
as VTK files, defined by :ref:`ModelOptions3d<model_options_3d>`.\ :py:attr:`.simulation_export_time` option.

In order to be able to restart a previous simulation, one has to export all
the prognostic variables that define the model state.
For a 2D simulation this implies::

    options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']

while a 3D simulation requires::

    options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d', 'uv_3d',
                                     'salt_3d', 'temp_3d', 'tke_3d', 'psi_3d']


Restarting a simulation
~~~~~~~~~~~~~~~~~~~~~~~

If you have stored the required HDF5 files, you can continue a simulation
using :py:meth:`~.FlowSolver.load_state` method, provided that you use the same
mesh and the same number of MPI processes. This call replaces the
:py:meth:`~.FlowSolver.assign_initial_conditions` call.
If initial conditions are not set, add ``load_state`` call above
the :py:meth:`~.FlowSolver.iterate` call.

In the simplest form, one only defines the export index that is used as initial
condition::

    solver_obj.load_state(155)

This also loads simulation time from the stored state.
It is also possible to load the initial state from another (sub-) directory::

    solver_obj.load_state(155, outputdir='other_outputdir')
