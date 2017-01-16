Visualising results
===================

By default Thetis stores outputs in `VTK <http://www.vtk.org/>`_
format, suitable for visualisation with `ParaView <http://www.paraview.org/>`_.

By default results are stored to ``outputs`` sub-directory.
Users can define a custom output directory with :py:attr:`~.ModelOptions.outputdir`
option. The fields to be exported are defined with
:py:attr:`~.ModelOptions.fields_to_export` list.

For example to store only 2D water elevation and 3D temperature fields to ``mydir``
directory one would set::

    solver_obj = FlowSolver(...)
    options = solver_obj.options
    options.outputdir = 'mydir'
    options.fields_to_export = ['elev_2d', 'temp_3d']

For a full list of available fields, refer to :doc:`physical fields <field_documentation>`.

See Firedrake's
`visualisation documentation <http://firedrakeproject.org/visualisation.html>`_
for more information.

Visualising stored ParaView state files
---------------------------------------

Creating a complex visualization is a time consuming task.
ParaView allows users to store the active visualization state into a file
(File → Save State), and restore it later (File → Load State).

Thetis has an helper utility `visualize_output.py` that loads a prevously
stored ParaView state file, using outputs in any user-defined directory::

    visualize_output.py <outputdir> <statefile.pvsm>

The script replaces all ``*.pvd`` file paths in the ``statefile.pvsm`` to point
to ``outputdir``, and launches ParaView.
For more information see **TODO**:script-visualize_output-docs.

