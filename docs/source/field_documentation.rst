Physical fields
===============

This page lists all supported physical fields.
Every field is identified with an unique key, e.g. ``temp_3d``.
The metadata is defined in :py:mod:`~.field_defs` module.

Fields are stored in solver's :py:attr:`~.FlowSolver.fields` dictionary using
the unique keys.
Therefore one can access a field :py:class:`Function` with::

    # dictionary access
    solver_obj.fields['temp_3d']
    # or attribute access
    solver_obj.fields.temp_3d

List of fields
--------------

.. include:: field_list.rst
