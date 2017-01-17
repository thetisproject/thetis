======================
 Thetis documentation
======================

Installation
============

See :doc:`obtaining Thetis <download>` for installation instructions.

Tutorials
=========

Once Thetis is successfully installed, you can start running some example
simulations that demonstrate basic functionality.

.. note::

    Setting up Thetis simulations makes use of Firedrake objects,
    such as meshes, functions, and expression.
    In order to be able to follow these demo simulations,
    it is essential to have a basic understanding of these objects.
    Please refer to the
    `Firedrake manual <http://firedrakeproject.org/documentation.html>`_
    when reading these demos.

.. toctree::
    :maxdepth: 1

    2D channel with closed boundaries.<demo_2d_channel>
    2D channel with boundary conditions.<demo_2d_channel_bnd>
    3D channel with boundary conditions.<demo_3d_channel>

Manual
======

.. toctree::
    :maxdepth: 1

    Model outputs and visualization.<outputs_and_visu>
    List of physical fields.<field_documentation>

API documentation
=================

The complete list of all the classes and methods is
available at the :doc:`thetis` page. The same information is :ref:`indexed
<genindex>` in alphabetical order. Another very effective mechanism is
the site :ref:`search engine <search>`.
