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

    2D channel with closed boundaries.<demo_2d_channel.rst>
    2D channel with boundary conditions.<demo_2d_channel_bnd.rst>
    Visualising model results.<demo_visualisation.rst>

Manual
======

- Model formulation
    - 2D depth averaged model
    - 3D baroclinic model
    - Turbulence closure models

- Test cases
    - Lock exchange benchmark
    - Idealized estuary simulation
    - Idealized river plume simulation

API documentation
=================

The complete list of all the classes and methods is
available at the :doc:`thetis` page. The same information is :ref:`indexed
<genindex>` in alphabetical order. Another very effective mechanism is
the site :ref:`search engine <search>`.
