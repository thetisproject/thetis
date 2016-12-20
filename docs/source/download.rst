==================
 Obtaining Thetis
==================

Thetis requires installation of `Firedrake
<http://firedrakeproject.org>`_ (available for Ubuntu, Mac, and in
principle other Linux and Linux-like systems) and must be run from
within the Firedrake virtual environment.

If you installed Firedrake yourself
-----------------------------------

You can directly install Thetis in your Firedrake installation by
activating the Firedrake virtualenv and running::

    firedrake-update --install git+ssh://github.com/thetisproject/thetis#egg=thetis

The Thetis source will be installed in the ``src/thetis`` subdirectory
of your Firedrake install. Using this install method you should
**not** add add Thetis to your ``PYTHONPATH``. Instead, Thetis will
automatically be available to import whenever your Firedrake
virtualenv is active.


If you are using a shared, pre-installed Firedrake (such as on some clusters)
-----------------------------------------------------------------------------

Check out the `Thetis <http://github.com/thetisproject/thetis>`_
repository on Github. Then, start the Firedrake virtualenv and add the
``thetis`` subdirectory to your ``PYTHONPATH``.
