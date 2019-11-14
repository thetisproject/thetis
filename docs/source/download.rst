==================
 Obtaining Thetis
==================

.. highlight:: none

Thetis requires installation of `Firedrake
<http://firedrakeproject.org>`_ (available for Ubuntu, Mac, and in
principle other Linux and Linux-like systems) and must be run from
within the Firedrake virtual environment.

Install Firedrake and Thetis
-----------------------------

You can install both Firedrake and Thetis by running::

    curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
    python3 firedrake-install --install thetis

See  `Firedrake website <http://firedrakeproject.org/download.html>`_ for more
information on the installation procedure. Note that the install proceduce may
take up to one hour depending on your system.

The Thetis source will be installed in the ``src/thetis`` subdirectory
of your Firedrake install.
In order to use Firedrake and Thetis you need to activate the Firedrake
virtualenv::

    source <your-firedrake-install-dir>/firedrake/bin/activate

Using this install method you should
**not** add Thetis to your ``PYTHONPATH``. Instead, Thetis will
automatically be available to import whenever your Firedrake
virtualenv is active.

If you have already installed Firedrake
---------------------------------------

You can install Thetis in your Firedrake installation by
activating the Firedrake virtualenv and running::

    firedrake-update --install thetis


If you are using a shared, pre-installed Firedrake (such as on some clusters)
-----------------------------------------------------------------------------

Check out the `Thetis <http://github.com/thetisproject/thetis>`_
repository from Github.
You then need to add the Thetis repository to your ``PYTHONPATH`` in the
Firedrake virtualenv. You can do this with ``pip``::

    pip install -e <path-to-thetis-repository>
