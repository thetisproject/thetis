==================
 Obtaining Thetis
==================

.. highlight:: none

Thetis requires installation of `Firedrake
<http://firedrakeproject.org>`_ (available for Ubuntu, Mac, and in
principle other Linux and Linux-like systems) and must be run from
within the Firedrake virtual environment.

Installing Firedrake
---------------------

You can install Firedrake by following the download documentation on the
`Firedrake website <http://firedrakeproject.org/download.html>`_.

.. note::

   **Installing Firedrake can take up to an hour depending on the system.**

In order to use Firedrake and install Thetis you need to activate the Firedrake
virtualenv::

   source <your-firedrake-venv-dir>/bin/activate

.. warning::

   **Please check that Firedrake is working before trying to install Thetis.**


Installing Thetis
------------------

.. warning::

   ``pip install thetis`` will **not** install the desired Thetis package!

You can install Thetis in your Firedrake installation by activating the Firedrake virtualenv and running:

.. code-block:: none

   pip install git+https://github.com/thetisproject/thetis.git

Using this install method, you should **not** add Thetis to your
``PYTHONPATH``. Instead, Thetis will automatically be available to import whenever your Firedrake virtualenv is active.

.. _editable-install:

Editable install
=============================================================================

If you would like to link your Thetis installation to the GitHub repository and contribute to the project, first activate
your Firedrake environment and then clone and install the repository with:

.. code-block:: none

   git clone https://github.com/thetisproject/thetis
   python -m pip install -e <path_to_thetis>

An IDE such as PyCharm will not recognize Thetis when installed in this fashion for any project outside the cloned repository,
as the source is not in site-packages. It will still run, and if you would like to enable full code navigation, you
can add the Thetis cloned repository as a content root, then add the ``thetis`` sub-directory as a sources root.

.. _shared-preinstalled-firedrake:

If you are using a shared, pre-installed Firedrake (such as on some clusters)
=============================================================================

Check out the `Thetis repository <http://github.com/thetisproject/thetis>`_ from GitHub.
You then need to add the Thetis repository to your ``PYTHONPATH`` in the Firedrake virtualenv. You can do this with ``pip``:

.. code-block:: none

   pip install -e <path-to-thetis-repository>

