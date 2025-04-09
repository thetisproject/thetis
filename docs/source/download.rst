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

After installation, in order to use Firedrake and install Thetis you need to activate the Firedrake
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

If you want to install Thetis from a local checkout of the repository that you can directly edit, update (pull) from github, switch branches, etc., it is recommended to use an editable install using:

.. code-block:: none

   git clone https://github.com/thetisproject/thetis
   pip install -e <path_to_thetis>

An IDE such as PyCharm will not recognize Thetis when installed in this fashion for any project outside the cloned repository,
as the source is not in site-packages. It will still run, and if you would like to enable full code navigation, you
can add the Thetis cloned repository as a content root, then add the ``thetis`` sub-directory as a sources root.

.. _alternative-installation-methods:

Alternative installation methods
=================================

As well as being installable through ``pip``, Firedrake also provides Docker containers which typically contain the
latest Thetis at the time of release.

If there are any problems with the installation of Firedrake and Thetis, the Slack workspace for Firedrake contains both
the general channel for Firedrake and a specific channel for Thetis. GitHub can also be used to report issues. Please
follow this `link <https://thetisproject.org/contact.html>`_ for contact details and we will be happy to help.

