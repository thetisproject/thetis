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
`Firedrake website <http://firedrakeproject.org/install.html>`_.

.. note::

   **Installing PETSc and Firedrake can take up to 30 minutes depending on the system.**

After installation, in order to use Firedrake and install Thetis you need to activate the Firedrake
virtual environment::

   source <your-firedrake-venv-dir>/bin/activate

.. warning::

   **You should check that the Firedrake install has been successful by running:**

   ::

      firedrake-check

.. note::

   If you want to install the developer **master** branch of Thetis, you will need to
   install the corresponding development branch (**main**) of Firedrake.
   Please follow the instructions on the `Firedrake website <http://firedrakeproject.org/install.html>`_
   to do this.

Installing Thetis
------------------

You can install Thetis in your Firedrake installation by activating the Firedrake virtual environment and running:

.. code-block:: none

   pip install git+https://github.com/thetisproject/thetis.git@release

This will install the latest (stable) release branch.

.. _editable-install:

Editable install
------------------

If you want to install Thetis from a local checkout of the repository that you can directly edit, update (pull) from
GitHub, switch branches, etc., it is recommended to use an editable install using:

.. code-block:: none

   git clone https://github.com/thetisproject/thetis
   cd thetis
   git checkout <branch_name>
   pip install -e .

If you have SSH keys set up with GitHub, you can use the SSH-based clone instead:

.. code-block:: none

   git clone git@github.com:thetisproject/thetis
   cd thetis
   git checkout <branch_name>
   pip install -e .

.. note::

   For development, the default branch of Thetis is **master**. This tracks the **main** branch of Firedrake. If
   you do not intend on developing Thetis, please ensure that you checkout the release branch for compatibility.


An IDE such as PyCharm will not recognize Thetis when installed in this fashion for any project outside the cloned repository,
as the source is not in site-packages. It will still run, and if you would like to enable full code navigation, you
can add the Thetis cloned repository as a content root, then add the ``thetis`` sub-directory as a sources root.

.. _alternative-installation-methods:

Alternative installation methods
---------------------------------

As well as being installable through ``pip``, Firedrake also provides Docker containers.
Thetis is no longer distributed in Docker containers with the "latest" image, so please
follow the standard installation instructions above.

If there are any problems with the installation of Firedrake and Thetis, the Slack workspace for Firedrake contains both
the general channel for Firedrake and a specific channel for Thetis. GitHub can also be used to report issues. Please
follow this `link <https://thetisproject.org/contact.html>`_ for contact details and we will be happy to help.

