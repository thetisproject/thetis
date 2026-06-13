==================
 Obtaining Thetis
==================

.. highlight:: none

Thetis requires installation of `Firedrake
<http://firedrakeproject.org>`_ (available for Ubuntu, Mac, and in
principle other Linux and Linux-like systems) and must be run from
within the Firedrake virtual environment.

.. note::

   Before reading these instructions and installing Firedrake, please decide
   whether you would like to use the ``release`` branch or the ``main`` branch.
   Firedrake ``release`` is not typically compatible with Thetis ``main``, and vice
   versa.

   ``release`` is stable and is updated when a major Firedrake release triggers
   a Thetis release update. ``main`` is the development branch, which requires a
   different installation method. It has all the latest updates but depends on
   upstream stability. If you want to work on ``main``, please see
   :doc:`developer_notes`.

Installing Firedrake (**release**)
----------------------------------

You can install Firedrake by following the download documentation on the
`Firedrake website <http://firedrakeproject.org/install.html>`_.

After installation, in order to use Firedrake and install Thetis you need to activate the Firedrake
virtual environment::

   source <your-firedrake-venv-dir>/bin/activate

.. warning::

   **You should check that the Firedrake install has been successful by running:**

   ::

      firedrake-check

Installing Thetis (**release**)
-------------------------------

To install Thetis in your Firedrake installation, activate the Firedrake virtual
environment and run:

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

In an editable install, ``git pull`` updates the Thetis source code and these changes take immediate effect in your (run-time) python environment, but it does
not update the installed package metadata used by ``pip list`` or ``pip show``.
Re-run ``pip install -e .`` after switching branches, pulling new commits, or
checking out a release tag if the version reported by pip needs to match the
current checkout.

An IDE such as PyCharm will not recognize Thetis when installed in editable
mode for any project outside the cloned repository, as the source is not in
site-packages. It will still run, and if you would like to enable full code
navigation, you can add the Thetis cloned repository as a content root, then add
the ``thetis`` sub-directory as a sources root.

Downloading the examples and demos
----------------------------------

If you only want to run the examples or demos, and do not want to use git, you
can download an archive of the ``release`` branch:

.. code-block:: none

   curl -JLO https://api.github.com/repos/thetisproject/thetis/tarball/release

Or download and extract it in one command:

.. code-block:: none

   curl -L https://api.github.com/repos/thetisproject/thetis/tarball/release | tar xz

The extracted archive contains the ``examples/`` and ``demos/`` directories, as
well as the rest of the Thetis source tree, without requiring a ``git checkout``.

.. _alternative-installation-methods:

Alternative installation methods
---------------------------------

As well as being installable through ``pip``, Firedrake also provides Docker images.
Thetis is no longer included in the default Firedrake Docker images, so after starting
a container from a Firedrake image you should install Thetis inside that container
using the standard installation instructions above.

If there are any problems with the installation of Firedrake and Thetis, the Slack workspace for Firedrake contains both
the general channel for Firedrake and a specific channel for Thetis. GitHub can also be used to report issues. Please
follow this `link <https://thetisproject.org/contact.html>`_ for contact details and we will be happy to help.
