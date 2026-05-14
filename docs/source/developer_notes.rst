================
 Developer notes
================

This page is aimed at people developing Thetis, maintaining CI, and doing
releases.


Installing Firedrake
---------------------

Installation of Firedrake for the ``main`` branch follows a different set
of instructions. Firedrake has a different `website address
<https://www.firedrakeproject.org/firedrake>`_ for the development ``main``
branch. You can install Firedrake by following the download documentation
`there
<https://www.firedrakeproject.org/firedrake/install#developer-install>`_.

After installation, in order to use Firedrake and install Thetis you need to
activate the Firedrake virtual environment::

   source <your-firedrake-venv-dir>/bin/activate

.. warning::

   **On the main branch, it is critical to check the Firedrake install has been successful by running:**

   ::

      firedrake-check


Installing Thetis
------------------

For development work, clone the repo and install in editable mode:

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

If you are a developer, you may also want to directly install dependencies for testing
Thetis and building the website:

.. code-block:: none

   pip install -e ".[docs,lint,test]"

Alternatively, the GitHub Actions workflow installs Thetis into a fresh venv *that can
still see the Firedrake site-packages* by using ``--system-site-packages``::

  python3 -m venv --system-site-packages venv-thetis
  . venv-thetis/bin/activate
  pip install .

This pattern is useful when you want to reproduce CI locally without polluting
your base Firedrake environment.


CI testing
-----------

CI is implemented with GitHub Actions workflows under ``.github/workflows/``:

* ``push.yml`` runs on pushes to ``main`` and ``release``.
* ``pr.yml`` runs on pull requests.
* ``weekly-main.yml`` schedules a weekly run on ``main``.
* ``weekly-release.yml`` schedules a weekly run on ``release``.
* ``core.yml`` is the reusable workflow that does the actual work.

The reusable workflow (``core.yml``):

* runs on a self-hosted Linux runner (physically situated at Imperial College London),
  inside a Firedrake Docker image
* checks out the requested ref into a directory called ``thetis-repo`` (to avoid
  false positives from ``import thetis`` working without installation)
* creates ``venv-thetis`` with ``--system-site-packages`` and installs Thetis
* runs linting via ``make -C thetis-repo lint``
* runs tests:

  * serial tests via pytest-xdist::

      python -m pytest -n 12 --verbose --durations=0 --durations-min=60.0 \
        -m "parallel[1] or not parallel" thetis-repo/test

  * MPI-parallel tests (2 ranks)::

      mpiexec -n 2 python -m pytest --verbose --durations=0 --durations-min=60.0 \
        -m parallel[2] thetis-repo/test

  * adjoint tests::

      python -m pytest -n 8 --verbose --durations=0 thetis-repo/test_adjoint

The workflow sets ``PYTEST_MPI_MAX_NPROCS=2`` to avoid silently skipping tests
that request more ranks than are available.

Pull requests
--------------

Most changes should go via a pull request (PR) to ``main``.

* Target branch: PRs should usually target ``main``. If a change needs to ship
  on the stable branch, it must land on ``release`` first (see the release-branch
  policy below).
* CI: PRs are tested by ``.github/workflows/pr.yml`` against the appropriate
  Firedrake Docker image (based on the PR base branch).
* Before opening/merging a PR, it is expected that lint is clean (``make lint``),
  tests are run where practical (see the CI commands above), and docs changes
  render as expected (build the Sphinx site locally if you edited
  ``docs/source/*``).


Release vs main, and updating release
--------------------------------------

Branch intent
~~~~~~~~~~~~~

* ``main`` is the development branch and is tested against the Firedrake
  ``dev-main`` Docker image in CI.
* ``release`` is the stable branch and is tested against the Firedrake
  ``latest`` Docker image in CI.

Policy (No Cherry-Picks)
~~~~~~~~~~~~~~~~~~~~~~~~

Thetis follows a Firedrake-style branching policy:

* ``release`` must always be an ancestor of ``main`` (everything in ``release``
  is also in ``main``).
* We do not cherry-pick between ``main`` and ``release``. Changes that should
  ship on both branches should land on ``release`` first, then ``release`` is
  merged into ``main``.
* ``release`` is not periodically "restarted". Users should be able to
  ``git pull`` their local ``release`` branch normally.

This avoids duplicated commits and keeps merges between the branches simple.

When Do We Update ``release``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``release`` branch is updated in two main cases:

1. When Firedrake makes a new stable (major) release (e.g. ``2025.10`` -> ``2026.4``),
   so Thetis ``release`` stays compatible with the corresponding Firedrake stable
   Docker image. This typically advances Thetis ``release`` to a recent, known-good
   commit from ``main``, so users on ``release`` pick up the main-branch developments
   that are compatible with the new Firedrake stable release.
2. When a user-facing bugfix is needed on the stable branch.

Policy: keep ``release`` changes minimal and compatibility-driven. Prefer
targeted fixes over large feature merges.

Updating ``release`` On GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Thetis updates to the stable branch are done with PRs and must preserve history
(use merge commits, not squash/rebase).

To update ``release``:

1. Open a PR with base branch ``release``.
2. Wait for CI to pass and the PR to be reviewed.
3. Merge using "Create a merge commit" (do not squash merge or rebase merge).
4. Sync ``release`` into ``main`` when needed:

   * Required: if the PR introduced commits that do not already exist on
     ``main`` (for example a hotfix implemented on ``release``).
   * Optional: if the PR only advanced ``release`` to a vetted commit that was
     already on ``main`` (for example during a Firedrake stable release update).

   Important: if branch protection requires PR branches to be "up to date",
   GitHub may show the ``release`` -> ``main`` sync PR as out-of-date and offer
   an "Update branch" button. Do not click this: it merges ``main`` into
   ``release``, which breaks the "``release`` is an ancestor of ``main``"
   policy.

   Instead, create a temporary sync branch *from* ``main`` and merge ``release``
   into it, then open a PR to ``main``:

   .. code-block:: none

      git checkout main
      git pull
      git checkout -b sync/release-into-main
      git merge --no-ff release
      git push -u origin sync/release-into-main

   Open a PR targeting ``main`` from ``sync/release-into-main`` and merge it
   using "Create a merge commit".

If you are cutting a tagged release, bump the packaged version in
``pyproject.toml``, push a tag, create a GitHub Release, and verify Zenodo
archived the release (see the Zenodo section below).

If the release includes documentation changes, publish the updated rendered site
to ``thetisproject.github.io`` at the same time as merging the corresponding PRs
in ``thetisproject/thetis``.

This ensures ``main`` always contains everything that shipped on ``release``.

Worked examples
~~~~~~~~~~~~~~~

1. Firedrake stable release update (advance ``release`` to a vetted ``main`` commit)

   Scenario: ``main`` has progressed ``A -> B -> C -> D``. Firedrake makes a new
   stable release, and we want ``release`` users to pick up the compatible
   developments from ``main``.

   Action: open a PR targeting ``release`` that advances it to the chosen
   ``main`` commit (often the current ``main`` tip), using a merge commit. A
   typical local workflow is:

   .. code-block:: none

      git checkout release
      git pull
      git checkout -b release-update
      git merge --no-ff main
      git push -u origin release-update

   Then follow the sync-to-``main`` guidance (step 4 above). In this scenario
   the sync is often optional (because the commits already exist on ``main``),
   but can be used to ensure ``main`` also contains the merge commit that
   advanced ``release``.

2. Stable-branch bugfix (land on ``release``, then sync to ``main``)

   Scenario: ``main`` is at ``A -> ... -> D``. A user-facing bugfix ``Z`` is
   needed on the stable branch, and must also end up on ``main``.

   Action: open a PR targeting ``release`` with the bugfix (merge commit), then
   sync ``release`` back into ``main`` using the procedure in step 4 above.


Tag/version convention
~~~~~~~~~~~~~~~~~~~~~~

The repository currently contains multiple historical tag naming schemes (for
example ``2026.4.0`` / ``2025.10.1`` as well as older ``Thetis_YYYYMMDD``-style
tags). For new releases, prefer a tag that matches the packaged version in
``pyproject.toml`` and align the version scheme with Firedrake's release cadence.


Thetis website
---------------

Content
~~~~~~~

Thetis documentation content lives in this repository under ``docs/source`` and
is built with Sphinx (see ``docs/Makefile`` and ``docs/source/conf.py``).

Thetis has a single published documentation website. The published site is
updated whenever the ``release`` branch is updated, and it must be built from
the ``release`` branch (do not publish a site built from ``main``).

If you need the equivalent rendered documentation for ``main``, build it locally
from a ``main`` checkout using the same instructions below.

Deployment
~~~~~~~~~~

Thetis is published from a separate `rendered-site repository
<https://github.com/thetisproject/thetisproject.github.io>`_. The workflow is:

1. Check out the ``release`` branch of this repository and build the website
   locally from source.

   Install the doc-build dependencies if needed.::

     pip install -e ".[docs]"


   Build the website locally.::

     make -C docs html

   The rendered site will be in ``docs/build/html``. You can inspect the local
   build, e.g.::

     firefox docs/build/html/index.html


2. Make the relevant changes under ``docs/source/`` (for example
   ``docs/source/download.rst``).

3. Rebuild locally (step 1) to check the rendered output.

4. Commit the changes in this repo and open a PR.

   If the change is intended for the published website, the PR must target
   ``release`` (since the published website is built from ``release``). It is
   fine to iterate on documentation changes in a PR targeting ``main``, but do
   not publish a website build from ``main``.

5. When the PR is approved, merge it. At the same time, copy the contents of
   ``docs/build/html/`` into the ``thetisproject.github.io`` repository and
   merge there as well.

Zenodo
-------

Zenodo can archive GitHub releases/tags and mint a DOI per version (plus a
concept DOI across all versions).

In practice for Thetis:

1. Create and push a git tag for the release.
2. Create a GitHub Release for that tag.
3. Verify Zenodo created/updated the corresponding record and that metadata
   (authors, title, description) is correct.

Sometimes the GitHub-Zenodo integration breaks and GitHub Releases do not get
archived on Zenodo. To fix it, a GitHub organization owner for ``thetisproject``
(not just a repository collaborator) may need to:

1. Delete the Zenodo webhook in the GitHub repository settings.
2. Disconnect and reconnect *their* GitHub <-> Zenodo account connection and
   re-authenticate.
3. Re-enable the Thetis repository in Zenodo's GitHub integration settings (if
   it does not re-enable automatically).

It is often also worth the person doing the release reconnecting their personal
GitHub/Zenodo connection.
