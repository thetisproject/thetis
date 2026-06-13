================
 Developer notes
================

This page is aimed at people developing Thetis, maintaining continuous
integration (CI), and doing releases.

.. _branches-main-vs-release:

Branches (**main** vs **release**)
----------------------------------

Thetis has two long-lived branches:

* ``main`` is the development branch. New features should be developed here.
  Thetis ``main`` tracks Firedrake ``main`` and generally requires Firedrake
  ``main``.
* ``release`` is the stable branch intended for users. It is kept compatible
  with the latest stable Firedrake release.

The temporary branch ``release-candidate`` is used only while preparing a new
Thetis release after a Firedrake release. It is force-updated to the intended
``main`` release commit, and GitHub Actions tests it against the same Firedrake
``latest`` Docker image used for ``release``. This gives maintainers a tested
candidate before the public ``release`` branch is force-updated and tagged.

CI reflects this by testing ``main`` against the Firedrake ``dev-main`` Docker
image and testing ``release`` and ``release-candidate`` against the Firedrake
``latest`` Docker image.


Installing Firedrake
---------------------

If you are developing Thetis on ``main``, you will generally need Firedrake
``main`` as well. Firedrake ``main`` follows a different set of installation
instructions from the stable Firedrake release.

Firedrake has a different `website address
<https://www.firedrakeproject.org/firedrake>`_ for the development ``main``
branch. You can install Firedrake by following the download documentation
`there
<https://www.firedrakeproject.org/firedrake/install#developer-install>`_.

After installation, in order to use Firedrake and install Thetis you need to
activate the Firedrake virtual environment::

   source <your-firedrake-venv-dir>/bin/activate

.. warning::

   **On the main branch, it is critical to check the Firedrake install has**
   **been successful by running:**

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

If you have SSH keys set up with GitHub, you can use the SSH-based clone
instead:

.. code-block:: none

   git clone git@github.com:thetisproject/thetis
   cd thetis
   git checkout <branch_name>
   pip install -e .

If you are a developer, you may also want to directly install dependencies for
testing Thetis and building the website:

.. code-block:: none

   pip install -e ".[docs,lint,test]"


CI testing
-----------

CI is implemented with GitHub Actions workflows under ``.github/workflows/``:

* ``push.yml`` runs on pushes to ``main``, ``release``, and
  ``release-candidate``.
* ``pr.yml`` runs on pull requests.
* ``weekly-main.yml`` schedules a weekly run on ``main``.
* ``weekly-release.yml`` schedules a weekly run on ``release``. Note that as it
  triggered on ``main``, the tag in the Actions tab will show as ``main``.
* ``core.yml`` is the reusable workflow that does the actual work.

The reusable workflow (``core.yml``):

* runs on a self-hosted Linux runner (physically situated at Imperial College
  London),
  inside a Firedrake Docker image
* checks out the requested ref into a directory called ``thetis-repo`` (to
  avoid false positives from ``import thetis`` working without installation)
* creates ``venv-thetis`` with ``--system-site-packages`` (this is only
  appropriate where Firedrake is installed in system packages) and installs
  Thetis
* runs linting via ``make -C thetis-repo lint``
* runs tests:

  * serial tests via pytest-xdist::

      python -m pytest -n 12 --verbose --durations=0 --durations-min=60.0 \
        -m "parallel[1] or not parallel" thetis-repo/test

  * MPI-parallel tests (2 ranks)::

      mpiexec -n 2 python -m pytest --verbose --durations=0 \
        --durations-min=60.0 \
        -m parallel[2] thetis-repo/test

  * adjoint tests::

      python -m pytest -n 8 --verbose --durations=0 thetis-repo/test_adjoint

The workflow sets ``PYTEST_MPI_MAX_NPROCS=2`` to avoid silently skipping tests
that request more ranks than are available.

Pull requests
--------------

Most changes should go via a pull request (PR) to ``main``.

* Target branch: PRs should usually target ``main``. PRs to ``release`` should
  only be used for release-management work, release-only fixes, or exceptional
  backports that must ship before the next Firedrake release cycle.
* CI: PRs are tested by ``.github/workflows/pr.yml`` against the appropriate
  Firedrake Docker image (based on the PR base branch).
* Before opening/merging a PR, it is expected that lint is clean
  (``make lint``),
  tests are run where practical (see the CI commands above), and docs changes
  render as expected (build the Sphinx site locally, using ``make html`` in the ``docs/`` directory,  if you edited
  ``docs/source/*``).

**Branch hygiene**: on feature branches it is fine (and often encouraged) to
clean up history (interactive rebase, squash, fixups) and force-push while
iterating on a PR.

**Merging**: for normal PRs into ``main`` or ``release``, prefer squash merge or
rebase merge so that project history stays linear. This is especially important
for ``release`` hotfix/backport PRs, where the result should be a single
release-side fix commit rather than an extra PR merge commit. The
Firedrake-triggered ``release`` update is not a PR merge: it is an explicit
branch reset by someone with the necessary repository permissions.


Maintaining **release**
-----------------------

Policy (Hard Reset at Firedrake Release Points)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Thetis follows Firedrake's major release cadence:

* ``main`` is the normal development branch. New features and ordinary bug fixes
  should land on ``main``.
* ``release`` is the current stable branch. It tracks the latest stable
  Firedrake release and is the branch used for the published website.
* When Firedrake makes a major release, Thetis ``release`` is hard-reset
  to a ``main`` commit that has been tested in the latest stable Firedrake
  release container.
* Between these reset points, keep ``release`` conservative. Bug fixes may be
  cherry-picked when they should be available on the stable line, but do not
  backport new functionality, API changes, substantial rewrites, or changes
  whose risk is out of proportion to the bug being fixed.

After Firedrake has made a major release, avoid merging further changes to
Thetis ``main`` until the corresponding Thetis release has been completed. The
release should be based on the ``main`` state that is intended to work with the
new Firedrake release, not on later development work against a moving Firedrake
``main``.

This policy keeps ``main`` linear and avoids merge commits whose only purpose is
to move changes back and forth between long-lived branches. The hard reset is
also acceptable because users will typically need to reinstall PETSc and
Firedrake at each major release cycle anyway. The tradeoff is that the public
``release`` branch is force-updated at release points. Tags are therefore the
stable install points for historical releases.


When Does **release** Move?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``release`` moves in three cases:

1. Firedrake makes a major release and Thetis ``release`` is hard-reset
   to the compatible, tested ``main`` commit.
2. A release-only fix is required for the current stable Firedrake stack.
3. A shared fix from ``main`` must be copied to the current stable branch before
   the next hard reset.

Keep ``release`` changes minimal and compatibility-driven. Each update to
``release`` creates a new installable stable state and should be tagged after
the corresponding CI run has passed.


Workflow: Normal Development and Bug Fixes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most changes should go to ``main`` only:

1. Create a branch from ``main``.
2. Open a PR with base branch ``main``.
3. Merge according to normal project practice.

If the change does not need to ship on the current stable Firedrake stack, do
not backport it. It will arrive on ``release`` at the next hard reset.


Workflow: Shared Fix from **main** to **release**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this only when a fix already on ``main`` must also ship on the current
stable branch before the next reset.

1. Create a backport branch from ``release`` and cherry-pick the ``main`` fix:

   .. code-block:: none

      git fetch origin
      git checkout -b release-backport/<short-description> origin/release
      git cherry-pick -x <main-fix-sha>
      git push -u origin release-backport/<short-description>

2. Open a PR with base branch ``release``.
3. Merge with a method that leaves a single release-side fix commit where
   possible (squash merge, or rebase merge if enabled). If using squash merge,
   keep the original ``cherry picked from`` line or mention the source ``main``
   PR/commit in the squash commit message.
4. Wait for the post-merge ``release`` push workflow to pass.
5. Tag the resulting ``release`` commit using the tag convention below.

The release-side commit will usually have a different SHA from the ``main`` fix.
This is expected because the parent commits differ; the same patch cannot have
the same commit SHA unless it has the same parents and metadata. If ``release``
was already ``N`` development commits behind ``main`` before the cherry-pick,
then after this workflow ``release`` is normally ``N + 1`` commits behind
``main`` and one commit ahead of ``main``.


Workflow: **release**-Only Fix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this when the issue only exists on the current stable branch, for example
because of the older Firedrake dependency stack.

1. Create a branch from ``release``:

   .. code-block:: none

      git fetch origin
      git checkout -b release-fix/<short-description> origin/release

2. Commit the fix and open a PR with base branch ``release``.
3. Merge with a method that leaves a single release-side fix commit where
   possible.
4. Wait for the post-merge ``release`` push workflow to pass.
5. Tag the resulting ``release`` commit using the tag convention below.

If the same fix is also needed on ``main``, put it on ``main`` first and use the
shared-fix workflow above. For a true release-only fix, if ``release`` was
already ``N`` development commits behind ``main`` before the fix, then after the
fix ``release`` is normally ``N`` commits behind ``main`` and one commit ahead of
``main``.


Workflow: Hard-Reset **release** to **main**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the normal workflow when Firedrake makes a major release and
Thetis ``release`` should become the compatible state from ``main``. The goal is
to tag only a commit that has passed Thetis CI in the latest stable Firedrake
release container.

Before moving ``release``, check whether the final state of the old stable line
already has the intended tag. If it does not, preserve it with a tag using the
tag convention below.

.. code-block:: none

   git fetch origin
   git checkout release
   git pull --ff-only
   git tag -a <old-version-final-or-patch-tag> release -m "<tag message>"
   git push origin <old-version-final-or-patch-tag>

Prepare and test a temporary ``release-candidate`` branch. Pushes to this branch
run the same ``push.yml`` workflow as ``release`` and use the Firedrake
``latest`` Docker image.

.. code-block:: none

   git fetch origin
   git checkout -B release-candidate <main-sha-or-origin/main>
   git push --force-with-lease origin release-candidate

Wait for the ``release-candidate`` push workflow to pass. If it fails, fix the
problem on ``release-candidate`` and push again until CI passes. Keep
``release-candidate`` based on the selected ``main`` commit; it is fine to
rewrite or squash fix commits on ``release-candidate`` while preparing the
release.

If fixes were needed on ``release-candidate``, fast-forward ``main`` to the
passing candidate before moving ``release``:

.. code-block:: none

   git fetch origin
   git checkout main
   git pull --ff-only
   git merge --ff-only origin/release-candidate
   git push origin main

This push to ``main`` should still be a normal fast-forward push, not a
force-push. If GitHub rejects this due to branch protection rules, a
maintainer must temporarily relax the relevant rule, make the normal
fast-forward push, and restore branch protection immediately afterwards.
After the Thetis release is complete, fix any remaining ``main`` CI failure
through the normal PR workflow.

After ``release-candidate`` has passed, move ``release`` to exactly the passing
commit:

.. code-block:: none

   git fetch origin
   git checkout release
   git reset --hard origin/release-candidate
   git push --force-with-lease origin release

Wait for the ``release`` push workflow to pass, then tag that exact commit and
push the tag. If the ``release`` workflow fails unexpectedly, do not tag it; fix
the problem on ``release-candidate`` and repeat the move to ``release``. The
package version is derived from git tags, so no version-only commit is needed at
the hard reset point.

.. code-block:: none

   git tag -a <new-release-version> release -m "Thetis <new-release-version>"
   git push origin <new-release-version>

Immediately restore/confirm the branch protection settings after the force-push
if they had to be relaxed.


History Shape
~~~~~~~~~~~~~

Immediately after a hard reset, ``release`` contains the vetted ``main`` state.
If ``release`` receives a release-only fix, it will diverge from ``main`` until
the next reset:

.. code-block:: none

   main:    A -- B -- C -- D
   release: A -- B -- R1

After the reset, active ``release`` contains the selected ``main`` state and the
old state remains available by the tag created before the reset:

.. code-block:: none

   main:    A -- B -- C -- D
   release: A -- B -- C -- D
   tag:     A -- B -- R1


Publishing A Tagged Release
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When publishing a tagged release, push a tag, create a GitHub Release, and
verify Zenodo archived the release (see the Zenodo section below). If the
release includes documentation changes, update the published website at the same
time.


Tag convention
~~~~~~~~~~~~~~

The repository currently contains multiple historical tag naming schemes (for
example ``2026.4.0`` / ``2025.10.1`` as well as older ``Thetis_YYYYMMDD``-style
tags). For new releases, the release tag is the source of the packaged version,
so use PEP 440-compatible tags.

New stable release tags should use:

.. code-block:: none

   YYYY.M.postN

where ``YYYY.M`` is the Firedrake major release series used by that Thetis
release branch and ``N`` is incremented for each Thetis release update on that
branch. For example, the first Thetis release against the Firedrake ``2026.4``
line should be tagged ``2026.4.post0``. The next Thetis update to that release
line should be tagged ``2026.4.post1``, regardless of whether the compatible
Firedrake stack is ``2026.4.0``, ``2026.4.1``, or a later Firedrake patch. This
keeps Thetis release numbering visibly separate from Firedrake patch numbering.

Tags are normally created only:

* before a hard reset, if the final installable state of the old stable line has
  not already been tagged;
* after a hard reset, after CI has passed for the new Thetis release against the
  latest stable Firedrake release container; or
* after a tested release-only fix or backported bug fix has been merged to
  ``release``.

Do not use tags such as ``2026.4.rel1`` for Thetis releases: the release tag is
used as the Python package version, and ``rel1`` is not valid in that position
under PEP 440.


Python package versions
~~~~~~~~~~~~~~~~~~~~~~~

Thetis uses dynamic versioning from git tags and branch state. Developers
normally only choose the release tags described above; untagged development
versions are derived automatically.

The automatic package-version rules are:

* A tagged commit has the exact release version from the tag.
* Untagged commits on ``main`` and ordinary feature branches report
  ``YYYY.M.devN``. ``YYYY.M`` is based on the commit date of ``HEAD`` (the Git
  committer date, not the author date), and ``N`` is the number of commits in
  that month.
* Untagged commits on ``release`` and ``release-*``/``release/*`` maintenance
  branches, including ``release-candidate``, report the next ``postN``
  development version for that release line.

If ``main`` and ``release`` point at the same tagged commit, they report the
same exact version. Once ``main`` receives new commits, it reports a development
version again. Do not manually tag arbitrary ``main`` commits to create
development versions.

For editable installs, ``git pull`` updates the source code but does not rewrite
the installed Python distribution metadata that commands such as ``pip list``
and ``pip show`` read. Re-run ``pip install -e .`` after switching branches,
pulling new commits, or creating/checking out a release tag if the version shown
by pip needs to match the current checkout.


Thetis website
---------------

Content
~~~~~~~

Thetis documentation content lives in this repository under ``docs/source`` and
is built with Sphinx (see ``docs/Makefile`` and ``docs/source/conf.py``).

Thetis has a single published documentation website. The published site is
updated whenever the documentation content in the ``release`` branch is updated,
and it must be built from the ``release`` branch (do not publish a site built
from ``main``).

If you need the equivalent rendered documentation for ``main``, build it
locally from a ``main`` checkout using the same instructions below.

Deployment
~~~~~~~~~~

Thetis is published from a separate `rendered-site repository
<https://github.com/thetisproject/thetisproject.github.io>`_. Documentation
content changes should normally be merged to ``main`` first, like other Thetis
changes. If the website needs to be updated before the next release reset,
backport or repeat the documentation change on ``release`` and build the
published site from that ``release`` checkout.

The workflow is:

1. Make the relevant changes under ``docs/source/`` (for example
   ``docs/source/download.rst``) on a branch based on ``main``.

2. Build the website locally from source.

   Install the doc-build dependencies if needed.::

     pip install -e ".[docs]"


   Build the website locally.::

     make -C docs html

   The rendered site will be in ``docs/build/html``. You can inspect the local
   build, e.g.::

     firefox docs/build/html/index.html


3. Commit the changes in the Thetis repo and open a PR to ``main``.

4. If the change should appear on the published website before the next release
   reset, apply the same content change to ``release`` using the shared-fix
   workflow above, then check out ``release`` and rebuild the website locally.

5. Copy the contents of ``docs/build/html/`` from the ``release`` build into the
   ``thetisproject.github.io`` repository and merge there as well.

   It is encouraged to do this via a PR in the rendered-site repository and to
   check on GitHub that the rendered changes are as intended. If unsure, ask for
   a review.

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
archived on Zenodo. To fix it, a GitHub organization owner for
``thetisproject`` (not just a repository collaborator) may need to:

1. Delete the Zenodo webhook in the GitHub repository settings.
2. Disconnect and reconnect *their* GitHub <-> Zenodo account connection and
   re-authenticate.
3. Re-enable the Thetis repository in Zenodo's GitHub integration settings (if
   it does not re-enable automatically).

It is often also worth the person doing the release reconnecting their personal
GitHub/Zenodo connection.
