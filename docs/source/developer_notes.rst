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

CI reflects this by testing ``main`` against the Firedrake ``dev-main`` Docker
image and testing ``release`` against the Firedrake ``latest`` Docker image.


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

* ``push.yml`` runs on pushes to ``main`` and ``release``.
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

* Target branch: PRs should usually target ``main``. If a change needs to ship
  on the stable branch, it must land on ``release`` first (see the
  release-branch policy below).
* CI: PRs are tested by ``.github/workflows/pr.yml`` against the appropriate
  Firedrake Docker image (based on the PR base branch).
* Before opening/merging a PR, it is expected that lint is clean
  (``make lint``),
  tests are run where practical (see the CI commands above), and docs changes
  render as expected (build the Sphinx site locally if you edited
  ``docs/source/*``).

**Branch hygiene**: on feature branches it is fine (and often encouraged) to
clean up history (interactive rebase, squash, fixups) and force-push while
iterating on a PR.

**Merging**: for normal PRs into ``main`` or ``release``, choose a merge
strategy that keeps history readable. Hard rule: PRs that merge one long-lived
branch into another (``release`` -> ``main`` sync, or advancing ``release``
to a vetted ``main`` commit) must use "Create a merge commit" and must not use
squash/rebase merge.


Maintaining **release**
-----------------------

Policy (No Cherry-Picks)
~~~~~~~~~~~~~~~~~~~~~~~~

Thetis follows a Firedrake-style branching policy:

* ``release`` must always be an ancestor of ``main`` (everything in ``release``
  is also in ``main``).
* Changes that should ship on both branches (e.g. bug fixes) should land on
  ``release`` first, then ``release`` is merged into ``main``.
* ``release`` is not periodically "restarted". Users should be able to
  ``git pull`` their local ``release`` branch normally.

Alternatives approaches would reduce merge-commit noise, but they trade away
invariants that this policy is built around:

* Cherry-pick hotfixes from ``release`` to ``main``:
  This creates different SHAs for the same change and can make later merges and
  "advance release" operations harder to reason about.
* Periodically reset ``release`` to ``main`` (force-updating ``release``):
  This rewrites public history. Users cannot rely on ``git pull`` for
  ``release`` and must hard-reset their local branch. It also increases the
  risk of accidental history rewriting.


When Does **release** Move?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``release`` is updated in two main cases:

1. Firedrake makes a new stable (major) release (e.g. ``2025.10`` ->
   ``2026.4``) and Thetis ``release`` is advanced to a recent, known-good
   commit from ``main`` that is compatible with that Firedrake stable stack.
2. A user-facing bugfix is needed on the stable branch.

Keep ``release`` changes minimal and compatibility-driven.

Workflow: Hotfix on **release**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Create a branch from ``release`` (not from ``main``) and open a PR with base
   branch ``release``.
2. Merge the PR (squash merge is preferable as it removes unnecessary merge
   commits).
3. Sync ``release`` into ``main`` (required, see below).

Workflow: Special Case (**release**-Only Workaround)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Occasionally, ``release`` may need a workaround for a bug that only exists on
``release`` (for example due to an older Firedrake dependency stack). In this
case we still follow the "no cherry-picks" policy (so ``release`` remains an
ancestor of ``main``), but we immediately undo the workaround on ``main`` after
syncing.

1. Create a branch from ``release`` and open a PR with base branch ``release``.
2. Merge the PR (squash and merge the hot-fix PR).
3. Sync ``release`` into ``main`` as usual (required).
4. Immediately create a commit on ``main`` that undoes the hot-fix (typically a
   ``git revert`` of the hot-fix commit or the squash-merge commit).

This maintains history and keeps branch ancestry correct, at the expense of
additional (arguably wasteful) commits on ``main``.

Workflow: Advance **release** to a Vetted **main** Commit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the common workflow when Firedrake makes a new stable release and we
want ``release`` users to get the compatible developments from ``main``.

1. Ensure ``release`` is updated and merge ``main`` into it, then checkout
   another branch
   (saves an unnecessary merge commit):

   .. code-block:: none

      git checkout release
      git pull
      git merge --no-ff main
      git checkout -b release-update
      git push -u origin release-update

2. Open a PR with base branch ``release`` from ``release-update`` and merge it
   using "Create a merge commit" (do not squash merge or rebase merge). Make
   sure ``pyproject.toml`` is updated with the new version number if you are
   cutting a new tagged/user-visible Thetis release (so ``pip list`` reports
   the intended version).
3. Sync ``release`` into ``main`` (required). This keeps ``release`` an
   ancestor of ``main`` and ensures the merge commit that advanced ``release``
   is recorded in ``main`` history.
4. Cut a tagged release (see below).

Workflow: Syncing **release** into **main**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If branch protection requires PR branches to be "up to date", GitHub may show
the ``release`` -> ``main`` sync PR as out-of-date and offer an "Update branch"
button. Do not click this: it merges ``main`` into ``release``.

Instead, create a temporary sync branch *from* ``main`` *after* merging
``release`` into it, then open a PR to ``main``:

.. code-block:: none

   git checkout main
   git pull
   git merge --no-ff release
   git checkout -b sync/release-into-main
   git push -u origin sync/release-into-main

Merge the sync PR using "Create a merge commit" (do not squash/rebase).


History Shape (Why Cross-Branch PRs Add Merge Commits)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under this policy we intentionally preserve ancestry (no cherry-picks) and keep
``release`` pullable for users (no history rewrites). The tradeoff is that
cross-branch operations are recorded as merge commits.

Commit accounting (typical GitHub setup):

* Hotfix (``release`` then sync to ``main``):
  The sync step typically adds two merge commits on the ``main`` side of
  history. ``release`` will end up 2 behind main, per hotfix. This is reset at
  each advance.
* Advance ``release`` (merge ``main`` into ``release-update`` then sync back):
  Advancing ``release`` and then syncing back typically adds one merge
  commit on ``release`` and one merge commit on ``main``. ``release`` will end
  up 1 behind main at the end of this process.

This is expected and is the cost of:

* no cherry-picking between long-lived branches, and
* enforcing cross-branch synchronization via PRs under branch protection.


Cutting A Tagged Release
~~~~~~~~~~~~~~~~~~~~~~~~

When cutting a tagged release, bump the packaged version in ``pyproject.toml``,
push a tag, create a GitHub Release, and verify Zenodo archived the release
(see the Zenodo section below). If the release includes documentation changes,
update the published website at the same time.


Tag/version convention
~~~~~~~~~~~~~~~~~~~~~~

The repository currently contains multiple historical tag naming schemes (for
example ``2026.4.0`` / ``2025.10.1`` as well as older ``Thetis_YYYYMMDD``-style
tags). For new releases, we prefer a tag that matches the packaged version in
``pyproject.toml`` and align the version scheme with Firedrake's release
cadence.


Thetis website
---------------

Content
~~~~~~~

Thetis documentation content lives in this repository under ``docs/source`` and
is built with Sphinx (see ``docs/Makefile`` and ``docs/source/conf.py``).

Thetis has a single published documentation website. The published site is
updated whenever the ``release`` branch is updated, and it must be built from
the ``release`` branch (do not publish a site built from ``main``).

If you need the equivalent rendered documentation for ``main``, build it
locally from a ``main`` checkout using the same instructions below.

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

4. Commit the changes in the Thetis repo and open a PR.

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
archived on Zenodo. To fix it, a GitHub organization owner for
``thetisproject`` (not just a repository collaborator) may need to:

1. Delete the Zenodo webhook in the GitHub repository settings.
2. Disconnect and reconnect *their* GitHub <-> Zenodo account connection and
   re-authenticate.
3. Re-enable the Thetis repository in Zenodo's GitHub integration settings (if
   it does not re-enable automatically).

It is often also worth the person doing the release reconnecting their personal
GitHub/Zenodo connection.
