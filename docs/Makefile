# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXAPIDOC  = sphinx-apidoc
SPHINXPROJ    = Thetis
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean

GENERATED_FILES = source/teamgrid.rst

source/teamgrid.rst: source/team.py
	cd source; python team.py

SPHINX_TARGETS = html dirhtml singlehtml pickle json htmlhelp qthelp devhelp epub \
	latex latexpdf latexpdfja text man texinfo info gettext changes \
	xml pseudoxml linkcheck doctest coverage

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
$(SPHINX_TARGETS): Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html: apidoc $(GENERATED_FILES)

.PHONY: apidoc
apidoc:
	$(SPHINXAPIDOC) $$(python -c 'import thetis; import os; print os.path.dirname(thetis.__file__)') -o source -f -T


clean:
	-git clean -fdx $(BUILDDIR)/html/
	-rm -rf $(BUILDDIR)/doctrees
	-rm -rf $(GENERATED_FILES)
