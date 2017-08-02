lint:
	@echo "    Linting firedrake codebase"
	@python -m flake8 firedrake
	@echo "    Linting firedrake test suite"
	@python -m flake8 tests
	@echo "    Linting firedrake scripts"
	@python -m flake8 scripts --filename=*

