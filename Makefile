lint:
	@echo "    Linting cofs codebase"
	@flake8 cofs
	@echo "    Linting cofs test suite"
	@flake8 test
	@echo "    Linting cofs examples"
	@flake8 examples
	@echo "    Linting cofs scripts"
	@flake8 scripts

