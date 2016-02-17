lint:
	@echo "    Linting thetis codebase"
	@flake8 thetis
	@echo "    Linting thetis test suite"
	@flake8 test
	@echo "    Linting thetis examples"
	@flake8 examples
	@echo "    Linting thetis scripts"
	@flake8 scripts

