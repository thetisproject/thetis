lint:
	@echo "    Linting thetis codebase"
	@python -m flake8 thetis
	@echo "    Linting thetis test suite"
	@python -m flake8 test
	@echo "    Linting thetis scripts"
	@python -m flake8 scripts --filename=*
	@echo "    Linting thetis examples"
	@python -m flake8 examples

