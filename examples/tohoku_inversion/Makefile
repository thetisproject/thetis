all: invert plot

MODELS	= DG0 CG1 radial box okada

invert:
	for model in $(MODELS); do \
		python3 inverse_problem.py --source-model $$model --no-consistency-test --no-taylor-test; \
	done

plot:
	python3 plot_elevation_progress.py
	python3 plot_elevation_optimised.py
	python3 plot_convergence.py

clean:
	rm -rf outputs*
	rm -rf __pycache__
