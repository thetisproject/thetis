all: invert plot

CASE        ?= Uniform # Regions IndependentPointsScheme GradientReg HessianReg
OPTIMISER   ?= ROL # L-BFGS-B
STATIONS    = stationA stationB stationC stationD stationE
PARALLEL    = 1

invert:
	mpiexec --use-hwthread-cpus -np $(PARALLEL) python inverse_problem.py --case $(CASE) --optimiser $(OPTIMISER) --no-consistency-test --no-taylor-test

plot:
	for station in $(STATIONS); do \
		python3 plot_velocity_progress.py -s $$station --case $(CASE) --optimiser $(OPTIMISER); \
	done; \

clean:
	rm -rf __pycache__
	rm -rf *.png
