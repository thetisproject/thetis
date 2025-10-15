all: invert plot

CASE        ?= Uniform # Regions IndependentPointsScheme GradientReg HessianReg
STATIONS    = stationA stationB stationC stationD stationE
PARALLEL    = 1

invert:
	mpiexec --use-hwthread-cpus -np $(PARALLEL) python inverse_problem.py --case $(CASE) --no-consistency-test --no-taylor-test

plot:
	for station in $(STATIONS); do \
		python3 plot_velocity_progress.py -s $$station --case $(CASE); \
	done; \

clean:
	rm -rf __pycache__
	rm -rf *.png
