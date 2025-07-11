all: invert plot

CASE        ?= Constant # Regions IndependentPointsScheme NodalFreedom
STATIONS    = stationA stationB stationC stationD stationE
PARALLEL    = 1

invert:
	mpiexec -np $(PARALLEL) python inverse_problem.py --case $(CASE) --no-consistency-test --no-taylor-test

plot:
	for station in $(STATIONS); do \
		python3 plot_velocity_progress.py -s $$station --case $(CASE); \
	done; \

clean:
	rm -rf __pycache__
	rm -rf *.png
