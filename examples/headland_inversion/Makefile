all: invert plot

CASE        ?= IndependentPointsScheme # Constant Regions NodalFreedom
CONTROLS	= Manning # This option is not used in this example - Manning only
STATIONS    = stationA stationB stationC stationD stationE

invert:
	python3 inverse_problem.py --case $(CASE) --no-consistency-test --no-taylor-test; \

plot:
	for station in $(STATIONS); do \
		python3 plot_velocity_progress.py -s $$station --case $(CASE); \
	done; \

clean:
	rm -rf __pycache__
	rm -rf *.png
