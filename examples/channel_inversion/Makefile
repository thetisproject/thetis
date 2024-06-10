all: invert plot

CONTROLS	= Bathymetry Manning InitialElev
STATIONS    = stationA stationB stationC stationD stationE

invert:
	for controls in $(CONTROLS); do \
		python3 inverse_problem.py -c $$controls --no-consistency-test --no-taylor-test; \
	done

plot:
	for controls in $(CONTROLS); do \
		for station in $(STATIONS); do \
			python3 plot_elevation_progress.py -c $$controls -s $$station; \
		done; \
	done

clean:
	rm -rf __pycache__
	rm -rf outputs_Bathymetry* outputs_Manning* outputs_InitialElev*
	cd outputs_forward && rm -rf init_bathymetry_2d Elevation2d Velocity2d log
