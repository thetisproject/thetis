all: forward plot_obs invert plot

CASE              	?= GradientReg # Regions IndependentPointsScheme GradientReg HessianReg
STATIONS          	= stationA stationB stationC stationD stationE stationF stationG
ENSEMBLE          	?= false
RANKS_PER_MEMBER  	?= 2
N_MEMBERS         	?= 7
PARALLEL          	?= $(shell expr $(N_MEMBERS) \* $(RANKS_PER_MEMBER))

ifeq ($(ENSEMBLE),true)
PARALLEL          	?= $(shell expr $(N_MEMBERS) \* $(RANKS_PER_MEMBER))
MPIEXEC				= mpiexec --use-hwthread-cpus -np $(PARALLEL)
RUN_MODE_FORWARD 	= --ensemble --ranks-per-member $(RANKS_PER_MEMBER)
RUN_MODE_INVERT  	= --ensemble --ranks-per-member $(RANKS_PER_MEMBER)
RUN_MODE_PLOT	  	= --ensemble
else
PARALLEL          	?= 4
MPIEXEC          	= mpiexec --use-hwthread-cpus -np $(PARALLEL)
RUN_MODE_FORWARD 	=
RUN_MODE_INVERT  	=
RUN_MODE_PLOT	  	=
endif

forward:
	$(MPIEXEC) python forward_run.py $(RUN_MODE_FORWARD)

plot_obs:
	python plot_observed_elev.py $(RUN_MODE_PLOT)

invert:
	$(MPIEXEC) python inverse_problem.py $(RUN_MODE_INVERT) --case $(CASE) --no-consistency-test --no-taylor-test

plot:
	for station in $(STATIONS); do \
		python3 plot_velocity_progress.py -s $$station --case $(CASE) $(RUN_MODE_PLOT); \
	done; \

clean:
	rm -rf __pycache__
	rm -rf *.png
