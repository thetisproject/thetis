all: invert plot

CASE        ?= Uniform # Regions IndependentPointsScheme GradientReg HessianReg
STATIONS    = stationA stationB stationC stationD stationE stationF stationG
PARALLEL    = 1
IP_INTERPOLATION ?= linear # linear cubic rbf
IP_RBF_KERNEL ?= thin_plate_spline
IP_RBF_SMOOTHING ?= 0.0

invert:
	mpiexec --use-hwthread-cpus -np $(PARALLEL) python inverse_problem.py --case $(CASE) --no-consistency-test --no-taylor-test --ip-interpolation $(IP_INTERPOLATION) --ip-rbf-kernel $(IP_RBF_KERNEL) --ip-rbf-smoothing $(IP_RBF_SMOOTHING)

plot:
	for station in $(STATIONS); do \
		python3 plot_velocity_progress.py -s $$station --case $(CASE) --ip-interpolation $(IP_INTERPOLATION); \
	done; \

clean:
	rm -rf __pycache__
	rm -rf *.png
