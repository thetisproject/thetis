#!/usr/bin/env bash
# Single-turbine centreline wake-recovery sensitivity.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
firedrake_python=${FIREDRAKE_PYTHON:-/home/aangelou/firedrake/bin/python}
water_depth=${1:-50}
results_dir="results/depth_${water_depth}m"
average_steps=5

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONPATH="${script_dir}/../../..${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${script_dir}/results/matplotlib"
cd "${script_dir}"

gmsh -2 -format msh2 -o single_turbine.msh single_turbine.geo

"${firedrake_python}" single_turbine.py --label constant --depth "${water_depth}" --results-dir "${results_dir}" --average-steps "${average_steps}"
"${firedrake_python}" single_turbine.py --label cs_0p1 --cs 0.1 --depth "${water_depth}" --results-dir "${results_dir}" --average-steps "${average_steps}"
"${firedrake_python}" single_turbine.py --label cs_1 --cs 1 --depth "${water_depth}" --results-dir "${results_dir}" --average-steps "${average_steps}"
"${firedrake_python}" single_turbine.py --label cs_5 --cs 5 --depth "${water_depth}" --results-dir "${results_dir}" --average-steps "${average_steps}"
"${firedrake_python}" single_turbine.py --label cs_10 --cs 10 --depth "${water_depth}" --results-dir "${results_dir}" --average-steps "${average_steps}"
"${firedrake_python}" plot_velocity_deficit.py --results-dir "${results_dir}" --depth "${water_depth}" --average-steps "${average_steps}"
