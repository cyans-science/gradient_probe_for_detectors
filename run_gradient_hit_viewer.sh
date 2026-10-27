#!/usr/bin/env bash
# run_gradient_hit_viewer.sh
#
# Purpose: run gradient_hit_viewer.py for the 20261019_Fisher_depth_probe run inside the plotting container. Needs the outputs of run_build_event_index.sh and run_gradient_statistics.sh in the run directory.
#
# Arguments on the command line override the default arguments
#   ./run_gradient_hit_viewer.sh --event-id 1234 --animate none
#
# cyan.jo, Claude Opus5.5 Extra, 2026-10-21

CONTAINER=/ceph/cjo/containers/plotting.sif
SOURCE_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P )
UTILS_DIR=/work/cjo/Utils   # PlotUtils.py, ExternalFunctions.py

ARGS=(
    --run-dir   /ceph/cjo/GradientProbe/runs/20261019_Fisher_depth_probe
    --converter /work/cjo/e2ereco_bernoulli_reproduction/source/modules/datastructures/TrainData_DICE_cellparent.py
    --fps 2
    --selection top20_min1 --events-per-stage 20
)

apptainer exec \
    -B /work \
    -B /ceph \
    --env PYTHONPATH="${UTILS_DIR}" \
    --pwd "${SOURCE_DIR}" \
    "${CONTAINER}" \
    python3 gradient_hit_viewer.py "${ARGS[@]}" "$@"