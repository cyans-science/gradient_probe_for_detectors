#!/usr/bin/env bash
# run_gradient_statistics.sh
#
# Purpose: 
# run gradient_statistics.py for the 20261019_Fisher_depth_probe run inside the plotting container. Run run_build_event_index.sh first, so the selection gets root file and entry per hit.
#
# Arguments on the command line override the default arguments 
#   ./run_gradient_statistics.sh --top-n 10 --min-stages 2
#
# cyan.jo, Claude Opus5.5 Extra, 2026-10-21

CONTAINER=/ceph/cjo/containers/plotting.sif
SOURCE_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P )
UTILS_DIR=/work/cjo/Utils   # PlotUtils.py, ExternalFunctions.py

ARGS=(
    --probe-dir  /work/cjo/e2ereco_topas/20261019_Fisher_depth_probe/Results/20261019_Fisher_depth_probe_per_hit/grad
    --out-dir    /ceph/cjo/GradientProbe/runs/20261019_Fisher_depth_probe
    --top-n      20
    --min-stages 1
)

apptainer exec \
    -B /work \
    -B /ceph \
    --env PYTHONPATH="${UTILS_DIR}" \
    --pwd "${SOURCE_DIR}" \
    "${CONTAINER}" \
    python3 gradient_statistics.py "${ARGS[@]}" "$@"