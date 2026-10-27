#!/usr/bin/env bash
# run_argmax_position.sh
#
# Purpose: run diagnostics/argmax_position.py in the plotting container on the
# probe CSVs of the 20261019_Fisher_depth_probe run. Extra arguments are
# passed on, e.g.
#   bash diagnostics/run_argmax_position.sh --examples 10
#
# cyan.jo, Claude Opus 5.5, 2026-10-23

CONTAINER=/ceph/cjo/containers/plotting.sif
REPO_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd -P )
PROBE_DIR=/work/cjo/e2ereco_topas/20261019_Fisher_depth_probe/Results/20261019_Fisher_depth_probe_per_hit/grad

apptainer exec \
    -B /work \
    -B /ceph \
    --pwd "${REPO_DIR}" \
    "${CONTAINER}" \
    python3 diagnostics/argmax_position.py --probe-dir "${PROBE_DIR}" "$@"