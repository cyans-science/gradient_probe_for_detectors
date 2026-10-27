#!/usr/bin/env bash
# run_compare_djctd_root.sh
#
# Purpose: run diagnostics/compare_djctd_root.py in the plotting container for
# both validation files of the 20261019_Fisher_depth_probe run.
# Extra arguments are passed on, e.g.
#   bash diagnostics/run_compare_djctd_root.sh --n-events 10
#
# cyan.jo, Claude Opus 5.5, 2026-10-21

CONTAINER=/ceph/cjo/containers/plotting.sif
REPO_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd -P )

ARGS=(
    --run-dir   /ceph/cjo/GradientProbe/runs/20261019_Fisher_depth_probe
    --converter /work/cjo/e2ereco_bernoulli_reproduction/source/modules/datastructures/TrainData_DICE_cellparent.py
)

for file in singlejet_81 singlejet_75; do
    apptainer exec \
        -B /work \
        -B /ceph \
        --pwd "${REPO_DIR}" \
        "${CONTAINER}" \
        python3 diagnostics/compare_djctd_root.py "${ARGS[@]}" --file "${file}" "$@"
    echo
done