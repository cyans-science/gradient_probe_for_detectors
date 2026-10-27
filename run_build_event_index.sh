#!/usr/bin/env bash
# run_build_event_index.sh
#
# Purpose:
# run build_event_index.py for the 20261019_Fisher_depth_probe run inside the plotting container, which has djcdata. If uproot is in the same container, check L3 runs too; otherwise it is reported as skipped.
# Arguments given on the command line are appended to the ones below and override them
#
# cyan.jo, Claude Opus5.5 Extra, 2026-10-21

CONTAINER=/ceph/cjo/containers/plotting.sif
SOURCE_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P )
PROBE_DIR=/work/cjo/e2ereco_topas/20261019_Fisher_depth_probe/Results/20261019_Fisher_depth_probe_per_hit/grad

ARGS=(
    --djcdc     /ceph/cjo/DICEv1_4_0_small/SingleJet_pt30_eta1p2_1p7_100k/ten_files_conversion/dataCollection.djcdc
    --converter /work/cjo/e2ereco_bernoulli_reproduction/source/modules/datastructures/TrainData_DICE_cellparent.py
    --root-dir  /ceph/cjo/DICEv1_4_0/SingleJet_pt30_eta1p2_1p7_100k
    --probe-csv "${PROBE_DIR}/post_initial.csv"   # any one stage: all stages list the same hits
    --out-dir   /ceph/cjo/GradientProbe/runs/20261019_Fisher_depth_probe
)

apptainer exec \
    -B /work \
    -B /ceph \
    --pwd "${SOURCE_DIR}" \
    "${CONTAINER}" \
    python3 build_event_index.py "${ARGS[@]}" "$@"