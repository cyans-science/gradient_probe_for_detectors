#!/usr/bin/env bash
# run_beta_statistics.sh
#
# Purpose: run beta_statistics.py for the 20261118_Fisher_depth_probe run inside the plotting container. Reads the same probe dump as run_gradient_statistics.sh (stage CSVs) and the beta CSV of that dump; writes to <run-dir>/beta/.
#
# Arguments on the command line override the default arguments
#   ./run_beta_statistics.sh --max-points 200000 --no-per-stage
#
# cyan.jo, Claude Opus5.5 Extra, 2026-11-19

CONTAINER=/ceph/cjo/containers/plotting.sif
SOURCE_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P )
UTILS_DIR=/work/cjo/Utils   # PlotUtils.py, ExternalFunctions.py

ARGS=(
    # Same --probe-dir as in run_gradient_statistics.sh (the <dump>/grad directory).
    # --probe-dir /work/cjo/e2ereco_local/20261113_Fisher_depth_probe/Results/20261113_Fisher_depth_probe_per_hit/grad
    # --beta-file /work/cjo/e2ereco_local/20261113_Fisher_depth_probe/Results/20261113_Fisher_depth_probe_per_hit/beta.csv
    # --out-dir   /ceph/cjo/GradientProbe/runs/20261113_Fisher_depth_probe
    # --probe-dir /work/cjo/e2ereco_local/20261118_Fisher_depth_probe/Results/20261118_Fisher_depth_probe_per_hit/grad
    # --beta-file /work/cjo/e2ereco_local/20261118_Fisher_depth_probe/Results/20261118_Fisher_depth_probe_per_hit/beta.csv
    # --out-dir   /ceph/cjo/GradientProbe/runs/20261118_Fisher_depth_probe
    # --probe-dir /work/cjo/e2ereco_topas/20261119_Fisher_probe_gross/Results/20261119_Fisher_probe_gross_per_hit/grad
    # --beta-file /work/cjo/e2ereco_topas/20261119_Fisher_probe_gross/Results/20261119_Fisher_probe_gross_per_hit/beta.csv
    # --out-dir   /ceph/cjo/GradientProbe/runs/20261119_Fisher_probe_gross
    # --stages prop_gn1 prop_gn2 prop_gn3 prop_gn4 post_output_head
    --probe-dir /work/cjo/topas_gradient_probe/20261126_gradient_probe/Results/20261126_gradient_probe_per_hit/grad
    --beta-file /work/cjo/topas_gradient_probe/20261126_gradient_probe/Results/20261126_gradient_probe_per_hit/beta.csv
    --out-dir   /ceph/cjo/GradientProbe/runs/20261126_gradient_probe
    # init, gn1-gn4, output. 20261118 probes block N at prop_gnN (sender gate,
    # before the kNN aggregation), not post_gnN; names = CSV names in --probe-dir.
    --stages gradient_gn1 gradient_gn2 gradient_gn3 gradient_gn4 gradient_post_output
)

apptainer exec \
    -B /work \
    -B /ceph \
    --env PYTHONPATH="${UTILS_DIR}" \
    --pwd "${SOURCE_DIR}" \
    "${CONTAINER}" \
    python3 beta_statistics.py "${ARGS[@]}" "$@"