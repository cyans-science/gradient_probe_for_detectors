#!/usr/bin/env bash
# run_all.sh
#
# Purpose: run the whole chain for one run.
#   1. run_build_event_index.sh    event_id -> root file and entry (checks L1, L3)
#   2. run_gradient_statistics.sh  gradient distributions and hit selection
#   3. run_beta_statistics.sh      beta histogram and (z, beta) per stage
#   4. run_gradient_hit_viewer.sh  event views (check L5)
# The run is set here, in the "Run settings" block below: run_all passes those
# values to every step as arguments. Each wrapper appends them after its own
# ARGS and argparse keeps the last value, so these override the wrappers'
# defaults. The wrappers stay unchanged and still run alone with their own
# defaults.
#
# Any run setting can be changed for one call from the environment:
#   RUN_NAME=20261113_Fisher_depth_probe ./run_all.sh
#   TOP_N=10 MIN_STAGES=2 ./run_all.sh
# Stops at the first step that fails. Arguments go to the viewer, after the
# settings below, so they win, e.g.
#   ./run_all.sh --event-id 1234 --animate none
# Start at a later step when the earlier outputs are already there:
#   FROM=gradient_statistics ./run_all.sh
#   FROM=beta_statistics ./run_all.sh
#   FROM=gradient_hit_viewer ./run_all.sh
# Each step's output is also saved in LOG_DIR (default <RUN_DIR>/logs), and
# the resolved run settings in LOG_DIR/<stamp>_settings.txt.
# In the background, so you can log out (combine with FROM and arguments):
#   BACKGROUND=1 FROM=gradient_hit_viewer ./run_all.sh
# It prints the PID and how to follow or stop the run; the PID is also kept
# in LOG_DIR/run_all.pid.
#
# cyan.jo, Claude Opus5.5 (Extra), 2026-10-21
# 2026-10-23: BACKGROUND=1
# 2026-11-19: step beta_statistics (needs only the probe dump, runs before the viewer)
# 2026-11-20: one block of run settings, passed to every step as overriding arguments

set -euo pipefail

# ============================== Run settings ==============================
# Everything that changes from run to run. Each line keeps a value already
# set in the environment.
RUN_NAME=${RUN_NAME:-20261119_Fisher_probe_gross}
# Where the training ran: e2ereco_local or e2ereco_topas.
TRAIN_BASE=${TRAIN_BASE:-/work/cjo/e2ereco_topas}
DUMP_DIR=${DUMP_DIR:-${TRAIN_BASE}/${RUN_NAME}/Results/${RUN_NAME}_per_hit}
PROBE_DIR=${PROBE_DIR:-${DUMP_DIR}/grad}
BETA_FILE=${BETA_FILE:-${DUMP_DIR}/beta.csv}
RUN_DIR=${RUN_DIR:-/ceph/cjo/GradientProbe/runs/${RUN_NAME}}
LOG_DIR=${LOG_DIR:-${RUN_DIR}/logs}
# Dataset of the training run, its source .root files, and the converter
# that made the .djctd files.
DJCDC=${DJCDC:-/ceph/jkiesele/datasets/dice/qcd_singlejet_pt20-200_eta3_noPromptLep/merged_global_tR0p65/djc/dataCollection.djcdc}
ROOT_DIR=${ROOT_DIR:-/ceph/jkiesele/datasets/dice/qcd_singlejet_pt20-200_eta3_noPromptLep/merged_global_tR0p65}
CONVERTER=${CONVERTER:-/work/cjo/e2ereco_bernoulli_reproduction/source/modules/datastructures/TrainData_DICE_cellparent.py}
# Loader settings of the training script; they fix the event_id order.
BATCH_SIZE=${BATCH_SIZE:-10000}
VAL_FRACTION=${VAL_FRACTION:-0.2}
# Probe stages in network order (= CSV names in PROBE_DIR). YYYYMMDD probes
# block N at prop_gnN (sender gate, before the kNN aggregation).
STAGES=${STAGES:-prop_gn1 prop_gn2 prop_gn3 prop_gn4 post_output_head}
# Selection: the viewer's --selection is derived, so it always names the
# selection that the statistics step of this call writes.
TOP_N=${TOP_N:-30}
MIN_STAGES=${MIN_STAGES:-1}
EVENTS_PER_STAGE=${EVENTS_PER_STAGE:-30}
SELECTION=${SELECTION:-top${TOP_N}_min${MIN_STAGES}}
SETTINGS=(RUN_NAME DUMP_DIR PROBE_DIR BETA_FILE RUN_DIR LOG_DIR
          DJCDC ROOT_DIR CONVERTER BATCH_SIZE VAL_FRACTION
          STAGES TOP_N MIN_STAGES EVENTS_PER_STAGE SELECTION)
# ==========================================================================

read -ra stages <<< "${STAGES}"

# Per-step overrides. Not listed here (same for every run, so they stay in the
# wrappers): --djcdc, --converter, --root-dir, --fps, --per-event-histograms.
BUILD_ARGS=(
    --djcdc        "${DJCDC}"
    --root-dir     "${ROOT_DIR}"
    --converter    "${CONVERTER}"
    --batch-size   "${BATCH_SIZE}"
    --val-fraction "${VAL_FRACTION}"
    --probe-csv    "${PROBE_DIR}/${stages[0]}.csv"
    --out-dir      "${RUN_DIR}"
)
STATISTICS_ARGS=(
    --probe-dir  "${PROBE_DIR}"
    --out-dir    "${RUN_DIR}"
    --top-n      "${TOP_N}"
    --min-stages "${MIN_STAGES}"
)
BETA_ARGS=(
    --probe-dir "${PROBE_DIR}"
    --beta-file "${BETA_FILE}"
    --out-dir   "${RUN_DIR}"
    --stages    "${stages[@]}"
)
VIEWER_ARGS=(
    --run-dir          "${RUN_DIR}"
    --converter        "${CONVERTER}"
    --selection        "${SELECTION}"
    --events-per-stage "${EVENTS_PER_STAGE}"
)

SOURCE_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P )
FROM=${FROM:-build_event_index}
STAMP=$(date +%Y%m%d_%H%M%S)

STEPS=(build_event_index gradient_statistics beta_statistics gradient_hit_viewer)
if [[ ! " ${STEPS[*]} " == *" ${FROM} "* ]]; then
    echo "FROM must be one of: ${STEPS[*]}" >&2
    exit 1
fi
# Fail now rather than after the first steps when the dump path is wrong.
if [[ ! -d ${PROBE_DIR} ]]; then
    echo "PROBE_DIR does not exist: ${PROBE_DIR}" >&2
    exit 1
fi
mkdir -p "${LOG_DIR}"

if [[ ${BACKGROUND:-0} == 1 ]]; then
    # Restart this script detached from the terminal: nohup ignores the hang-up
    # signal sent at logout; setsid gives it its own process group, so one kill
    # stops every step, container and worker process together. The resolved
    # settings are passed on, so the restart uses exactly these values.
    out="${LOG_DIR}/${STAMP}_run_all.out"
    env_settings=()
    for name in "${SETTINGS[@]}"; do env_settings+=("${name}=${!name}"); done
    env "${env_settings[@]}" BACKGROUND=0 FROM="${FROM}" \
        setsid nohup bash "${BASH_SOURCE[0]}" "$@" \
        > "${out}" 2>&1 < /dev/null &
    pid=$!
    echo "${pid}" > "${LOG_DIR}/run_all.pid"
    echo "Running ${RUN_NAME} in the background (PID ${pid}, FROM=${FROM}); you can log out."
    echo "  follow:   tail -f ${out}"
    echo "  running?  ps -o pid,etime,cmd -g ${pid}"
    echo "  stop:     kill -- -${pid}"
    exit 0
fi

# Show the settings and keep them next to the logs.
{
    for name in "${SETTINGS[@]}"; do
        printf '%-17s %s\n' "${name}" "${!name}"
    done
} | tee "${LOG_DIR}/${STAMP}_settings.txt"

# Print Python output as it happens instead of in blocks when piped to tee.
export PYTHONUNBUFFERED=1

started=0
run_step() {
    local name=$1
    shift
    if [[ ${started} == 0 && ${name} != "${FROM}" ]]; then
        echo "=== ${name} skipped (FROM=${FROM})"
        return
    fi
    started=1
    local log="${LOG_DIR}/${STAMP}_${name}.log"
    local start=${SECONDS}
    echo "=== ${name} started $(date +%H:%M:%S), log: ${log}"
    # The command with the overriding arguments, at the top of the step's log.
    if ! { echo "+ $*"; "$@"; } 2>&1 | tee "${log}"; then
        echo "=== ${name} FAILED after $(( SECONDS - start )) s, see ${log}" >&2
        exit 1
    fi
    echo "=== ${name} done in $(( SECONDS - start )) s"
}

run_step build_event_index   bash "${SOURCE_DIR}/run_build_event_index.sh"   "${BUILD_ARGS[@]}"
run_step gradient_statistics bash "${SOURCE_DIR}/run_gradient_statistics.sh" "${STATISTICS_ARGS[@]}"
run_step beta_statistics     bash "${SOURCE_DIR}/run_beta_statistics.sh"     "${BETA_ARGS[@]}"
run_step gradient_hit_viewer bash "${SOURCE_DIR}/run_gradient_hit_viewer.sh" "${VIEWER_ARGS[@]}" "$@"