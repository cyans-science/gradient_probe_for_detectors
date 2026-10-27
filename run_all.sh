#!/usr/bin/env bash
# run_all.sh
#
# Purpose: run the whole chain for the 20261019_Fisher_depth_probe run.
#   1. run_build_event_index.sh    event_id -> root file and entry (checks L1, L3)
#   2. run_gradient_statistics.sh  gradient distributions and hit selection
#   3. run_gradient_hit_viewer.sh  event views (check L5)
# Stops at the first step that fails. Arguments go to the viewer, e.g.
#   ./run_all.sh --event-id 1234 --animate none
# Start at a later step when the earlier outputs are already there:
#   FROM=gradient_statistics ./run_all.sh
#   FROM=gradient_hit_viewer ./run_all.sh
# Each step's output is also saved in LOG_DIR (override: LOG_DIR=... ./run_all.sh).
# In the background, so you can log out (combine with FROM and arguments):
#   BACKGROUND=1 FROM=gradient_hit_viewer ./run_all.sh
# It prints the PID and how to follow or stop the run; the PID is also kept
# in LOG_DIR/run_all.pid.
#
# cyan.jo, Claude Opus5.5 (Extra), 2026-10-21
# 2026-10-23: BACKGROUND=1

set -euo pipefail

SOURCE_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P )
LOG_DIR=${LOG_DIR:-/ceph/cjo/GradientProbe/runs/20261019_Fisher_depth_probe/logs}
FROM=${FROM:-build_event_index}
STAMP=$(date +%Y%m%d_%H%M%S)

STEPS=(build_event_index gradient_statistics gradient_hit_viewer)
if [[ ! " ${STEPS[*]} " == *" ${FROM} "* ]]; then
    echo "FROM must be one of: ${STEPS[*]}" >&2
    exit 1
fi
mkdir -p "${LOG_DIR}"

if [[ ${BACKGROUND:-0} == 1 ]]; then
    # Restart this script detached from the terminal: nohup ignores the hang-up
    # signal sent at logout; setsid gives it its own process group, so one kill
    # stops every step, container and worker process together.
    out="${LOG_DIR}/${STAMP}_run_all.out"
    BACKGROUND=0 FROM=${FROM} setsid nohup bash "${BASH_SOURCE[0]}" "$@" \
        > "${out}" 2>&1 < /dev/null &
    pid=$!
    echo "${pid}" > "${LOG_DIR}/run_all.pid"
    echo "Running in the background (PID ${pid}, FROM=${FROM}); you can log out."
    echo "  follow:   tail -f ${out}"
    echo "  running?  ps -o pid,etime,cmd -g ${pid}"
    echo "  stop:     kill -- -${pid}"
    exit 0
fi

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
    if ! "$@" 2>&1 | tee "${log}"; then
        echo "=== ${name} FAILED after $(( SECONDS - start )) s, see ${log}" >&2
        exit 1
    fi
    echo "=== ${name} done in $(( SECONDS - start )) s"
}

run_step build_event_index   bash "${SOURCE_DIR}/run_build_event_index.sh"
run_step gradient_statistics bash "${SOURCE_DIR}/run_gradient_statistics.sh"
run_step gradient_hit_viewer bash "${SOURCE_DIR}/run_gradient_hit_viewer.sh" "$@"