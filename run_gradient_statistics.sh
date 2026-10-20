#!/bin/bash
# Run analysis/gradient_statistics.py inside the plotting container.
# Input CSVs (/work) and output plots (/ceph) are set in the script's main().

set -euo pipefail

CONTAINER=/ceph/cjo/containers/plotting.sif
SOURCE_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P )
UTILS_DIR=/work/cjo/Utils   # PlotUtils.py, ExternalFunctions.py

apptainer exec \
    -B /work \
    -B /ceph \
    --env PYTHONPATH="${UTILS_DIR}" \
    --pwd "${SOURCE_DIR}" \
    "${CONTAINER}" \
    python3 gradient_statistics.py "$@"
