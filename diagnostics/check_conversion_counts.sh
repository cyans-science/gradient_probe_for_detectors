#!/usr/bin/env bash
# check_conversion_counts.sh
#
# Purpose: diagnostic for check L3. For each validation file listed in
# events_meta.json, print the number of .djctd events, root entries, and
# root entries with SimShowers. The conversion drops entries without
# SimShowers and, in the 20261019 run, about 40 more per file by another rule.
#
# Usage: ./diagnostics/check_conversion_counts.sh [path/to/events_meta.json]
#
# cyan.jo, Claude Opus5.5 (Extra), 2026-10-21

CONTAINER=/ceph/cjo/containers/plotting.sif
META_FILE=${1:-/ceph/cjo/GradientProbe/runs/20261019_Fisher_depth_probe/events_meta.json}

# The quoted 'EOF' stops bash from expanding anything inside the Python code,
# so the meta file path is passed as an argument instead.
apptainer exec -B /ceph "${CONTAINER}" python3 - "${META_FILE}" <<'EOF'
import json, sys
import uproot, awkward as ak

meta = json.load(open(sys.argv[1]))
for f in meta["checks"]["L3"]["per_file"]:
    with uproot.open(f["root_file"]) as r:
        n_showers = ak.num(r["SimShowers"]["sim_shower_pdg_id"].array())
    print(f["djctd_file"].split("/")[-1], "djctd:", f["djctd_events"], "root:", f["root_entries"],
          "root with SimShowers:", int(ak.sum(n_showers > 0)))
EOF