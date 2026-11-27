# gradient_probe_for_detectors

Per-hit loss sensitivities inside a GravNet + object-condensation (OC) reconstruction model,
for simulated calorimeter events.

Which hits does the loss depend on, and at which depth of the network? A scalar gate per hit
is placed at chosen points of the forward pass. One autograd call gives the derivative of the
OC loss with respect to every gate. Analysis scripts map these gradients back to the simulated
events and draw them in the detector.

> Research code, under development. Paths refer to the KIT ETP cluster (`/work`, `/ceph`).

---

## 1. Method

**Model.** GravNet on [e2ereco](https://github.com/jkiesele/e2ereco) and
[FastGraphCompute](https://github.com/jkiesele/FastGraphCompute): dense input layer, four
GravNet blocks (k = 32 neighbours in a learned 3-d space, 64 propagated features), output head
→ `beta` (sigmoid) and 3-d clustering coordinates.
**Loss.** Object condensation ([Kieseler 2020](https://arxiv.org/abs/2002.03605)), `Loss_OC`
with `q_min = 0.01`. Clustering only; no energy or PID regression yet.

**Probes.** At location *l*, hit *i* gets a gate `m_i^(l) = 1` (a temporary leaf, so the
forward pass is unchanged) that multiplies all of its features. After the loss, one
`torch.autograd.grad` call returns

```
g_i^(l) = dL_OC / dm_i^(l)   at all m = 1
```

Scaling hit *i* at *l* by (1 − ε) changes the loss by −ε g_i: `g > 0` means weakening the hit
lowers the loss, `g < 0` means the hit helps. `g²` is an empirical, Fisher-style importance
proxy (squared gradient of one sample's loss, not an expectation).

| key | where | what the gate scales |
|---|---|---|
| `post_initial` | after the input layer (Linear + ELU) | the whole feature vector entering the GravNet stack |
| `prop_gn1` … `prop_gn4` | inside GravNet op N, on the propagate features, before kNN aggregation | only what hit *i* sends to its neighbours (itself included); not its coordinates, not its skip path |
| `post_output_head` | after the output head | the features from which `beta` and the coordinates are computed |

Older runs also used `post_gnN` (after BN2) and `agg_gnN` (on the aggregated message); the
scripts still read them, in the order `prop_gnN → agg_gnN → post_gnN`.

**Reading the gradients**

- *BatchNorm* couples hits in training mode. The dump runs in eval mode, so it has no BN
  cross-talk; training freezes every BN layer after epoch 20.
- *No normalisation.* Gradients are stored as autograd returns them. Their scale depends on
  how `Loss_OC` reduces over events and objects, so raw `|g|` may favour small events.
- *Hit order.* `hit_id` is the row order of the model input, after preprocessing.

---

## 2. Pipeline and layout

```
DICE (.root) ── djcdata conversion ── .djctd + dataCollection.djcdc
  └─ training ── Results/<run>_per_hit/   grad/<location>.csv, beta.csv, meta.json
       ├─ 1 build_event_index.py     event_id → .djctd event → root entry
       ├─ 2 gradient_statistics.py   gradient distributions, hit/event selection
       ├─ 3 beta_statistics.py       beta distribution, (z, beta) per stage
       └─ 4 gradient_hit_viewer.py   3D views and rotation videos per event
```

```
training/                 copied into an e2ereco working copy (e2ereco itself unchanged)
  models/model_gravnet_fisher_november.py
  training/training_example_november.py
  run_local.sh
build_event_index.py      gradient_statistics.py   beta_statistics.py   gradient_hit_viewer.py
run_<step>.sh             one wrapper per step: container, PYTHONPATH, defaults in ARGS
run_all.sh                steps 1–4 with one block of run settings
diagnostics/              check_conversion_counts.sh, compare_djctd_root.py, argmax_position.py
```

---

## 3. Environment

| step | runs in | needs |
|---|---|---|
| training | `docker://jkiesele/e2ereco_base` (apptainer `--nv`) | e2ereco, djctools, djcdata, FastGraphCompute, W&B (optional) |
| analysis | `/ceph/cjo/containers/plotting.sif` | djcdata, uproot, awkward, pandas, matplotlib, plotly, particle, Pillow; `imageio-ffmpeg` for mp4 (`pip install --user imageio-ffmpeg`) |

Wrappers put `/work/cjo/Utils` (`PlotUtils`, `ExternalFunctions`) on `PYTHONPATH` and keep
their defaults in an `ARGS` array; command-line arguments are appended and override them.
`djcdata`, `djctools`, `e2ereco` and the converter (`TrainData_DICE_cellparent`) are used
unchanged.

---

## 4. Data

| dataset | `.djcdc` | used for |
|---|---|---|
| DICE v1.4.0 SingleJet, pT 30 GeV, 1.2 < \|η\| < 1.7, ten-file subset | `/ceph/cjo/DICEv1_4_0_small/SingleJet_pt30_eta1p2_1p7_100k/ten_files_conversion/` | 20261019, 20261118 |
| QCD single jet, pT 20–200 GeV (J. Kiesele) | `/ceph/jkiesele/datasets/dice/qcd_singlejet_pt20-200_eta3_noPromptLep/merged_global_tR0p65/djc/` | 20261119 (`run_all.sh` default) |

SingleJet subset: validation = singlejet_75, 81 (1847 events), training = 8 files (7407
events). Its conversion dropped about 40 root entries per file, many without SimShowers;
step 1 handles this.

The SingleJet generation settings and geometry were **reconstructed after the fact** and are
not confirmed: barrel and both endcaps, each 1 EM layer (30 cm PbWO₄) + 3 HAD layers (barrel
6 cm brass + 6 cm scintillator, endcap 15 cm brass + 10 cm scintillator), minimum distance
150 cm, granularity 512² (barrel) / 1024² (endcap). The viewer draws this geometry for every
dataset and labels it provisional.

---

## 5. Training

```bash
# Topas (HTCondor)
python3 scripts/topas_submit.py -f <path>/dataCollection.djcdc -n <run_name> \
    -t training_example_november.py --submit --wandb --gpu 1 --cpu 1
# Local GPU node (edit the CONFIG block first; code is snapshotted, job runs under nohup)
GPU_IDS=0 ./run_local.sh
```

Batch size 10,000 hits, validation = last 20 % of the sample list. 100 epochs, Adam, cosine
warm-up over 10 epochs to 7.5e-4, then cosine decay; BN frozen after epoch 20; validation
every epoch with probes active.

Outputs in `Results/`:

- `<run>_epoch<NNN>.pt`: best (lowest `val/L_OC`), epoch 50 and last; an epoch with several
  roles is stored once. `<run>_checkpoints.json` maps roles to files.
- `<run>_per_hit/`: made once after training from the best checkpoint.
  `grad/<location>.csv` (signed `g`), `beta.csv`, and `meta.json` (written last, marks a
  complete dump). All CSVs are `event_id,hit_id,value` with identical rows.
  `event_id` = position in the unshuffled validation pass; `hit_id` = 0 … n−1 in model input
  order.

---

## 6. Analysis

Outputs go to `/ceph/cjo/GradientProbe/runs/<run>/`; inputs are never modified.

```bash
./run_all.sh                                   # steps 1–4
RUN_NAME=20261118_Fisher_depth_probe TRAIN_BASE=/work/cjo/e2ereco_local ./run_all.sh
TOP_N=10 MIN_STAGES=2 ./run_all.sh             # any run setting, for one call
FROM=beta_statistics ./run_all.sh              # build_event_index | gradient_statistics |
                                               # beta_statistics | gradient_hit_viewer
./run_all.sh --event-id 1234 --animate none    # extra arguments go to the viewer
BACKGROUND=1 ./run_all.sh                      # survives logout; stop with
                                               # kill -- -$(cat <run>/logs/run_all.pid)
```

The **run settings** block at the top of `run_all.sh` (`RUN_NAME`, `TRAIN_BASE`, `DUMP_DIR`,
`PROBE_DIR`, `BETA_FILE`, `RUN_DIR`, `DJCDC`, `ROOT_DIR`, `CONVERTER`, `BATCH_SIZE`,
`VAL_FRACTION`, `STAGES`, `TOP_N`, `MIN_STAGES`, `EVENTS_PER_STAGE`) is passed to every step as
overriding arguments; each value can be set from the environment. `SELECTION` is derived
(`top<TOP_N>_min<MIN_STAGES>`), so the viewer always reads the selection written in the same
call. `STAGES` reaches only `beta_statistics` (and its first entry is the L1 probe CSV);
`gradient_statistics` and the viewer use every stage in the dump. The resolved settings and
each step's log are saved in `<run>/logs/`. The wrappers still run alone with their own
defaults.

### 1 `build_event_index.py`

Replays djcdata's validation order (last 20 % of the sample list, batches filled greedily up to
`--batch-size` elements across file boundaries, oversized events skipped) and writes
`events.csv` (`event_id, djctd_file, local_idx, root_file, entry, n_hits, energy_sum,
batch_idx, n_events_in_batch`) and `events_meta.json`. Exits 1 if a check fails.

### 2 `gradient_statistics.py`

Reads every stage CSV in `--probe-dir`, in network order, and checks that all stages have the
same rows. Writes the overlay histogram of `z = sign(g) · log10(1 + |g| / 1e-8)` and a
selection: per stage, the hit with the largest `|g|` of each event, then the `--top-n` events
(default 20); hits in the top-n of ≥ `--min-stages` stages (default 1) are kept. `beta` and,
if L1 passed, root file and entry are joined. `--per-event-histograms` adds one overlay per
selected event.

### 3 `beta_statistics.py`

`beta` exists only at the output; it is compared with `g` at every stage. Writes to
`<run>/beta/` (pdf): the `beta` histogram (50 bins), one (z, beta) scatter per stage with
marginal histograms plus all stages in one figure (the same 1,000,000 random hits at every
stage), and an all-stage (z, beta) density over all hits (50 × 50 bins, log counts), with its
bin counts as CSV. Options: `--beta-file`, `--stages`, `--beta-threshold`, `--zlim`,
`--linear-counts`, `--seed`.

### 4 `gradient_hit_viewer.py`

Per stage, picks the `--events-per-stage` (default 10) selection events with the largest `|z|`
at that stage and draws each picked event at every stage (`--animate all`). Hits are the exact
`.djctd` model input; truth (`t_idx`, `t_pid`) comes from the `.djctd` or, matched by
position, from the converted root entry.

- Colour `z` (Spectral, shared symmetric scale), size = energy, marker = true PID.
- Rings per stage: circle = largest positive `z`, square = most negative `z`, from the event's
  own gradients.
- Views `detector` and `zoom`; rotation videos as webp (VS Code) and mp4 (Keynote/PowerPoint),
  `--fps 10`, `--rotation-speed 10`, `--workers 4`. Existing outputs are skipped unless
  `--overwrite`; a missing format is re-encoded from an existing webp/gif.

### Run directory

```
runs/<run>/
  events.csv, events_meta.json                                   step 1
  figures/gradient_histograms_overlay.{png,pdf}                  step 2
  selections/<name>/selection.csv, selection_meta.json,
                    top_hits_evolution.{png,pdf}, per_event_histograms/
  beta/beta_histogram.pdf, z_vs_beta.pdf, z_vs_beta/<i>_<stage>.pdf,
       z_vs_beta_density.pdf, z_vs_beta_density_counts.csv,
       beta_summary.csv, beta_meta.json                          step 3
  events/<root stem>_e<entry>/                                   step 4, one per event
      <i>_<stage>__<view>.{webp,mp4}, interactive_<view>.html,
      annotated_hits.csv (g_<stage>, z_<stage>, beta), event_info.json
  stages/picked_by_stage.csv
  stages/<i>_<stage>/<rank>_<event> -> ../../events/<event>
  logs/<stamp>_settings.txt, <stamp>_<step>.log, run_all.pid
```

### Consistency checks

| check | what | where |
|---|---|---|
| L1 | hits per `event_id` in `events.csv` == rows per `event_id` in the probe CSV | step 1 |
| L3 | `.djctd` event → root entry: identity, or aligned by ≥ 99 % hit-position overlap when the conversion dropped entries | step 1 |
| L4 | preprocessing keeps hit order | assumed (read once in the code) |
| L5 | ≥ 99 % of the `.djctd` hit positions are in the root entry; failing events are not drawn | step 4 |

---

## 7. Open questions

- How does `Loss_OC` reduce over events and objects? Does raw `|g|` scale with 1/n_hits?
- The last `hit_id` is 2× to 26× more often the extreme hit than chance, growing towards the
  output head. Noise (`t_idx = −1`), lowest energy, or `.djctd` hit ordering?
- The current converter does not reproduce the training conversion exactly (~2 % of hit
  energies differ, a few extra hits): other version or a random step?
- 20261113 run: best validation loss at epoch 11, right after warm-up. Overfitting or the BN
  freeze?
- Does the reconstructed DICE v1.4.0 geometry drawn by the viewer match the QCD dataset?
- Hit coordinates are in mm (`coordinate_units_per_cm = 10`); confirm against the converter.

---

## 8. Conventions

- Script header: purpose, then `cyan.jo, <date>`; with Claude:
  `cyan.jo, Claude Opus5.5 Extra, <date>`.
- Commits with Claude-written code carry `Co-authored-by: Claude <noreply@anthropic.com>`.
- Outputs are written to a hidden `.part` file and renamed when complete.

## License
MIT License. See `LICENSE`.