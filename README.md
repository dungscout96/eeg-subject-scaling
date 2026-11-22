# Scaling subjects in cross-modal alignment: video decoding with EEG foundation model

Code, configurations and measured results for the paper by Dung Truong, Kuntal
Kokate and Arnaud Delorme (UC San Diego / CNRS).

We align 2 s of 129-channel EEG to frozen V-JEPA-2 embeddings of the movie being
watched (soft-target CLIP), on Healthy Brain Network (HBN) recordings of up to
1863 subjects, and measure how decoding scales with the number of training
subjects for three encoder arms:

| arm | encoder | initialisation |
|---|---|---|
| **pretrained** | REVE, depth 22, 200-sample patches (20 overlap) | public REVE-base weights |
| **random-init, depth 22** | same architecture | random |
| **random-init, depth 12** | REVE-like, depth 12, 400-sample patches | random |

Every arm is trained on cohorts of S ∈ {10, 20, 50, 100, 200, 400, 701, 1000,
1400, 1863} subjects (three nested draws per S), and each cell is evaluated on
the untouched test release (R6) with a ridge feature probe and fit-free
retrieval, within-task (*The Present*) and cross-task (*Despicable Me*).

---

## Two ways to reproduce

### A. From the shipped results — figures and every number (CPU, ~1 min)

```bash
uv sync
bash scripts/reproduce_from_results.sh
```

This re-derives the checkpoint selection from the held-out epoch curves,
regenerates the three paper figures into `figures/`, and recomputes every number
in the main text and appendix tables, checking each against the value printed in
the paper (`283/283 checks agree with the paper.`). The tables are written to
`results/paper_tables.txt`. `uv run pytest` runs the same checks plus unit
tests of the data subsampling, selection and feature code.

### B. From raw EEG — the full pipeline (GPU)

102 training runs (one ~44 GB GPU each; a 400-epoch depth-22 run takes about
80 min on an A40), then evaluation. Every step is a plain script; loop over
cells on your scheduler of choice.

**0. Environment.** Python 3.12, [uv](https://docs.astral.sh/uv/):

```bash
uv sync                      # add  --extra wandb  to log runs to Weights & Biases
```

**1. EEG data.** HBN-EEG releases R1–R10 (OpenNeuro ds005505–ds005515; R9 is
ds005514), tasks `ThePresent` and `DespicableMe`. `scripts/preprocess_hbn.py`
downloads each release through [EEGDash](https://github.com/sccn/EEGDash) and
applies the offline pipeline of Appendix A.1 (drop recordings < 10 s, resample
to 200 Hz, 0.5–99.5 Hz FIR band-pass, per-channel z-score pooled over the
release, clip at ±15 SD):

```bash
export HBN_PREPROCESS_DIR=/path/to/hbn_preprocessed
uv run python scripts/preprocess_hbn.py --multirun \
    release=R1,R2,R3,R4,R5,R6,R7,R8,R9,R10 task=ThePresent,DespicableMe \
    output_dir=$HBN_PREPROCESS_DIR
```

After the load-time screen (movie onset/offset markers present, duration within
1 s of the film), the training pool is 2156 *The Present* recordings, one per
subject (R1–R5, R7–R10), and the test release R6 has 108.

**2. Movie artifacts** are shipped in `movie_artifacts/` (see
[Movie artifacts](#movie-artifacts)); the movie files themselves are not
redistributed.

**3. REVE weights** for the pretrained arm (the Hugging Face repository may
require accepting its terms and `huggingface-cli login`):

```bash
uv run python scripts/prepare_reve_checkpoint.py --output checkpoints/reve_base_eet_init.pth.tar
```

The script fails unless every converted tensor loads into the depth-22 encoder.

**4. Train.** `--list` prints all 102 runs (arm, cell, S, draw, budget, seed):

```bash
uv run python scripts/train_cell.py --list
# one run; keep its log in the run directory -- step 5 reads it
d=runs/pretrained_d22/e05_s100_a101_av_d11 && mkdir -p $d
uv run python scripts/train_cell.py --arm pretrained_d22 --S 100 --draw 11 \
    --reve-ckpt checkpoints/reve_base_eet_init.pth.tar > $d/train.log 2>&1
uv run python scripts/train_cell.py --arm random_d22 --S 1000 --draw 11 --epochs 800
```

| arm | budgets |
|---|---|
| `pretrained_d22` | 400 epochs (4400 steps), every S |
| `random_d22` | 400 epochs every S, and 800 epochs (8800 steps) at S ≥ 701 — the 800-epoch cells are reported there |
| `random_d12` | 400 epochs for S < 701, 800 epochs for S ≥ 701 |

An epoch is a fixed 703-window random draw from the cohort (11 steps at batch
64), so the step budget does not depend on S. Checkpoints are saved every 25
epochs; nothing is early-stopped.

**5. Collapse screen** (Appendix C.1). About 10 % of pretrained runs stay at
the chance loss ln 64 from the first epoch. Flag them and re-run with the next
seed of the fixed sequence 2026 → 7 → 2025 (`--seed`); the re-runs of the
reported runs are recorded in `scripts/cells.py` and applied automatically.

```bash
uv run python scripts/screen_collapse.py
```

**6. Held-out selection curves.** Each draw is scored at every checkpoint on
the 293 subjects outside its S = 1863 cohort, which no cell of that draw trained
on (cohorts nest within a draw):

```bash
for arm in pretrained_d22 random_d22 random_d12; do
  for d in 11 22 33; do uv run python scripts/run_epoch_curve.py run --arm $arm --draw $d; done
  uv run python scripts/run_epoch_curve.py merge --arm $arm
done
```

**7. Select** each cell's checkpoint — argmax of held-out time-pool e→v@1 (the
paper), and the held-out probe for the robustness check of Appendix C.2:

```bash
uv run python scripts/select_checkpoints.py --write
uv run python scripts/select_checkpoints.py --metric probe_mean_r --write
```

**8. Test evaluation** on R6, plus the untrained-encoder floors:

```bash
for arm in pretrained_d22 random_d22 random_d12; do
  uv run python scripts/evaluate_test.py cells --arm $arm                    # paper
  uv run python scripts/evaluate_test.py cells --arm $arm --selector probe   # App. C.2
done
uv run python scripts/evaluate_test.py random --depth 22
uv run python scripts/evaluate_test.py random --depth 12
```

**9. Analysis:** `bash scripts/reproduce_from_results.sh`.

Training is not bit-deterministic on GPU, so a re-run reproduces the results
up to run-to-run variation (between-draw SD ≈ 0.003 r on the probe and ≈ 0.002
on retrieval).

---

## Repository layout

```
eb_jepa/                     library code
  architectures.py           REVE encoder (EEGEncoderTokens) and the CLIP projections
  clip.py                    soft-target CLIP objective (App. B)
  datasets/hbn.py            HBN loading, windowing, cohort subsampling
  training/clip_pretrain.py  training entry point
  evaluation/clip_probe/     probe_traintest.py (ridge probe), retrieval.py,
                             epoch_curve.py (held-out selection curve)
configs/
  train/depth22.yaml         pretrained and random-init depth-22 arms
  train/depth12.yaml         random-init depth-12 arm
  eval/select_*.yaml         selection on the cohort complement (2156 pool)
  eval/test_*.yaml           test-time probe and retrieval (head fit excludes R5)
  preprocess_hbn.yaml        offline preprocessing
scripts/                     pipeline steps (above); cells.py is the experimental design
analysis/                    figures and paper_numbers.py (all reported numbers)
movie_annotation/            frame-feature extraction (12 probe features)
movie_artifacts/             per-movie features, V-JEPA-2 embeddings, centring mean
results/                     every measured JSON the paper reads (651 files)
figures/                     rendered paper figures
tests/                       unit tests and regression tests of results/
```

### Naming in `results/`

Cell and file names keep the identifiers the runs were made under, so a fresh
run lines up with the shipped files one-for-one.

| paper arm | cell prefix | selection / curve files |
|---|---|---|
| pretrained, depth 22 | `e05_…` | `addval_*` |
| random-init, depth 22 | `e05fs_…` | `fromscratch_*`, `fromscratch800_*` |
| random-init, depth 12 | `e03_…` | `d12av_*` |

A cell is `<prefix>_s<S>_a101_av[_ep800]_d<draw>`. Test readouts are
`<readout>_<cell>_epretr.json` (paper selector) or `…_epprb.json` (probe
selector), where `<readout>` is `e04_tt_test` (within-task probe),
`e04_retr_test` (within-task retrieval), `xtask_tt_DMtest` (cross-task probe)
or `xtask_retr_DMtest` (cross-task zero-shot retrieval). `*_random*.json` are
the untrained-encoder floors (`e04`/`_e04` = depth 22, `e03`/unsuffixed =
depth 12).

---

## Movie artifacts

`movie_artifacts/<ThePresent|DespicableMe>/`:

| file | content |
|---|---|
| `features_enriched.parquet` | per-frame features at native frame rate, including the 12 probe features (App. A.2) |
| `vjepa2_embeddings.npz` | frozen V-JEPA-2 embeddings, 1408-d, 2 Hz |
| `vjepa2_recipe.npz` | `global_mean`, the mean embedding used to centre the targets (`scripts/precompute_vjepa2_recipe.py`) |
| `transnetv2_boundaries.csv`, `vjepa2_scenes_map.csv` | shot and scene boundaries (inputs to the recipe script) |

Ten of the twelve probe features come from `movie_annotation/annotate.py`
(OpenCV statistics and optical flow, DPT-Large depth, SSD faces, CLIP
ViT-B/32 naturalness; `uv sync --extra annotation`); `position_in_movie` and
`narrative_event_score` are derived by `movie_annotation/enrich_features.py`,
which reproduces the shipped values exactly (`tests/test_movie_features.py`).

---

## Notes

- **Appendix C.3, "What the shorter budget cost".** These numbers are measured
  on a separate depth-22 random-init replication on the 1863-subject pool (R5
  held out), trained for 800 epochs and scored on R5: the best held-out probe
  reachable by epoch 400 against by epoch 775 within the same run. Its curve is
  `results/e05rand800_epoch_curve.json`; it is not part of the 2156-pool arms.
- **Rounded inputs in Table A2.** The "gain" column and the cross-task
  zero-shot retrieval row were computed from the rung means as printed in
  Tables A3/A4. That row sits within ~0.005 of chance, so rounding moves its
  statistics; `results/paper_tables.txt` prints the unrounded fit beside it.
- **Not recomputable from `results/`:** the collapse-screen statistics of
  Appendix C.1 (failure rate and loss ratios), which come from the training
  logs of the reported runs.
- The result JSONs record the checkpoint paths of the original runs on the
  compute cluster; they are provenance only and are not read by the analysis.

## Citation

```bibtex
@misc{truong2026scaling,
  title  = {Scaling subjects in cross-modal alignment: video decoding with EEG foundation model},
  author = {Truong, Dung and Kokate, Kuntal and Delorme, Arnaud},
  year   = {2026},
  note   = {To appear}
}
```

## License

Apache 2.0 (see `LICENSE`). The encoder code builds on the EB-JEPA codebase.
HBN-EEG data are distributed by the Child Mind Institute under their own terms;
REVE weights and V-JEPA-2 are subject to their respective licenses.
