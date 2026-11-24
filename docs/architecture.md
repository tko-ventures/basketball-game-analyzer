# Architecture

*What is built in this repository as of 2026-11-24 (M0–M9 shipped, M10 next).*

This document is the canonical human-oriented overview of the system design and current active stack. Exact backend schemas, CLI flags, and config inventories live in the code (`src/bga/*/base.py`, `bga backends`, `configs/`).

- **C4 visual diagrams:** [`docs/architecture-c4-view.md`](architecture-c4-view.md) *(companion view — keep in sync whenever this document changes)*
- **Why choices were made:** [`docs/adr/`](adr/README.md)
- **Roadmap & milestone targets:** [`docs/PLAN.md`](PLAN.md)
- **Evaluation scores & training runs:** [`docs/metrics.md`](metrics.md), [`docs/train.md`](train.md)
- **Cloud compute spend:** [`docs/costs.md`](costs.md)
- **Gold-set details:** [`docs/gold.md`](gold.md)

---

## System overview

The repository has three loosely coupled parts:

1. **The `bga` pipeline (`src/bga/`)** — a modular, config-driven video analysis pipeline with swappable layer backends, gold-set evaluation, Modal remote inference, and Colab/Modal training orchestration.
2. **Labelling tools (`tools/labeler/`, `scripts/`)** — a local HTTP labeller (`http://127.0.0.1:8765`) and contact-sheet / batch scripts used to build and audit the gold set and training manifests.
3. **The demo site (`scripts/demo_*`, `scripts/coach_*`, `docs/roadmap.html`)** — a static showcase published to Vercel (`https://basketball-analyzer.demo.tko.ventures`), built outside the `bga` pipeline.

---

## Pipeline architecture

### Execution unit: sequences

The pipeline processes video in **sequences** (`bga.stage.Sequence`): one continuous broadcast shot of one clip, sampled on a list of source frame indices.

- **`shots/gold`** selects the 12 gold shots across 4 PLK games (9 dev shots, 3 held-out shots from `Tnuscrm9RyU`) at stride 5 (every 5th frame of 50 fps video).
- **`shots/clips`** selects shots (≥ 3 s) from named non-gold clips for training-data harvesting, enforcing a gold-leakage check.
- `FrameReader` decodes sequentially across small frame steps (≤ 64 frames) instead of seeking, cutting decode latency from ~122 ms to ~4 ms per frame.

### Layer DAG

Layers and their upstream dependencies are fixed in `bga.config.LAYERS`. Optional upstreams are `court`, `appearance`, and `numbers` globally (`OPTIONAL_UPSTREAM`), plus `link` for `positions` and `link`/`court`/`positions` for `ball` (`OPTIONAL_BY_LAYER`):

```text
shots ─┬─> court ────────────────┐
       └─> detect ─┬─> team      │
                   ├─> appearance│
                   ├─> track <───┘   (reads detect, optional court)
                   │     ├─> numbers   (reads detect, track)
                   │     ├─> link      (reads track, team; optional appearance, court, numbers)
                   │     │     └─> identity (reads link, track, team; optional numbers, appearance; shot or clip scope)
                   │     ├─> positions (reads link or track, court)
                   │     └─> ball      (reads detect, track, team; optional link, court, positions)
```

Each layer defines its exact JSON/array output contract in `src/bga/<layer>/base.py`.

### Current active stack

Every layer supports multiple swappable backends registered via `@register(layer, name)` (`bga backends` lists all registered backends). The table below tracks the **current leading choice** per layer:

| Layer | Active backend & config | What it does | Current benchmark / status | ADR |
|---|---|---|---|---|
| `shots` | `gold` / `clips` | Selects gold eval shots or non-gold training shots | M10 adds raw full-game ingest | [2026-11-23](adr/2026-11-23-highlights-footage-and-m10-ingest-milestone.md) |
| `court` | `rfdetr_keypoint` (`court-plk-v1-smooth4`) | PLK fine-tuned RF-DETR Keypoint → RANSAC homography + 4-frame median smoothing | 0.21 m paint error dev, 0.20 m held-out (vs. 0.89 m NBA baseline) | [2026-11-02](adr/2026-11-02-court-keypoint-finetune-and-smoothing.md) |
| `detect` | `rfdetr_torch` (`detect-plk-r3-poss`) | PLK fine-tuned RF-DETR Large @ 1280 px on MPS; 6 active classes incl. `player-in-possession` | AP50 dev: people 0.966, ball 0.766, number 0.848, possession 0.559 | [2026-11-19](adr/2026-11-19-detector-rfdetr-large-1280px-mps.md), [2026-11-20](adr/2026-11-20-detector-classes-and-possession-schema.md) |
| `team` | `siglip_kmeans` (`fit_fps: 5.0`, fixed seed) | SigLIP + UMAP + KMeans per shot; aligns clusters via clip `light`/`dark` kit metadata | ~1.00 track accuracy on dev & held-out | [2026-11-20](adr/2026-11-20-team-classification-siglip-5fps.md) |
| `appearance` | `reid` (`plk-r1-shot-last`) | OSNet x1.0 fine-tuned on PLK crops; float16 L2-normalised embeddings | Used in BoT-SORT & kept for M7 cross-shot identity (0.946 rank-1) | [2026-11-22](adr/2026-11-22-tracking-botsort-plk-reid-and-greedy-link.md) |
| `track` | `trackers` (`BoTSORTTracker` + PLK re-ID) | BoT-SORT with camera-motion compensation and tuned re-ID (runs on Modal L4) | ~64 ms/frame on L4 | [2026-11-22](adr/2026-11-22-tracking-botsort-plk-reid-and-greedy-link.md) |
| `link` | `greedy` (`link-botsort-plk-r1-shot-last-tuned-noapp-cap5`) | Joins tracklets by team, gap, and court speed + 5-players-per-team cap | HOTA 0.743 dev / 0.742 held-out; IDF1 0.873 / 0.849; IDSW 52 / 26 | [2026-11-22](adr/2026-11-22-tracking-botsort-plk-reid-and-greedy-link.md) |
| `numbers` | `parseq` (`numbers-parseq-r2`) | Fine-tuned PARSeq scene-text reader (`stop_on_confirm: false`, logging crop probabilities) | Crop top-1 under roster: 0.963 dev / 0.970 held-out (4–7 ms/crop); refusal 0.447 / 0.500 | [2026-11-23](adr/2026-11-23-jersey-number-reading-parseq-and-roster-scoring.md) |
| `identity` | `game` / `covisible_assign` (`identity-game`) | Layer 5b: roster log-prob voting, co-visible LSA uniqueness (`identity-covisible-assign`), and clip-scoped cross-shot appearance clustering (`identity-game`) | Dev (69 tracks): 0.841 shot / 0.913 game (2 / 3 wrong); held-out (26 tracks): 0.846 / 0.846 (2 wrong, 0.923 conf) | [2026-11-23](adr/2026-11-23-jersey-number-reading-parseq-and-roster-scoring.md) |
| `positions` | `bbox_bottom` (`positions-plk-bbox`) | Projects linked-track box bottom-centre via PLK homography (`w > 0` guard) + `clean_paths` (`smooth_window: 9`); `ankles_pose` kept as comparator | 0.358 m keyframe median dev (vs. 1.540 m baseline), 99.4% on-court, 0.0 ms/frame | [2026-11-23](adr/2026-11-23-positions-bbox-bottom-over-coco-pose-and-parquet-export.md) |
| `ball` | `fused` (`ball-plk-fused`) | Static rim-FP filter + DAG DP ball trajectory, Viterbi HMM possession, rim-trajectory made/missed shot classification & FIBA 2/3-pt scoring (`skalski` kept as baseline) | Dev (9 shots): ball F1 0.794, poss acc 0.618, holder F1 0.617, team acc 0.703, shot F1 1.000 (11/11), outcome acc 0.818, 0.36 ms/frame | [2026-11-24](adr/2026-11-24-ball-fused-tracking-possession-and-rim-shots.md) |

Other registered backends remain available for baselines, ablations, and reproducibility: Roboflow NBA models (`roboflow_keypoints`, `roboflow_inference`), vendored/upstream ByteTrack (`bytetrack`, `trackers`), SAM 2.1 propagation (`sam2`, CUDA-only), SmolVLM2 OCR (`smolvlm_ocr`), and crop classifiers (`crop_cls`).

---

## Core mechanics

### Content-addressed caching and execution

- **Configs:** A run is defined by a YAML `RunConfig` (`configs/*.yaml`) specifying `backend`, `params`, and `weights` per layer. CLI overrides use `--set layer.params.key=value`.
- **Layer hashes:** Each layer gets an 8-char hash covering its own config plus upstream layer hashes (excluding sequence-selection parameters so dev and held-out runs share directories).
- **Outputs:** Written atomically to `runs/<clip>/<layer>/<backend>@<hash8>/<sequence>.json`, with heavy arrays saved as sibling `.npy` files and provenance recorded per sequence in `meta.json`.
- **Memory discipline:** `bga.pipeline.run` executes one layer at a time across all sequences (`load()` → infer → `unload()`), keeping at most one heavy model resident in unified memory on the M1 Max. Existing outputs are reused unless `--force <layer>` is passed.

### Local Mac + cloud GPU compute (Colab & Modal)

- Inference backends auto-select devices via `bga.device.pick_device` (CUDA → MPS → CPU).
- `--remote modal` routes CUDA-only backends (`requires = "cuda"`, e.g. `track/sam2`) to the `bga-runner` Modal app; `--remote-layer <layer> --remote-gpu L4` offloads any heavy layer (such as BoT-SORT with re-ID) in parallel per sequence.
- Clips and trained weights sync once to the `bga-data` Modal volume; results stream back into local `runs/` as each sequence completes so partial runs resume cleanly.
- Modal spend is tracked against Modal's own billing (`scripts/modal_spend.py`) and Colab training spend is tracked in Compute Units (`colab usage`) in [`docs/costs.md`](costs.md).

### Training and provenance (`bga train`)

- Every fine-tune is declared in `configs/train/<layer>-<backend>-<name>.yaml` (`TrainConfig`), pinning base weights, dataset sources/manifest sha256 hashes, recipe, and default platform (`colab` or `modal`) and GPU.
- Trainable setups (`bga.train.run.SETUPS`): `court/rfdetr_keypoint`, `detect/rfdetr_torch`, `appearance/reid`, `numbers/crop_cls`, and `numbers/parseq`, all sharing the same core trainers in `bga.train.dispatch`.
- **Gold-leakage guard:** `bga.train.leakage` (and `tests/test_gold_leakage.py`) blocks any gold clip—dev or held-out—from entering a training manifest or config.
- **Alternating Colab & Modal:** `--platform colab|modal` and `--gpu <GPU>` (or env vars `BGA_TRAIN_PLATFORM` / `BGA_TRAIN_GPU`) override where a run executes without mutating the YAML or its `weight_id()`. Colab runs (`bga.remote.colab_train`, via `google-colab-cli`) stream logs live, transfer archives/checkpoints in chunks, pull outputs into `runs/train/<id>/output/`, and always stop the session on exit; Modal runs (`bga.remote.modal_train`) spawn detached on `bga-train` and support `--wait` / `--resume` and multi-GPU DDP (`recipe.devices > 1`).
- `bga train --collect` writes checkpoints, metrics, and a provenance `train.json` into `weights/<layer>/<backend>/<name>@<hash8>/`, invalidating stale cached outputs, dropping any stale Modal volume copy, and logging rows to `docs/train.md` and `docs/costs.md`.

### Evaluation (`bga eval*`)

Seven dedicated CLI commands evaluate layers against the gold set and append rows to [`docs/metrics.md`](metrics.md) (with `--write`). Every `eval` command refuses weights that are neither `pretrained:` nor backed by a local `weights/<id>/train.json`:

| Command | Gold target | Primary metrics |
|---|---|---|
| `bga eval` | `data/gold/v0` (tracks) | HOTA, DetA, AssA, MOTA, IDF1, IDSW, team/number accuracy, churn, ms/frame |
| `bga eval-court` | Gold court keyframes (18 dev, 6 held-out) | Median & paint reprojection error (m) per arena, share < 0.5 m |
| `bga eval-detect` | `data/gold/v1/boxes.json` (100 frames) | Per-class COCO AP50:95 and AP50 across the 10-class schema + merged `people` |
| `bga eval-appearance` | Clean `v0` gold boxes | Teammate re-association (1 s, 3 s) and cross-shot rank-1 / MRR |
| `bga eval-numbers` | `data/gold/v1/numbers.json` (376 crops) | Free & roster top-1, non-number refusal rate, calibration (ECE), ms/crop |
| `bga eval-positions` | Gold court keyframes + linked tracks | Keyframe median/p90 error (m), on-court share, coverage, speed p95 & >10 m/s spikes, ms/frame |
| `bga eval-ball` | `data/gold/v1/boxes.json` + `events.json` | Ball F1/R/med px, possession acc, holder F1, team acc, shot F1, outcome acc, shooter acc, ms/frame |

Other top-level CLI commands: `bga run`, `bga train`, `bga export` (17-column per-player per-frame Parquet + sibling `<seq>.ball.parquet` when `ball` is configured), `bga radar` (broadcast + 2D FIBA court overlay video with team colours, `#jersey` labels, ball/holder ring, and cumulative shot markers), and `bga backends`.

---

## Data, labelling, and demo

### Data layout

- **Source clips (`data/clip_ids.txt` → `data/raw/`):** 39 PLK YouTube highlight clips (1080p50 H.264, gitignored).
- **Labels & rosters (`data/labels/`, `data/rosters/`):** Per-clip shot/kit/box JSONs, 19 PLK club rosters (2024/25–2026/27), versioned training manifests (`detect/`, `appearance/`, `numbers/`, `court/`), and replayable audit logs (`data/labels/review/*.ops.txt`).
- **Gold set (`data/gold/v0/`, `data/gold/v1/`):** 4 PLK games (3 dev + 1 held-out `Tnuscrm9RyU`), 3 central shots each. Track gold (`v0`) is MOTChallenge format; box gold, court keyframes, number crops, and verified shot events (`events.json`) are `v1`.
- **Synthetic fixtures (`data/fixtures/synthetic/`):** 8 deterministic simulated possessions (`bga.tracking2d/v0`, see [`docs/fixtures.md`](fixtures.md)) used solely by the coach demo page.

### Labeller (`tools/labeler/server.py`)

A local stdlib HTTP server on port `8765` with five focused web pages:

- `/` — track review, shot tagging, and identity confirmation
- `/court` — landmark dots + painted-line pins with live Levenberg-Marquardt court fitting
- `/boxes` — box geometry (`box_gold_v1`), ball queues (`mode: ball`), and possession/dead-ball queues (`mode: possession`)
- `/numbers` — crop-level jersey-number verification against context frames
- `/events` — shot-event verification (`made`/`missed`, 2/3-pt, click-on-player shooter, scorebug start/end comparison) writing `data/gold/v1/events.json`

### Demo site

Published to `https://basketball-analyzer.demo.tko.ventures` via `scripts/deploy_demo.sh` (from a clean `main` only). English-only, built outside the `bga` pipeline:

- `/` — analysis showcase (3 shots of `OSp445e0mXU` with broadcast labels + live 2D court powered by `court-plk-v1-smooth4`)
- `/coach/` — interactive coach prototype built from the synthetic 2D fixtures (`Watch` / `Insights` / `Feedback`)
- `/roadmap/` — static milestone view (`docs/roadmap.html`)

---

## Repository map

| Path | Responsibility | Tracked in git |
|---|---|---|
| `src/bga/` | Pipeline package: layers, registry, config/hashing, eval, training, Colab & Modal runners | Yes |
| `configs/` | Inference run configs (`*.yaml`) and training configs (`train/*.yaml`) | Yes |
| `tools/labeler/` | Local labelling HTTP server and 5 web UI pages | Yes |
| `scripts/` | Labelling/review helpers, gold export, sweeps, probes, demo builders | Yes |
| `tests/` | Pytest suite + doctests (including gold-leakage and contract tests) | Yes |
| `data/labels/`, `data/rosters/`, `data/gold/`, `data/fixtures/` | Committed labels, rosters, gold sets, synthetic fixtures | Yes |
| `docs/` | Architecture (`architecture.md`, `architecture-c4-view.md`), ADRs (`adr/`), plan, metrics, training log, costs, archived logs | Yes |
| `data/raw/`, `weights/`, `runs/` | Downloaded videos, trained weights, cached stage outputs | No |
