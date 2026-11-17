# Architecture: what is built

This page describes the code and data that exist, as of 2026-11-17 on `feat/m3-court`. `main` holds M0 to M2. It records only what is in the repository. Intentions, targets and milestones are in `docs/PLAN.md`, decisions in `docs/decisions.md`, and scores in `docs/metrics.md`.

The repository has three parts:

1. **The `bga` package** (`src/bga/`). A pipeline of swappable layers, the gold-set evaluation, and a runner for Modal.
2. **Labelling tools** (`tools/labeler/`, most of `scripts/`). These build and check the gold set and the court keyframes.
3. **The demo site** (`scripts/demo_*`, `scripts/coach_*`). Web pages published to Vercel. They are built by their own scripts, not by the `bga` pipeline.

Every model in use is a public pretrained one. There is no training code: no `bga train`, no `configs/train/` and no `weights/` directory.

## Repository layout

| Path | Contents | In git |
|---|---|---|
| `src/bga/` | the pipeline package, about 2,400 lines | yes |
| `configs/baseline.yaml` | the only run config | yes |
| `tools/labeler/` | the local labelling server and its two pages | yes |
| `scripts/` | labelling, gold export, court, probe, demo and coach scripts | yes |
| `tests/` | 12 test files, 57 tests | yes |
| `data/clip_ids.txt` | the 31 YouTube highlight clips | yes |
| `data/labels/` | per-clip labels, the gold manifest, review ops, court keyframes | yes |
| `data/gold/v0/` | the exported gold set | yes |
| `data/rosters/` | 19 PLK clubs, seasons 2024/25 to 2026/27 | yes |
| `data/fixtures/synthetic/` | 8 simulated possessions for the coach page | yes |
| `data/raw/` | downloaded videos | no |
| `runs/` | every stage output, probe, sheet and demo build | no |
| `docs/` | plan, decisions, metrics, costs, gold set, fixtures, roadmap, this page | yes |

## The `bga` pipeline

### Run units

A **sequence** (`bga.stage.Sequence`) is one shot of one clip, processed on a list of source frame indices. `FrameReader` decodes frames with a small cache, so sequential reads stay cheap.

The only sequence source is `shots/gold` (`bga.ingest.gold_shots`). It produces the 12 gold shots from `data/gold/<version>/manifest.json`, from each shot's first gold frame to its last. The stride must divide the gold stride of 5, so every gold frame is processed. A `held_out` parameter selects the held-out game instead of the 9 dev shots.

### Layers

The layers and the upstream outputs each one reads are fixed in `bga.config.LAYERS`:

```
shots ─┬─> court ──────────────┐
       └─> detect ─┬─> team    │
                   ├─> track <─┤   (track reads detect and court)
                   │     ├─> numbers   (reads detect and track)
                   │     └─> positions (reads track and court)
```

Each layer's output format is documented in its `base.py`:

| Layer | Output per sequence |
|---|---|
| `shots` | `{"sequences": [...]}` |
| `court` | `{"frames": {frame: 3×3 image→court matrix in cm, or null}, "units": "cm"}` |
| `detect` | `{"frames": {frame: [[x1, y1, x2, y2, score, class_name], ...]}, "classes": [...]}`, using the 10-class schema in `bga.types` |
| `team` | `{"frames": {frame: ["home" \| "away" \| "ref" \| "", ...]}}`, aligned with the detect boxes |
| `track` | `{"frames": {frame: [[track_id, x1, y1, x2, y2, score, class_name, det_index], ...]}}` |
| `numbers` | `{"tracks": {track_id: "23" or null}, "reads": n, "ms_per_read": ms}` |
| `positions` | `{"tracks": {track_id: [[frame, x_m, y_m], ...]}, "court_m": [28.0, 15.0]}` |

### Backends

A backend is a class registered with `@register("<layer>", "<name>")` (`bga.registry`). The registry imports the modules listed in `BACKEND_MODULES` lazily. Each layer has exactly one registered backend, and `configs/baseline.yaml` selects all of them:

| Layer | Backend | What it does | Model |
|---|---|---|---|
| shots | `gold` | gold shots at stride 5 | none |
| court | `roboflow_keypoints` | keypoints above `kp_conf` (at least 4), then a RANSAC homography to the FIBA vertices of `sports.CourtConfiguration`, threshold 40 cm | `basketball-court-detection-2/14` (Roboflow, NBA frames) |
| detect | `roboflow_inference` | Roboflow `inference` run locally (weights fetched with `ROBOFLOW_API_KEY`, from the environment or `~/.config/bga/roboflow.env`); confidence 0.4, IoU 0.9 | `basketball-player-detection-3-ycjdo/6` (RF-DETR-S, NBA frames) |
| team | `siglip_kmeans` | `sports.TeamClassifier` (SigLIP, UMAP, KMeans) fitted per sequence on crops sampled at 1 fps, referees excluded | `google/siglip-base-patch16-224` |
| track | `bytetrack` | supervision ByteTrack on person boxes, in pixels | none |
| numbers | `smolvlm_ocr` | reads `number` boxes lying at least 90% inside a track's box; a number is confirmed after 3 consecutive equal reads; runs on MPS with `max_new_tokens=4` | `basketball-jersey-numbers-ocr/3` (SmolVLM2, NBA crops) |
| positions | `bbox_bottom` | box bottom-centre through the frame's homography, then `sports` `clean_paths` (jumps over 0.7 m, smoothing window 9) | none |

Code that exists but no config uses:
- `bga.detect.onnx_rfdetr`, an RF-DETR decoder on ONNX Runtime with the CoreML provider. It is not registered.
- `bga.detect.nms`, nested-box suppression, used only by `onnx_rfdetr`.

Both are covered by `tests/test_nms.py`.

### Configs, hashes and run directories

- **Config model:** a config is a pydantic `RunConfig` (`bga.config`). Each layer is a `LayerConfig` with `backend`, `params` and `weights`. `weights` is a declared id such as `pretrained:roboflow/<model>`.
- **Layer hash:** covers the layer's own entry plus the hashes of its upstream layers. The `shots` parameters that only choose sequences (`version`, `held_out`, `root`, `labels`) are excluded, so dev and held-out runs share directories.
- **Config hash:** the hash of all layer hashes.
- **Overrides:** `--set layer.params.key=value` changes one value and appends the override to the config name.
- **Outputs:** each layer writes `runs/<clip>/<layer>/<backend>@<hash8>/<sequence>.json`, plus a `meta.json`. The meta file holds the layer entry, upstream hashes, config name, git commit and wall time, plus platform, GPU, seconds and USD for remote runs.

`bga.pipeline.run` executes one layer at a time over all sequences, calling `load()` before and `unload()` after. Only one model is resident at a time. An existing output is reused unless `--force` names its layer. A backend that declares `requires = "cuda"` is skipped locally, and runs only with `--remote modal`.

### Command line

`bga` is a project script (`bga.cli`, typer):

| Command | What it does |
|---|---|
| `bga run --config configs/baseline.yaml [--set ...] [--seq ...] [--layer ...] [--force ...] [--remote modal] [--remote-layer ...]` | runs the layers |
| `bga eval --config a.yaml [--config b.yaml] [--gold v0] [--held-out] [--write]` | runs, reusing cached outputs, and scores each config side by side; `--write` appends rows to `docs/metrics.md` |
| `bga radar --config ...` | renders a radar video per sequence (`bga.export.radar`): the broadcast frame above, the FIBA court with players below |
| `bga backends` | lists the registered backends |

### Evaluation (`bga.eval`)

- **Gold export** (`gold.py`): label files to MOTChallenge sequences, filling gaps of up to 3 sampled frames by interpolation.
- **Tracking** (`mot.py`): HOTA, DetA, AssA, MOTA, IDF1 and IDSW from TrackEval, pinned to a commit. Players and referees are one class; scored on every 5th frame.
- **Other scores** (`scores.py`):
  - class-agnostic person AP50 on the 100 box-gold frames;
  - team accuracy per gold track (majority of matched boxes);
  - number accuracy per gold track with a legible number (confirmed fraction, wrong count);
  - milliseconds per frame per layer;
  - churn (`churn.py`): id-set changes and track lifetimes.

`docs/metrics.md` holds two rows, both for `baseline@70498c00` on gold v0: dev and held-out.

### Remote execution (`bga.remote`)

- `modal_runner.py` is the `bga-runner` Modal app:
  - its image is `debian_slim` on Python 3.12;
  - clips go up once to the `bga-data` volume;
  - it runs one layer per sequence and returns the outputs, which the pipeline writes into local `runs/`;
  - cost per call comes from a list-price table.
- `sam2_prefill.py` is a separate Modal app (`bga-sam2-prefill`, L40S). It runs SAM 2.1 hiera-large video propagation for the gold-track prefill, and is a labelling aid, not a layer.
- Spend is logged by hand in `docs/costs.md`: 3.60 USD of GPU time, all in October 2026.

### Court helpers (`bga.court.propagate`)

These are labelling aids, not a layer:
- `Propagator` carries a keyframe's homography through its shot. It chains frame-to-frame homographies from SIFT matches inside the court area, and stops when RANSAC inliers drop below 40.
- `court_to_image` inverts an image→court homography. It fixes the sign at a floor point in the image, not at a court point, so a court partly behind the camera still projects correctly.
- `court_mask` and `landmarks` project the court outline and its 33 vertices into the image.

## Data

### Clips and per-clip labels

- **Clips:** `scripts/fetch_highlights.sh` downloads the 31 clips in `data/clip_ids.txt` into `data/raw/`. They are mostly 1080p50 H.264; `scripts/transcode_h264.sh` re-encodes AV1 or VP9.
- **Per-clip labels:** `data/labels/<clip>.json`, label format version 2, with these fields:
  - `clip`, `fps`, `frame_count`, `width`, `height`, `stride`;
  - `meta`: season, home and away teams, source, title, `review_seconds` per day;
  - `shots`: start, end, camera tag;
  - `identities`: id to team, number and name;
  - `frames`: frame to boxes.

  `tools/labeler/prepare_clips.py` writes the shots, with cuts from `bga.ingest.hist_diff`. The 4 gold clips also have `.identities.csv` and `.mot.txt` exports.
- **Rosters:** `data/rosters/<club>.json` has `team`, `team_id_plk`, `aliases`, `retrieved`, and `seasons` (2024/25, 2025/26 and 2026/27). It is written by `scripts/fetch_rosters.py` from plk.pl.

### Gold set (details in `docs/gold.md`)

- **Manifest:** `data/labels/gold.json` lists 4 games with 3 central shots each, at stride 5. `Tnuscrm9RyU` is flagged held-out.
- **Track gold:** `data/gold/v0/<youtube_id>_s<shot>/` is one MOTChallenge sequence per shot: `gt/gt.txt`, `frames.txt`, `seqinfo.ini` and `identities.csv`. A `manifest.json` sits alongside the sequences, and box gold is in `boxes.json`.
- **Audit trail:**
  - `data/labels/review/*.ops.txt` holds every review operation, applied in order by `scripts/review_ops.py`;
  - `data/labels/sheet_ledger.tsv` logs every contact sheet read;
  - `data/labels/box_gold.json` lists the 100 box-gold frames.

### Court keyframes

- **Queue:** `data/labels/court/queue.json` lists 42 keyframes, one per shot: 24 gold (6 per gold game) and 18 training frames (1 or 2 from each of 12 other halls). Each entry carries the NBA model's homography as its starting point (`prefill_H`). `queue_redo.json` lists 9 keyframes picked for a second pass.
- **Labels:** `data/labels/court/<clip>.json` maps each frame to:
  - `H`: image pixels to court cm, normalised so h33 = 1;
  - `fixed`: vertex index to `[x, y]`;
  - `lines`: pins, each `{el, p}`;
  - `status`: `ok` or `unusable`;
  - `fit_rms_px` and `fit_unc_px` (saved by the current page);
  - `saved_at`.

  Plus `meta.review_seconds`. 41 of the 42 are `ok`.
- **Geometry:** `tools/labeler/static/fiba_court.json` is written by `scripts/court_geometry.py`. It has 33 vertices, labelled `01` to `41` in `sports`' numbering, plus edges and the arcs as polylines, in cm on a 2800 × 1500 court with vertex `01` at (0, 0).

### Synthetic fixtures

`data/fixtures/synthetic/*.json` holds 8 simulated possessions, in schema `bga.tracking2d/v0` (format in `docs/fixtures.md`). `scripts/synth_plays.py` writes them deterministically, and `tests/test_synthetic_plays.py` fails if they drift. They are input to the coach page only, never gold or training data.

## Labelling tools

### Labeller server

`tools/labeler/server.py` is a stdlib HTTP server on port 8765, serving two pages:

- **`/` (track labeller):** `static/index.html` and `app.js`. Shot tagging, boxes, identities, and track-level review: K confirm, P split, M merge, ⇧Del delete a track, ⇧D accept a shot. A filmstrip shows 8 crops per track, and active time is logged in `meta.review_seconds`.
- **`/court` (court keyframes):** `static/court.html`.
  - **Dots:** drag a landmark dot onto its true spot.
  - **Pins:** shift-click any painted line or circle. A pin attaches to the nearest of 20 painted elements (sidelines, centre line and circle, and at each end the baseline, key sides, free-throw line and circle, three-point straights and arc); Tab or a list reassigns it.
  - **Fit:** Levenberg-Marquardt on the dots' reprojection error and the pins' distance to their line. It starts from the current court, the anchor and a DLT, and restarts from shifted courts when the fit stays poor.
  - **Uncertainty:** the top bar shows the fit's rms and how far the lines in view could move given 2 px click error. It warns above 15 px.
  - **Folds:** a fit is refused if it folds between the dots and pins or mirrors the court there; parts of the court may be behind the camera.
  - **Queues:** `?set=<name>` opens `queue_<name>.json`.

API endpoints:

| Method | Path |
|---|---|
| GET | `/api/clips`, `/api/clip`, `/api/frame`, `/api/shots`, `/api/pipeline`, `/api/court/queue`, `/api/court/labels` |
| POST | `/api/labels`, `/api/shots`, `/api/court/labels` |

Paths are confined to the repository by `safe_path`.

### Scripts by job

| Job | Scripts |
|---|---|
| Clips and shots | `fetch_highlights.sh`, `transcode_h264.sh`, `fetch_rosters.py`, `compare_shots.py`, `shot_diff_sheet.py`, `shot_sheets.py`, `check_labeller.py` |
| Track prefill | `prefill_tracks.py` (Skalski's detector plus ByteTrack), `link_prefill.py`, `extract_shot_frames.py`, `sam_to_tracks.py` (SAM 2.1 identities fused with detector boxes), `import_prefill.py` |
| Review | `track_sheets.py`, `review_sheets.py` (with `--propose` kit-colour ops), `review_ops.py` |
| Gold export | `export_gold.py`, `box_gold.py` |
| Court | `court_geometry.py`, `court_prefill.py` (the NBA court model over every clip), `court_keyframes.py`, `court_propagate.py`, `qa_court_overlay.py` |
| Probes (not metrics rows) | `probe_reid.py` (Roboflow `reid` encoders on gold v0), `probe_court_keyframes.py` (the NBA court model and propagation against the court keyframes) |

Contact sheets for reading by eye come from `bga.review.sheets`.

## Demo site

The site is published to the Vercel project `basketball-analyzer-demo` by `scripts/deploy_demo.sh`. It is built outside the `bga` pipeline:

- **Video and analysis pages:** `scripts/demo_web.py` builds `/` and `/analysis/`, plus `/en/` and `/en/analysis/`, from three shots of `OSp445e0mXU`. It runs Skalski's models through the helpers in `scripts/demo_video.py` and caches number reads and homographies in `runs/demo/cache`.
  - Templates: `demo_page_video.html` and `demo_page_analysis.html`.
  - Text: `demo_strings.py` (Polish at `/`, English at `/en/`).
  - Check: `demo_audit.py` compares the teams and numbers shown with gold v0.
- **Coach page:** `scripts/coach_web.py` builds `/coach/` and `/en/coach/` from the synthetic fixtures, using `coach_insights.py` (plays, coverage, shot quality, key moments), `coach_strings.py` and `coach_page.html`. `deploy_demo.sh` rebuilds it on every deploy.
- **Roadmap:** `docs/roadmap.html` is published as `/roadmap/`.
- **Local serving:** `scripts/serve_demo.py` serves the build with HTTP Range support, which video seeking needs.

## Environment

- **Python:** 3.12 with `uv`.
- **Core dependencies:** numpy, opencv-python-headless, pydantic, pyyaml, typer.
- **Extras:**

  | Extra | Contents |
  |---|---|
  | `onnx` | onnxruntime |
  | `prefill` | Roboflow `inference` |
  | `remote` | modal |
  | `eval` | TrackEval, pycocotools |
  | `demo` | `sports`, transformers, umap-learn, scikit-learn, sentencepiece |

  `uv sync --all-extras` is needed for the full test suite.
- **OpenCV:** `opencv-python` and `opencv-contrib-python` are overridden away, leaving one headless cv2.
- **Checks:** `ruff check .`, `ruff format --check .` and `pytest`, with tests on config hashing, shot cuts, gold export, MOT scoring, churn, NMS, review ops, rosters, the labeller server, court orientation and the synthetic fixtures.
