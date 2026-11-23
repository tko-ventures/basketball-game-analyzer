# Architecture: what is built

This page describes the code and data that exist in the repository, as of 2026-11-21 (M0 to M5, and M6 so far). It records only what is in the repository. Intentions, targets and milestones are in `docs/PLAN.md`, decisions in `docs/decisions.md`, and scores in `docs/metrics.md`.

The repository has three parts:

1. **The `bga` package** (`src/bga/`). A pipeline of swappable layers, the gold-set evaluation, and a runner for Modal.
2. **Labelling tools** (`tools/labeler/`, most of `scripts/`). These build and check the gold set and the court keyframes.
3. **The demo site** (`scripts/demo_*`, `scripts/coach_*`). Web pages published to Vercel. They are built by their own scripts, not by the `bga` pipeline.

`configs/baseline.yaml` uses only public pretrained models. Five training setups exist (`bga train`, below): court keypoints, whose weights `configs/court-plk-v1*.yaml` use; detection, whose weights the ten `configs/detect-*.yaml` use; the appearance encoder (`configs/train/appearance-reid-*.yaml`), whose weights the `*-plk-r1*` track and appearance configs and the configs built on them use; the number crop classifier (`configs/train/numbers-crop_cls-*.yaml`), whose weights `configs/numbers-crop_cls-*.yaml` use; and the PARSeq number reader (`configs/train/numbers-parseq-*.yaml`), whose weights `configs/numbers-parseq-r1.yaml` uses.

## Repository layout

| Path | Contents | In git |
|---|---|---|
| `src/bga/` | the pipeline package, about 8,700 lines | yes |
| `configs/baseline.yaml`, `configs/court-plk-v1*.yaml`, `configs/detect-*.yaml`, `configs/team-*.yaml`, `configs/track-*.yaml`, `configs/appearance-*.yaml`, `configs/link-*.yaml`, `configs/crops-*.yaml` | run configs: the public-model baseline; the baseline with the PLK court model, unsmoothed and with smoothing windows of 2 and 4; the baseline with the detect layer swapped for one trained detector each (`rfdetr_torch` at confidence 0.05: `nba-704`, `nba-1280`, `nba-tile2`, `plk-r1`, `plk-r2`, `plk-r2-fast`, `plk-r2-fast-es`, `plk-r2-xl`, `plk-r3`, `plk-r3-poss`); `detect-plk-r3-poss` with the team layer seeded 0 to 4 (`team-plk-r3-poss-seed0` to `seed4`), the same with the team crop on the chest (`team-plk-r3-poss-chest-seed0` to `seed4`), and fitted on 5 frames a second (`team-plk-r3-poss-fit5-seed0` to `seed4`); `track-bytetrack`, which is `team-plk-r3-poss-fit5-seed0` without the numbers and positions layers, and `track-bytetrack-tuned`, the same with tuned ByteTrack params; `track-trackers-bytetrack` and `track-trackers-bytetrack-tuned`, the same on the `trackers` backend at library defaults and tuned; `track-trackers-botsort` (BoT-SORT at library defaults), `track-trackers-botsort-precmc` (the same with precomputed camera motion), `track-trackers-botsort-tuned`, and `track-trackers-botsort-reid` with its twin `track-trackers-botsort-noreid` (tuned BoT-SORT with internal camera motion, with and without the `osnet_x1_0_sportsmot` re-ID encoder), `track-trackers-botsort-reid-tuned` (re-ID settings tuned on dev on Modal) and `track-trackers-botsort-reid-tuned-numbers` (the same with the numbers layer) and `-numbers-all` (the reader not stopping at a confirmation); `track-sam2` (the `sam2` tracker over the same detector and team layers); `track-trackers-botsort-plk-r1` and `-plk-r1-shot-last` (BoT-SORT's re-ID model swapped for the PLK encoder `plk-r1` or `plk-r1-shot-last`) and `-plk-r1-shot-last-tuned` (the latter with its re-ID settings re-tuned on dev on Modal); `appearance-reid-sportsmot`, which is `track-trackers-botsort-reid-tuned` with the appearance layer (`reid`, `osnet_x1_0_sportsmot`), and `appearance-reid-plk-r1`, `-plk-r1-team85` and `-plk-r1-shot-last`, the same with the PLK fine-tunes; `link-botsort-reid-tuned` (the same plus the `greedy` link layer) and its twin `link-botsort-reid-tuned-noapp` (appearance `none`), `link-botsort-reid-tuned-plk-r1` and `-plk-r1-shot-last` (the same with a PLK encoder in the appearance layer), `link-botsort-reid-tuned-cap5` (the same with `max_per_team` 5), `link-botsort-plk-r1-shot-last-tuned-cap5` (the same over `track-trackers-botsort-plk-r1-shot-last-tuned`, with `plk-r1-shot-last` in the appearance layer) and its twin `-noapp-cap5` (appearance `none`), `link-sam2-plk-cap5` and `link-sam2-noapp-cap5` (that linking over `track-sam2`, with and without appearance), `link-botsort-reid-tuned-numbers` (the same with the numbers layer as a link term) `link-botsort-reid-tuned-numbers-owner` (the same with `one_owner`) `link-botsort-reid-tuned-numbers-roster` (the same with `roster`) `link-botsort-reid-tuned-numbers-roster-all` (the same over `-numbers-all`) and its twin `-nojoin` (joins off: `max_gap_s` 0), and `link-bytetrack-tuned` and `link-bytetrack-tuned-noapp` (the same over `track-trackers-bytetrack-tuned`); and `crops-reid-r1`, `link-botsort-reid-tuned` without the court layer over the shots of four non-gold 2025/26 games (`shots/clips`), the source of the appearance crops; and `crops-numbers-r2`, `link-botsort-plk-r1-shot-last-tuned-noapp-cap5` without the court layer over the shots of eight more non-gold 2025/26 games, for the round-2 number crops | yes |
| `configs/train/` | training configs, one per training setup | yes |
| `weights/` | trained weights, one directory per weight id | no |
| `tools/labeler/` | the local labelling server and its three pages | yes |
| `scripts/` | labelling, gold export, court, probe, demo and coach scripts | yes |
| `tests/` | 28 test files, 380 tests | yes |
| `data/clip_ids.txt` | the 39 YouTube highlight clips: 31 up to the 2026 Superpuchar, and the eight league games of 2026/27 rounds 1 to 8 added on 2026-11-23 | yes |
| `data/labels/` | per-clip labels, the gold manifest, review ops, court keyframes, training manifests (`detect/`, `appearance/`) | yes |
| `data/gold/v0/` | the exported gold set | yes |
| `data/rosters/` | 19 PLK clubs, seasons 2024/25 to 2026/27 | yes |
| `data/fixtures/synthetic/` | 8 simulated possessions for the coach page | yes |
| `data/raw/` | downloaded videos | no |
| `runs/` | every stage output, probe, sheet and demo build | no |
| `docs/` | plan, decisions, metrics, training runs, costs, gold set, fixtures, roadmap, this page | yes |

## The `bga` pipeline

### Run units

A **sequence** (`bga.stage.Sequence`) is one shot of one clip, processed on a list of source frame indices. `FrameReader` decodes frames with a small cache. A read up to 64 frames ahead of the last one decodes and drops the frames between; only a longer step or a step back seeks, which restarts decoding at the previous keyframe (122 ms a frame on the gold shots at stride 5 against 3.7 ms without seeking, the same frames either way).

There are two sequence sources. `shots/clips` (`bga.ingest.clip_shots`) gives the shots of named non-gold clips from their label files, those of at least `min_s` (3) seconds, on the multiples of the stride and named `<youtube id>_s<shot index>`; it refuses gold games (`bga.train.leakage.assert_no_gold`) and is for training data. `shots/gold` (`bga.ingest.gold_shots`) produces the 12 gold shots from `data/gold/<version>/manifest.json`, from each shot's first gold frame to its last. The stride must divide the gold stride of 5, so every gold frame is processed. A `held_out` parameter selects the held-out game instead of the 9 dev shots.

### Layers

The layers and the upstream outputs each one reads are fixed in `bga.config.LAYERS`. A config may leave out the upstream layers in `bga.config.OPTIONAL_UPSTREAM` (court, appearance and numbers); every other upstream layer must be configured:

```
shots ─┬─> court ──────────────┐
       └─> detect ─┬─> team    │
                   ├─> appearance
                   ├─> track <─┤   (track reads detect and court)
                   │     ├─> numbers   (reads detect and track)
                   │     ├─> link      (reads track and team; appearance, court and numbers if configured)
                   │     └─> positions (reads track and court)
```

Each layer's output format is documented in its `base.py`:

| Layer | Output per sequence |
|---|---|
| `shots` | `{"sequences": [...]}` |
| `court` | `{"frames": {frame: 3×3 image→court matrix in cm, or null}, "units": "cm"}` |
| `detect` | `{"frames": {frame: [[x1, y1, x2, y2, score, class_name], ...]}, "classes": [...]}`, using the 10-class schema in `bga.types` |
| `team` | `{"frames": {frame: ["home" \| "away" \| "ref" \| "", ...]}, "clusters": {cluster: side}, "separation": {"silhouette": s, "minor_share": f}, "ms_per_frame": ms}`, labels aligned with the detect boxes; outputs written before 2026-11-20 lack `separation` |
| `appearance` | `{"boxes": [[frame, det_index], ...], "dim": d, "model": alias, "ms_per_crop": ms, "arrays": {"embeddings": record}}`; row *i* of the `(boxes, d)` float16 array of unit-length embeddings belongs to box *i*; `none` writes no boxes and no array |
| `track` | `{"frames": {frame: [[track_id, x1, y1, x2, y2, score, class_name, det_index], ...]}}` |
| `link` | the `track` format with linked pieces under one id, plus `"splits": [[track_id, frame, new_id, distance], ...]`, `"links": [[earlier_id, later_id, cost, gap_s], ...]` and `"ids": {piece_id: linked_id}` and `"numbers": {linked_id: "23" or null}` and `"dropped_over_cap": n`; a chain takes its earliest piece's id |
| `numbers` | `{"tracks": {track_id: "23" or null}, "log": [[frame, track_id, "23" or null], ...], "reads": n, "ms_per_read": ms}`; `log` lists every read, and outputs written before 2026-11-21 lack it |
| `positions` | `{"tracks": {track_id: [[frame, x_m, y_m], ...]}, "court_m": [28.0, 15.0]}` |

### Backends

A backend is a class registered with `@register("<layer>", "<name>")` (`bga.registry`). The registry imports the modules listed in `BACKEND_MODULES` lazily. The shots, court, detect and appearance layers have two registered backends each, the track and numbers layers three, and every other layer one. `configs/baseline.yaml` selects the first of each, except appearance, which it does not configure:

| Layer | Backend | What it does | Model |
|---|---|---|---|
| shots | `gold` | gold shots at stride 5 | none |
| shots | `clips` | shots of named non-gold clips, at least `min_s` long; `clips` and `min_s` only select, so they are kept out of the hashes with the gold selection params | none |
| court | `roboflow_keypoints` | keypoints above `kp_conf` (at least 4), then a RANSAC homography to the FIBA vertices of `sports.CourtConfiguration`, threshold 40 cm | `basketball-court-detection-2/14` (Roboflow, NBA frames) |
| detect | `roboflow_inference` | Roboflow `inference` run locally (weights fetched with `ROBOFLOW_API_KEY`, from the environment or `~/.config/bga/roboflow.env`); confidence 0.4, IoU 0.9 | `basketball-player-detection-3-ycjdo/6` (RF-DETR-S, NBA frames) |
| team | `siglip_kmeans` | `sports.TeamClassifier` (SigLIP, UMAP, KMeans) fitted per sequence on crops sampled at 1 fps, referees excluded; the cluster with the brighter kit takes the side the clip meta marks `light`, and a clip whose meta does not mark exactly one side `light` is refused (`light_side`). SigLIP is loaded once in `load()`; each sequence gets a new UMAP and KMeans. `crop` (0.4) is the share of the box kept around the point `crop_cy` (0.5, the box centre) of its height from the top. `seed` (default unset) seeds UMAP and KMeans. Reports the separation of the fitted clusters (silhouette in UMAP space, smaller cluster's share), which changes no label | `google/siglip-base-patch16-224` |
| appearance | `none` | no embeddings: an empty output | none |
| appearance | `reid` | a Roboflow `reid` 0.1.0 encoder (`model`, a `reid` alias) on the person boxes scoring at least `min_score` (0.1); crops of one frame go through in batches of `batch_size` (32) on the device `bga.device.pick_device` picks; embeddings L2-normalised and stored as float16. The weights id is `pretrained:roboflow/reid/<model>`, or a trained `appearance/reid/<name>@<hash8>` loaded from its folder (`weights.safetensors` and `reid_config.json`, `reid`'s folder format). Used by `configs/appearance-reid-sportsmot.yaml` | `osnet_x1_0_sportsmot` (OSNet x1.0 fine-tuned on SportsMOT) |
| track | `bytetrack` | ByteTrack on person boxes, in pixels: supervision 0.30.9's implementation, vendored unchanged in `bga.track._sv_bytetrack` (MIT, Roboflow) because supervision 0.31 removed it | none |
| track | `trackers` | Roboflow `trackers` 2.7 on person boxes, in pixels, chosen by `tracker`: `bytetrack` (`ByteTrackTracker`), `botsort` (`BoTSORTTracker`, given each frame for camera-motion compensation and optional re-ID through `reid_model` (a `reid` source, or `weights` for the entry's trained `appearance/reid` weight id); with `cmc: precomputed` the motion transforms are estimated once per shot and cached in `runs/_cache/cmc/`, masking person boxes at `cmc_mask_score`, which gives identical tracks at the default threshold and lets sweeps skip frame decoding), `ocsort`, `sort`; constructor arguments pass through `args`; unconfirmed tracks (id -1) are dropped and ids start at 1. Used by `configs/track-trackers-*.yaml` | none, or a `reid` encoder |
| track | `sam2` | SAM 2.1 video propagation (official predictor, `facebook/sam2.1-hiera-large`, commit `2b90b9f5` as in the gold prefill), CUDA only, on an L40S on Modal: the detector's person boxes scoring at least `min_score` (0.4) and no wider than tall prompt the shot's first grid frame; masks propagate through every source frame (JPEGs at a 1024 px long side); a detection no mask covers on two consecutive grid frames becomes a new object, at most `max_rounds` (12) passes. Each mask box on a grid frame is a row with the class, score and index of the highest-scoring detection it overlaps at IoU 0.5 or more (the detector's low-score duplicates carry no team label), else a `player` row with score 0.5 and index -1. Used by `configs/track-sam2.yaml` and `configs/link-sam2-*.yaml` | SAM 2.1 hiera large (Apache 2.0) |
| link | `greedy` | joins tracklets of one shot, cheapest first, each with at most one successor and one predecessor. A join needs the same majority team (from `team`), no overlap in time and a gap of at most `max_gap_s` (10); its cost is the cosine distance between the mean embeddings of the earlier tracklet's last `ends` (5) boxes and the later one's first, or without embeddings the gap as a share of `max_gap_s`; joins above `max_cost` (0.3) are refused, and so are joins whose foot points, where both ends have a homography, are further apart than `max_speed_mps` (8) times the gap plus `court_slack_m` (2). With `split_cost` set (default off), a tracklet is first cut where the distance between the mean embeddings of the `split_window` (5) boxes before and after a point peaks above it. With a numbers layer, a tracklet carries its tracker id's confirmed number (a cut piece only when the read log has a read of it inside the piece); tracklets or chains with different numbers are never joined, the same number joins at cost 0 past the `max_cost` gate, and with `one_owner` (default off) a (team, number) on several final tracks stays on the one with the most boxes. With `roster` (default off), a tracklet's number is instead confirmed again from the logged reads inside it, after dropping numbers not on its team's roster for the clip's season (`data/rosters`, through `bga.numbers.base.roster_numbers`), by `confirm_reads` (3) equal reads in a row. With `max_per_team` (default off), in any frame where more than that many linked tracks carry one team (each track's majority team), the rows of the team's shortest tracks are dropped until that many are left, counted in `dropped_over_cap`. Used by `configs/link-*.yaml` | none |
| numbers | `smolvlm_ocr` | reads `number` boxes (cut with `pad`, 10 px, by `bga.numbers.base.number_crop`) lying at least 90% inside a track's box; `read_crops` reads single crops for `bga eval-numbers`, without probabilities; a number is confirmed after 3 consecutive equal reads, after which the track is no longer read unless `stop_on_confirm` is off (at most `max_reads_per_track`, 12); runs on MPS with `max_new_tokens=4`. Used by `configs/track-trackers-botsort-reid-tuned-numbers*.yaml`, `configs/link-botsort-reid-tuned-numbers*.yaml` and `configs/numbers-smolvlm-nba.yaml` | `basketball-jersey-numbers-ocr/3` (SmolVLM2, NBA crops) |
| numbers | `crop_cls` | a crop classifier over `unreadable` and 0 to 99 trained by `bga train`, loaded from its weight id with its `cls_config.json` (any `timm` architecture, and the head: `flat`, one output per class, or `factorised`, 21 outputs for `unreadable`, the tens digit and the ones digit, which `class_probs` turns into a probability per class: a number's is the readable share times its tens and ones probabilities) on MPS, crops in batches of `batch_size` (64); `read_crops` gives a probability per class, `read_crop` the most probable class (none for `unreadable`); the track matching, sampling and confirmation are `smolvlm_ocr`'s, shared through `bga.numbers.base.TrackNumbers` | weights `numbers/crop_cls/<name>@<hash8>` |
| numbers | `parseq` | PARSeq (`strhub` at commit `1902db04` of `baudm/parseq`, the `parseq` extra), a scene-text recogniser: a crop is stretched to 32 x 128 and decoded into a distribution over 94 characters and an end token per position; `number_probs` scores each number from them (its digits at their positions times the end token after them, the rest is `unreadable`), so `read_crops` gives a probability per class as `crop_cls` does; crops in batches of `batch_size` (64) on MPS; the track matching, sampling and confirmation are `smolvlm_ocr`'s. Loads the released checkpoint (fetched once by `torch.hub`, hash checked) or a fine-tune's `weights.safetensors` with `parseq_config.json` trained by `bga train`. Used by `configs/numbers-parseq-zeroshot.yaml` and `configs/numbers-parseq-r1.yaml` | `pretrained:github/baudm/parseq/v1.0.0/parseq-bb5792a6` (scene text, not fine-tuned), or weights `numbers/parseq/<name>@<hash8>` |
| detect | `rfdetr_torch` | RF-DETR weights trained by `bga train`, loaded from their weight id; class names from the checkpoint, tiling from `train.json`; on MPS, confidence 0.05 by default. Weights trained on square tiles run on the same tiles (`bga.detect.tiles`: two overlapping crops of the full frame height), and the per-tile detections are merged with a per-class suppression that prefers boxes not cut by a tile edge. Used by `configs/detect-*.yaml` | weights `detect/rfdetr_torch/<name>@<hash8>` |
| court | `rfdetr_keypoint` | a court keypoint model trained by `bga train`, loaded from its weight id on MPS (the class and keypoint counts come from its `train.json`); the most confident court detection gives the landmarks, then the same RANSAC homography as `roboflow_keypoints`. Used by `configs/court-plk-v1*.yaml` | weights `court/rfdetr_keypoint/<name>@<hash8>` |
| positions | `bbox_bottom` | box bottom-centre through the frame's homography, then `sports` `clean_paths` (jumps over 0.7 m, smoothing window 9) | none |

Code that exists but no config uses:
- `bga.detect.onnx_rfdetr`, an RF-DETR decoder on ONNX Runtime with the CoreML provider. It is not registered.
- `bga.detect.nms`, nested-box suppression, used by `onnx_rfdetr`; its `iou` is also used by `bga.detect.tiles`.

Both are covered by `tests/test_nms.py`.

### Configs, hashes and run directories

- **Config model:** a config is a pydantic `RunConfig` (`bga.config`). Each layer is a `LayerConfig` with `backend`, `params` and `weights`. `weights` is a declared id such as `pretrained:roboflow/<model>`.
- **Layer hash:** covers the layer's own entry plus the hashes of its upstream layers. The `shots` parameters that only choose sequences (`version`, `held_out`, `root`, `labels`) are excluded, so dev and held-out runs share directories.
- **Config hash:** the hash of all layer hashes.
- **Overrides:** `--set layer.params.key=value` changes one value and appends the override to the config name.
- **Outputs:** each layer writes `runs/<clip>/<layer>/<backend>@<hash8>/<sequence>.json`, plus a `meta.json`. Arrays a backend returns under `arrays` are saved next to the JSON as `<sequence>.<name>.npy` by `bga.pipeline.write_output`, and the JSON keeps a `{file, shape, dtype}` record in their place; upstream outputs reach a backend with their arrays loaded (`load_output(..., arrays=True)`), locally and on Modal. Each file is written to a temp file and renamed (`bga.paths.write_atomic`), the JSON last, so an output that exists is complete. The meta file holds the layer entry, upstream hashes, and the config name, git commit and wall time of the last run that wrote to the directory. Under `sequences` it keeps one record per sequence written since 2026-11-22 (commit, config, time written, wall time, and `platform`: `local`, or `modal` with GPU, seconds and USD); a run over some sequences adds or replaces only theirs. `remote` totals the seconds and USD of the sequences made remotely.

`bga.pipeline.run` executes one layer at a time over all sequences, calling `load()` before and `unload()` after. Only one model is resident at a time. An existing output is reused unless `--force` names its layer. A backend that declares `requires = "cuda"` is skipped locally, and runs only with `--remote modal`.

### Command line

`bga` is a project script (`bga.cli`, typer):

| Command | What it does |
|---|---|
| `bga run --config configs/baseline.yaml [--set ...] [--seq ...] [--layer ...] [--force ...] [--remote modal] [--remote-layer ...]` | runs the layers |
| `bga eval --config a.yaml [--config b.yaml] [--gold v0] [--held-out] [--write]` | runs, reusing cached outputs, and scores each config side by side; `--write` adds rows to the tracking table in `docs/metrics.md` |
| `bga eval-court --config a.yaml [--config b.yaml] [--held-out] [--write]` | scores each config's court layer on the gold court keyframes; `--write` adds rows to the court table in `docs/metrics.md` |
| `bga eval-appearance --config a.yaml [--config b.yaml] [--gold v0] [--held-out] [--write]` | embeds the gold boxes with each config's appearance backend and scores re-association and cross-shot identity; `--write` adds rows to the appearance table in `docs/metrics.md` |
| `bga eval-numbers --config a.yaml [--config b.yaml] [--held-out] [--write]` | reads the gold number crops with each config's numbers backend and scores top-1 under the roster, refusal and calibration; `--write` adds rows to the number-reader table in `docs/metrics.md` |
| `bga eval-detect --config a.yaml [--set ...] [--gold v1] [--held-out] [--write]` | runs the detect layer (reusing cached outputs) and scores per-class AP on the box-gold frames; `--write` adds rows to the detection table in `docs/metrics.md` |
| `bga radar --config ...` | renders a radar video per sequence (`bga.export.radar`): the broadcast frame above, the FIBA court with players below |
| `bga backends` | lists the registered backends |
| `bga train --config configs/train/<layer>-<backend>-<name>.yaml [--build-only] [--collect [--partial]] [--wait] [--no-resume]` | builds the dataset, checks it for gold clips, trains on Modal and pulls the weights back (section below) |

The five `eval` commands refuse a config whose layer weights are neither a `pretrained:` id nor a weight id with a `weights/<id>/train.json`.

### Evaluation (`bga.eval`)

- **Gold export** (`gold.py`): label files to MOTChallenge sequences, filling gaps of up to 3 sampled frames by interpolation.
- **Tracking** (`mot.py`): HOTA, DetA, AssA, MOTA, IDF1 and IDSW from TrackEval, pinned to a commit. Players and referees are one class; scored on every 5th frame. Tracks come from the `link` layer when a config has one, else from `track`, and so do their numbers: after linking, number accuracy reads the link output's `numbers`.
- **Detection** (`detect.py`): COCO AP50:95 and AP50 per class of the 10-class schema on the box-gold frames of gold v1 inside the run's sequences, plus `people`, with the six person classes merged on both sides. Each result carries its gold box count.
- **Court** (`court.py`): each gold court keyframe Tomasz labelled `ok` (18 dev, 6 held-out) is mapped by the config's court backend, which runs on the shot's frames within 25 frames of the keyframe at a stride of 5, so smoothing has its neighbours. The court vertices visible under his mapping go through the predicted mapping. Scores per arena and overall: the median over keyframes of the per-frame median error in metres; the same over the paint (key corners, free-throw line ends and centres); the share of keyframes under 0.5 m in the paint; and the unmapped count.
- **Appearance** (`appearance.py`): the config's appearance backend embeds the clean gold boxes (visibility 1) of the selected shots, frame by frame, so the scores depend on neither detector nor tracker. Candidates are ranked by cosine distance; each question reports rank-1, MRR, query count and chance. Re-association: a player's crop, queried every 0.5 s, against his teammates' crops 1 s and 3 s later in the same shot. Cross-shot: the mean embedding of a numbered player (at least 5 clean boxes) against his numbered teammates' means in another shot of the same game, also reported per game. Plus ms per crop. This is the metric form of `scripts/probe_reid.py`, and gives the probe's figures for `osnet_x1_0_sportsmot`.
- **Number readers** (`numbers.py`): the config's numbers backend reads the gold number crops (`data/gold/v1/numbers.json`) of the selected shots through its `read_crops`, each cut by `bga.numbers.base.number_crop` with the backend's `pad`, so the scores depend on neither detector nor tracker.
  - **Legible crops:** top-1 of the free read, top-1 under the roster of the crop's team (`roster_choice`), and the share answered under the roster.
  - **Illegible and not-a-number crops:** the share declined under the roster.
  - **Calibration:** for readers that report probabilities, the expected calibration error (10 bins) of the roster answer's probability. An answer counts as right when it is the label, or when it declines a crop with no digits.
  - **Speed:** ms per crop.
- **Other scores** (`scores.py`):
  - class-agnostic person AP50 on the 100 box-gold frames;
  - team accuracy per gold track (majority of matched boxes);
  - number accuracy per gold track with a legible number (confirmed fraction, wrong count);
  - milliseconds per frame per layer;
  - churn (`churn.py`): id-set changes and track lifetimes.

`docs/metrics.md` has five tables:
- **Tracking:** on gold v0, 53 rows: `baseline@70498c00` on dev and held-out; the 15 team configs `team-plk-r3-poss-seed*`, `-chest-seed*` and `-fit5-seed*` on dev; `team-plk-r3-poss-fit5-seed0` on held-out; and `track-bytetrack`, `track-bytetrack-tuned`, `track-trackers-bytetrack`, `track-trackers-bytetrack-tuned`, `track-trackers-botsort`, `track-trackers-botsort-precmc`, `track-trackers-botsort-tuned`, and `track-trackers-botsort-noreid`, `track-trackers-botsort-reid` and `track-trackers-botsort-reid-tuned` (track layer on Modal) on dev; and `link-botsort-reid-tuned` (also with `split_cost` 0.4), `link-botsort-reid-tuned-noapp`, `link-bytetrack-tuned` and `link-bytetrack-tuned-noapp` on dev; and `track-trackers-botsort-reid-tuned-numbers`, `link-botsort-reid-tuned-numbers`, `link-botsort-reid-tuned-numbers-owner`, `link-botsort-reid-tuned-numbers-roster`, `track-trackers-botsort-reid-tuned-numbers-all`, `link-botsort-reid-tuned-numbers-roster-all` `link-botsort-reid-tuned-numbers-roster-all-nojoin`, `link-botsort-reid-tuned-plk-r1`, `track-trackers-botsort-plk-r1`, `track-trackers-botsort-plk-r1-shot-last`, `link-botsort-reid-tuned-plk-r1-shot-last`, `link-botsort-reid-tuned-cap5`, `track-trackers-botsort-plk-r1-shot-last-tuned`, `link-botsort-plk-r1-shot-last-tuned-cap5`, `link-botsort-plk-r1-shot-last-tuned-noapp-cap5`, `track-sam2`, `link-sam2-noapp-cap5` and `link-sam2-plk-cap5` on dev; and `link-botsort-plk-r1-shot-last-tuned-noapp-cap5` and `link-botsort-reid-tuned-cap5` on held-out.
- **Court:** dev rows for `baseline`, `court-plk-v1` and its two smoothing windows, and held-out rows for `baseline` and `court-plk-v1-smooth4`.
- **Appearance:** on gold v0, a dev row for `appearance-reid-sportsmot`.
- **Detection:** on gold v1, 14 dev rows (`baseline` at its confidence of 0.4 and at 0.05, Roboflow's NBA detector versions 13 and 16, and the ten `configs/detect-*.yaml`) and 2 held-out rows (`baseline` at 0.05 and `detect-plk-r2`).
- **Number readers:** on the gold number crops (v1), dev rows for `numbers-smolvlm-nba`, `numbers-crop_cls-resnet18-r1` and `numbers-crop_cls-dinov2s-r1`, their factorised-head twins `-resnet18-r1-fact` and `-dinov2s-r1-fact`, `numbers-parseq-zeroshot` and `numbers-parseq-r1`.

### Remote execution (`bga.remote`)

- `modal_runner.py` is the `bga-runner` Modal app:
  - its image is `python:3.12-slim-bookworm` (Modal's `debian_slim` is bullseye, whose apt archives are gone) with git, then `uv sync --frozen` of the repository's `uv.lock` with the extras `prefill`, `demo`, `court`, `track` and `remote`, so every layer runs the Mac's exact package versions; then the local `bga` package and `tools/labeler/static/fiba_court.json`;
  - clips go up once to the `bga-data` volume under `clips/`, and trained weights once per weight id under `weights/`, read through `BGA_WEIGHTS`: `train.json` and every file a training setup collects that is not a trainer log (`bga.train.run.weight_files`). A file the volume already has is not compared, so `bga train --collect` removes the volume's copy of the weight id it rewrites;
  - model downloads go to the container's disk (`HF_HOME`, `MODEL_CACHE_DIR`), because `inference_models` and Hugging Face lock their caches with file locks, which the volume does not support;
  - the `roboflow` Modal secret provides `ROBOFLOW_API_KEY`;
  - it runs one layer per sequence in parallel calls and returns the outputs, which the pipeline writes into local `runs/`;
  - cost per call comes from a list-price table.
- **Running layers remotely:** `bga run` and `bga eval` take `--remote modal`, which sends CUDA-only backends to Modal, and `--remote-layer <layer>` with `--remote-gpu <type>` (default `L4`, or `cpu`) for any other layer. Backends pick their device with `bga.device.pick_device` (CUDA, then MPS, then CPU), so one config runs in both places. `bga eval` names the layers made on Modal in a row's notes ("ms on Modal: numbers (L4)"), since their ms per frame is not the Mac's. Each sequence is one parallel call; results are taken as they finish and each output is written at once, so a run stopped part way keeps its finished sequences (with their `meta.json` and cost rows) and the next run computes only the rest. A failed call does not cancel the others: the batch finishes, then raises `RemoteCallsFailedError` with the count, and running again retries the missing ones. `scripts/sweep_track.py --remote-gpu` writes its outputs the same way.
- **Mac and Modal outputs differ slightly.** On 2026-11-20 the number reader on one shot (`OSp445e0mXU_s10`, `baseline`) confirmed the same 5 numbers on the Mac (MPS) and on an L4, plus one more on the L4. Rows compare within one platform. The L4 read 1.39 s per crop against the Mac's 2.34 s; the gain is the parallel shots.
- The runner image also holds SAM 2.1 at commit `2b90b9f5`, built without its CUDA extension, with the hiera-large weights baked in, for `track/sam2`.
- `modal_train.py` is the `bga-train` Modal app, used by `bga train`. Each training container gets 8 CPU cores and 32 GB of memory, and the detection trainer 8 data-loader workers: about 16 s per epoch for RF-DETR Large at 704 px on the NBA set on an L40S, against about 20 s with Modal's default reservation (from Modal's billed GPU time). CPU and memory are billed on top of the GPU. The appearance trainer `train_appearance` runs in its own image (torch, `reid` 0.1.0 and their dependencies at the Mac's versions) on an L4 by default. The PARSeq trainer `train_parseq` has its own image as well (torch and `timm` at the Mac's versions, `strhub` at the pinned commit with `pytorch-lightning` 2.6.6 and `nltk` 3.10.4).
- `sam2_prefill.py` is a separate Modal app (`bga-sam2-prefill`, L40S). It runs SAM 2.1 hiera-large video propagation for the gold-track prefill, and is a labelling aid, not a layer. Its image does not hold `bga`, so the remote call returns seconds and the local entrypoint prices them from `modal_runner.USD_PER_HOUR`, as `sam2_boxmask.py` does.
- `sam2_boxmask.py` is another labelling aid (`bga-sam2-boxmask`, L40S). It masks each box of the box gold v1 prefill with SAM 2.1 hiera-large in image mode, one box prompt per box, and returns each mask's bounding box next to its prompt box in `runs/box_gold/v1/sam_boxes.json`.
- Spend goes to `docs/costs.md` as per-job estimates: `modal_runner.append_cost_rows` inserts rows above the summary line, which is updated by hand from Modal's own figures (`modal billing summary`, `modal billing report --show-resources`). Per Modal: 4.18 USD in October 2026, 10.86 USD in November to 2026-11-19, all within credits.

### Court smoothing (`bga.court.smooth`)

Both court backends take `smooth_window` (default 0, off; the baseline leaves it off). When it is set, the mappings of a sequence, which is one shot, are median-smoothed after the per-frame fits:
- each frame's mapping is evaluated at a fixed grid of 12 × 8 image points; a point counts only where the mapping puts it on the floor, within 2 m of the court;
- each point's court position is replaced by its median over the frames within `smooth_window` each way;
- the frame's homography is refitted from the grid to those medians.

A frame that jumps is outvoted, a frame without a mapping gets one from its neighbours, and nothing crosses a cut. `configs/court-plk-v1-smooth2.yaml` and `-smooth4.yaml` set windows of 2 and 4.

### Court helpers (`bga.court.propagate`)

These are labelling aids, not a layer:
- `Propagator` carries a keyframe's homography through its shot. It chains frame-to-frame homographies from SIFT matches inside the court area, and stops when RANSAC inliers drop below 40.
- `court_to_image` inverts an image→court homography. It fixes the sign at a floor point in the image, not at a court point, so a court partly behind the camera still projects correctly.
- `court_mask` and `landmarks` project the court outline and its 33 vertices into the image.
- `disagreement_m` and `round_trip_m` measure drift in metres: between two mappings, and after carrying a keyframe away and back.
- `court_line_points` and `line_support` measure how well a mapping fits the frame: the share of projected line points within 2 px of a painted line (white top-hat), compared with the keyframe's own share.

### Training (`bga.train`)

- **Config:** `configs/train/<layer>-<backend>-<name>.yaml`, a pydantic `TrainConfig` (`bga.train.config`):
  - `base_weights` (name, URL, licence) and `package` (the trainer, pinned);
  - `sources`: Roboflow dataset versions (`workspace/project/version`, exported as COCO) and committed label manifests pinned by sha256;
  - `recipe`: resolution, epochs, batch size, gradient accumulation, learning rate, seed, and the trainer metric used to select the checkpoint;
  - `platform`: `modal`, with GPU and timeout.

  The weight id is `<layer>/<backend>/<name>@<hash8>`, the hash covering the whole config.
- **Leakage guard** (`bga.train.leakage`): refuses any manifest that names a gold clip, held out or not; it reads the clips and videos of frames, keyframes, appearance identities, number-crop tracklets and strays. `tests/test_gold_leakage.py` checks every committed config and every committed detection, appearance and number-crop manifest.
- **Dataset (detection):** `bga.train.detect_data` builds a COCO detection dataset from Roboflow exports in the 10-class schema (categories matched by name, id 0 the placeholder), optionally cut into square tiles (`recipe.tiles`). Detection manifests (`data/labels/detect/train_manifest_<name>.json`) are checked against their sha256 and the gold clips, and their frames decoded from the local videos, resized with their boxes to the source's `image_width` (every detection config sets 1920, the clips' width, so frames stay at full size), into the split each frame names.
- **Dataset (appearance)** (`bga.train.appearance_data`): identity folders `train/<identity>/<frame>.jpg` and `valid/...` (the layout `reid.load_identity_folders` reads), cut from the local videos by crop manifests (`data/labels/appearance/crops_*.json`, from `scripts/reid_crops.py`) pinned by sha256; identities marked `dropped` are left out, and an identity validates when its game is in `recipe.val_clips`, so validation never shares a game with training.
- **Appearance trainer** (`bga.train.reid_trainer`, run by `train_appearance`): starts from the `reid` alias in `base_weights.name`; P x K batches (`batch_size` / `crops_per_id` identities, from the whole split or, with `shot_batches`, from one shot at a time, the shot read from the identity folder's name), identity loss with label smoothing 0.1 through a batch-norm neck and a bias-free classifier (weighted by `id_loss_weight`, 1 by default), plus batch-hard triplet loss on cosine distance (`triplet_margin`); Adam with weight decay 5e-4 and a cosine schedule; crops stretched to the encoder input, flipped, padded and cropped, randomly erased. Before the first epoch and every `eval_interval` epochs it scores the encoder on the `valid` identities with `reid`'s `RetrievalProtocol` (2 queries per identity): `val/map`, `val/rank1` and `val/minp` in `metrics.csv`, the best on `select_on` kept.
- **Dataset (court)** (`bga.train.court_data`): merges the Roboflow export and the manifest frames into one COCO-keypoint dataset with `train/`, `valid/` and `test/` folders under `runs/train/<weight id>/dataset/`. Manifest frames are decoded from the local videos and resized to 1280 px wide.
- **Dataset (numbers)** (`bga.train.number_data`): class folders `train/<class>/<n>.jpg` and `valid/...` of `bga.numbers.base.NUMBER_CLASSES` (`unreadable` and 0 to 99), cut from the local videos with `recipe.pad` pixels (10) around each box, from number-crop manifests (`data/labels/numbers/train_*.json`, from `scripts/number_crops.py`) pinned by sha256. Crops of a tracklet read as a number take that number; crops of `not_number` tracklets, the manifest's `strays` and, with `recipe.illegible_as_unreadable`, crops of `illegible` tracklets train `unreadable`; `mixed` and unread tracklets are left out. A crop validates when its game is in `recipe.val_clips`.
- **Number trainer** (`bga.train.cls_trainer`, run by `train_numbers`): a `timm` classifier named by `base_weights.name`, pretrained weights from `timm`, with 101 outputs, or 21 with `recipe.head: factorised` (`unreadable`; tens digit: none or 1 to 9; ones digit: 0 to 9). Crops go through `bga.numbers.crop_cls.to_input` (stretched to the square input) and then augmentation: affine slant, rotation, shift and scale, blur, colour and contrast, grayscale, no flips. Batches are drawn with weights 1 / sqrt(class size); cross-entropy with label smoothing 0.1 (the factorised head: binary cross-entropy for `unreadable` on every crop, plus that cross-entropy for the tens and for the ones digit on the crops of numbers), AdamW (weight decay 0.05), one warm-up epoch then cosine, bf16 autocast on CUDA. After each epoch, `val/top1`, `val/top1_numbers`, `val/refused`, and `val/top1_seen` and `val/top1_unseen` (the crops of numbers that have training crops, and of those that have none) go to `metrics.csv`; the best epoch on `select_on` is saved as `weights.safetensors` with `cls_config.json` (architecture, input size, classes, head, normalisation). The trainer's result lists the validation numbers without training crops (`unseen`).
- **PARSeq trainer** (`bga.train.parseq_trainer`, run by `train_parseq`): the released PARSeq (`base_weights.name` `parseq-bb5792a6`, fetched by `torch.hub` in the container, hash checked) fine-tuned on the same class folders as the number trainer. A crop's text is its number, and the empty string for `unreadable`. Crops go through `bga.numbers.parseq.to_input` (stretched to 32 x 128) and the number trainer's augmentation (`cls_trainer.augmentation`); batches are drawn with weights 1 / sqrt(class size). The loss is PARSeq's permutation language modelling (`plm_loss`, after `strhub`'s `training_step`; a batch of empty labels only is scored in reading order). AdamW without weight decay, one warm-up epoch then cosine, full precision. After each epoch the validation crops are read as the backend reads them (`number_probs`, the most probable class) and scored by `cls_trainer.score`, the same five numbers as the number trainer's in `metrics.csv`; the best epoch on `select_on` is saved as `weights.safetensors` with `parseq_config.json` (the model's hyperparameters, the base and the epoch). `recipe.resolution` is the input height and must be 32.
- **Setups:** `bga.train.run.SETUPS` maps each trainable layer and backend to its dataset module and Modal trainer: `court/rfdetr_keypoint` to `court_data` and `train_keypoint`, `detect/rfdetr_torch` to `detect_data` and `train_detect`, `appearance/reid` to `appearance_data` and `train_appearance` (whose collect pulls `weights.safetensors`, `reid_config.json`, `metrics.csv` and `result.json`, `PULLS`), `numbers/crop_cls` to `number_data` and `train_numbers` (collect pulls `weights.safetensors`, `cls_config.json`, `metrics.csv` and `result.json`), `numbers/parseq` to `number_data` and `train_parseq` (collect pulls `weights.safetensors`, `parseq_config.json`, `metrics.csv` and `result.json`). Any other pair raises `NotImplementedError`.
- **Run** (`bga.train.run`):
  1. Upload the dataset to the Modal volume.
  2. Train in a detached Modal app. The call is spawned and its id saved to `call.json` in the working directory; the local process polls for the result in two-minute waits, which survive Modal's auth token expiring, and `--wait` resumes waiting after a restart. A run cut short continues from the `last.ckpt` its trainer left on the volume, unless `--no-resume` is given; only Modal's `NotFoundError` counts as no checkpoint, and any other volume error stops the launch. Court: `RFDETRKeypointPreview` with the schema read from the annotations, `KeypointTrainConfig`, and the `rfdetr` Lightning trainer. Detection: the `rfdetr` class named by `recipe.variant`, trained with its `train()`.
  3. Collect (re-collecting a weight id first deletes cached layer outputs made with its earlier checkpoint, and everything downstream of them; every collect also removes the weight id's copy from the volume, `drop_remote_weights`, so the next remote run uploads the collected files; `--partial` takes a cut-short run's best EMA checkpoint so far and records the last epoch done in `train.json`): the best checkpoint, `metrics.csv` and `result.json` go into `weights/<weight id>/`, with a `train.json` (config, dataset counts and sha256, commit, GPU, time, cost, the trainer's best and last validation rows). A row goes to `docs/train.md` and one to `docs/costs.md`.
- **Recipe fields for detection:** `variant` (the `rfdetr` class), `tiles`, `devices` (GPUs per run, spawn-based DDP; CPU and memory scale with them), `strategy`, `eval_interval`, `early_stopping_patience` (in validations) and `init_weights` (warm start from a trained weight id, whose checkpoint is uploaded next to the dataset). All are optional and left out of the config hash when unset, so older weight ids are unchanged.
- **Recipe fields for appearance:** `val_clips` (validation games), `crops_per_id` (K of the P x K batches), `triplet_margin`, `min_team_share` (identities with less of their boxes on one team are left out), `shot_batches` and `id_loss_weight`; `resolution` is not read (the encoder's preprocessing fixes the input).
- **Configs:** `detect-rfdetr_torch-nba-704.yaml`, `-nba-1280.yaml` and `-nba-tile2.yaml`: RF-DETR Large on Skalski's NBA set version 18 (CC BY 4.0) at 704 px, at 1280 px, and at 704 px on two square tiles, the resolution experiment of M4. `detect-rfdetr_torch-plk-r1.yaml`: the 1280 px setting plus `train_manifest_r1.json`; `-plk-r2.yaml`: the same plus `train_manifest_r2.json` (238 frames, validation clips the same as round 1's); `-plk-r2-xl.yaml`: `plk-r2` with RF-DETR XL, the M4 bake-off; `-plk-r2-fast.yaml` and `-plk-r2-fast-es.yaml`: `plk-r2` with batch 16 and validation every 5 epochs, the second also with early stopping (speed check). `-plk-r3.yaml` and `-plk-r3-poss.yaml`: `plk-r2-fast-es`'s recipe on all three rounds, as drafted and with the possession manifests (the possession experiment). `court-rfdetr_keypoint-plk-v1.yaml`: RF-DETR Keypoint (preview, XLarge, Apache 2.0) on Roboflow's NBA court set version 20 (CC BY 4.0) plus `train_manifest_plk-v1.json`.
- **Appearance configs:** `appearance-reid-plk-r1.yaml` fine-tunes `osnet_x1_0_sportsmot` on `crops_r1.json` without its dropped identities, validating on the King game (30 epochs, batches of 16 identities x 4 crops, Adam at 1.5e-4, triplet margin 0.3, selected on `val/map` every 3 epochs, L4); `appearance-reid-plk-r1-team85.yaml` is the same with `min_team_share` 0.85; `appearance-reid-plk-r1-shot.yaml` draws each batch from one shot with the identity loss off, for 15 epochs selected on `val/rank1` (it kept epoch 0, the starting encoder); `appearance-reid-plk-r1-shot-last.yaml` is the same keeping the last epoch (`select_on: epoch`).

## Data

### Clips and per-clip labels

- **Clips:** `scripts/fetch_highlights.sh` downloads the 39 clips in `data/clip_ids.txt` into `data/raw/`. They are mostly 1080p50 H.264; `scripts/transcode_h264.sh` re-encodes AV1 or VP9.
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
- **Detection manifest, round 1:** `data/labels/detect/train_manifest_r1.json`, written by `autolabel.py manifest`: the 231 finished frames of the ball queue, each with `clip`, `video`, `frame`, `split` (3 whole clips are `valid`), `size` and `boxes` as `[x1, y1, x2, y2, class]` (Tomasz's ball, if any, plus the draft's other boxes), and the sha256 of the ball queue it was built from.
- **Possession queue:** `data/labels/box_queue_possession.json` (`autolabel.py possqueue --rounds ...`): every finished frame with a ball from the given rounds' manifests, with its people boxes and its ball; adding rounds leaves earlier decisions untouched and refreshes the instructions. 643 frames: 199 from round 1, 218 from round 2, 226 from round 3.
  - **Each frame** adds to the box-queue fields `holder` (the id of the box holding or dribbling the ball, or null) and `dead_ball` (the ball is out of live play: a throw-in from behind a line, the referee has it). The holder box has class `player-in-possession`, or keeps a shot class; an inbounder is a holder too.
  - **Review:** `scripts/possession_ops.py migrate` gave the frames answered before `holder` existed their holder from the class; `apply` runs an ops file once (holder, nobody, dead, live, and review, which sends a frame back with a note) and records its sha256 in `meta.ops_applied`; each frame it touches gets a new `saved_at`, so an open page cannot save over it: `possession.ops.txt` (the implementer's pass, 31 frames back to Tomasz), then `possession.r2.ops.txt` (after his review) and `possession.r3.ops.txt` (his dribbler rule). 624 done, 19 skipped.
- **Detection manifest, round 3:** `data/labels/detect/train_manifest_r3.json`: 244 finished frames of the round-3 ball queue (32 of them validation frames, from the same clips as round 1's).
- **Detection manifests with possession:** `data/labels/detect/train_manifest_r<N>_poss.json` (`autolabel.py possmanifest --rounds 1 2 3`): each round's manifest with the possession queue applied to its done frames. The queue's people boxes replace the manifest's, with boxes the reviewer moved and holders drawn by hand: the holder is `player-in-possession` or keeps a shot class, every other player is `player`, and a second box on the holder (IoU 0.8 or more, the draft's other class for the same person; 18 boxes) is dropped. Other frames and other classes are the round's. `player-in-possession` boxes: 136, 143 and 157 in rounds 1, 2 and 3.
- **Appearance crop manifest, round 1:** `data/labels/appearance/crops_r1.json` (`reid_crops.py manifest` over `crops-reid-r1`): 1,090 identities, 25,458 crops from 108 shots of four non-gold 2025/26 games, each identity with `clip`, `video`, `shot`, majority `team` and `team_share`, `boxes`, `span` and up to 24 crops as `[frame, x1, y1, x2, y2]`. A seeded sample of 96 was checked by eye on filmstrips (`review` in the file): 18 marked `dropped`, 13 of them tracker swaps and 5 referees, coaches or bench close-ups.
- **Ball queue, round 3:** `data/labels/box_queue_r3balls.json`: 262 frames, prefilled by `plk-r2`, 99 balls accepted without review (threshold 0.85, calibrated for `plk-r2`), 163 frames for review.
- **Ball queue, round 2:** `data/labels/box_queue_r2balls.json`: 261 frames from 22 clips, 115 balls accepted without review (status `done`, source `auto`), 146 frames for review.
- **Ball queue, round 1:** `data/labels/box_queue_r1balls.json`, in the box-queue format with `mode: ball`, `title` and `instructions`; each frame holds the draft's ball boxes and its three best weaker ball guesses as candidates.
- **Kits:** every clip label file marks one side's kit `light` and the other `dark` in its meta (`data/labels/review/kits.ops.txt` for the 35 non-gold clips, read from sheets on 2026-11-23); the home side wears the darker kit in five of the 39 clips. Team outputs cached for a non-gold clip before that date were made without kits.
- **Number training crops, round 1:** `data/labels/numbers/train_r1.json` (`number_crops.py manifest` over `crops-reid-r1`, four non-gold 2025/26 games): 955 tracklets with 18,670 `plk-r3-poss` number boxes scoring at least 0.3, each inside exactly one non-referee linked row (3,876 boxes inside two people dropped), plus 162 `strays` inside no person. Each tracklet has `clip`, `video`, `sequence`, majority `team` and `team_share`, `person` (frame and box), `crops` as `[frame, x1, y1, x2, y2, score]`, `label` and `note`; `sample` lists the 300 tracklets read on sheets (seed 0). Labels (`data/labels/review/number_train_r1.ops.txt`): 208 numbers (5,233 crops, 28 numbers), 39 illegible, 36 mixed, 17 not numbers. The team sides in these clips are named the wrong way round (their meta has no kits), so `check` tests a read against both rosters.
- **Gold number crops:** `data/gold/v1/numbers.json` (`number_gold.py export`): 376 crops on the box gold v1 frames, 292 dev and 84 held-out, each with `id`, `source` (`box_gold_v1` for Tomasz's 287 `number` boxes, or the detector and weight id for the 89 `plk-r3-poss` number boxes at 0.3 or more that overlap no gold box), `sequence`, `clip`, `frame`, `held_out`, `xyxy`, `team` and `label`: the digits, `illegible` or `not_number`. 282 carry digits, 48 are illegible and 46 not numbers. Each crop also has `review`: the reviewer's verdict from the `/numbers` page (`ok` or `fix`), or null. The working file `data/labels/numbers/gold_crops.json` adds the holders' gold track hints, notes, the reviewer's answers and the spot-check sample; `data/labels/review/number_gold.ops.txt` holds every label. Details in `docs/gold.md`.
- **Box gold v1 queue:** `data/labels/box_gold_v1.json` is written by `scripts/box_gold_v1.py prefill` and edited at `/boxes`.
  - **Top level:** `version`, `classes`, the SAM model, `frames` and `meta.review_seconds`.
  - **Each frame:** `youtube_id`, `clip`, `video`, `shot`, `frame`, `status` (`todo`, `done` or `skip`), `note`, `boxes` and `candidates`.
  - **Each box:** `id`, `xyxy`, `cls`, `team`, `tid`, `src`, `flags`, `hints`, `alt` (SAM or prefill geometry), `checked` and `edited`.

  `data/labels/review/box_gold_v1.ops.txt` holds the implementer's first pass, applied by `box_gold_v1.py apply`. `data/labels/review/box_gold_v1.qa.ops.txt` holds the QA fixes after Tomasz's review, applied with `apply --include-done`. `box_gold_v1.py export` writes `data/gold/v1/boxes.json` (100 frames, 1,669 boxes, each with `xyxy`, `cls`, `team`, `tid`) from the frames marked done. Conventions and counts are in `docs/gold.md`.

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
- **Training manifest:** `data/labels/court/train_manifest_<name>.json`, written by `scripts/court_train_manifest.py` from the usable training keyframes.
  - **Frames:** each keyframe is propagated every 5th frame, up to 1.5 s each way. A frame is kept only while its line support stays at or above 85% of the keyframe's; where the keyframe's own support is under 0.4, only frames within 0.1 s are kept. One keyframe each from King, Legia and WKS Śląsk is held out as validation.
  - **Fields:** per frame, the clip, video, frame, keyframe, split, size, H, line support and the 33 projected landmarks (visibility 2 inside the frame, 0 outside); plus per-keyframe summaries and the sha256 of every source label file.
  - **`plk-v1`:** 77 frames, 71 train and 6 valid.
- **Geometry:** `tools/labeler/static/fiba_court.json` is written by `scripts/court_geometry.py`. It has 33 vertices, labelled `01` to `41` in `sports`' numbering, plus edges and the arcs as polylines, in cm on a 2800 × 1500 court with vertex `01` at (0, 0).

### Synthetic fixtures

`data/fixtures/synthetic/*.json` holds 8 simulated possessions, in schema `bga.tracking2d/v0` (format in `docs/fixtures.md`). `scripts/synth_plays.py` writes them deterministically, and `tests/test_synthetic_plays.py` fails if they drift. They are input to the coach page only, never gold or training data.

## Labelling tools

### Labeller server

`tools/labeler/server.py` is a stdlib HTTP server on port 8765, serving four pages:

- **`/` (track labeller):** `static/index.html` and `app.js`. Shot tagging, boxes, identities, and track-level review: K confirm, P split, M merge, ⇧Del delete a track, ⇧D accept a shot. A filmstrip shows 8 crops per track, and active time is logged in `meta.review_seconds`.
- **`/court` (court keyframes):** `static/court.html`.
  - **Dots:** drag a landmark dot onto its true spot.
  - **Pins:** shift-click any painted line or circle. A pin attaches to the nearest of 20 painted elements (sidelines, centre line and circle, and at each end the baseline, key sides, free-throw line and circle, three-point straights and arc); Tab or a list reassigns it.
  - **Fit:** Levenberg-Marquardt on the dots' reprojection error and the pins' distance to their line. It starts from the current court, the anchor and a DLT, and restarts from shifted courts when the fit stays poor.
  - **Uncertainty:** the top bar shows the fit's rms and how far the lines in view could move given 2 px click error. It warns above 15 px.
  - **Folds:** a fit is refused if it folds between the dots and pins or mirrors the court there; parts of the court may be behind the camera.
  - **Queues:** `?set=<name>` opens `queue_<name>.json`.
- **`/boxes` (box gold v1 and other box queues):** `static/boxes.html` edits `data/labels/box_gold_v1.json`, one frame at a time; `?set=<name>` opens `data/labels/box_queue_<name>.json` instead. A frame's status is `todo`, `done` or `skip` (the reviewer could not decide; left out of every export), checked against `BoxStatus` in `server.py` when a frame is saved.
  - **Possession mode** (`mode: possession`, the queue `possession`): people boxes and the frame's ball, which is shown but cannot be chosen. `Tab` steps through the players, the model's holder first, then nearest the ball. `Space` or a click makes that player the holder (`player-in-possession`; a shooter keeps the shot class) and demotes any other holder to `player`. `N` records that nobody holds the ball, `S` skips the frame, and `Shift`-drag then `Enter` adds a holder without a box. Each answer also saves the frame's `holder`. `D` toggles `dead_ball`; the top bar shows the saved holder, the dead-ball flag and the frame's note, and the list marks the holder, which comes first in the Tab order.
  - **Ball mode** (a queue with `mode: ball`, such as `r1balls`): only ball boxes, the queue's own instructions, and one decision per frame. `Space` or a click on a guess makes it the frame's ball and moves on; `Shift`-drag then `Enter` keeps only the boxes drawn or adjusted; `N` records no visible ball and `S` skips the frame.
  - **Display:** boxes are coloured by class. Flagged boxes not yet checked are outlined in orange and come first in the Tab order and the list, with their flags and hints shown.
  - **Loupe:** shows the selected box at up to 8×, without smoothing, with the alternative box (SAM or prefill) dashed.
  - **Editing:** a click selects the smallest box under the cursor; `Esc` or a click on empty floor deselects. Edges and corners drag on the frame or in the loupe; arrow keys nudge one edge by 1 or 5 px; Alt-drag moves the selected box. Keys `1`–`0` set the selected box's class, `T` the team, `A` swaps in the alternative box, Shift-drag draws a new box anywhere, `X` deletes, and `C` shows the detector candidates (low-score objects, and people no box covers), adopted with a click. Scroll zooms, and dragging empty space pans.
  - **Saving:** each change autosaves the frame. Enter marks it done, checks all its boxes and opens the next frame not done.
  - **Time:** active time is logged in `meta.review_seconds`.
- **`/numbers` (gold number crops):** `static/numbers.html` shows the crops of `data/labels/numbers/gold_crops.json` one at a time, for the reviewer to check the implementer's label. `?set=spotcheck` (the default) opens the sample `number_gold.py spotcheck` stored, `illegible` the crops the implementer labelled illegible, `all` every crop, and `reviewed` the crops already answered.
  - **Views:** a close-up (the box plus 40% on each side) and the box in context (5×, `W` for 13×), drawn from the frame in the browser. `S` switches between smooth and sharp pixels, and `B` raises the contrast. A strip of thumbnails below shows each crop's label and answer, and a click opens it.
  - **Answers:** `Enter` or `Space` marks the label right. Digits then `Enter` give another number, `I` marks the crop illegible and `X` not a number. `U` clears the answer. Each answer saves at once and opens the next open crop.
  - **Storage:** the answer goes to the crop's `review` (`verdict` `ok`, or `fix` with the reviewer's `label`); the implementer's `label` is kept. Active time is logged in `meta.review_seconds`.

API endpoints:

| Method | Path |
|---|---|
| GET | `/api/clips`, `/api/clip`, `/api/frame`, `/api/shots`, `/api/pipeline`, `/api/court/queue`, `/api/court/labels`, `/api/boxes/queue[?set=]`, `/api/numbers/queue[?set=]` |
| POST | `/api/labels`, `/api/shots`, `/api/court/labels`, `/api/numbers/review` (rejects an unknown crop or verdict, a label that is not one or two digits, `illegible` or `not_number`, and an answer whose `prev_at` differs from the crop's stored review stamp), `/api/boxes/frame[?set=]` (rejects classes outside the schema, empty boxes, a record for another frame, a `holder` that is not one of the frame's boxes, a non-boolean `dead_ball`, a record lacking a field the stored frame has, and a record whose `saved_at` differs from the stored frame's, as a page that loaded the frame before a later change would send; the reply carries the new `saved_at`, which the page keeps) |

Paths are confined to the repository by `safe_path`.

### Scripts by job

| Job | Scripts |
|---|---|
| Clips and shots | `fetch_highlights.sh`, `transcode_h264.sh`, `fetch_rosters.py`, `compare_shots.py`, `shot_diff_sheet.py`, `shot_sheets.py`, `check_labeller.py`, `kit_sheets.py` (kits in the clip meta: sheets of four frames per clip without kits, the round-1 number reads per kit cluster against both rosters, and `apply` writing `data/labels/review/kits.ops.txt` into the label files) |
| Track prefill | `prefill_tracks.py` (Skalski's detector plus ByteTrack), `link_prefill.py`, `extract_shot_frames.py`, `sam_to_tracks.py` (SAM 2.1 identities fused with detector boxes), `import_prefill.py` |
| Tuning | `sweep_track.py` (grid sweep of tracker params on gold dev in one process, CSV in `runs/_sweeps/`; refuses the held-out game; `--remote-gpu` first runs every missing setting and shot on Modal in one parallel batch, through `modal_runner.run_layer` with one entry per sequence) |
| Review | `track_sheets.py`, `review_sheets.py` (with `--propose` kit-colour ops), `team_gold_sheets.py` (team audit of the gold tracks), `review_ops.py`, `possession_ops.py` (possession queue: explicit holder, review ops) |
| Gold export | `export_gold.py`, `box_gold.py`, `box_gold_v1.py` (box gold v1: prefill with SAM merge, crop sheets, ops, candidate refresh, export), `number_crops.py` (number training crops: manifest of number boxes on linked tracklets of non-gold games, seeded sample, blind sheets of one tracklet per row, ops, roster check), `number_gold.py` (gold number crops: prefill from box gold v1, detector negatives, blind reading sheets, ops, compare against the gold tracks, a seeded spot-check sample stored for the `/numbers` page with sheets, export taking the reviewer's label where it differs) |
| Court | `court_geometry.py`, `court_prefill.py` (the NBA court model over every clip), `court_keyframes.py`, `court_propagate.py`, `qa_court_overlay.py`, `court_train_manifest.py` (training frames from the training keyframes) |
| Auto-labelling (M4) | `autolabel.py --round N`; each step gets the round's files from a `Round` record. Round 2 on: frames at least 1 s from earlier rounds' frames, prefill by the previous round's PLK detector (`plk-r1`, then `plk-r2`) at confidence 0.1, draft at score 0.5, and a frame's ball accepted without review when one ball scores at least the round's threshold (0.7 for `plk-r1`, 0.85 for `plk-r2`, each calibrated on gold v1 dev) and no other 0.2. Round 1: central shots of the non-gold clips by the PLK court model's confident landmark count; 275 frames sampled, 12 per clip at most and 1 s apart; prefill by Roboflow's detector versions 16 (all but the ball) and 6 (the ball) at confidence 0.1; draft at score 0.3 with per-class NMS, people off the court and numbers outside every person dropped; the ball queue `data/labels/box_queue_r1balls.json` (264 frames, close-ups with fewer than 4 people left out) |
| Spend | `modal_spend.py` (Modal's own billing: the month and each app) |
| Appearance crops (M6) | `reid_crops.py manifest` (one identity per linked track of a crops config: non-referee detected rows scoring at least 0.5 and at least 64 px tall, 10 or more per identity, 24 spread over its life; boxes only, written to `data/labels/appearance/`), `sheets` (filmstrips of a seeded sample for the purity check by eye) and `drop` (marks identities seen to be mixed) |
| Probes (not metrics rows) | `probe_train_speed.py` (seconds per epoch of a detection fine-tune on Modal), `probe_reid.py` (all six Roboflow `reid` encoders on gold v0, with a first-half against second-half question; `bga eval-appearance` scores one config's encoder), `probe_court_keyframes.py` (the NBA court model and propagation against the court keyframes), `probe_parseq.py` (PARSeq's released scene-text weights read zero-shot on the gold dev number crops, the check that preceded the `numbers/parseq` backend; it reads the checkpoint from a file) |

Contact sheets for reading by eye come from `bga.review.sheets`.

## Demo site

The site is published to the Vercel project `basketball-analyzer-demo` by `scripts/deploy_demo.sh`. It is served at https://basketball-analyzer.demo.tko.ventures, and `basketball-analyzer-demo.vercel.app` redirects there. The Vercel project has no Git connection. The script deploys only from a clean `main` that matches `origin/main`, and tags each deployment with the commit (`BGA_DEPLOY_ANY_BRANCH=1` overrides the check). The site is built outside the `bga` pipeline:

- **Analysis page:** `scripts/demo_web.py` builds `/` from three shots of `OSp445e0mXU`: the broadcast video with labels burnt in, beside a 2D court the page draws from per-frame positions (coverage, spacing, trails, player selection). It runs Skalski's detector, team and number models through the helpers in `scripts/demo_video.py`. The court mapping comes from the court layer of a run config, `--court-config`, by default `configs/court-plk-v1-smooth4.yaml` (the PLK court model with smoothing), run on the rendered frames of each shot as one sequence. Number reads and homographies are cached in `runs/demo/cache`, and the homographies are redone when the court config changes.
  - Template: `demo_page_analysis.html`.
  - Text: `demo_strings.py`. The site is English only; `deploy_demo.sh` redirects the old `/analysis/`, `/analiza/` and `/en/` paths to `/`, and refuses to publish a build whose `index.html` is not English.
  - Check: `demo_audit.py` compares the teams and numbers shown with gold v0.
- **Coach page:** `scripts/coach_web.py` builds `/coach/` from the synthetic fixtures, using `coach_insights.py` (plays, coverage, shot quality, key moments), `coach_strings.py` and `coach_page.html`. `deploy_demo.sh` rebuilds it on every deploy.
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
  | `court` | `rfdetr==1.12.1`, for running trained court and detection weights on the Mac |
  | `plus` | `rfdetr_plus==1.3.0`: RF-DETR XL and 2XL, under Roboflow's Platform Model Licence 1.0 (treated as Apache 2.0 by Tomasz's ruling, plan question 5) |
  | (Modal image only) | `rfdetr[train]==1.12.1`, `rfdetr_plus==1.3.0`, torch 2.15 |
  | `eval` | TrackEval, pycocotools |
  | `demo` | `sports`, transformers, umap-learn, scikit-learn, sentencepiece |
  | `track` | `trackers==2.7.0` (Roboflow: BoT-SORT, ByteTrack, OC-SORT, McByte, camera-motion compensation), `reid==0.1.0` (appearance encoders) |
  | `parseq` | `strhub` from `baudm/parseq` at commit `1902db04` (PARSeq), with `pytorch-lightning` and `nltk`, which it imports without declaring |

  `uv sync --all-extras` is needed for the full test suite.
- **OpenCV:** `opencv-python` and `opencv-contrib-python` are overridden away, leaving one headless cv2.
- **Overrides for `trackers` 2.7:** `inference` pins `trackers<2.5` and `supervision<0.31`; `pyproject.toml` overrides them to `trackers==2.7.0` and `supervision>=0.31,<0.32` (0.31.7 locked). Only `inference.get_model` is called. With the upgraded set, `baseline` reproduces its 2026-11-02 dev row in every column but team accuracy (0.990 against 1.000, the unseeded team fit), and the `track-bytetrack-tuned` track outputs are identical.
- **Lint and format:** `.pre-commit-config.yaml` pins the hooks: pre-commit-hooks file hygiene, ruff check and format (ruff 0.17.5), docformatter and codespell (report only). `pyproject.toml` `[tool.ruff]` follows Borda's vision repos: 120-character lines, Google docstrings (`D`), annotations (`ANN`), naming (`N`), bandit (`S`) and complexity limits (McCabe 12, branches 12, statements 50, returns 6, arguments 12). `scripts/` is exempt from `ANN`, `D`, `N` and `S`, and `tests/` from `ANN`, `D`, `N` and `S101`. The dev group has pre-commit, pytest and ruff.
- **Checks:** `uv run pre-commit run --all-files` and `pytest`, which runs the doctests in `src/` and `tools/` (`--doctest-modules`) as well as the tests in `tests/`, with tests on config hashing, shot cuts, gold export, MOT scoring, churn, NMS, review ops, rosters, the labeller server, court orientation, the synthetic fixtures, gold leakage in training configs and manifests, court smoothing, square tiles and their merge (`test_tiles.py`), per-class detection AP (`test_eval_detect.py`), the appearance layer: array storage, the encoder's weights check and the gold-box scores (`test_appearance.py`), the link layer: splits, join gates and the greedy pass (`test_link.py`), the frame reader on a synthetic clip (`test_stage.py`), whole-file output writes and the per-sequence `meta.json` (`test_pipeline.py`), the team backend's kit check and single SigLIP load (`test_team.py`), and the files a remote run uploads for a weight id and the removal of the volume's copy (`test_train_run.py`).
