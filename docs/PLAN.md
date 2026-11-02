# Basketball game analyzer: implementation plan

**Status:** handoff draft, 2026-10-27. Revised 2026-10-28 (Liquid, basal, backends) and 2026-10-29 (remote CUDA, training setups, SAM 3.x withdrawn after Meta declined access, Tomasz's answers in section 12).
**Written by:** Claude Fable 5.1 with Tomasz Kopacki (tko).
**Written for:** Claude Opus 5.5 or any engineer picking this up with no prior conversation context.

Everything marked **verified** was checked against the live source on 2026-10-27. Numbers marked **provisional** are starting targets to be replaced by measured baselines. Section 12 lists questions only Tomasz can answer; ask them before the milestone that needs them, not before starting.

---

## 1. Goal

Offline analysis of ORLEN Basket Liga (PLK, Polish men's top league) games involving Dziki Warszawa, from the broadcast main camera. For every central-camera shot the system must produce:

1. Every player and referee detected, with a mask or box.
2. Each player assigned to a team by kit.
3. Each player tracked across frames without identity confusion.
4. Each player identified by jersey number, mapped to a name via the roster.
5. Every player's position and the ball's position projected onto a flat 2D FIBA court in metres.

Longer term, the 2D data feeds tactical reports about opponents. That is out of scope for this plan.

## 2. Fixed facts and decisions

| Topic | Decision | Source |
|---|---|---|
| Footage | Broadcast main ("central") camera. Currently 32 YouTube highlight clips of Dziki games, 1920x1080 at 50 fps, about 3 min each, 25 to 45 shots per clip with replays and close-ups mixed in. Full games via Hudl Instat or Synergy may come later. | prior repo, stage 1 summary |
| Camera consequence | The camera pans, zooms and cuts. Court mapping needs a per-frame keypoint model and shot filtering, not a one-time calibration. | this plan |
| Processing mode | Offline, after the game. Two-pass and global optimisation are allowed. No live requirement. | Tomasz |
| Hardware | Apple M1 Max, 64 GB, no NVIDIA GPU. Inference runs locally where a backend runs on the Mac, one heavy model in memory at a time. Training and CUDA-only backends run on a rented GPU through `bga ... --remote` (section 5, "Where things run": Modal by default, including RF-DETR training with the `rfdetr` package; RunPod for interactive sessions; Roboflow Train only if a workspace with credits exists; Colab only for the reference notebook). | prior repo; Tomasz 2026-10-29 |
| Model policy | Open-weight models fine-tuned into PLK-specialised detectors. Closed VLM APIs only for labelling and offline QA, never as the per-frame production model. | Tomasz |
| Reference stack | Follow Piotr Skalski's Roboflow basketball pipeline (section 3). Deviate only where his own posts list a limitation or where PLK differs from the NBA. | Tomasz |
| Court geometry | FIBA court, 28 x 15 m. The `sports` package already ships a FIBA preset with the same 33-landmark vertex order as the NBA one. **Verified.** | roboflow/sports `feat/basketball` branch, `sports/basketball/config.py` |
| Identity | `player_id = (team, jersey_number)`. Numbers are unique within a team only. FIBA allows 0 to 99. | prior repo |
| Ground truth first | No heuristic is added to any layer until the gold set in section 8 exists and the change is measured against it. | review of prior repo |
| Data provenance | Start from scratch. Clips are re-downloaded with the fetch script, weights re-downloaded from their sources, shot files regenerated. Nothing under the prior checkout's `data/`, `weights/` or `runs/` is copied; the old shot files serve only as a cross-check. | Tomasz 2026-10-29 |
| Labelling budget | Tomasz corrects the gold labels himself in about 90 minutes total, measured by a timed dry run on one shot. The implementer does every first pass itself. | Tomasz 2026-10-29 |
| Labelling VLM | No third-party VLM API keys. The implementer's own vision through Claude Code (Tomasz's subscription) reads contact sheets of crops and review frames. Labelling and QA only, never per-frame production. | Tomasz 2026-10-29 |
| Licences | RF-DETR Nano to Large, RF-DETR Seg, RF-DETR Keypoint: Apache 2.0. RF-DETR XL and 2XL: Roboflow Platform Model Licence 1.0, terms to be checked. YOLO26 and YOLO11: AGPL-3.0. D-FINE and DEIM: Apache 2.0. DEIMv2: non-commercial only. SAM 3 and SAM 3.1: SAM License, gated; Meta declined Tomasz's access request, so neither is used, and third-party re-uploads of gated weights are off limits. SAM 2.1 and EdgeTAM: Apache 2.0, open downloads. **Verified** 2026-10-29. | section 13 |

## 3. Reference stack: Skalski's basketball pipeline (verified 2026-10-27)

Blog post: https://roboflow.com/blog/identify-basketball-players (2025-09-30).
Notebook: https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb
Video: https://www.youtube.com/watch?v=yGQb9KkvQ1Q
Court keypoint fine-tuning post: https://roboflow.com/blog/train-rf-detr-keypoint (2026-06-26).

What the notebook does, in order:

| Stage | His implementation | Reuse for PLK |
|---|---|---|
| Detection | RF-DETR-S fine-tuned on Roboflow Universe dataset `roboflow-jvuqo/basketball-player-detection-3-ycjdo`, 10 classes: `ball`, `ball-in-basket`, `number`, `player`, `player-in-possession`, `player-jump-shot`, `player-layup-dunk`, `player-shot-block`, `referee`, `rim`. | Keep the class schema. Retrain on PLK frames. |
| Tracking | SAM2 real-time (`Gy920/segment-anything-2-real-time`) prompted with RF-DETR boxes on frame one; masks cleaned with `sv.filter_segments_by_distance`. Needs CUDA. Works on short clips only. | Run as published on Modal as the `sam2_realtime` candidate (layer 4). ByteTrack in court space stays the local default. |
| Teams | `sports.TeamClassifier`: crop the central 40% of each player box (`sv.scale_boxes(factor=0.4)`), SigLIP embeddings, UMAP to 3D, K-means into 2. Fitted once per video on crops sampled at 1 fps. Team assigned per track once. | Reuse unchanged, fit per clip. |
| Numbers | RF-DETR `number` boxes converted to masks and matched to player masks with `sv.mask_iou_batch(overlap_metric=sv.OverlapMetric.IOS)` at IoS >= 0.9. Crops padded 10 px, read by SmolVLM2 (fine-tuned on ~3.6k NBA crops, 86%) or a ResNet classifier (93%). Read every 5th frame. `ConsecutiveValueTracker(n_consecutive=3)` confirms. `TEAM_ROSTERS[team][number]` gives the name. | Reuse the matching rule and validator. Train the classifier on PLK crops. Add the on-floor constraint (layer 5). |
| Court | Keypoint model `basketball-court-detection-2/14` from Universe dataset `roboflow-jvuqo/basketball-court-detection-2`, 33 landmarks. Keep landmarks with confidence > 0.5, need >= 4. `sports.ViewTransformer(source=frame_landmarks, target=config.vertices[mask])` with `CourtConfiguration(league=League.NBA, measurement_unit=MeasurementUnit.FEET)`. | Use `League.FIBA`, `MeasurementUnit.CENTIMETERS`. Fine-tune the keypoint model on PLK arena frames with the same 33-landmark schema. |
| 2D positions | `detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)` through `transform_points`. `sports.clean_paths(jump_sigma=3.5, min_jump_dist=0.6, max_jump_run=18, pad_around_runs=2, smooth_window=9, smooth_poly=2)`. `draw_court`, `draw_points_on_court`, `draw_paths_on_court`. | Reuse. Upgrade ground point to ankles in layer 6. |
| Shots | `player-jump-shot` class plus `ShotEventTracker(reset_time_frames=1.7 s, minimum_frames_between_starts=0.5 s, cooldown_frames_after_made=0.5 s)`; `draw_made_and_miss_on_court`. | Layer 7, last. |

Environment the notebook pins: `supervision==0.27.0`, `inference-gpu`, `git+https://github.com/roboflow/sports.git@feat/basketball`, `transformers`, `num2words`. The keypoint post uses `rfdetr>=1.8.1` and `supervision>=0.29.1`. Pick one `supervision` version per environment and record it.

His stated limitations, which this plan treats as the work list: 1 to 2 FPS on a T4 with SAM2; SAM2 degrades with many objects and in high-action moments; tracking needs re-prompting on longer clips; numbers fail on distant players; number predictions should be constrained to who is on the court.

## 4. Assets to import from the prior repository

Private repo `tko-ventures/basketball-tracking`, branch `feat/labeller-rosters`, reviewed 2026-10-27. It is a one-week prototype (2026-10-01 to 10-07) tested on a single 10 s clip, with a strong stage-1 data layer and a pipeline whose identity heuristics were never measured against ground truth. All 32 label files have shots detected but zero boxes labelled.

Import as-is, then adapt:

| Path in prior repo | What it is | Change needed |
|---|---|---|
| `tools/labeler/` (server.py, prepare_clips.py, sam_service.py, static/) | Browser ground-truth labeller: shot tagging, persistent identities, roster autocomplete, copy-forward and interpolation, prefill from any `tracks.json`, SAM 3.1 Core ML assist with identity linking, MOTChallenge export plus `identities.csv`. | Drop the SAM 3.1 Core ML assist and `sam_service.py` (no licence access, section 6); prefill comes from `tracks.json`. Add tracklet-level review actions (confirm, split, merge, assign team and number once per track) so the human pass fits the 90-minute budget. Generalise team labels from `dziki/opp/ref/other` to `home/away/ref/other` with a per-clip mapping to team names. Add export of number crops from boxes flagged `num_visible`. Add a court-landmark mode later or use Roboflow Annotate for landmarks. |
| `scripts/fetch_highlights.sh` | yt-dlp fetch of Dziki highlights with a manifest. | None. Check footage rights (section 12). |
| `data/rosters/dziki_warszawa.json` | Rosters 2024/25 to 2026/27 from plk.pl with `team_id_plk`. | Extend to every PLK team, same schema. |
| `data/labels/*.json` | Shot boundaries for all 32 clips, season and opponent filled from titles. | Keep. |
| `tools/labeler/server.py::ShotJobs` | Shot-cut detector: HSV histogram Bhattacharyya distance plus mean pixel change against a 25-frame rolling median baseline, min shot length 0.4 s. | Lift into layer 0. |
| `src/basketball_tracking/export/churn.py` | Tracker-health metrics: unique ids, id-set change rate, short-lived id fraction, survival. | Lift into the evaluation module. |
| `src/basketball_tracking/detect/onnx_rfdetr.py` | ONNX Runtime adapter with CoreML provider and nested-box NMS. | Pattern for Mac inference of RF-DETR exports. |
| Identity rules (docs/architecture.md §2.2) | Unique within team; conflicting reads mark `conflict` and never overwrite; a lost track keeps its identity for a rebind window (75 frames at 50 fps). | Adopt as rules in layer 4 linking. Do not port the code. |

Do not port: the OCR gate stack (confirm-reject gate, confusion-pair margin, spaced-K, crop diversity, sharpness floor, crop preprocessing, Apple Vision x EasyOCR dual-confirm), the HSV kit-colour team cue, the fixed court trapezoid filter, the NBA jersey ONNX head (42 classes, cannot emit 38), and the homography no-go decision.

## 5. Architecture

Pipeline is a sequence of stages. Every stage writes its output to disk (parquet or JSON under `runs/<clip>/<stage>/`) so any stage can be re-run alone. Heavy models are loaded one at a time on the Mac.

```
video -> L0 ingest + shots -> L1 court homography -> L2 detect + pose (+ seg)
      -> L3 team -> L4 track + link -> L5 number + identity -> L6 2D positions
      -> L7 ball + possession -> export (parquet, radar video) -> L8 eval
```

### Backends and configuration

Every layer is a backend behind a small interface, so approaches can be swapped and compared per layer without touching the pipeline.

- One interface per layer in `src/bga/<layer>/base.py`: a `Protocol` with the layer's input and output types from this section, plus `load()` and `unload()` so one heavy model is resident at a time.
- Implementations register by name in `src/bga/registry.py`. Adding an approach means adding a backend module and a config entry, never an `if` inside the pipeline.
- A run config `configs/<name>.yaml` (validated with pydantic) names one backend per layer with its parameters and weight paths. The config hash is part of the run directory: `runs/<clip>/<layer>/<backend>@<hash8>/`. Downstream layers read upstream outputs by that key, so swapping the number reader does not rerun detection.
- `bga run --config a.yaml` runs a pipeline. `bga eval --config a.yaml --config b.yaml` prints metrics side by side and appends one row per config to `docs/metrics.md`. A metrics row cites the config name and hash. Configs that produced a row are committed.
- Backends that need CUDA declare `requires: cuda`. Locally they are skipped with a warning, not a crash. `bga run --remote modal` (or `--remote runpod`) executes those stages on a rented GPU and syncs their `runs/` output back, so a CUDA backend is evaluated exactly like a local one (see "Where things run").

Initial backends. Defaults follow Skalski's stack; candidates are compared only through the eval harness.

| Layer | Default backend | Candidates to compare |
|---|---|---|
| L0 shots | `hist_diff` (prior repo detector) | `pyscenedetect` |
| L1 court | `rfdetr_keypoint` | none planned |
| L2 detect | `rfdetr_onnx` (CoreML provider) | `rfdetr_torch` (MPS), `yolo26`, `deim`, `d1_boxes` (labelling cross-check only, see layer 2) |
| L2 segment | `none` | `rfdetr_seg`, `sam21_boxmask` (SAM 2.1 masks from detector boxes, `requires: cuda`; mask labelling and mask source) |
| L3 team | `siglip_kmeans` (`sports.TeamClassifier`) | `d1_choice` (kit descriptions as options) |
| L4 track | `bytetrack_court` | `sam2_realtime` (Skalski's path, `requires: cuda`), `edgetam` (SAM 2 variant with Core ML export, Mac), Roboflow `trackers` |
| L5 numbers | `resnet_cls` | `d1_choice`, `lfm3vl` (zero-shot and SFT), `smolvlm2`, `basal_choice` (`requires: cuda`, low priority) |
| L6 ground point | `ankles_pose` | `bbox_bottom` |

### Where things run

The Mac is the default. Anything that needs CUDA runs on a rented GPU and writes to the same `runs/` layout. Decided with Tomasz on 2026-10-29. List prices below were checked the same day; verify before budgeting.

| Need | Platform | Why | Price |
|---|---|---|---|
| Scripted CUDA stages and fine-tunes: SAM 2.1, RF-DETR training with the open-source `rfdetr` package, LFM3-VL SFT, bake-off trainers | **Modal** (default for `--remote`) | Serverless, per-second billing, GPU and image declared in Python, volumes for clips and weights, scales to zero, nothing to manage. The natural fit for an agent-driven workflow. | H100 about 3.95 USD/h, A100 80 GB about 2.50, L40S about 1.95, L4 about 0.80; Starter plan has 30 USD free credit per month; volumes 0.09 USD/GiB/month; region pinning 1.15x to 1.75x. |
| Interactive debugging, long notebooks, cheapest raw hours | **RunPod** pod with a network volume | SSH and Jupyter on a persistent box; Community Cloud is the cheapest per hour. | H100 PCIe 1.99 to 2.89 USD/h, A100 80 GB 1.19 to 1.79, L40S 0.79 to 1.09, RTX 4090 0.34 to 0.89 (Community vs Secure); network storage 0.07 USD/GB/month. |
| RF-DETR fine-tunes, optional | **Roboflow Train** | Only if a workspace with training credits exists (section 12). Otherwise RF-DETR trains on Modal with the `rfdetr` package, which is the default. | Plan credits. |
| Skalski's reference notebook, unchanged | **Colab Pro** (or a Lightning AI Studio) | Only to reproduce his cells as published. | About 10 USD/month for 100 compute units; A100 availability not guaranteed. |

Rules: clips go up once to a volume and stay there; `runs/` outputs and weights come back through `bga` sync, never by hand; every remote stage writes GPU type, wall time and cost into its `meta.json`; the monthly spend cap is proposed at 50 USD until Tomasz confirms (section 12). Credentials: the Modal token lives in `~/.modal.toml` on Tomasz's Mac (profile `default`, CLI installed) and is picked up automatically there; on any other machine set `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` from a dedicated token created in the Modal dashboard; never copy a token into the repo. Keys that remote jobs need go into Modal secrets. Sizing: the gold shots total under 10 minutes of video, so measure the SAM2 path on an L40S first; cost is small either way.

### Training setups and their evaluation

A fine-tune is reproducible and evaluable only if its setup is recorded the same way an inference config is.

- Every trainable backend has training configs in `configs/train/<layer>-<backend>-<name>.yaml` (pydantic): `base_weights` (name, URL, sha256), `dataset` (Roboflow `workspace/project/version`, or a local label manifest, plus the sha256 of the manifest), `recipe` (resolution, epochs, batch, lr, augmentations, seed), `platform` (`roboflow`, `modal`, `runpod`, `colab`, `local_mps`).
- `bga train --config configs/train/x.yaml` launches the job on the named platform, or prints the exact steps when the platform is a UI (Roboflow Train, Colab), then pulls the weights into `weights/<layer>/<backend>/<name>@<hash8>/` next to a `train.json` holding the config, dataset sha, git commit, platform, GPU, wall time, cost and the trainer's own validation metrics. The hash is of the training config.
- An inference config names weights by that id, for example `weights: detect/rfdetr_large/plk-v1@3f9a1c2e`. Public pretrained checkpoints are declared as `source: pretrained` with URL and sha256. `bga eval` refuses weights that have neither a `train.json` nor a pretrained declaration.
- A metrics row therefore cites the inference config name and hash and, for each trained backend, the training config name and hash. Trainer validation numbers go to `docs/train.md` (append-only, one row per training run); only gold-set numbers from `bga eval` go to `docs/metrics.md`.
- Leakage guard: `bga train` refuses a dataset manifest that contains any frame of a gold clip, and `tests/test_gold_leakage.py` checks every committed manifest. The held-out game is scored once per milestone, never while choosing between training setups.
- Comparing training setups: `bga eval --config det.yaml --set detect.weights=A --set detect.weights=B` runs one inference config over several weight ids and prints the rows side by side, so a resolution or epoch sweep is one command and one table.
- Weights are gitignored. `configs/train/*.yaml`, `docs/train.md` and the label manifests are committed, which is enough to retrain any row.

### Layer 0: ingest and scene filtering
- **Input:** clip. **Output:** per-frame timestamps, shot id, camera tag (`central` / `other`), optional game clock.
- **Approach:** shot-cut detector from the prior repo; a small frame classifier (central wide shot vs replay/close-up/graphics) trained from the labeller's camera tags; scoreboard-clock OCR later for play-by-play alignment.
- **Done when:** on the gold clips, every shot has the right camera tag with no missed cuts inside central shots.

### Layer 1: court homography
- **Input:** central-camera frames. **Output:** per-frame 3x3 homography image to court plane in centimetres, with a validity flag.
- **Approach:** RF-DETR Keypoint fine-tuned on the 33-landmark schema; keep points with confidence > 0.5 and at least 4 visible; `ViewTransformer` with `cv2.findHomography` RANSAC; smooth across frames within a shot (exponentially weighted on keypoint positions, or reject frames whose homography jumps); never smooth across a cut.
- **PLK adaptation:** label landmarks on frames from every PLK arena, since floor paint and logos differ per hall. Keypoint visibility flags: visible = 2, occluded but geometrically known = 1, off-screen = 0 skipped.
- **Training recipe (verified from the June 2026 post):** resolution 576, 50 epochs, batch 2 with gradient accumulation 2, lr 1e-4, select on `val/keypoint_map_50_95`.
- **Data:** provisional 300 to 1,000 frames across all arenas, started by pre-labelling with the NBA court model and correcting.
- **Done when:** median reprojection error on held-out PLK frames is reported per arena. Provisional target under 0.5 m near the paint.

### Layer 2: detection, pose, segmentation
- **Input:** central frames. **Output:** boxes with the 10-class schema, 17-point COCO pose per player, optional masks.
- **Approach:** RF-DETR Large for boxes, PLK fine-tuned. RF-DETR Keypoint default COCO-pose checkpoint for ankles and torso. RF-DETR Seg only if masks are needed for a downstream consumer; team colour and number reads work from pose-derived torso crops.
- **Filtering:** drop any detection whose projected foot point lies outside the court polygon plus a margin. This replaces the old fixed trapezoid.
- **Resolution:** small objects (ball, numbers, far-side players) matter more than architecture. Train and infer at a higher input size than default, or tile. Measure per-class AP at two resolutions before choosing.
- **Liquid d1-2B is not a detector or segmenter here (checked 2026-10-28).** Its `boxes` answer returns at most 16 boxes per question as fractions of the image, with no masks, at about 100 ms per frame on an Apple M5 Pro, and it cannot be fine-tuned without Liquid. Keep it as the `d1_boxes` backend for labelling cross-checks and never for production frames. Masks, when needed, come from RF-DETR Seg or SAM 2.1.
- **SAM 3.x is not available (2026-10-29).** Meta declined Tomasz's Hugging Face access request for `facebook/sam3.1`, and the SAM 3 weights are gated the same way. Neither model, nor any third-party re-upload or Core ML conversion of them, is used anywhere in this project. SAM 3 remains reachable only as a hosted labelling option inside Roboflow Auto Label, if a workspace is ever set up. Masks come from RF-DETR Seg trained on masks that SAM 2.1 (Apache 2.0, open download) draws from detector boxes on Modal, backend `sam21_boxmask`.
- **Done when:** per-class AP50:95 on the gold set is tabulated, with `ball` and `number` reported separately, plus ms per frame on the Mac.

### Layer 3: team classification
- **Input:** player boxes per clip. **Output:** team label per track.
- **Approach:** `sports.TeamClassifier` unchanged, fitted per clip on crops sampled at 1 fps, referees excluded by detector class. Majority vote per track. Add a cluster-separation check with a manual kit-colour fallback for near-identical kits.
- **Cross-check backend:** `d1_choice` asks Liquid d1-2B one `choice` question per crop with the two kit descriptions as options. It needs no fitting, so it is a cheap second opinion when the clusters do not separate.
- **Done when:** team accuracy on gold tracks is reported. Provisional target 0.98.

### Layer 4: tracking and tracklet linking
- **Input:** filtered player boxes, foot points in court coordinates. **Output:** stable track ids per clip.
- **Approach, two stages:**
  1. Short tracklets from ByteTrack (`supervision` or Roboflow `trackers`) on per-frame RF-DETR boxes. Run the motion model in court metres after the homography, not in pixels.
  2. Offline linking pass that merges tracklets within a shot using team, confirmed number, court-space motion continuity, and the hard constraint of 5 players per team on the floor. Identity rules from section 4 apply: one live owner per `(team, number)`, conflicts flagged, rebind window after occlusion.
- **Candidates (2026-10-29):** `sam2_realtime` is Skalski's SAM2 real-time path as published (`Gy920/segment-anything-2-real-time`, SAM 2.1 weights, Apache 2.0), prompted with RF-DETR boxes on the first frame of each central shot, with a re-prompt monitor that adds objects when the on-floor count drops below 5 per team and closes the session at every cut. It needs CUDA and runs through `--remote`; his stated limits (1 to 2 fps on a T4, degradation with many objects and in high action) are the things to measure. `edgetam` is Meta's EdgeTAM (CVPR 2025, Apache 2.0, open checkpoint, box prompts, 22x faster than SAM 2, SA-V val J&F 72.3 against 76.8 for SAM 2.1-B+, Core ML export for macOS), a mask tracker that can run on the Mac; try it after `sam2_realtime` has a metrics row. Both produce masks plus ids per frame, so they feed L2 masks and L4 tracks. The offline linking pass and identity rules apply unchanged. Compare with `bytetrack_court` on HOTA and IDF1 in M6 and keep whichever wins per shot type.
- **Done when:** HOTA and IDF1 on the gold set, plus churn metrics, are reported before and after the linking pass.

### Layer 5: jersey number and identity
- **Input:** `number` boxes matched to tracks by IoS >= 0.9, torso crops from pose as fallback. **Output:** jersey number per track, player name via roster.
- **Approach:** a 0 to 99 classifier (ResNet-18 class, as in Skalski's result) on number crops, read every 5th frame, confirmed by consecutive agreement, voted per tracklet. Restrict outputs to the roster of that team. Where play-by-play exists, restrict further to the 5 numbers on the floor at that game-clock time.
- **PLK adaptation:** European kits carry sponsor logos next to the number. Include those in the crop training set.
- **Backends to compare (M7):** `resnet_cls` is the default because Skalski measured 93% for a classifier against 86% for fine-tuned SmolVLM2. `d1_choice` asks Liquid d1-2B one `choice` question per crop with the roster numbers as options and gets a probability per option in a single forward pass with no token generation (18.6 ms per question and 45 ms with a 384 px picture on an Apple M5 Pro; no OCR benchmark published; no self-serve fine-tuning). The same pass can carry extra questions on the same crop: `noul` number visible, `choice` team by kit. The probabilities feed the consecutive-agreement vote directly. `lfm3vl` is Liquid LFM3-VL-2B, generative, OCRBench 86.3 vendor-reported, fine-tunable with Liquid's TRL notebook. `basal_choice` is the same `choice` pattern on basal-omni (Polish open-source decision models by Remigiusz Kinas, Apache 2.0, Gemma 4 base, same `/v1/systemone` API as d1). It is low priority: images need `--mode eager` on an NVIDIA GPU (about 16 GB for mini, 24 GB for max), so it is Colab-only; the vision encoder is used zero-shot with no multimodal training and the author reports plain Gemma 4 scoring close to it on photos; confidence thresholds were calibrated on text only. Worth keeping because basal's independent Werdykt benchmark is the only third-party measurement of decision models so far: Liquid d1-ultra via API scored 88.1% on Polish and English text decisions while the local d1-omni-600M scored 33.3%, so small local decision models can fall far below their vendor claims and d1-2B must be measured, not assumed. Score every backend on the gold number crops: top-1 accuracy under the roster constraint, calibration of the reported probabilities, ms per crop on the M1 Max. Measure before choosing; none of the Liquid numbers has an independent evaluation yet.
- **Done when:** per-track identity accuracy on the held-out gold game is reported, and the fraction of tracks that reach a confirmed number.

### Layer 6: 2D positions
- **Input:** tracks, homographies, pose. **Output:** x, y in metres per player per frame; radar video.
- **Approach:** ground point = ankle midpoint from pose, falling back to box bottom-centre when ankles are missing; hold last floor position while both ankles are airborne; project through the per-frame homography; `clean_paths` smoothing within a shot.
- **Done when:** positions for every gold shot are exported and the radar video is visually checked against the footage.

### Layer 7: ball, possession, shots
- **Input:** `ball`, `rim`, `ball-in-basket`, `player-in-possession` detections. **Output:** ball x, y per frame, holder track id, shot events.
- **Approach:** tiled inference for the ball if recall is low; possession = nearest `player-in-possession` box; while held, ball position = holder position; in flight, interpolate between release and reception; `ShotEventTracker` for made/miss.
- **Done when:** possession accuracy on gold shots is reported. Build this last.

### Layer 8: evaluation and review
- Gold set and metrics in section 8. The labeller is the review tool. Every human correction is written back to the training sets.

### Export contract
Per frame per player: `game_id, clip, frame_idx, t_video_s, shot_id, camera, game_clock, track_id, player_uid, team, jersey_number, player_name, bbox_xyxy_px, foot_px, x_m, y_m, confidence`. Per frame ball: `frame_idx, x_m, y_m, holder_track_id, in_flight`. Court coordinates from `CourtConfiguration(League.FIBA, MeasurementUnit.CENTIMETERS)` divided by 100.

## 6. Models and licences (verified 2026-10-27)

| Role | Model | Licence | Numbers |
|---|---|---|---|
| Detection, default | RF-DETR Large (DINOv2 backbone) | Apache 2.0 | COCO 56.5 AP, RF100-VL 62.2, 6.8 ms T4 TensorRT FP16 |
| Detection, if licence allows | RF-DETR XL / 2XL | PML 1.0 | COCO 58.6 / 60.1 AP, RF100-VL 62.9 / 63.2 |
| Segmentation | RF-DETR Seg Nano to 2XL | Apache 2.0 | COCO mask 40.3 to 49.9 AP, 3.4 to 21.8 ms T4 |
| Keypoints | RF-DETR Keypoint (preview) | Apache 2.0 | COCO keypoint 71.8 AP, 9.7 ms T4, 40.7M params |
| Bake-off alternatives | YOLO26-L (AGPL-3.0), DEIM-D-FINE-L (Apache 2.0) | see licence | COCO comparable; no published RF100-VL transfer numbers |
| Excluded | DEIMv2 (non-commercial) | | |
| Not available (2026-10-29) | SAM 3 and SAM 3.1 (`facebook/sam3`, `facebook/sam3.1`) | SAM License, gated; Meta declined Tomasz's access request. Third-party re-uploads and Core ML conversions are not used either. | Assessed before the refusal: SAM 3.1 (2026-03-27) adds Object Multiplex, up to 16 objects per pass, 16 to 32 fps on an H100. Kept here so nobody re-adds it. SAM 3 stays usable only as a hosted option inside Roboflow Auto Label. |
| Tracking and mask candidates (checked 2026-10-29) | SAM 2.1 (`facebook/sam2.1-hiera-large`, 0.2B) through Skalski's real-time fork; EdgeTAM (Meta, CVPR 2025) | Apache 2.0, open downloads | SAM 2.1: Skalski's path, 1 to 2 fps on a T4 with his settings, CUDA only. EdgeTAM: SA-V val J&F 72.3 against 76.8 for SAM 2.1-B+, 16 fps on an iPhone 15 Pro Max, Core ML export script in the repo, so it can run on the Mac. |
| Number reading, candidates (checked 2026-10-28) | Liquid LFM3-VL-2B (generative VLM, 1.92B) and Liquid d1-2B (decision model on the same backbone) | LFM Open License v1.0: Apache-style, commercial use granted only while annual revenue is under USD 10M, inherited by fine-tunes | LFM3-VL-2B: OCRBench 86.3, TextVQA 87.9, RefCOCO 90.5, Video-MME 62.8, all vendor-reported. d1-2B: no OCR or detection benchmark; M5 Pro 18.6 ms per question, 45 ms with a 384 px picture, 54 ms `box`, 100 ms `boxes`. Both released 2026-10-27, no independent evaluation yet. |
| Number reading, low-priority candidate (checked 2026-10-28) | basal-omni 1.0 mini / max / ultra (decision models on Gemma 4 E4B / 12B / 31B, author Remigiusz Kinas) | Apache 2.0 per the project's NOTICE | No OCR, detection or sports probe. Own multimodal probe: real-photo task 66.7% mini, 80.0% ultra on 150 questions; vision zero-shot, calibration from text. Images need an NVIDIA GPU in eager mode; Apple silicon runs text only. No media latency published. |
| Labelling assist | The implementer's own vision through Claude Code on contact sheets (Tomasz's subscription, no API keys); SAM 2.1 box-to-mask on Modal; SAM 3 inside Roboflow Auto Label only if a workspace exists | subscription; Apache 2.0; hosted | For reference, Roboflow Vision Evals object detection mAP@50: Gemini 4 Argon 85.1 high effort, GPT-6.1 Terra 84.7 low effort; best open-weight VLM Qwen3.8 27B 65.7. VLM boxes are loose, so geometry comes from detectors and SAM 2.1, classes and number values from vision review. |

Decision rule: pick the production detector by a bake-off on the PLK validation split once a few thousand labelled frames exist, comparing per-class AP (especially `ball` and `number`) and ms per frame on the Mac. Until then, RF-DETR Large.

Mac inference: export RF-DETR to ONNX and run with `onnxruntime` CoreML provider, or run torch on MPS. The notebook's SAM2 cells require CUDA; they run through `--remote` (section 5, "Where things run"). EdgeTAM's Core ML export is the Mac-side mask tracker to try.

Liquid models on the Mac (status 2026-10-28): `transformers` with `trust_remote_code` on MPS works today (`>=5.19` for d1-2B, `>=5.21` for LFM3-VL-2B). llama.cpp support for both was still in review, and the d1 llama.cpp path serves `noul`, `choice` and `score` over pictures through `/v1/systemone` but not `box`, extraction or video. MLX-VLM support for LFM3-VL-2B was in progress. No published speed figures exist for the M1 Max; measure and record.

## 7. Labelling strategy

Goal: near-zero hand-drawn boxes outside the gold set. Humans review and correct; models draw.

| Label type | Automatic source | Human work |
|---|---|---|
| Player, referee, ball, rim boxes | Skalski's NBA-trained 10-class model on PLK frames; the implementer reviews contact sheets of low-confidence and crowded frames with its own vision and fixes classes; from round two the PLK-trained v0 detector labels and the NBA model is the second opinion; keep boxes where both agree | Tomasz sees only frames the implementer flags, inside the 90-minute budget |
| Masks | SAM 2.1 on Modal draws a mask inside every reviewed box (`sam21_boxmask`); Roboflow Auto Label with SAM 3 only if a workspace exists | Spot checks on contact sheets by the implementer |
| Number boxes | `number` class from his model; torso crop from pose as fallback | Review flagged crops |
| Number values | The implementer reads contact sheets of 3 to 4 legible crops per tracklet (about 20 crops per sheet) with its own vision; majority vote propagates to every crop of that tracklet; reads outside the roster are conflicts, not labels | Tomasz resolves conflicts only |
| Court landmarks | Pre-label with the NBA court model; the implementer drops points with a high RANSAC reprojection residual and keeps frames with at least 6 consistent points; 6 frames per arena go to Tomasz | The only real clicking in the project, about 20 minutes |
| Pose | COCO-pretrained keypoint model | None |
| Gold set (section 8) | Prefill from the walking skeleton's `tracks.json`; the implementer does the first correction pass on contact sheets; Tomasz does the final pass at tracklet level | Confirm, split, merge, assign identities; fix boxes only on the box-gold frames |

Roboflow Auto Label is optional (no workspace confirmed, no third-party VLM keys). Facts, verified, in case one is set up: models offered are Gemini 4 Argon, GPT-6 Astra, GPT-6.1 Sol, Gemini 3.8 Flash, Claude Opus 5.5, SAM 3, CLIP, VLM + SAM 3 combinations, your own models trained on the same dataset, and saved Workflows with SAHI or consensus blocks. Boxes, masks, polygons and classification are supported; keypoints are not. Test runs on four images are free; batch jobs show cost before starting; your own API key avoids credit charges.

Bootstrap loop: auto-label a few hundred frames, review, train RF-DETR v0; use v0 as the labeller on a few thousand more, review only what it is unsure about, train v1. Each round shrinks the review queue.

Cautions: VLM boxes are loose at mAP@50, so let the detector or SAM 2.1 set geometry and use vision review for class and number value. Reading images through Claude Code spends context and subscription limits per image, so always batch crops into contact sheets and keep a ledger of sheets read. Confusions 6/9 and 1/7 and sponsor logos must be in the gold crops. The gold set must never be produced by the model under test.

## 8. Evaluation

Gold set, built first (milestone 1):

| Component | Spec |
|---|---|
| Games | 4, different seasons, opponents, arenas and kits. One held out for final testing only. |
| Shots | 3 central-camera shots per game, 8 to 12 s each, 12 shots in total |
| Track gold | every 5th frame at 50 fps (10 frames per second) through each shot, prefilled and interpolated; every player and referee carries a persistent identity, team and number, with `num_visible` and `occluded` flags. Human work is per track, not per frame: confirm, split, merge, assign. |
| Box gold | about 100 frames drawn evenly from the 12 shots, every box corrected by hand to tight geometry; detection AP is computed on this subset |
| Number crops | ~300 with digit labels, cut from boxes flagged `num_visible`; labelled by the implementer on contact sheets, 30 spot-checked by Tomasz |
| Court landmarks | 6 frames per arena, 24 in total, 33 landmarks with visibility flags, prefilled by the NBA court model; Tomasz corrects only the points the implementer flags |
| Human budget | about 90 minutes of Tomasz's time in total, in two sessions. Dry-run one shot first and time it; if the projection exceeds the budget, cut box-gold frames and shots, never quality. |
| Format | MOTChallenge gt (`frame,id,x,y,w,h,1,class,visibility`) plus `identities.csv` (`tid,team,number,name`), as the labeller already writes |

Metrics, computed by a single `eval` command and written to `docs/metrics.md` with date and commit:

| Layer | Metric | Tool |
|---|---|---|
| Detection | per-class AP50 and AP50:95 on the box-gold frames | pycocotools or supervision |
| Tracking | HOTA, IDF1, MOTA, plus churn metrics | TrackEval on MOT gt |
| Team | accuracy per gold track | custom |
| Identity | per-track number accuracy, confirmed fraction, wrong-confirmed count | custom |
| Court | median and p90 reprojection error in metres on held-out landmarks | custom |
| 2D positions | position error in metres against landmark-derived gold where available; otherwise visual radar check | custom |
| Speed | ms per frame per stage on the Mac | custom |

Rule: no change to any layer merges without a before/after row in `docs/metrics.md`. A row cites the inference config and, for every trained backend, the training config (section 5, "Training setups and their evaluation"). Trainer validation numbers go to `docs/train.md` and never stand in for gold-set numbers.

## 9. Milestones

Each milestone ends with a metrics table or an artifact checked into the repo.

Progress is tracked in GitHub issues, one per milestone, created 2026-10-29 (label `milestone`):

| M0 | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 | M9 |
|---|---|---|---|---|---|---|---|---|---|
| [#1](https://github.com/tko-ventures/basketball-game-analyzer/issues/1) | [#2](https://github.com/tko-ventures/basketball-game-analyzer/issues/2) | [#3](https://github.com/tko-ventures/basketball-game-analyzer/issues/3) | [#4](https://github.com/tko-ventures/basketball-game-analyzer/issues/4) | [#5](https://github.com/tko-ventures/basketball-game-analyzer/issues/5) | [#6](https://github.com/tko-ventures/basketball-game-analyzer/issues/6) | [#7](https://github.com/tko-ventures/basketball-game-analyzer/issues/7) | [#8](https://github.com/tko-ventures/basketball-game-analyzer/issues/8) | [#9](https://github.com/tko-ventures/basketball-game-analyzer/issues/9) | [#10](https://github.com/tko-ventures/basketball-game-analyzer/issues/10) |

The issue holds the checklist, comments, partial progress and dead ends; it closes when the milestone's done-when line has its metrics row or artifact. This file changes only when a decision changes, with a dated entry in `docs/decisions.md` and the `decision` label on the issue or pull request that changed it. Work the lowest open milestone issue first.

**M0. Repository bootstrap.**
Import the code assets from section 4 into this repo under the layout in section 10, with no media, weights, runs or caches from the prior checkout (section 2, data provenance). Set up `uv`, `ruff`, `pytest`, a `.gitignore` that excludes media, weights, runs and label caches. Re-download the clips with `fetch_highlights.sh`, regenerate the shot files with `prepare_clips.py` and diff them against the imported ones as a cross-check. Done: labeller opens every clip with shots ready to tag, from fresh downloads.

**M1. Gold set.**
Propose the 4 games from the clip manifest and get Tomasz's confirmation. Run Skalski's NBA player model over them (train it from the public dataset on Modal and export ONNX, or use `inference` if a Roboflow key exists). Write `tracks.json` in the labeller's prefill format, add the tracklet-level review actions to the labeller, and do the first correction pass yourself on contact sheets. Time a dry run with Tomasz on one shot, then run his two sessions. Label the number crops on contact sheets and prefill the landmark frames. Done: gold set per section 8 committed, one game marked held-out, Tomasz's time logged.

**M2. Walking skeleton and baseline.**
RF-DETR NBA weights, ByteTrack, NBA court keypoint model with `League.FIBA` targets, bottom-centre ground point, `clean_paths`, radar video for every gold shot. This milestone also lands the backend registry, the per-layer interfaces, `configs/baseline.yaml` and the `--remote` runner with a Modal app skeleton from section 5. Run `eval`. Done: first full metrics table, its row citing `baseline.yaml`. This is the baseline everything else must beat.

**M3. Court layer.**
Landmark dataset across arenas, RF-DETR Keypoint fine-tune per section 5 layer 1, temporal smoothing within shots. As the first fine-tune, this milestone lands `bga train`, `configs/train/`, `docs/train.md` and the gold-leakage test. Done: reprojection error per arena in the metrics table, the row citing both configs.

**M4. Detection fine-tune.**
Auto-label per section 7, review, fine-tune RF-DETR Large on the 10-class schema. Bake-off against YOLO26-L and DEIM-L on the same split, one training config each, trained on Modal. Export ONNX, run on the Mac. Done: per-class AP and ms per frame.

**M5. Team layer.**
`TeamClassifier` per clip with separation check. Done: team accuracy.

**M6. Tracking and linking.**
Court-space ByteTrack, then the offline linking pass with the 5-per-team constraint and identity rules. Run `sam2_realtime` on Modal as the second tracker config over the same shots, then `edgetam` on the Mac if time allows. Done: HOTA and IDF1 for both trackers, before and after linking.

**M7. Numbers and identity.**
VLM-bootstrapped crop labels with tracklet propagation, then a number-reader bake-off per section 5 layer 5: `resnet_cls`, `d1_choice`, `lfm3vl` zero-shot and SFT, each as its own config with a metrics row (top-1 under the roster constraint, calibration, ms per crop). Wire the winner into identity with the roster constraint and the play-by-play on-floor constraint if the feed is available. Done: identity accuracy on the held-out game.

**M8. 2D positions.**
Ankle ground points from pose, airborne handling, export contract, radar video. Done: exported parquet for all gold shots and a visual check.

**M9. Ball, possession, shots.**
Done: possession accuracy on gold shots.

After M2, prefer finishing each milestone over starting the next. If a milestone's metric does not improve on the baseline, record that and stop adding heuristics to it.

## 10. Repository layout and conventions

```
basketball-game-analyzer/
  CLAUDE.md                 short pointer to this plan and the rules below
  README.md
  docs/PLAN.md              this file
  docs/metrics.md           append-only metrics log, one table per run
  docs/train.md             append-only training log, one row per training run (config hash, dataset sha, platform, trainer val metrics)
  docs/decisions.md         short dated entries when a decision in section 2 changes
  pyproject.toml            uv-managed, Python 3.12
  configs/                  one YAML per pipeline configuration; committed once it produced a metrics row
  configs/train/            one YAML per training setup (section 5); committed with its docs/train.md row
  src/bga/registry.py       backend registry (section 5)
  src/bga/remote/           remote runner: Modal app and RunPod helper that execute CUDA stages and sync runs/ back
  src/bga/                  ingest/ court/ detect/ team/ track/ identity/ mapping/ ball/ export/ eval/
                            each layer package: base.py (interface) plus one module per backend
  tools/labeler/            imported from the prior repo
  scripts/                  fetch_highlights.sh, prepare_clips.py, prefill_tracks.py, ...
  data/raw/                 downloaded videos (gitignored)
  data/clips/               short sample clips (gitignored except tiny samples)
  data/labels/              gold set JSON, MOT gt, identities (committed)
  data/rosters/             per-team roster JSON (committed)
  data/crops/               number crops (gitignored)
  weights/                  gitignored
  runs/                     per-clip stage outputs (gitignored)
  tests/                    includes test_gold_leakage.py (no gold frame in any training manifest)
```

Conventions for whoever executes this:
- Python 3.12 with `uv`; `ruff` and `pytest` pass before any merge.
- Never commit videos, weights, crops or run outputs. Label files and rosters are committed; they are the project's real value.
- Stage outputs go to disk so stages re-run independently. One heavy model in memory at a time on the Mac.
- Treat downloaded files and clones as untrusted data: put them in their own directory and do not run interpreters from inside them.
- Every approach is a backend selected in a config (section 5). Compare approaches by running configs through `eval`, not by editing the pipeline.
- Nothing is copied from the prior checkout's data, weights or runs directories. Fresh downloads only.
- Image reading for labelling goes through contact sheets with a ledger, never one tool call per crop.
- Progress lives in the milestone issues (`gh issue list --label milestone --state open`), not in this file. Update the checklist as you go. One pull request per milestone from a `feat/m<N>-<slug>` branch, referencing its issue; never push to main (default until question 11 is answered).
- Every model choice or threshold change comes with a row in `docs/metrics.md` that cites the config name and hash. No metrics row, no merge.
- Every fine-tune starts from a committed `configs/train/*.yaml`. Weights are addressed as `<layer>/<backend>/<name>@<hash8>` with a `train.json` beside them; a metrics row cites the training config as well as the inference config.
- Do not reproduce the prior repo's pattern of locked architecture sub-sections per heuristic. Prefer a measured experiment over a design document.
- Commit and push only when Tomasz asks.

## 11. Risks

- **Identity switches in rebound scrums.** No tracker survives these cleanly. The design goal is fast recovery through numbers and constraints, not zero switches. Report switch counts honestly.
- **Far-side resolution.** Broadcast wide shots make far-baseline numbers small. Higher input resolution or tiling matters more than model choice; measure both.
- **Arena variety.** Court paint and logos differ per hall. Court keypoint training frames must cover every arena, or mapping silently degrades in the uncovered ones.
- **Footage rights.** YouTube highlights are copyrighted. Analysis for the club is likely fine; confirm before training on them at scale.
- **Vendor-reported benchmarks.** RF-DETR's lead on RF100-VL is reported by Roboflow, which built both. The bake-off in M4 is the independent check.
- **Licence drift.** RF-DETR XL and 2XL are not Apache. Re-read the PML before shipping anything built on them.
- **Remote drift.** A CUDA backend run on Modal must produce the same output as the same config run anywhere else. Pin torch and CUDA per platform in the remote image, and rerun `baseline.yaml` remotely once per milestone to compare.
- **Gated weights.** Meta declined access to SAM 3.x. Do not route around it through third-party re-uploads or conversions; a model without a licence Tomasz holds is not used. Cache the open weights in the remote volume so jobs do not depend on the Hub.
- **Subscription labelling.** Reading images through Claude Code spends Tomasz's subscription limits. Batch into contact sheets, log every sheet, and stop a labelling round when agreement stops improving.
- **Prior-repo gravity.** 18k lines of heuristics are tempting to port. Only section 4's list comes over.

## 12. Open questions for Tomasz

Ask these when the milestone needs them.

1. **Roboflow account (M1, M4).** Partly answered 2026-10-29: no third-party VLM API keys; the implementer uses its own vision through Claude Code. Answered 2026-10-29: workspace `tomasz-kopacki` exists; the API key is `ROBOFLOW_API_KEY` from `~/.config/bga/roboflow.env` on the Mac and the Modal secret `roboflow`. It serves Universe dataset downloads and `inference` prefill for labelling. Training credits are not visible through the API, so fine-tunes stay on Modal unless Tomasz says otherwise.
2. **Runtime cloud policy (M2).** Answered 2026-10-29: rented CUDA machines (Modal, RunPod) are allowed for training and for batch stages that need them; closed VLM APIs stay labelling and QA only; prefer the Mac whenever a backend runs there at acceptable speed. Still open: a monthly spend cap for rented GPUs.
3. **Play-by-play feed (M7).** plk.pl publishes per-game play-by-play. Confirm it includes substitutions with game clock, and whether scraping it is acceptable.
4. **Full-game footage (M1 onward).** Does the club have Hudl Instat or Synergy access? Full games change the data mix and allow play-by-play alignment.
5. **RF-DETR XL licence (M4).** Is the Roboflow Platform Model Licence acceptable, or stay on Large?
6. **Opponent rosters (M1).** Collect all PLK team rosters now, same schema as the Dziki file?
7. **Labeller future (M0).** Default stands: keep the local labeller for video ground truth; Roboflow Annotate only if a workspace exists. Tomasz corrects the gold labels himself within about 90 minutes (answered 2026-10-29).
8. **Hugging Face gated access.** Answered 2026-10-29: Meta declined the request, so SAM 3.x is out and no HF token is needed; every remaining model is an open download.
9. **Modal credentials (M2).** Answered 2026-10-29: the token is in `~/.modal.toml` on Tomasz's Mac and the CLI is installed, so an implementer running there needs nothing else. Still open: confirm the 50 USD monthly cap.
10. **Data provenance (M0).** Answered 2026-10-29: start from scratch; nothing is copied from the prior checkout except code, rosters and the shot files used as a cross-check.
11. **Commit policy (M0).** Answered 2026-10-29, default confirmed: the implementer commits on `feat/m<N>-<slug>` branches and opens one pull request per milestone referencing its issue, never pushing to main or merging without Tomasz.
12. **Gold games (M1).** Answered 2026-10-29: `4XMD9FWmQws` PGE Start – Dziki (2024/25), `OSp445e0mXU` Dziki – Legia (2025/26), `1jDaJBkUGmc` Energa Trefl – Dziki (2025/26), `Tnuscrm9RyU` Dziki – Zastal (2026/27, held out). Rationale on issue #2.
13. **Footage rights (M4).** Open: training on the YouTube highlights at scale. Default: proceed; the gold set is small.
14. **Rosters (M1).** Answered 2026-10-29: yes, collect every PLK team's roster.

## 13. Sources (checked 2026-10-27; Liquid AI and basal entries 2026-10-28; SAM 3.1 and GPU platform entries 2026-10-29)

- Skalski basketball tutorial: https://roboflow.com/blog/identify-basketball-players
- Notebook: https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb
- Video: https://www.youtube.com/watch?v=yGQb9KkvQ1Q
- RF-DETR keypoint fine-tuning: https://roboflow.com/blog/train-rf-detr-keypoint
- Camera calibration with keypoints (football, method reference): https://roboflow.com/blog/camera-calibration-sports-computer-vision
- Ball tracking (football, method reference): https://roboflow.com/blog/tracking-ball-sports-computer-vision
- Sports package, basketball branch: https://github.com/roboflow/sports (branch `feat/basketball`, `sports/basketball/config.py` has `League.FIBA`)
- Player detection dataset: https://universe.roboflow.com/roboflow-jvuqo/basketball-player-detection-3-ycjdo
- Court keypoint dataset: https://universe.roboflow.com/roboflow-jvuqo/basketball-court-detection-2
- RF-DETR repository and benchmark table: https://github.com/roboflow/rf-detr
- Roboflow Vision Evals, object detection: https://playground.roboflow.com/evals/object-detection
- Roboflow Auto Label docs: https://docs.roboflow.com/annotate/ai-labeling/automated-annotation-with-autodistill
- Multi-model auto labeling with Workflows: https://blog.roboflow.com/multi-model-auto-labeling/
- Roboflow task-by-task model guide 2026: https://roboflow.com/blog/best-computer-vision-models
- DEIMv2 paper and licence: https://arxiv.org/pdf/2509.20787 , https://github.com/Intellindust-AI-Lab/DEIMv2 (LICENSE.md, non-commercial)
- YOLO26 paper: https://arxiv.org/pdf/2601.12882
- SAM3 zero-shot vs fine-tuned YOLO study: https://arxiv.org/pdf/2512.11884
- Prior repo: https://github.com/tko-ventures/basketball-tracking/tree/feat/labeller-rosters (private)
- Skalski's post that prompted the model-selection discussion: https://x.com/skalskip92/status/2115163498435219553
- Liquid AI LFM3 launch thread: https://x.com/liquidai/status/2115128329003303107 and the d1 post https://x.com/liquidai/status/2115128344769687701
- LFM3 blog: https://www.liquid.ai/blog/lfm3
- LFM3-VL-2B model card (benchmarks, TRL fine-tuning notebooks, MLX / GGUF / ONNX repos): https://huggingface.co/LiquidAI/LFM3-VL-2B
- d1-2B model card (question types, `box` / `boxes` output, M5 Pro timings): https://huggingface.co/LiquidAI/d1-2B
- d1-2B GGUF and llama.cpp limits: https://huggingface.co/LiquidAI/d1-2B-GGUF
- LFM Open License v1.0 text and explainer: https://huggingface.co/LiquidAI/LFM3-VL-2B/blob/main/LICENSE , https://liquid.ai/lfm-license
- basal decision models (site, repo, benchmark with Liquid d1 entries, basal-omni card): https://basal.si5.pl/ , https://github.com/rkinas/basal , https://basal.si5.pl/benchmark/ , https://huggingface.co/Remek/basal-omni-1.0-mini
- SAM 3.1 (assessed, access declined, not used): repository https://github.com/facebookresearch/sam3 , release notes https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md , checkpoints https://huggingface.co/facebook/sam3.1 (gated, no transformers integration), licence https://github.com/facebookresearch/sam3/blob/main/LICENSE , fine-tuning https://github.com/facebookresearch/sam3/blob/main/README_TRAIN.md , video example https://github.com/facebookresearch/sam3/blob/main/examples/sam3.1_video_predictor_example.ipynb , Meta blog https://ai.meta.com/blog/segment-anything-model-3/ , Ultralytics notes (3.1 images only there) https://docs.ultralytics.com/models/sam-3
- SAM 3.1 Core ML port used by the prior labeller, not reused: https://huggingface.co/AllanVester/SAM3.1-CoreML-FP16
- SAM 2.1 weights (Apache 2.0): https://huggingface.co/facebook/sam2.1-hiera-large ; Skalski's real-time fork: https://github.com/Gy920/segment-anything-2-real-time
- EdgeTAM (Apache 2.0, Core ML export): https://github.com/facebookresearch/EdgeTAM
- Roboflow Inference SAM 3 docs (no 3.1 yet): https://github.com/roboflow/inference/blob/main/docs/foundation/sam3.md
- GPU platform prices: https://modal.com/pricing , https://www.runpod.io/pricing
