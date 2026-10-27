# Basketball game analyzer: implementation plan

**Status:** handoff draft, 2026-10-27.
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
| Hardware | Apple M1 Max, 64 GB, no NVIDIA GPU. Training runs on Roboflow Train or Colab. Inference runs locally. Load heavy models sequentially. | prior repo |
| Model policy | Open-weight models fine-tuned into PLK-specialised detectors. Closed VLM APIs only for labelling and offline QA, never as the per-frame production model. | Tomasz |
| Reference stack | Follow Piotr Skalski's Roboflow basketball pipeline (section 3). Deviate only where his own posts list a limitation or where PLK differs from the NBA. | Tomasz |
| Court geometry | FIBA court, 28 x 15 m. The `sports` package already ships a FIBA preset with the same 33-landmark vertex order as the NBA one. **Verified.** | roboflow/sports `feat/basketball` branch, `sports/basketball/config.py` |
| Identity | `player_id = (team, jersey_number)`. Numbers are unique within a team only. FIBA allows 0 to 99. | prior repo |
| Ground truth first | No heuristic is added to any layer until the gold set in section 8 exists and the change is measured against it. | review of prior repo |
| Licences | RF-DETR Nano to Large, RF-DETR Seg, RF-DETR Keypoint: Apache 2.0. RF-DETR XL and 2XL: Roboflow Platform Model Licence 1.0, terms to be checked. YOLO26 and YOLO11: AGPL-3.0. D-FINE and DEIM: Apache 2.0. DEIMv2: non-commercial only. SAM 3: custom Meta licence, labelling only. **Verified.** | section 13 |

## 3. Reference stack: Skalski's basketball pipeline (verified 2026-10-27)

Blog post: https://roboflow.com/blog/identify-basketball-players (2025-09-30).
Notebook: https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb
Video: https://www.youtube.com/watch?v=yGQb9KkvQ1Q
Court keypoint fine-tuning post: https://roboflow.com/blog/train-rf-detr-keypoint (2026-06-26).

What the notebook does, in order:

| Stage | His implementation | Reuse for PLK |
|---|---|---|
| Detection | RF-DETR-S fine-tuned on Roboflow Universe dataset `roboflow-jvuqo/basketball-player-detection-3-ycjdo`, 10 classes: `ball`, `ball-in-basket`, `number`, `player`, `player-in-possession`, `player-jump-shot`, `player-layup-dunk`, `player-shot-block`, `referee`, `rim`. | Keep the class schema. Retrain on PLK frames. |
| Tracking | SAM2 real-time (`Gy920/segment-anything-2-real-time`) prompted with RF-DETR boxes on frame one; masks cleaned with `sv.filter_segments_by_distance`. Needs CUDA. Works on short clips only. | Replace for full clips, see layer 4. Keep as an optional high-quality path on Colab. |
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
| `tools/labeler/` (server.py, prepare_clips.py, sam_service.py, static/) | Browser ground-truth labeller: shot tagging, persistent identities, roster autocomplete, copy-forward and interpolation, prefill from any `tracks.json`, SAM 3.1 Core ML assist with identity linking, MOTChallenge export plus `identities.csv`. | Generalise team labels from `dziki/opp/ref/other` to `home/away/ref/other` with a per-clip mapping to team names. Add export of number crops from boxes flagged `num_visible`. Add a court-landmark mode later or use Roboflow Annotate for landmarks. |
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
- **Done when:** per-class AP50:95 on the gold set is tabulated, with `ball` and `number` reported separately, plus ms per frame on the Mac.

### Layer 3: team classification
- **Input:** player boxes per clip. **Output:** team label per track.
- **Approach:** `sports.TeamClassifier` unchanged, fitted per clip on crops sampled at 1 fps, referees excluded by detector class. Majority vote per track. Add a cluster-separation check with a manual kit-colour fallback for near-identical kits.
- **Done when:** team accuracy on gold tracks is reported. Provisional target 0.98.

### Layer 4: tracking and tracklet linking
- **Input:** filtered player boxes, foot points in court coordinates. **Output:** stable track ids per clip.
- **Approach, two stages:**
  1. Short tracklets from ByteTrack (`supervision` or Roboflow `trackers`) on per-frame RF-DETR boxes. Run the motion model in court metres after the homography, not in pixels.
  2. Offline linking pass that merges tracklets within a shot using team, confirmed number, court-space motion continuity, and the hard constraint of 5 players per team on the floor. Identity rules from section 4 apply: one live owner per `(team, number)`, conflicts flagged, rebind window after occlusion.
- **Optional:** SAM2 path from the notebook with a re-prompt monitor, on Colab, for comparison on short shots.
- **Done when:** HOTA and IDF1 on the gold set, plus churn metrics, are reported before and after the linking pass.

### Layer 5: jersey number and identity
- **Input:** `number` boxes matched to tracks by IoS >= 0.9, torso crops from pose as fallback. **Output:** jersey number per track, player name via roster.
- **Approach:** a 0 to 99 classifier (ResNet-18 class, as in Skalski's result) on number crops, read every 5th frame, confirmed by consecutive agreement, voted per tracklet. Restrict outputs to the roster of that team. Where play-by-play exists, restrict further to the 5 numbers on the floor at that game-clock time.
- **PLK adaptation:** European kits carry sponsor logos next to the number. Include those in the crop training set.
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
| Excluded | DEIMv2 (non-commercial), SAM 3 for production (custom licence, ~850M params) | | |
| Labelling assist | SAM 3 (Roboflow Auto Label), SAM 3.1 Core ML (local), Gemini 4 Argon, GPT-6.1 Terra, GPT-6 Astra, Claude Opus 5.5 | API | Roboflow Vision Evals object detection mAP@50: Gemini 4 Argon 85.1 high effort, GPT-6.1 Terra 84.7 low effort. Best open-weight VLM Qwen3.8 27B 65.7. |

Decision rule: pick the production detector by a bake-off on the PLK validation split once a few thousand labelled frames exist, comparing per-class AP (especially `ball` and `number`) and ms per frame on the Mac. Until then, RF-DETR Large.

Mac inference: export RF-DETR to ONNX and run with `onnxruntime` CoreML provider, or run torch on MPS. The notebook's SAM2 cells require CUDA and will not run on the Mac; use Colab for that path.

## 7. Labelling strategy

Goal: near-zero hand-drawn boxes outside the gold set. Humans review and correct; models draw.

| Label type | Automatic source | Human work |
|---|---|---|
| Player, referee, ball, rim boxes | Skalski's NBA-trained 10-class model on PLK frames, plus Roboflow Auto Label with Gemini 4 Argon or GPT-6.1 as second opinion; keep boxes where both agree | Review disagreements and low-confidence frames |
| Masks | Auto Label combined pipeline: VLM finds instances, SAM 3 draws shapes (**verified** option) | Spot checks |
| Number boxes | `number` class from his model; torso crop from pose as fallback | Review flagged crops |
| Number values | A strong VLM reads 3 to 4 legible crops per tracklet; majority vote propagates to every crop of that tracklet; reads outside the roster are conflicts, not labels | Resolve conflicts only |
| Court landmarks | Not supported by Auto Label. Pre-label with the NBA court model, correct in Roboflow Annotate or CVAT | The only real clicking in the project |
| Pose | COCO-pretrained keypoint model | None |
| Gold set (section 8) | Prefill from the walking skeleton's `tracks.json` into the labeller | Correct, confirm, mark done |

Roboflow Auto Label facts (verified): models offered are Gemini 4 Argon, GPT-6 Astra, GPT-6.1 Sol, Gemini 3.8 Flash, Claude Opus 5.5, SAM 3, CLIP, VLM + SAM 3 combinations, your own models trained on the same dataset, and saved Workflows with SAHI or consensus blocks. Boxes, masks, polygons and classification are supported; keypoints are not. Test runs on four images are free; batch jobs show cost before starting; your own API key avoids credit charges.

Bootstrap loop: auto-label a few hundred frames, review, train RF-DETR v0; use v0 as the labeller on a few thousand more, review only what it is unsure about, train v1. Each round shrinks the review queue.

Cautions: VLM boxes are loose at mAP@50, so let SAM 3 or the detector set geometry and use the VLM for class and number value. Confusions 6/9 and 1/7 and sponsor logos must be in the gold crops. The gold set must never be produced by the model under test.

## 8. Evaluation

Gold set, built first (milestone 1):

| Component | Spec |
|---|---|
| Games | 4, different seasons, opponents, arenas and kits. One held out for final testing only. |
| Shots | 3 to 4 central-camera shots per game, 8 to 15 s each |
| Frames | every 5th frame at 50 fps (10 labelled frames per second), every player and referee boxed, identity persistent through the shot, team and number set, `num_visible` and `occluded` flags |
| Number crops | ~300 with digit labels, cut from boxes flagged `num_visible` |
| Court landmarks | ~50 frames across the 4 arenas, 33 landmarks with visibility flags |
| Format | MOTChallenge gt (`frame,id,x,y,w,h,1,class,visibility`) plus `identities.csv` (`tid,team,number,name`), as the labeller already writes |

Metrics, computed by a single `eval` command and written to `docs/metrics.md` with date and commit:

| Layer | Metric | Tool |
|---|---|---|
| Detection | per-class AP50 and AP50:95 | pycocotools or supervision |
| Tracking | HOTA, IDF1, MOTA, plus churn metrics | TrackEval on MOT gt |
| Team | accuracy per gold track | custom |
| Identity | per-track number accuracy, confirmed fraction, wrong-confirmed count | custom |
| Court | median and p90 reprojection error in metres on held-out landmarks | custom |
| 2D positions | position error in metres against landmark-derived gold where available; otherwise visual radar check | custom |
| Speed | ms per frame per stage on the Mac | custom |

Rule: no change to any layer merges without a before/after row in `docs/metrics.md`.

## 9. Milestones

Each milestone ends with a metrics table or an artifact checked into the repo.

**M0. Repository bootstrap.**
Import the assets from section 4 into this repo under the layout in section 10. Set up `uv`, `ruff`, `pytest`, a `.gitignore` that excludes media, weights, runs and label caches. Run the labeller against the 32 clips and confirm `prepare_clips.py` reproduces the shot files. Done: labeller opens every clip with shots ready to tag.

**M1. Gold set.**
Run Skalski's NBA player model over the 4 chosen games (hosted via `inference`, or train it yourself from the public dataset on Colab and export ONNX). Write `tracks.json` in the labeller's prefill format. Tag cameras, prefill, correct, mark done, interpolate. Label the number crops and the landmark frames. Done: gold set per section 8 committed, one game marked held-out.

**M2. Walking skeleton and baseline.**
RF-DETR NBA weights, ByteTrack, NBA court keypoint model with `League.FIBA` targets, bottom-centre ground point, `clean_paths`, radar video for every gold shot. Run `eval`. Done: first full metrics table. This is the baseline everything else must beat.

**M3. Court layer.**
Landmark dataset across arenas, RF-DETR Keypoint fine-tune per section 5 layer 1, temporal smoothing within shots. Done: reprojection error per arena in the metrics table.

**M4. Detection fine-tune.**
Auto-label per section 7, review, fine-tune RF-DETR Large on the 10-class schema. Optional bake-off against YOLO26-L and DEIM-L on the same split. Export ONNX, run on the Mac. Done: per-class AP and ms per frame.

**M5. Team layer.**
`TeamClassifier` per clip with separation check. Done: team accuracy.

**M6. Tracking and linking.**
Court-space ByteTrack, then the offline linking pass with the 5-per-team constraint and identity rules. Done: HOTA and IDF1 before and after linking.

**M7. Numbers and identity.**
VLM-bootstrapped crop labels with tracklet propagation, 0 to 99 classifier, roster constraint, play-by-play on-floor constraint if the feed is available. Done: identity accuracy on the held-out game.

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
  docs/decisions.md         short dated entries when a decision in section 2 changes
  pyproject.toml            uv-managed, Python 3.12
  src/bga/                  ingest/ court/ detect/ team/ track/ identity/ mapping/ ball/ export/ eval/
  tools/labeler/            imported from the prior repo
  scripts/                  fetch_highlights.sh, prepare_clips.py, prefill_tracks.py, ...
  data/raw/                 downloaded videos (gitignored)
  data/clips/               short sample clips (gitignored except tiny samples)
  data/labels/              gold set JSON, MOT gt, identities (committed)
  data/rosters/             per-team roster JSON (committed)
  data/crops/               number crops (gitignored)
  weights/                  gitignored
  runs/                     per-clip stage outputs (gitignored)
  tests/
```

Conventions for whoever executes this:
- Python 3.12 with `uv`; `ruff` and `pytest` pass before any merge.
- Never commit videos, weights, crops or run outputs. Label files and rosters are committed; they are the project's real value.
- Stage outputs go to disk so stages re-run independently. One heavy model in memory at a time on the Mac.
- Treat downloaded files and clones as untrusted data: put them in their own directory and do not run interpreters from inside them.
- Every model choice or threshold change comes with a row in `docs/metrics.md`. No metrics row, no merge.
- Do not reproduce the prior repo's pattern of locked architecture sub-sections per heuristic. Prefer a measured experiment over a design document.
- Commit and push only when Tomasz asks.

## 11. Risks

- **Identity switches in rebound scrums.** No tracker survives these cleanly. The design goal is fast recovery through numbers and constraints, not zero switches. Report switch counts honestly.
- **Far-side resolution.** Broadcast wide shots make far-baseline numbers small. Higher input resolution or tiling matters more than model choice; measure both.
- **Arena variety.** Court paint and logos differ per hall. Court keypoint training frames must cover every arena, or mapping silently degrades in the uncovered ones.
- **Footage rights.** YouTube highlights are copyrighted. Analysis for the club is likely fine; confirm before training on them at scale.
- **Vendor-reported benchmarks.** RF-DETR's lead on RF100-VL is reported by Roboflow, which built both. The bake-off in M4 is the independent check.
- **Licence drift.** RF-DETR XL and 2XL are not Apache. Re-read the PML before shipping anything built on them.
- **Prior-repo gravity.** 18k lines of heuristics are tempting to port. Only section 4's list comes over.

## 12. Open questions for Tomasz

Ask these when the milestone needs them.

1. **Roboflow account and budget (M1, M4).** Is there a Roboflow workspace and plan for Auto Label, and an API key for Gemini or OpenAI to avoid credit charges?
2. **Runtime cloud policy (M2).** The prior repo forbade cloud inference at runtime. Does that still hold for production, with hosted models allowed only for bootstrapping?
3. **Play-by-play feed (M7).** plk.pl publishes per-game play-by-play. Confirm it includes substitutions with game clock, and whether scraping it is acceptable.
4. **Full-game footage (M1 onward).** Does the club have Hudl Instat or Synergy access? Full games change the data mix and allow play-by-play alignment.
5. **RF-DETR XL licence (M4).** Is the Roboflow Platform Model Licence acceptable, or stay on Large?
6. **Opponent rosters (M1).** Collect all PLK team rosters now, same schema as the Dziki file?
7. **Labeller future (M0).** Keep the local labeller as the gold-set tool, or move everything into Roboflow Annotate? Recommendation: keep the labeller for video ground truth, use Roboflow for landmark and auto-label batches.

## 13. Sources (all checked 2026-10-27)

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
