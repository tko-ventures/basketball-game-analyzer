# Layer 7 (`ball`): adopt `ball/fused` over `ball/skalski`; skip tiled ball inference; ship gold `v1/events.json` and `.ball.parquet`

- **Date:** 2026-11-24
- **Status:** Accepted
- **Area:** Pipeline (ball, eval, export)
- **Decider:** Tomasz / Implementer

## Context

M9 (Issue #10) adds Layer 7 (`ball`) for per-frame ball tracking, ball-holder possession, shot detection (made/missed, 2/3-point value, shooter), per-frame ball Parquet export, and radar overlays. The reference stack (`docs/PLAN.md` §3 & §5) proposed tiled inference if ball recall was low, nearest `player-in-possession` box matching for possession, holder-anchored 2D ball interpolation, and Roboflow `sports.basketball.ShotEventTracker` (`player-jump-shot` / `player-layup-dunk` + `ball-in-basket`) for made/missed shots.

Measuring `detect-plk-r3-poss` (`detect/rfdetr_torch/plk-r3-poss@c62744fe`) directly on `data/gold/v1/boxes.json` (dev split: 76 frames, 58 ball boxes, 37 `player-in-possession` boxes) showed:
- **Ball recall is already 91.4% at `conf >= 0.05`** (AP50 = 0.766), so tiled ball inference is unnecessary overhead.
- **Two PLK-specific failure modes hurt the reference baseline:** (a) static rim-relative false positives (arena fixtures/rim hardware firing repeatedly at constant offset from the rim); (b) `detect-plk-r3-poss` was intentionally trained on 6 active classes (`player-jump-shot`, `player-layup-dunk`, and `ball-in-basket` were dropped in the [2026-11-20 detector schema ADR](2026-11-20-detector-classes-and-possession-schema.md)), leaving `ShotEventTracker` largely blind on PLK shots.

## Decision

1. **Skip tiled ball inference:** Use the existing full-frame 1280 px `detect-plk-r3-poss` detections down to `conf >= 0.05` and filter persistent static rim-relative false positives temporally instead of paying multi-tile detector latency.
2. **Add Layer 7 (`ball`) with two swappable backends (`skalski` baseline vs. `fused` active):**
   - `ball/skalski` (`configs/ball-plk-skalski.yaml`): reference baseline matching `player-in-possession` boxes to nearest tracks and driving `sports.basketball.ShotEventTracker` scaled to effective FPS.
   - `ball/fused` (`configs/ball-plk-fused.yaml`, **active**): 4-stage tracker running in **0.36 ms/frame** on CPU:
     1. Rim selection + short-gap interpolation.
     2. Static rim-relative FP suppression + globally optimal DAG DP single-ball trajectory over `conf >= 0.05`.
     3. Viterbi HMM possession over `{non-referee tracks, None}` fusing `player-in-possession` cues, normalized ball-to-player proximity, airborne/free-ball cues, and temporal continuity.
     4. Rim-approach clustering + post-rim net-cylinder descent classification (restricted to rim neighborhood `dist_rw <= 3.0`) + FIBA 2/3-point geometry (`1.575 m` rim, `6.75 m` arc, `6.60 m` corner).
   - Both backends produce 2D FIBA ball coordinates (`x_m, y_m`) anchored to the holder when held and linearly interpolated through rim anchors while in flight.
3. **Add `data/gold/v1/events.json`, `bga eval-ball`, and `/events` verification UI:** Score ball tracking and possession against `data/gold/v1/boxes.json` and shot attempts (detection F1, outcome accuracy, and strict argmax-IoU `shooter_gt_tid` accuracy against `data/gold/v0`) against human-verified `data/gold/v1/events.json` (11 shots across the 9 dev sequences, verified via `tools/labeler/static/events.html`).
   - Dev benchmark (9 shots, 76 box frames, 11 verified shot events):
     - `ball-plk-skalski`: ball F1 **0.759** (R 0.76), poss acc **0.579**, holder F1 **0.533**, team acc **0.568**, shot F1 **0.400** (4/11), outcome acc **0.000**, shooter acc **0.667** (on 3 matched), **0.04 ms/frame**.
     - `ball-plk-fused`: ball F1 **0.794** (R 0.86), poss acc **0.618**, holder F1 **0.617**, team acc **0.703**, shot F1 **1.000** (11/11), outcome acc **0.818** (9/11), shooter acc **0.455** (5/11), **0.36 ms/frame**.
4. **Ship sibling `.ball.parquet` export and radar ball/shot overlays:** `bga export` writes `<seq>.ball.parquet` (`BALL_SCHEMA`: `frame_idx, x_m, y_m, holder_track_id, in_flight`) alongside the 17-column player Parquet when `ball` is configured; `bga radar` draws the orange ball dot, holder highlight ring, and cumulative made/missed shot markers.

## Consequences

- `configs/ball-plk-fused.yaml` is the active end-to-end PLK stack through Layer 7; all upstream Layer 0–6 cache hashes remain unchanged.
- `bga eval-ball` adds a `## Ball, possession & shots layer` table in `docs/metrics.md`.
- Human review of the 9 dev shots surfaced targeted edge cases (crowded-paint player/ball misses in `OSp445e0mXU_s10` `[3320, 3475]` and `OSp445e0mXU_s2` `[1055, 1145]`, alley-oop shooter handoff in `OSp445e0mXU_s2` `[740, 760]`, and ball tracking around rim contact in `1jDaJBkUGmc_s7` `[2920, 2960]`, plus contested-shooter defender overlap), tracked in a post-M9 fine-tuning issue.
- Held-out `Tnuscrm9RyU` remains untouched until explicit sign-off for the one-shot milestone close run.
