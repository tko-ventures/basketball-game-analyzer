# Layer 7 (`ball`): adopt `ball/fused` over `ball/skalski`; skip tiled ball inference; ship gold `v1/events.json` and `.ball.parquet`

- **Date:** 2026-11-24
- **Status:** Accepted
- **Area:** Pipeline (ball, eval, export)
- **Decider:** Tomasz / Implementer

## Context

M9 (Issue #10) adds Layer 7 (`ball`) for per-frame ball tracking, ball-holder possession, shot detection (made/missed, 2/3-point value, shooter), per-frame ball Parquet export, and radar overlays. The reference stack (`docs/PLAN.md` §3 & §5) proposed tiled inference if ball recall was low, nearest `player-in-possession` box matching for possession, holder-anchored 2D ball interpolation, and Roboflow `sports.basketball.ShotEventTracker` (`player-jump-shot` / `player-layup-dunk` + `ball-in-basket`) for made/missed shots.

Measuring `detect-plk-r3-poss` (`detect/rfdetr_torch/plk-r3-poss@c62744fe`) directly on `data/gold/v1/boxes.json` (dev split: 76 frames, 58 ball boxes, 37 `player-in-possession` boxes) showed:
- **Ball recall is already 91.4% at `conf >= 0.05`** (AP50 = 0.766), so tiled ball inference is unnecessary overhead.
- **Two PLK-specific failure modes hurt the reference baseline:** (a) static rim-relative false positives (arena fixtures/rim hardware firing repeatedly at constant offset from the rim); (b) `detect-plk-r3-poss` was intentionally trained on 6 active classes (`player-jump-shot`, `player-layup-dunk`, and `ball-in-basket` were dropped in the [2026-11-20 detector schema ADR](2026-11-20-detector-classes-and-possession-schema.md)), leaving action-based shot tracking largely blind to made/missed outcomes on PLK shots.

## Decision

1. **Skip tiled ball inference:** Use the existing full-frame 1280 px `detect-plk-r3-poss` detections down to `conf >= 0.05` and filter persistent static rim-relative false positives temporally instead of paying multi-tile detector latency.
2. **Add Layer 7 (`ball`) with two swappable backends (`skalski` baseline vs. `fused` active):**
   - `ball/skalski` (`configs/ball-plk-skalski.yaml`): reference baseline matching `player-in-possession` boxes to nearest non-referee tracks and driving a self-contained, stride-invariant equivalent of `sports.basketball.ShotEventTracker` (avoiding the ~5 s top-level `sports` import cost on every `bga` CLI invocation). Missed shots emit `rim_frame: None` so timeout deadlines do not distort 2D interpolation or shot matching.
   - `ball/fused` (`configs/ball-plk-fused.yaml`, **active**): 4-stage tracker running in **0.17 ms/frame** on CPU with all temporal windows parameterized in seconds (stride-invariant across stride 5 and stride 1):
     1. Rim selection + short-gap interpolation (`rim_max_gap_s`).
     2. Static rim-relative FP suppression (excluding player-attached candidates from forming static clusters and requiring co-occurrence with another ball or dominant rim-frame share, so held balls survive) + multi-segment DAG DP ball trajectory over `conf >= 0.05` with restart transitions across occlusions longer than `ball_max_gap_s`.
     3. Viterbi HMM possession over `{non-referee tracks, None}` (using shared `REF_TEAM = "ref"` from `bga.team.base`) fusing `player-in-possession` cues, normalized ball-to-player proximity, airborne/free-ball cues, and temporal continuity.
     4. Rim-approach clustering (`shot_approach_s`, `shot_arc_s`) + post-rim net-cylinder descent classification + FIBA 2/3-point geometry (`1.575 m` rim, `6.75 m` arc, `6.60 m` corner).
   - Both backends produce 2D FIBA ball coordinates (`x_m, y_m`) anchored to the holder when held and linearly interpolated through rim anchors while in flight (bounded to gaps with ball evidence or brief occlusions, leaving ungrounded frames `None`, and setting `in_flight` only when a tracked ball exists).
3. **Add `data/gold/v1/events.json`, `bga eval-ball`, and `/events` verification UI:** Score ball tracking and possession against `data/gold/v1/boxes.json` and shot attempts (detection F1, outcome accuracy, and strict argmax-IoU `shooter_gt_tid` accuracy against `data/gold/v0`) strictly on sequences listed in `verified_sequences` of `data/gold/v1/events.json` (11 dev shots + 5 held-out shots verified via `tools/labeler/static/events.html`), matching on `rim_frame` (or `start_frame` vs. gold `release_frame` when `rim_frame` is `None`).
   - Dev benchmark (commit `0737457`, 9 shots, 76 box frames, 11 verified shot events):
     - `ball-plk-skalski`: ball F1 **0.759** (R 0.76, med 1.3 px), poss acc **0.566**, holder F1 **0.526**, team acc **0.568**, shot F1 **0.696** (P 0.67 / R 0.73, 12/11), outcome acc **0.250**, shooter acc **0.625**, **0.05 ms/frame**.
     - `ball-plk-fused`: ball F1 **0.787** (R 0.86, med 1.7 px), poss acc **0.592**, holder F1 **0.646**, team acc **0.892**, shot F1 **1.000** (P 1.00 / R 1.00, 11/11), outcome acc **0.727** (8/11), shooter acc **0.455** (5/11), **0.17 ms/frame**.
4. **Ship sibling `.ball.parquet` export and radar ball/shot overlays:** `bga export` writes `<seq>.ball.parquet` (`BALL_SCHEMA`: `frame_idx, x_m, y_m, holder_track_id, in_flight`) alongside the 17-column player Parquet when `ball` is configured; `bga radar` draws the orange ball dot, holder highlight ring, and cumulative made/missed shot markers.

## Consequences

- `configs/ball-plk-fused.yaml` is the active end-to-end PLK stack through Layer 7; all upstream Layer 0–6 cache hashes remain unchanged.
- `bga eval-ball` adds a `## Ball, possession, and shots` table in `docs/metrics.md` and skips unverified `events.json` sequences with a warning.
- Human review of the 9 dev shots surfaced targeted edge cases (crowded-paint player/ball misses in `OSp445e0mXU_s10` `[3320, 3475]` and `OSp445e0mXU_s2` `[1055, 1145]`, alley-oop shooter handoff in `OSp445e0mXU_s2` `[740, 760]`, and ball tracking around rim contact in `1jDaJBkUGmc_s7` `[2920, 2960]`, plus contested-shooter defender overlap), tracked in post-M9 follow-up issue #40.
- **Held-out game (`Tnuscrm9RyU`, 3 shots, 24 box frames, 5 verified shot events, commit `0737457` re-score after PR #41 review fixes):** `ball-plk-fused` achieves **0.870** ball F1 (R 0.91, med 1.5 px, matching `skalski`), **0.846** team acc (vs. **0.769** `skalski`), **1.000** shot F1 (5/5, P=1.00, R=1.00, vs. **0.667** `skalski` at 10/5), **1.000** outcome acc (5/5, vs. **0.800** `skalski`), and **0.600** shooter acc (matching `skalski`) at **0.18 ms/frame**; frame-level possession accuracy remains higher on `skalski` (**0.750**, holder F1 **0.714**) than `fused` (**0.542**, holder F1 **0.647**), confirming Viterbi/proximity over-holding during passes and rebounds as a concrete target for follow-up issue #40.
