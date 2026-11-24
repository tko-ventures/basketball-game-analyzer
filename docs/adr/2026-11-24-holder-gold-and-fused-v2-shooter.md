# Holder gold under the 2026-11-20 definition; `ball-plk-fused-v2` for shooter attribution

- **Date:** 2026-11-24
- **Status:** Proposed
- **Area:** Data & Eval / Pipeline (ball, L7)
- **Decider:** Tomasz / Implementer

## Context

Post-M9 issue #40: `ball-plk-fused` scored shooter accuracy 0.455 (5/11) and frame possession 0.592 on dev, and on the held-out game possession was below `ball/skalski`, which looked like the Viterbi holding on during passes. Auditing every error frame showed two separate causes:
1. **Gold.** Box gold v1 (2026-11-18) predates the [2026-11-20 possession schema](2026-11-20-detector-classes-and-possession-schema.md). Shooters still holding the ball are boxed `player-jump-shot` / `player-layup-dunk` and were never holders, and some clear holds were left as `player`. Tuning against that gold would teach the model to drop real holds.
2. **Shooter search.** The shooter was the last holder at or before the rim frame. With the ball at the rim, possession goes to whoever stands under the basket, so 3 of the 6 wrong shooters had their release at the rim frame. A defender whose box overlaps the shooter at release took 1 more.

## Decision

1. **`data/gold/v1/holders.json`** ([gold.md](../gold.md#holder-gold-v1-2026-11-24)): each box-gold frame's holder under the 2026-11-20 definition, as a box index. Box gold is untouched. The implementer's first pass changed 11 of the 100 frames and flagged 10. Tomasz answered all 21 on the labeller's `/holders` page, keeping the 5 shooters and the 1 ball in flight and overruling the 5 holds called from a single frame. In the end 9 frames differ from box gold. `bga eval-ball` scores possession against it.
2. **`configs/ball-plk-fused-v2.yaml`** sets three opt-in `ball/fused` params: `shot_release_rim_excl_rw: 2.0` (search for the release before the ball enters 2 rim widths of the rim), `shot_team_window_s: 2.0` (the team holding most over 2 s wins a contested release), and `ball_far_penalty: 1.5` (a tracked ball far from a player counts against that player holding it). Their defaults are off, so `ball-plk-fused` keeps its hash and outputs.

Dev (final holder gold, commit `bc3b98d`): shooter accuracy 0.455 → **0.727** (8/11). Possession 0.645 → 0.658, holder F1 0.686 → 0.699. Shot F1 (1.000) and outcome accuracy (0.727) are unchanged. Plateaus: rim exclusion from 1.5 to 4 rim widths, team window 2–3 s.

## Consequences

- Held-out game `Tnuscrm9RyU` (scored once with Tomasz's approval, 2026-11-24, final holder gold): possession 0.667 → **0.750** and holder F1 0.743 → **0.788** against `ball-plk-fused` (`ball-plk-skalski`: 0.750 / 0.759). Shot F1 (1.000), outcome accuracy (1.000) and shooter accuracy (0.600, 3/5) are unchanged; the two wrong shooters are a crowded layup whose holder is lost before release and a jump shot with the wrong player boxed at release, both detection or tracking misses.
- To accept: Tomasz's sign-off. The held-out game has been scored and the holder questions are answered.
- On acceptance, `ball-plk-fused-v2` replaces `ball-plk-fused` as the active Layer 7 config in `docs/architecture.md`.
- Remaining shooter misses (an alley-oop, two crowded-paint layups) and outcome misses come from missed ball or player detections, so they go to a detector round (#40 `plk-r4`), not more Layer 7 logic.
