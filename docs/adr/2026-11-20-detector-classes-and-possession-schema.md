# Six active box classes, split visual holder from `dead_ball`, adopt `plk-r3-poss`

- **Date:** 2026-11-20
- **Status:** Accepted
- **Area:** Pipeline (detect / M9)
- **Decider:** Tomasz

## Context

Labelling the 643-frame possession queue revealed two issues with the 10-class detector schema: (1) four shot/action classes (`player-jump-shot`, `player-layup-dunk`, `player-shot-block`, `ball-in-basket`) are rare temporal moments scoring 0–0.25 AP50; (2) a single-frame detector cannot distinguish live possession from an inbounder behind the end line or a dead ball.

## Decision

1. **Six actively labelled box classes:** `player`, `referee`, `ball`, `rim`, `number`, and `player-in-possession` (all visible in a single frame).
2. **Four action classes kept in schema, no longer labelled:** `player-jump-shot`, `player-layup-dunk`, `player-shot-block`, `ball-in-basket` remain in `bga.types` so existing weights stay comparable and M9 can benchmark Skalski's class-based `ShotEventTracker` against a ball-trajectory + court-release backend.
3. **Possession split into two orthogonal labels:**
   - **Visual `holder` (`player-in-possession`):** Whoever holds or dribbles the ball (mid-bounce included, plus inbounders; a shooter still holding the ball keeps the shot class and is marked `holder`). A ball in flight has no holder.
   - **Frame-level `dead_ball` flag:** Independent boolean for out-of-play states (throw-in, referee holding ball, post-score). Consumed by M9 evaluation, not by the single-frame detector.
4. **Active downstream detector:** Adopt `configs/detect-plk-r3-poss.yaml` (`detect/rfdetr_torch/plk-r3-poss@c62744fe`) for M5+, lifting `player-in-possession` AP50 from 0.256 (`plk-r2`) to **0.559** while improving people (0.966), number (0.848), and referee (0.866) AP50.

## Consequences

- All M5/M6/M7 configs build on `detect-plk-r3-poss`.
- M9 starts with live/dead ball, team possession, shots, rebounds, passes, and turnovers; fouls are not read from video. Play extraction (pick-and-roll coverages, etc.) waits for coach Marco's priority list.
