# Detector round 4 (`plk-r4-l4`) stays a comparator; `plk-r3-poss` and `ball-plk-fused-v2` remain active

- **Date:** 2026-11-26
- **Status:** Proposed (the implementer's recommendation; Tomasz decides between this and promoting `plk-r4-l4`)
- **Area:** Pipeline (detect, L2; ball, L7)
- **Decider:** Tomasz / Implementer

## Context

Issue #40 sent the remaining shooter and outcome misses to a detector round. `plk-r4-l4@c5abaa28` adds 228 hard
frames (crowded paint, airborne shooters, unsure balls) to the `plk-r3-poss` data and recipe, trained on a Colab L4
(batch 2 x gradient accumulation 8, one seed). Swapping it into the active stack lost 3 of 11 dev shots, because its
scores run higher at the low end, so `ball-plk-fused-v3-r4` re-tunes two Layer 7 thresholds on dev (`ball_min_conf`
0.05 to 0.10, `static_cluster_px` 14.0 to 8.0). Rows are in `docs/metrics.md`.

| | dev, active | dev, r4 + v3 | held-out, active | held-out, r4 + v3 |
|---|---|---|---|---|
| ball AP50:95 / rim AP50:95 | 0.486 / 0.560 | 0.513 / 0.600 | 0.538 / 0.779 | 0.551 / 0.758 |
| in possession AP50 / referee AP50 | 0.576 / 0.866 | 0.598 / 0.854 | 0.846 / 0.810 | 0.791 / 0.743 |
| shot F1, outcome, shooter | 11/11, 8/11, 8/11 | 11/11, 9/11, 9/11 | 5/5, 5/5, 3/5 | 5/5, 5/5, 4/5 |
| ball F1 / poss acc / holder F1 | 0.791 / 0.658 / 0.699 | 0.780 / 0.684 / 0.738 | 0.870 / 0.750 / 0.788 | 0.889 / 0.625 / 0.688 |
| number acc / IDSW | 0.870 / 57 | 0.899 / 67 | 0.846 / 25 | 0.885 / 20 |

## Decision

1. **`plk-r3-poss` stays the active detector and `ball-plk-fused-v2` the active Layer 7 config.** The gains of
   `plk-r4-l4` are one shot per split (shooter on both, outcome on dev) and one to two jersey tracks; held-out frame
   possession falls by 3 of 24 frames (team accuracy by 2 of 14) and referee AP falls on both splits. Every one of
   these differences is within what 11 dev shots, 5 held-out shots and 24 held-out frames can resolve, so there is
   no clear win to promote, and the largest single move is a loss on a metric #40 names.
2. **`plk-r4-l4`, `detect-plk-r4`, `ball-plk-fused-v2-r4` and `ball-plk-fused-v3-r4` stay in the repo as
   comparators**, with their rows, so the round is not re-run.
3. **`ball_min_conf` 0.10 is not carried over.** It sits on a narrow margin (above 0.11 one dev shot is lost) and on
   `plk-r3-poss` it loses a shot (10/11).
4. **`static_cluster_px` 8.0 is a separate Layer 7 candidate.** On the active detector alone
   (`ball-plk-fused-v2-static8`, dev) it lifts ball F1 0.791 to 0.822, possession 0.658 to 0.684 and holder F1 0.699
   to 0.713 with shots unchanged, which is most of what `ball-plk-fused-v3-r4` gains on those three. It has not been
   scored on held-out and is not active.

Rejected: promoting `plk-r4-l4` with `ball-plk-fused-v3-r4` now, on the strength of the shooter and number gains.

## Consequences

- `docs/architecture.md` is unchanged for the detect and ball layers.
- Promoting `static_cluster_px` 8.0 needs one held-out score of `ball-plk-fused-v2-static8` with Tomasz's approval.
- The held-out comparison changes the detector and both thresholds at once. Telling them apart on held-out would
  take further held-out scores, which this ADR does not ask for.
- A further detector round is unlikely to be decidable on this gold: one shot moves dev shooter accuracy by 0.09
  and held-out by 0.20. More gold shots and holder frames come before another round.
- The Colab runner changes of this branch (detached trainer, reconnecting polls, retried downloads) are independent
  of this decision.
