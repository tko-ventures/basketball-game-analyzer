# `plk-r4-l4` and `ball-plk-fused-v6-r4` become the active detector and Layer 7 config

- **Date:** 2026-11-27
- **Status:** Accepted (Tomasz, 2026-11-27) on dev; held-out scored once with his approval the same day (mixed, below)
- **Superseded in part by:** [2026-11-27-fused-v8-r4-active.md](2026-11-27-fused-v8-r4-active.md) (the active Layer 7 config; the detector decision stands)
- **Area:** Pipeline (detect, L2; ball, L7)
- **Decider:** Tomasz / Implementer
- **Supersedes:** [2026-11-26-detector-r4-kept-as-comparator.md](2026-11-26-detector-r4-kept-as-comparator.md)

## Context

On 2026-11-26, `plk-r4-l4` stayed a comparator. Its gains were one shot per split on 11 dev and 5 held-out shots, which that gold could not resolve. Epic #46 then grew the dev gold to 67 shots, 477 holder frames, 60 field goals and 6 free throws. Of those shots, 40 come from the full-length Gliwice game, the M10 footage. On the same epic, Layer 7 went from v2 to v6:
- v2-loose: the ball is released around rim arrivals;
- v4: rim bonus in the ball tracker, and strict arc for rim arrivals;
- v5: a far tracker jump no longer calls a shot missed;
- v6: free-throw tag.

## Evidence (dev, `docs/metrics.md` 2026-11-26 and 2026-11-27)

| | ball F1 | poss acc | holder F1 | team acc | shot F1 (P/R) | outcome acc (made / missed R) | shooter acc | FT F1 |
|---|---|---|---|---|---|---|---|---|
| active `ball-plk-fused-v2` (r3) | 0.857 | 0.706 | 0.741 | 0.869 | 0.873 (0.83/0.92) | 0.709 (0.73/0.68) | 0.709 | – |
| `ball-plk-fused-v6` (r3) | 0.863 | 0.755 | 0.769 | 0.861 | 0.898 (0.85/0.95) | 0.789 (0.89/0.64) | 0.754 | 0.857 |
| `ball-plk-fused-v6-r4` | 0.875 | 0.790 | 0.808 | 0.903 | 0.903 (0.88/0.93) | 0.750 (0.83/0.62) | 0.804 | 0.923 |

- **Full-game shots only** (shot F1 / outcome / shooter): r3 0.848 / 0.750 / 0.714, r4 0.839 / 0.731 / 0.885.
- **Highlight shots only:** r3 0.951 / 0.828 / 0.793, r4 0.968 / 0.767 / 0.733.
- **Rest of the stack, same configs** (9 gold shots):
  - HOTA 0.740 → 0.739;
  - IDF1 0.866 → 0.864;
  - IDSW 57 → 67;
  - number acc 0.870 → 0.899.
- **Detector AP50 on dev:**
  - ball 0.787 → 0.810;
  - possession 0.576 → 0.598;
  - referee 0.866 → 0.854.

## Decision

1. **`detect-plk-r4` (`plk-r4-l4@c5abaa28`) is the active detector.**
2. **`ball-plk-fused-v6-r4` is the active Layer 7 config.** It carries over r4's two thresholds (`ball_min_conf` 0.10, `static_cluster_px` 8.0) from `ball-plk-fused-v3-r4`.
3. **`plk-r3-poss`, `ball-plk-fused-v2` and `ball-plk-fused-v6` stay as comparators.**

`v6-r4` beats the active `v2` on every Layer 7 column. Against `v6` on r3, it wins possession, holder, team, shooter and free throws, and gives up outcome by 0.04. Outcome comes from Layer 7, not the detector, so it is the next Layer 7 work (#48: missed recall 0.62).

## Consequences

- `docs/architecture.md` and `docs/architecture-c4-view.md` list r4 and v6-r4. The link and identity benchmarks there are still the r3 numbers; on r4 tracking is flat and IDSW rises by 10 on dev.
- Held-out, scored once with Tomasz's approval (2026-11-27): 31 shots, 233 frames, 35 field goals and 6 free throws, from `Tnuscrm9RyU` and the PGE Start full game.

  | | ball F1 | poss acc | holder F1 | team acc | shot F1 (P/R) | outcome acc | shooter acc | FT F1 (P/R) |
  |---|---|---|---|---|---|---|---|---|
  | `ball-plk-fused-v2` (r3) | 0.867 | 0.704 | 0.747 | 0.897 | 0.842 (0.78/0.91) | 0.844 | 0.719 | – |
  | `ball-plk-fused-v6` (r3) | 0.874 | 0.747 | 0.770 | 0.889 | 0.842 (0.78/0.91) | 0.844 | 0.750 | 0.444 (0.67/0.33) |
  | `ball-plk-fused-v6-r4` | 0.908 | 0.777 | 0.799 | 0.865 | 0.785 (0.70/0.89) | 0.871 | 0.774 | 0.400 (0.50/0.33) |

  - **Held on held-out:** the gains in ball, possession, holder and shooter, and on held-out outcome too.
  - **Did not hold:**
    - shot F1 (44 predicted shots against 41, for 35 field goals);
    - team accuracy;
    - free-throw recall: 2 of 6, against 6 of 6 on dev.
- Referee AP falls slightly on both splits (held-out 0.810 → 0.743, scored 2026-11-26). Referees are excluded from holders and teams, so Layer 7 does not depend on it.
