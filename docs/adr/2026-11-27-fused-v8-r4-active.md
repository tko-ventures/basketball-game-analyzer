# `ball-plk-fused-v8-r4` becomes the active Layer 7 config

- **Date:** 2026-11-27
- **Status:** Accepted (Tomasz, 2026-11-27) on dev; held-out not scored (already scored once this milestone)
- **Area:** Pipeline (ball, L7)
- **Decider:** Tomasz / Implementer
- **Supersedes:** the Layer 7 part of [2026-11-27-detector-r4-and-fused-v6-active.md](2026-11-27-detector-r4-and-fused-v6-active.md). Its detector decision (`plk-r4-l4`) stands.

## Context

The held-out run of `ball-plk-fused-v6-r4` (2026-11-27) held its ball, possession, holder, shooter and outcome gains. Two things did not hold:
- **Shot precision:** 0.70, from 44 predicted for 35 field goals.
- **Free-throw recall:** 2 of 6.

The held-out errors were not inspected. Instead, the 8 dev false shots of `v6-r4` were read:
- 2 had no shooter;
- 2 were the right shot with the rim frame on a second contact;
- 4 were a ball near the rim with no attempt.

Three `fused` options followed, each off by default:
- **`shot_require_shooter`:** a rim arrival with no shooter is no shot.
- **`shot_rim_pick: first`:** the reported rim frame is the first contact with the rim or board, as shot gold defines it. Outcome and shooter are still read from the step nearest the rim.
- **`ft_scene_min_window_s`:** when the camera cuts to the free-throw line less than `ft_scene_window_s` before the release, the stillness window starts at the cut, as long as this many seconds remain.

`ball-plk-fused-v7-r4` is `v6-r4` with the first two options. `ball-plk-fused-v8-r4` adds `ft_scene_min_window_s: 0.5`.

Tomasz then labelled a third full game as dev gold: Energa Trefl Sopot – Dziki (`vHfwZeG4BPw`, 2026-01-31, away arena). It has 40 shots, with 29 field goals and 10 free throws. Dev now holds 107 shots, 89 field goals and 16 free throws, from highlights and two full games.

## Evidence (dev, `docs/metrics.md` rows of 2026-11-27)

| | ball F1 | poss acc | holder F1 | team acc | shot F1 (P/R, pred/gt) | outcome acc (made / missed R) | shooter acc | FT F1 (P/R) | attempt F1, any kind (P/R) |
|---|---|---|---|---|---|---|---|---|---|
| `ball-plk-fused-v6-r4` | 0.884 | 0.766 | 0.790 | 0.894 | 0.860 (0.82/0.90, 97/89) | 0.688 (0.76/0.58) | 0.750 | 0.750 (0.75/0.75) | 0.846 (0.80/0.89) |
| `ball-plk-fused-v7-r4` | 0.884 | 0.766 | 0.790 | 0.894 | 0.895 (0.88/0.91, 92/89) | 0.667 (0.75/0.55) | 0.753 | 0.800 (0.86/0.75) | 0.893 (0.88/0.91) |
| **`ball-plk-fused-v8-r4`** | 0.884 | 0.766 | 0.790 | 0.894 | **0.910 (0.91/0.91, 89/89)** | 0.667 (0.75/0.55) | 0.753 | **0.909 (0.88/0.94)** | **0.893 (0.88/0.91)** |

- **On the Trefl game alone**, labelled after `v7-r4` was fixed and never used to tune its options, `v6-r4` → `v7-r4` scores:
  - field-goal F1 0.774 → 0.814 (P 0.73 → 0.80);
  - attempt F1 0.840 → 0.895;
  - shooter 0.625 → 0.667.

  The precision gain carries to a game it was not tuned on.
- **Outcome:** across matched full-game shots, `v7-r4` changes only rim frames, which move closer to gold, and no outcome call. The 0.02 outcome dip is one Trefl attempt, `s202`. There the first-contact frame now matches a missed shot instead of the made putback 0.75 s later.
- **`ft_scene_min_window_s: 0.5` was picked on dev from three values:**

  | Value | FT F1 |
  |---|---|
  | 0.5 | 0.909 |
  | 0.75 | 0.875 |
  | 1.0 | 0.800 |

  For this option the Trefl game is therefore not unseen.
- **Layers 1–6 are unchanged.** The ball, possession, holder and team columns are identical across the three configs, because the options act only on shots.

## Decision

1. **`ball-plk-fused-v8-r4` is the active Layer 7 config**, on the active detector `plk-r4-l4`.
2. **Comparators:** `ball-plk-fused-v6-r4` and `ball-plk-fused-v7-r4` stay as comparators, alongside `ball-plk-fused-v2` and `ball-plk-fused-v6` on r3.
3. **Held-out:** not scored again this milestone. The next held-out score of the active Layer 7 config waits for the next milestone and Tomasz's approval.

## Consequences

- **Docs:** `docs/architecture.md` and `docs/architecture-c4-view.md` list `ball-plk-fused-v8-r4`. Their held-out numbers stay those of `v6-r4`, the last config scored there.
- **Held-out evidence:** the held-out shot-precision and free-throw problems that started this work have dev evidence of a fix but no held-out confirmation yet.
- **Weaknesses:** outcome is now the weakest Layer 7 column (0.667 on dev, 0.50 on the Trefl game).

## Known shortcomings and directions for future ADRs

Each item states the shortcoming, the evidence, and directions a later ADR could decide between. Every direction would come in as a backend option or a new backend with a config and metrics rows, never as a branch in the pipeline.

1. **Made/missed outcome (#48).**
   - **Evidence:** outcome is 0.667 on dev and 0.50 on the Trefl game. Missed recall is 0.55, and made shots are often read as missed (and the reverse) when the ball is lost around the rim. The rule reads the tracked ball below the ring.
   - **Directions:**
     - a net-motion or ball-through-ring cue from a crop under the detector's rim box: a small open-weight classifier fine-tuned on labelled attempts, or pixel change in the net;
     - the detector's `ball-in-basket` class with its own threshold;
     - a learned classifier over rim-relative trajectory features, trained on dev attempts other than the scored ones;
     - a score-change check from the broadcast scorebug, which is offline-safe and confirms a make a few seconds later.
2. **Misses followed by putbacks or tips.**
   - **Evidence:** a rim-arrival cluster can hold two attempts within `shot_cooldown_s` (1.2 s). Then one prediction covers both, and only one matches (`vHfwZeG4BPw_s202`).
   - **Directions:**
     - split a cluster when the ball leaves the rim zone upward and a player touches it before it returns;
     - a shorter cooldown once the ball rises again after a contact.
3. **Ball near the rim with no attempt.**
   - **Evidence:** this was 4 of the 8 dev false shots before `v7`: rebounds, passes across the rim and dead-ball tosses after a whistle. The current rules see only geometry.
   - **Directions:**
     - require an upward release from the shooter's box toward the rim;
     - a learned accept/reject model on rim arrivals;
     - a "clock stopped" signal from the scorebug or a still scene to drop dead-ball tosses.
4. **Layups with the ball hidden until the rim.**
   - **Evidence:** when the ball is in the shooter's hands right up to contact, no arrival from outside the rim zone is seen. The true contact is not counted, and a later tracker jump can create a late rim frame (`hSXPCy5NedU_s582`).
   - **Directions:** accept an arrival when the holder's box overlaps the rim zone and the ball appears at the rim, with no approach from afar.
5. **Free-throw tag.**
   - **Evidence:** 15 of 16 dev free throws are found. The one miss had players still moving at the release estimate, and held-out recall was 2 of 6 before the window fix. The tag relies on scene stillness alone, so other set plays after a stoppage can pass. The one false tag on dev is a sideline inbound lob that dropped near the rim (`vHfwZeG4BPw_s135`).
   - **Directions:**
     - add the shooter's court position at the free-throw line, which also rules out inbounds from the sideline;
     - a stopped game clock from the scorebug;
     - pair consecutive free throws in one stoppage.
6. **Team accuracy and identity on r4.**
   - **Evidence:** on held-out, team accuracy fell from 0.889 to 0.865 between `v6` and `v6-r4`, and IDSW rose by 10 on dev tracking with r4. Neither is a Layer 7 change.
   - **Directions:** a team-layer or tracker ADR re-tuned for r4 detections, measured on tracking gold.
7. **Evaluation capacity.**
   - **Evidence:**
     - Dev precision differences are 2 to 5 shots.
     - Held-out has one full game (20 shots) plus highlights, and is now spent for this milestone.
     - Cached outputs from changed code were reused silently twice (#52).
   - **Directions:**
     - a fourth full game as a fresh held-out game before the next milestone score;
     - more full-game dev gold where outcome and free throws are the target;
     - the code hash in each layer's cache key (#52).
