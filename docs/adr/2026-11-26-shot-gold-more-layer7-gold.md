# Shot gold: 26 more shots of the gold games, labelled for the ball layer only

- **Date:** 2026-11-26
- **Status:** Accepted (Tomasz chose to label more gold, 2026-11-26; the design is the implementer's)
- **Area:** Data & Eval / Pipeline (ball, L7)
- **Decider:** Tomasz / Implementer

## Context

Detector round 4 could not be judged ([ADR](2026-11-26-detector-r4-kept-as-comparator.md)): gold v1 scores Layer 7 on
11 dev and 5 held-out shot attempts and on 76 dev and 24 held-out frames, so one shot moves shooter accuracy by 0.09
on dev and 0.20 on held-out, and 3 frames move held-out possession by 0.125. Full gold (tracks, tight boxes of every
class) costs hours per shot. Layer 7 needs far less: the attempts of a shot, and the ball and the holder on a frame.

## Decision

1. **A separate, lighter gold set, "shot gold"**, in one working file, `data/labels/shot_gold.json`
   (`bga.eval.shot_gold`): per shot its attempts (release frame, rim frame, outcome, points, shooter box and team),
   and on one frame a second the ball box and the holder's box and team, or nobody. No tracks, no other classes.
2. **Shots come from the four gold games only**: 6 from each dev game and 8 from the held-out game, central, at
   least 6 s long, not already in gold, spread evenly over the clip by shot index. The leakage guard already refuses
   every frame of these clips for training, so no training set changes, and the held-out game stays held out. The
   choice reads no model output, so it cannot favour shots the pipeline handles well.
3. **`shots/shot_gold`** is the shots backend over that file, so any run config can be pointed at the new sequences.
4. **Labelling** is on the labeller's `/shotgold` page. The boxes offered for clicking are the active detector's
   (`configs/shot-gold-prefill.yaml`, a labelling aid); the implementer's first pass proposes every label from
   contact sheets with the neighbouring frames in view; Tomasz confirms every shot's attempts and answers the frames
   the first pass could not settle. A frame or a shot counts as gold only once Tomasz has confirmed it.

Rejected: more full gold shots (too slow for what Layer 7 needs); shots from new games (each would have to leave the
training pool, and the held-out game would stay at 3 shots); picking shots where the pipeline finds a shot (biases
shot recall upwards).

## Consequences

- Holder and shooter boxes are the detector's unless Tomasz draws one, so their geometry leans towards the model.
  Scoring matches a predicted box at IoU 0.5 or 0.4, where that matters little; ball boxes snap to a detector box only
  when Tomasz clicks on it.
- The held-out shots were run through the detector to offer boxes. No held-out score was read from that run.
- Once labelled: an export to `data/gold/v1/`, `bga eval-ball` reading both sets, and a before row for the active
  config on the larger set. Shooter matching needs the shooter's box instead of a gold track id.
- `static_cluster_px` 8.0 and any later detector round are then judged on about 3 times the attempts and frames.
