# Layer 0 ingest: `shots/video` with TransNet V2 cuts (threshold 0.3) and the `siglip_game` camera check

- **Date:** 2026-11-27
- **Status:** Accepted on dev; held-out not scored yet
- **Area:** Pipeline (shots, L0)
- **Decider:** Implementer
- **Builds on:** [2026-11-23-highlights-footage-and-m10-ingest-milestone.md](2026-11-23-highlights-footage-and-m10-ingest-milestone.md)

## Context

Until M10 (#32), every sequence came from hand-labelled shot files (`shots/gold`, `shots/clips`, `shots/shot_gold`). A raw
full game could not run without someone first cutting it into shots and tagging the main-camera ones.

Layer 0 now has its own backend, `shots/video`. It cuts each raw video with a cut detector, scores every shot with a
camera classifier, and keeps the central shots of at least `min_s` as `<ytid>_v<i>` sequences. Both parts are
registered backends (pseudo-layers `cuts` and `camera`), chosen in the config, and cached per video.

Two choices had to be made: the cut detector and the camera classifier.

## Options

**Cut detector:**
- **`hist_diff`:** the prior labeller's HSV histogram detector, which cut every label file so far.
- **`transnetv2`:** TransNet V2 (Souček and Lokoč, 2020), official MIT weights from soCzech/TransNetV2 at `85cef72a`.
  The weights are converted locally from TensorFlow (`scripts/convert_transnetv2.py`, within 3e-7 of TensorFlow) into
  the vendored, unchanged PyTorch model. Its separable 3D convolutions run as 2D convolutions with identical output:
  1,314 frames/s on M1 Max MPS, about 12 min for a two-hour game.

**Camera classifier:**
- **`siglip_zeroshot`:** SigLIP prompts, no training.
- **`siglip_game`:** zero-shot SigLIP, then drops central-tagged shots unlike this game's own central shots. "Unlike"
  means below a cosine similarity `min_sim` to a robust centroid of either an arena-colour HSV histogram (`hsv`) or the
  SigLIP embedding.

## Evidence (dev, `docs/metrics.md` "Ingest (Layer 0)", 2026-11-27)

**Camera** (795 tagged shots of 5 games, on the label files' own shot bounds):

| config | acc | central P | central R |
|---|---|---|---|
| `ingest-siglip-zeroshot` | 0.804 | 0.756 | 1.000 |
| `ingest-siglip-game-emb` | 0.919 | 0.883 | 1.000 |
| `ingest-siglip-game-hsv` | 0.971 | 0.956 | 0.998 |

All zero-shot errors are `other` shots tagged central. Most are other games' half-time highlights, which also show a
main camera and a scorebug. The arena-colour check removes them.

**Cuts** (`data/labels/cut_gold.json`, game time of the 5 dev games, 1,385 real cuts):
- The gold is every cut either detector makes, merged within 0.2 s.
- All 370 disagreements were read by eye. 140 agreements were read on a random sample, and 138 were real.
- Every further cut `transnetv2` makes at threshold 0.3 (69) was read; 60 were real.

| config | cut P | cut R |
|---|---|---|
| `ingest-siglip-game-hsv` (`hist_diff`) | 0.891 | 0.861 |
| `ingest-transnetv2-game-hsv` (threshold 0.5) | 0.998 | 0.929 |
| `ingest-transnetv2-t030-game-hsv` (threshold 0.3) | 0.989 | 0.998 |

- **`hist_diff`:** misses most dissolves, and cuts on slow-motion replays, whip pans and LED boards.
- **TransNet V2 at 0.5:** misses ORLEN replay stingers, which it scores 0.3–0.45. A missed stinger merges live play
  with the replay that follows into one sequence.
- **TransNet V2 at 0.3:** catches them. Its extra cuts are mostly both ends of one stinger; the 36-frame sliver
  between them is shorter than `min_s` and dropped.

## Decision

The production Layer 0 config is `ingest-transnetv2-t030-game-hsv`:
- `shots/video`;
- cuts `transnetv2` at threshold 0.3;
- camera `siglip_game` on the HSV arena-colour feature with `min_sim` 0.95;
- 3 frames per shot, central shots of at least 3 s, stride 5.

`hist_diff` stays registered as the baseline. The hand-labelled backends (`gold`, `clips`, `shot_gold`) stay the
sequences every other layer is scored on.

## Consequences

- A raw full game runs end to end with no hand-labelled shots: `configs/e2e-video-fused-v8-r4.yaml` is the active
  full stack on `shots/video`.
  Court, detect, team and appearance run on Modal (`--remote-layer`, A10G after L4 sat in Modal's capacity queue);
  track, numbers and the later layers run on the Mac. The Modal image has no PARSeq (`strhub`), so `numbers` stays
  local.
- **Picked on dev, held-out not scored:** `min_sim` and the threshold were picked on dev. The held-out games
  (`Tnuscrm9RyU`, `1A79NXz9CFM`) are scored once at the end of M10.
- **What the cut gold can miss:** it only holds cuts some detector made. A cut every detector missed is not in it, so
  recall is over the cuts found. A new detector's false positives on no gold entry are counted as unread and need
  reading before its precision counts.
- **Weights:** the TransNet V2 weights are not committed. `cuts/transnetv2` checks the sha256 of
  `weights/pretrained/transnetv2/transnetv2-c639fb6d.pth`, and `scripts/convert_transnetv2.py` rebuilds it.
