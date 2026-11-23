# Fine-tune RF-DETR Keypoint on PLK halls with median smoothing

- **Date:** 2026-11-02 *(measured in M3, 2026-11-17)*
- **Status:** Accepted
- **Area:** Pipeline (court)
- **Decider:** Tomasz / Implementer

## Context

Visual QA (`scripts/qa_court_overlay.py`) showed Skalski's pretrained NBA court detector (`basketball-court-detection-2/14`) mapped Warsaw adequately (0.7 m median error) but systematically shifted landmarks in Sosnowiec (blue/red keys, low RANSAC residuals that couldn't self-detect) and failed in Sopot (4.2 m error) and Ostrów (6.4 m). Overall paint error on dev was 0.89 m.

## Decision

1. **Keyframe + propagation labelling:** Tomasz corrects 1 keyframe per shot (dots + painted-line pins in `/court`) across PLK halls; SIFT floor-match propagation expands each training keyframe up to ±1.5 s gated by painted-line support (≥ 85% of keyframe support) so drift stays bounded.
2. **PLK court model:** Fine-tune `RF-DETR Keypoint` (XLarge, Apache 2.0) on Roboflow's NBA court set + the PLK training manifest (`court-rfdetr_keypoint-plk-v1.yaml`), served via `court/rfdetr_keypoint`.
3. **Within-shot median smoothing:** Apply `bga.court.smooth` (`smooth_window: 4`) over a 12×8 floor grid before refitting each frame's homography, suppressing single-frame jumps without crossing shot cuts.

## Consequences

- `configs/court-plk-v1-smooth4.yaml` is the active court setup: **0.21 m** paint error on dev and **0.20 m** on the held-out game (down from 0.89 m NBA baseline).
- The analysis demo page also runs this config instead of the NBA model + Savitzky-Golay filter.
- `bga.court.propagate.court_to_image` determines camera-front orientation at a floor point in the image (never by the sign of stored `H`, since `h33=1` flips when the image origin is above the horizon).
