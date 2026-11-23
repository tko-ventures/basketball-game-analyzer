# Keep `positions/bbox_bottom` on linked tracks + PLK court; reject zero-shot COCO `ankles_pose`; ship Parquet export & upgraded radar

- **Date:** 2026-11-23
- **Status:** Accepted
- **Area:** Pipeline (positions, export)
- **Decider:** Tomasz / Implementer

## Context

Layer 6 (`positions`) previously consumed raw `track` (ignoring `link`), ran only on the NBA baseline homography, had no guard against projective `w <= 0` singularities, had no quantitative evaluator, and had no Parquet export. M8 (Issue #9) called for comparing box bottom-centre (`positions/bbox_bottom`) against ankle keypoints from pose estimation (`positions/ankles_pose`) with jump handling, exporting the final per-player per-frame Parquet schema (`docs/PLAN.md` §5), and upgrading the radar video to render linked tracks, team colours, and confirmed jersey numbers.

## Decision

1. **Wire `link` as an optional upstream of `positions` (`OPTIONAL_BY_LAYER`):** `positions` consumes `link` when configured and falls back to `track`, preserving legacy config hashes that omit `link`. Move `fiba_court.json` from `tools/labeler/static/` into `src/bga/court/fiba_court.json` as package data (`FIBA_COURT_PATH`, `load_fiba_court`).
2. **Guard homography projection against `w <= 0` singularities:** `bga.positions.base.project` returns `NaN` when the homogeneous denominator `w <= 1e-6`, preventing off-horizon points from blowing up or flipping signs before `clean_paths` interpolation/smoothing (`smooth_window=9`, `min_jump_m=0.7`).
3. **Keep `positions/bbox_bottom` as the active backend (`configs/positions-plk-bbox.yaml`); keep `positions/ankles_pose` registered as a comparator, not adopted:**
   - On gold dev (9 shots, 87 player points on gold court keyframes), swapping to the full PLK stack (`court/rfdetr_keypoint` `plk-v1@c26b7841` `smooth_window=4` + linked tracks) with `positions/bbox_bottom` cuts keyframe median error from **1.540 m → 0.358 m** (4.3×), p90 from **4.895 m → 0.968 m** (5.1×), on-court share from **0.887 → 0.994**, 95th-percentile speed from **23.19 → 5.00 m/s**, and >10 m/s spikes from **12.0% → 0.8%** at **0.0 ms/frame**.
   - Zero-shot COCO `positions/ankles_pose` (`pretrained:rfdetr/keypoint-preview-xlarge`, Apache 2.0, `configs/positions-plk-ankles.yaml`) adds **109.9 ms/frame** on MPS and loses on every dev metric (**0.414 m** median, **1.156 m** p90, **0.993** on-court, **5.51 m/s** speed p95, **1.0%** spikes). Three failure modes caused the regression: (a) COCO pose matched only **71.1%** of track-frames (missing far-side 130–180 px players), so fallback to `bbox_bottom` created vertical jumps; (b) COCO ankle keypoints sit a median **26.6 px (~13–17% of box height)** above the shoe sole (`y2`), biasing floor projections toward the far sideline; (c) takeoff-chord airborne detection fired **648** times on dev (11/87 keyframe points, median error **1.014 m** vs **0.383 m** grounded) mostly from ankle/bbox switching and normal running strides.
   - Revisit pose-based ground points only if a high-resolution crop-level foot/shoe-sole keypoint model becomes worthwhile after M9.
4. **Ship 17-column Parquet export (`bga export`) and upgraded radar (`bga radar`):** Export exact `docs/PLAN.md` §5 columns (`game_id, clip, frame_idx, t_video_s, shot_id, camera, game_clock, track_id, player_uid, team, jersey_number, player_name, bbox_xyxy_px, foot_px, x_m, y_m, confidence`) via PyArrow, resolving `player_uid` (`plk:<id_plk>` from `data/rosters/`, else `<game_id>:<cluster_id>` from `identity`) and `player_name`. Upgrade `bga radar` to render linked tracks, team-coloured boxes/dots, foot points, and `#<jersey>` labels on both the broadcast frame and the 2D FIBA court.

## Consequences

- `configs/positions-plk-bbox.yaml` is the active end-to-end PLK stack through Layer 6.
- `bga eval-positions` adds a dedicated `## Positions layer` table in `docs/metrics.md`.
- `bga export --all-splits` and `bga radar --all-splits` produce Parquet tables and side-by-side radar `.mp4` videos for all 12 gold shots (dev + held-out).
- `pyarrow>=26.0.1` is added to project dependencies.
