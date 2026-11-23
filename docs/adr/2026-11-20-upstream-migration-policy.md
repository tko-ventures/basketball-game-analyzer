# Follow upstream migration guides via new backends (`supervision` → `trackers`)

- **Date:** 2026-11-20
- **Status:** Accepted
- **Area:** Process / Dependencies
- **Decider:** Tomasz

## Context

Adding Roboflow `trackers` 2.7.0 (for BoT-SORT + re-ID in M6) required `supervision` 0.31, which removed `sv.ByteTrack` in favour of `trackers.ByteTrackTracker` — a rewrite with different parameters, defaults, and Kalman state that produces different track IDs on the same detections.

## Decision

1. **Follow upstream migrations rather than pinning indefinitely:** Handle upstream deprecations/removals by adding a new backend and config with a before/after row in `docs/metrics.md`. Treat deprecation warnings in run logs as work items.
2. **Reproducibility bridge:** Vendored `supervision` 0.30.9's ByteTrack unchanged in `bga.track._sv_bytetrack` (MIT) solely to reproduce legacy `bytetrack` metrics rows; all new configs use the `trackers` backend (`trackers==2.7.0`, `supervision>=0.31,<0.32`).

## Consequences

- `pyproject.toml` overrides `inference`'s legacy pins (`trackers<2.5`, `supervision<0.31`) while calling only `inference.get_model`.
- Vendored `_sv_bytetrack` is frozen and excluded from pre-commit linting; new tracking work builds on `track/trackers`.
