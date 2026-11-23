# Pipeline hardening invariants and immutable cache/weight hashes

- **Date:** 2026-11-22
- **Status:** Accepted
- **Area:** Data & Eval / Pipeline
- **Decider:** Tomasz

## Context

An independent pipeline review (issue #26) audited data integrity, caching, remote sync, and training reproducibility ahead of M7's multi-game crop harvest.

## Decision

1. **Adopted immediately (Part 1, #27 — shipped):**
   - `FrameReader` reads sequentially and drops intermediate frames for steps ≤ 64 instead of seeking (122 ms → 3.7 ms/frame locally, 110 ms → 4.5 ms on Modal; pixel-identical frames).
   - `team/siglip_kmeans` loads SigLIP once and requires `light`/`dark` kit metadata on every clip.
   - Layer outputs write atomically (temp file → rename, JSON last) with per-sequence records preserved across partial runs in `meta.json`.
   - Remote weight uploads follow each setup's collect list, and `bga train --collect` drops the Modal volume copy of any re-collected weight ID to prevent stale weights.
2. **Queued after M7 (Part 2):** Dirty-tree guard for `--write` and `bga train`, pre-run spend check, backend-defaults snapshot test with optional revision hash, per-layer `Recipe` validation, unified `clip_id()`, `--set` + per-game/bootstrap breakdown on `eval-*`, batch cost rows, and CI (#34).
3. **Explicitly declined (do not re-propose without strong reason):**
   - Adding resolved defaults, code version, or platform to layer hashes, or changing `TrainConfig.config_hash` (would invalidate every cached run and all 17 trained weight IDs).
   - Resume/early-stopping for short re-ID/number trainers (3–52 min runs), detached remote inference, perceptual-hash leakage checks, config inheritance, or unmeasured homography/off-court filters (heuristics wait for M8 with metrics rows).

## Consequences

- Existing `runs/` and `weights/` hashes remain stable and backward-compatible.
- M7 also picked up a factorised number head and unseen-number validation (#30) from this review.
