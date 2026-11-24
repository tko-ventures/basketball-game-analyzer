# Full-length YouTube game recordings are the M10 footage; layers before ingest

- **Date:** 2026-11-24
- **Status:** Accepted
- **Area:** Pipeline (shots / M10)
- **Decider:** Tomasz

## Context

The [2026-11-23 ADR](2026-11-23-highlights-footage-and-m10-ingest-milestone.md) starts M10 (#32, production video ingest) "when the first licensed full game arrives". Raw high-quality recordings (club, Hudl Instat, Synergy) are not available. Full-length PLK games are published on YouTube, for example `1A79NXz9CFM` (https://www.youtube.com/watch?v=1A79NXz9CFM).

## Decision

1. **Full-length YouTube recordings are the full-game footage** for M10 until raw recordings exist. They are the same broadcast main camera as the highlights, uncut, with replays, graphics, and breaks that Layer 0 must filter.
2. **Improve the existing layers first:** M10 is unblocked but follows work that improves Layers 1–7 on the gold set (post-M9 issue #40 first).
3. Footage rights are unchanged: development only; production still needs licensed high-quality video before anything ships.

## Consequences

- M10's "first full game" is a full-length YouTube game; `1A79NXz9CFM` is the first candidate.
- Highlights stay the gold and training source; no gold frame may come from a full game used for training.
