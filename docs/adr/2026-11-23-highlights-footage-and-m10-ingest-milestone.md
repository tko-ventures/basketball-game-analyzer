# Stay on YouTube PLK highlights for R&D; split full-game ingest into M10

- **Date:** 2026-11-23 *(incorporates 2026-11-18 footage decision)*
- **Status:** Accepted
- **Area:** Pipeline (shots / M10)
- **Decider:** Tomasz

## Context

Highlight clips cut every few seconds, capping cross-shot tracking and game-level identity, and production will require licensed full-game video (Hudl Instat, Synergy, or club recordings). Issue #26/#32 raised whether raw full-game ingest (shot-cut detection + central-camera classification) should be folded into M8 or M9.

## Decision

1. **YouTube highlights stay the R&D dataset:** Continue M7–M9 using PLK YouTube highlights, including current 2026/27 season league games (whose rosters and kits match the held-out Superpuchar game).
2. **Dedicated milestone M10 (#32) for production video ingest:** Layer 0 for raw full-game footage — a registered `shots` backend combining shot-cut detection with a central-camera classifier — is its own milestone after M9, starting when the first licensed full game arrives. It is not folded into M8 or M9.
3. **Clip-agnostic pipeline:** Keep all downstream layers independent of clip length so full games plug in as new sequences without layer changes.

## Consequences

- Eight 2026/27 PLK league highlight games were added to `data/clip_ids.txt` (39 clips total) to supply current-season jersey-number crops for M7.
- A sealed full-game test clip remains open until full-game footage is sourced.
