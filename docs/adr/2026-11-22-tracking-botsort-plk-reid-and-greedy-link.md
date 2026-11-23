# BoT-SORT with PLK re-ID encoder + greedy link (5-per-team cap)

- **Date:** 2026-11-22 *(incorporates 2026-11-10 appearance scope, 2026-11-21 re-ID tuning, 2026-11-22 `bytetrack_court` drop)*
- **Status:** Accepted
- **Area:** Pipeline (appearance, track, link)
- **Decider:** Tomasz / Implementer

## Context

In M6, tuned ByteTrack still produced 114–183 ID switches on gold v0 dev (HOTA 0.668–0.686). Roboflow `reid` 0.1.0 showed strong teammate discrimination (0.857 rank-1 at 1 s, 0.865 across cuts for `osnet_x1_0_sportsmot`), motivating an appearance layer (`3b`), tracker bake-off, and offline tracklet linking.

## Decision

1. **Add layer 3b (`appearance`):** `appearance/reid` extracts L2-normalised float16 embeddings per person box. Fine-tune `osnet_x1_0_sportsmot` on non-gold PLK crops using within-shot batches (`plk-r1-shot-last`).
2. **Active M6 tracking + linking stack (`link-botsort-plk-r1-shot-last-tuned-noapp-cap5`):**
   - **Tracker:** `track/trackers` running `BoTSORTTracker` with camera-motion compensation and the PLK re-ID encoder (`plk-r1-shot-last`), tuned with no IoU proximity gate (`proximity_threshold: 1.0`, adaptive fusion) so players re-link after occlusions/fast moves. Runs on Modal L4 (~64 ms/frame).
   - **Linker:** `link/greedy` joining tracklets by same team, gap, and court floor speed + `max_per_team: 5`. An extra appearance term in the linker adds no gain after tuned BoT-SORT, so it is left off here (`appearance: none` in the link step); the appearance layer stays for M7 cross-shot identity (0.946 rank-1).
3. **Rejected candidates:**
   - **SAM 2.1 (`track/sam2`):** Fewer switches (36) but lower HOTA (0.676 vs 0.743, tight mask boxes and lost players), ~3 s/frame on L40S, and upward bias on gold v0.
   - **`bytetrack_court` (court-space ByteTrack):** Dropped from M6; BoT-SORT's camera-motion compensation already handles broadcast pan/zoom in pixels (+0.05 HOTA over pixel ByteTrack). May be revisited in M8 if needed.
   - **`edgetam` and `mcbyte`:** Not run given SAM 2.1's deficit.

## Evidence (gold v0)

| Stack | Dev HOTA / IDF1 / IDSW | Held-out HOTA / IDF1 / IDSW |
|---|---|---|
| `trackers` ByteTrack + link | 0.693 / 0.797 / 108 | — |
| BoT-SORT (`osnet_x1_0_sportsmot`) + link + cap5 | 0.742 / 0.869 / 68 | 0.731 / 0.844 / 34 |
| **BoT-SORT (`plk-r1-shot-last`) + link (noapp) + cap5** | **0.743 / 0.873 / 52** | **0.742 / 0.849 / 26** |
| SAM 2.1 + link | 0.676 / 0.848 / 36 | — |

## Consequences

- M7 jersey-number work builds on `link-botsort-plk-r1-shot-last-tuned-noapp-cap5`.
- Because re-ID thresholds have a sharp optimum fitted on 9 dev shots, future footage changes should re-check threshold sensitivity.
