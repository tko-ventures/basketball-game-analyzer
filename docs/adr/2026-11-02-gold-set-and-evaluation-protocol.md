# Gold set v0/v1 protocol, 4 PLK games, and metrics-row gate

- **Date:** 2026-11-02 *(incorporates 2026-10-29 Q12 & 2026-11-18 updates)*
- **Status:** Accepted
- **Area:** Data & Eval
- **Decider:** Tomasz

## Context

Evaluating pipeline layers requires ground truth across distinct PLK arenas without overfitting to a single hall or spending excessive manual review time up front.

## Decision

1. **Four gold games (3 central shots each, stride 5):**
   - Dev (3 games): `4XMD9FWmQws` (Sosnowiec 2024/25, Polish Cup), `OSp445e0mXU` (Warsaw, 2025/26), `1jDaJBkUGmc` (Sopot, 2025/26).
   - Held-out (1 game): `Tnuscrm9RyU` (Sosnowiec 2026/27 Superpuchar — same arena as `4XMD9FWmQws` with different floor paint, never trained on, scored at most once per milestone close).
2. **Iterative gold versions (`v0` / `v1`):**
   - `gold=v0` (tracks, teams, numbers) was prefilled from public models and reviewed by the implementer so progress wasn't blocked on human budget. Scores are comparable across backends; external reporting must label them `gold v0`.
   - `gold=v1` adds human-tightened box geometry (100 frames), court keyframes (24 gold keyframes), and verified number crops (376 crops), labelled just-in-time as each milestone needs them.
3. **Metrics-row gate:** No heuristic or backend change lands in a pipeline layer without a before/after row in `docs/metrics.md` citing committed configs and proven weights (`pretrained:` or local `train.json`). Every training manifest is blocked from touching any gold clip (`bga.train.leakage` + `tests/test_gold_leakage.py`).

## Consequences

- `bga eval*` commands enforce weight provenance and tag every row with its gold version (`v0` or `v1`).
- Court metrics are reported per hall (Warsaw, Sosnowiec 2024/25, Sopot, Sosnowiec 2026/27 held-out).
- The original 90-minute human labelling cap was lifted on 2026-11-18, though agents still do every first pass and send only geometry/ambiguous cases to Tomasz.
