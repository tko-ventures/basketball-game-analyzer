# Architecture Decision Records (ADRs)

This directory holds the active architectural, modelling, data, and process decisions for `basketball-game-analyzer`. Historical M0–M6 chronological journal entries are preserved in [`docs/archive/decisions-log.md`](../archive/decisions-log.md).

## How we use ADRs (human + parallel agents)

- **One decision per file**, named `YYYY-MM-DD-<short-slug>.md` (date-slugs avoid numbering collisions when Tomasz and 1–2 independent agents work on parallel branches).
- **When to write an ADR:** changing a pipeline layer's active backend or model, adding/dropping a milestone or data contract, setting licence/compute/eval policies, or explicitly rejecting a major candidate so nobody re-tests dead ends. Do not write ADRs for routine bug fixes, single experiment configs, or minor refactors.
- **Lifecycle:**
  - `Proposed` — draft created by Tomasz (to steer an agent before coding) or by an agent (for Tomasz's sign-off).
  - `Accepted` — active decision in effect.
  - `Superseded by <filename>` — kept for history, never deleted, linked to the replacement ADR.
- **Keep ADRs short** (~20–40 lines): context, decision, key evidence/metrics, and consequences. Update the index below in the same PR.

### Template

```markdown
# <Short decision title>

- **Date:** YYYY-MM-DD
- **Status:** Proposed | Accepted | Superseded by [<file>](<file>)
- **Area:** Pipeline (<layer>) | Data & Eval | Infra | Process
- **Decider:** Tomasz / Implementer

## Context

What problem, constraint, or measurement triggered this decision?

## Decision

What was chosen — and what key alternatives were rejected?

## Consequences

What changes in code, configs, metrics, or future milestones?
```

---

## Active decisions

### Pipeline layers

| Date | Area | Decision | Status |
|---|---|---|---|
| 2026-11-02 | Court (L1) | [Fine-tune RF-DETR Keypoint on PLK halls with median smoothing](2026-11-02-court-keypoint-finetune-and-smoothing.md) | Accepted |
| 2026-11-19 | Detect (L2) | [RF-DETR Large at 1280 px on PyTorch/MPS; drop XL, DEIM, YOLO26, and court filter](2026-11-19-detector-rfdetr-large-1280px-mps.md) | Accepted |
| 2026-11-20 | Detect / M9 | [Six active box classes, split visual holder from `dead_ball`, adopt `plk-r3-poss`](2026-11-20-detector-classes-and-possession-schema.md) | Accepted |
| 2026-11-20 | Team (L3) | [Fit `siglip_kmeans` per shot at 5 fps with fixed seed and required kit metadata](2026-11-20-team-classification-siglip-5fps.md) | Accepted |
| 2026-11-22 | Track / Link (L3b, L4) | [BoT-SORT with PLK re-ID encoder + greedy link (5-per-team cap); drop SAM 2.1 and `bytetrack_court`](2026-11-22-tracking-botsort-plk-reid-and-greedy-link.md) | Accepted |
| 2026-11-23 | Numbers & Identity (L5/L5b) | [PARSeq fine-tuned from scene-text weights + roster log-prob, co-visible assignment & game-level clustering](2026-11-23-jersey-number-reading-parseq-and-roster-scoring.md) | Accepted *(open licence check)* |
| 2026-11-23 | Ingest (L0 / M10) | [Stay on YouTube PLK highlights for R&D; split full-game ingest into M10](2026-11-23-highlights-footage-and-m10-ingest-milestone.md) | Accepted |
| 2026-11-23 | Positions & Export (L6 / M8) | [Keep `positions/bbox_bottom` on linked tracks + PLK court; reject zero-shot COCO `ankles_pose`; ship Parquet export & upgraded radar](2026-11-23-positions-bbox-bottom-over-coco-pose-and-parquet-export.md) | Accepted |
| 2026-11-24 | Ball, Possession & Shots (L7 / M9) | [Adopt `ball/fused` over `ball/skalski`; skip tiled ball inference; ship gold `v1/events.json` and `.ball.parquet`](2026-11-24-ball-fused-tracking-possession-and-rim-shots.md) | Accepted |
| 2026-11-24 | Ingest (L0 / M10) | [Full-length YouTube game recordings are the M10 footage; improve existing layers first](2026-11-24-full-length-youtube-games-for-m10.md) | Accepted |

### Data & evaluation

| Date | Area | Decision | Status |
|---|---|---|---|
| 2026-11-02 | Gold & Eval | [Gold set v0/v1 protocol, 4 PLK games (1 held-out), and metrics-row gate](2026-11-02-gold-set-and-evaluation-protocol.md) | Accepted |
| 2026-11-22 | Pipeline / Cache | [Pipeline hardening invariants and immutable cache/weight hashes](2026-11-22-pipeline-hardening-and-cache-invariants.md) | Accepted |
| 2026-11-24 | Gold & Eval / Ball (L7) | [Holder gold under the 2026-11-20 definition; `ball-plk-fused-v2` for shooter attribution](2026-11-24-holder-gold-and-fused-v2-shooter.md) | Proposed |

### Infrastructure, licences & process

| Date | Area | Decision | Status |
|---|---|---|---|
| 2026-11-17 | Compute | [Modal as sole cloud GPU platform, billing truth, and 75 USD/month spend cap](2026-11-17-modal-for-training-and-remote-compute.md) | Superseded by [2026-11-23](2026-11-23-colab-for-training-alternating-with-modal.md) |
| 2026-11-18 | Licences | [Open-weight policy: drop AGPL (YOLO26); treat research/PML weights as Apache 2.0 until pre-ship audit](2026-11-18-open-weights-and-permissive-licence-assumptions.md) | Accepted |
| 2026-11-18 | Style & Git | [Borda Python coding/testing style, pre-commit hooks, and squash-merge workflow](2026-11-18-python-style-and-quality-standards.md) | Accepted |
| 2026-11-18 | Demo & Language | [English-only codebase and demo site; single analysis view + synthetic coach showcase](2026-11-18-english-only-codebase-and-demo.md) | Accepted |
| 2026-11-20 | Dependencies | [Follow upstream migration guides (`supervision` → `trackers`) via new backends](2026-11-20-upstream-migration-policy.md) | Accepted |
| 2026-11-23 | Compute | [Google Colab for training, alternating with Modal by credit availability](2026-11-23-colab-for-training-alternating-with-modal.md) | Accepted |
