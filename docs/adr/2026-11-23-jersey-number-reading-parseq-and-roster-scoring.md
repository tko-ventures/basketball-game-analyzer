# PARSeq fine-tuned from scene-text weights + roster log-prob, co-visible assignment & game-level clustering

- **Date:** 2026-11-23 *(incorporates 2026-11-17 M7 design & 2026-11-22 `crop_cls` bake-off)*
- **Status:** Accepted *(open: Tomasz to confirm interim licence assumption for PARSeq weights)*
- **Area:** Pipeline (numbers, identity)
- **Decider:** Tomasz / Implementer

## Context

Skalski's NBA SmolVLM2 reader (`smolvlm_ocr`) ran at ~1.9–2.3 s/crop on MPS, scored 0.653 top-1 under the roster on gold dev, and systematically confused PLK font digits (e.g. `2` vs `8`). Round-1 PLK crop classifiers (`numbers/crop_cls` with ResNet-18 and DINOv2, flat and factorised heads) topped out at 0.468 and left numbers without training crops unread because they had to learn digits from scratch.

## Decision

1. **Adopt PARSeq (`numbers/parseq`) in the M7 bake-off:** Fine-tune `baudm/parseq` (`strhub` commit `1902db04`, released `parseq-bb5792a6.pt`) on PLK crops. Because it already reads digits zero-shot (0.616 roster top-1, 0.882 refusal at 5 ms/crop), round-1 PLK fine-tuning (`numbers-parseq-r1`) jumped to **0.917 top-1 under the roster** at **4 ms/crop** (0.710 track accuracy on dev, 0.855 confirmed). Koshkina & Elder's SoccerNet/hockey checkpoints are not used.
2. **Keep `numbers/crop_cls` as the bake-off comparator:** One backend supporting both ImageNet ResNet-18 and Apache-2.0 DINOv2 ViT-S/14 (101-class flat or 21-output factorised head).
3. **Add layer 5b (`identity`) with swappable shot- and clip-scoped backends:**
   - `numbers` logs aligned per-read class probabilities (`arrays["probs"]`, float32 `(N, 101)`) when `stop_on_confirm: false`.
   - `identity/consecutive`: roster-filtered 3-in-a-row baseline (`identity-consecutive`).
   - `identity/roster_logprob`: renormalises per-crop probabilities over `roster U {unreadable}`, drops crops where `unreadable` wins, sums log-probabilities, and confirms at margin ≥ 8.0 (`identity-roster-logprob`; optional `owner_rule` comparator `identity-owner-rule`).
   - `identity/covisible_assign`: enforces teammate number uniqueness per co-visible component via `linear_sum_assignment` on negative margins with a per-track abstain column at `-margin` (`identity-covisible-assign`). Because only a track's argmax has positive margin, losers abstain and log a conflict rather than being silently reassigned.
   - `identity/game` (`scope = "clip"`): clusters each team's linked tracklets across all shots of one game by mean `plk-r1-shot-last` embedding (average linkage, `max_dist = 0.18`), never joining co-visible tracklets or tracklets with conflicting confirmed single-track numbers; pools crop probabilities per cluster; and assigns unique numbers per shot over the clusters present in that shot (`identity-game`). Force-merging same confirmed numbers across arbitrary embedding distance was tested and rejected (merged distinct misread players across shots, dropping dev accuracy from 0.913 to 0.899).
4. **Rejected from outside review:** HSV kit-colour k-means (replaced by `siglip_kmeans`), test-time shear/CLAHE sweeps (handled via training augmentation instead).

## Consequences

- **Round 2 reader (`numbers-parseq-r2`):** number crops from eight more 2025/26 games (`configs/crops-numbers-r2.yaml`; 864 number tracklets, 670 settled by `numbers-parseq-r1` at ≥ 0.8 of ≥ 10 reads and checked blind on a sample of 60 with no conflicting number) took `numbers-parseq-r2` to **0.963 top-1 under the roster** and **0.754 track accuracy** on dev (0.884 confirmed, 9 wrong; SmolVLM2 on the same stack: 0.551, 16 wrong). Trained on Colab (T4, 0.75 CCU).
- **Layer 5b (`identity`) progression on gold v0 dev (69 tracks):**
  - `identity-consecutive` (roster-filtered 3-in-a-row): **0.768** (53/69), 0.884 confirmed, 8 wrong.
  - `identity-roster-logprob` (`margin: 8.0`): **0.841** (58/69), 0.899 confirmed, 4 wrong (+0.073 acc, wrong halved).
  - `identity-owner-rule` (longest-track-wins): **0.812** (56/69), 0.855 confirmed, 3 wrong (drops a shorter true `#10` beaten by a longer misread track).
  - `identity-covisible-assign` (LSA with abstain at 8.0): **0.841** (58/69), 0.870 confirmed, **2 wrong** (+0.029 vs `owner_rule`, halves wrong confirmations from 4 to 2).
  - `identity-game` (cross-shot clustering `max_dist: 0.18` + pooled log-prob + per-shot LSA): **0.913** (63/69), **0.957 confirmed**, **3 wrong** (+0.072 over shot-level, +0.159 over `numbers-parseq-r2`).
- **Refusal at the crop level** remains 0.447 on `numbers-parseq-r2`, but skipping crops where `unreadable` wins and requiring an 8.0 log-prob margin + co-visible assignment cuts wrong track confirmations from 9 down to 2 (shot-level) / 3 (game-level).
- The held-out game (`Tnuscrm9RyU`) has not been scored yet; ready for the single M7 milestone-close held-out evaluation once approved.
- **Open (Tomasz):** Confirm treating PARSeq's scene-text weights under the interim R&D Apache-2.0 assumption ([licence ADR](2026-11-18-open-weights-and-permissive-licence-assumptions.md)), or specify a synthetic-only starting checkpoint.
