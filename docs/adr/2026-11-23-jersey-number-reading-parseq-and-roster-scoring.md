# PARSeq fine-tuned from scene-text weights + roster log-prob & co-visible assignment

- **Date:** 2026-11-23 *(incorporates 2026-11-17 M7 design & 2026-11-22 `crop_cls` bake-off)*
- **Status:** Accepted *(open: Tomasz to confirm interim licence assumption for PARSeq weights)*
- **Area:** Pipeline (numbers)
- **Decider:** Tomasz / Implementer

## Context

Skalski's NBA SmolVLM2 reader (`smolvlm_ocr`) ran at ~1.9–2.3 s/crop on MPS, scored 0.653 top-1 under the roster on gold dev, and systematically confused PLK font digits (e.g. `2` vs `8`). Round-1 PLK crop classifiers (`numbers/crop_cls` with ResNet-18 and DINOv2, flat and factorised heads) topped out at 0.468 and left numbers without training crops unread because they had to learn digits from scratch.

## Decision

1. **Adopt PARSeq (`numbers/parseq`) in the M7 bake-off:** Fine-tune `baudm/parseq` (`strhub` commit `1902db04`, released `parseq-bb5792a6.pt`) on PLK crops. Because it already reads digits zero-shot (0.616 roster top-1, 0.882 refusal at 5 ms/crop), round-1 PLK fine-tuning (`numbers-parseq-r1`) jumped to **0.917 top-1 under the roster** at **4 ms/crop** (0.710 track accuracy on dev, 0.855 confirmed). Koshkina & Elder's SoccerNet/hockey checkpoints are not used.
2. **Keep `numbers/crop_cls` as the bake-off comparator:** One backend supporting both ImageNet ResNet-18 and Apache-2.0 DINOv2 ViT-S/14 (101-class flat or 21-output factorised head).
3. **Track- and team-level M7 backends:**
   - `roster_logprob`: sum per-frame log-probabilities over roster candidates + `unreadable` (including end token, no length division) instead of hard 3-in-a-row free-text voting.
   - `covisible_assign`: enforce teammate number uniqueness per shot/team with an abstain option (a missed number is preferable to a wrong player name for coaches).
4. **Rejected from outside review:** HSV kit-colour k-means (replaced by `siglip_kmeans`), test-time shear/CLAHE sweeps (handled via training augmentation instead).

## Consequences

- **Round 2 (same day):** number crops from eight more 2025/26 games (`configs/crops-numbers-r2.yaml`; 864 number tracklets, 670 settled by `numbers-parseq-r1` at ≥ 0.8 of ≥ 10 reads and checked blind on a sample of 60 with no conflicting number) took `numbers-parseq-r2` to **0.963 top-1 under the roster** and **0.754 track accuracy** on dev (0.884 confirmed, 9 wrong; SmolVLM2 on the same stack: 0.551, 16 wrong). Trained on Colab (T4, 0.75 CCU).
- **Refusal is still open:** 0.447 on `numbers-parseq-r2` (0.461 on r1, 0.882 for the weights as released), although unreadable training crops grew from 700 to 3,075. Of its 42 answers on the 76 illegible and not-a-number dev crops, 19 are the holding track's gold number and 16 another number; 15 of 37 not-a-number crops still get a number. Track-level rules (`roster_logprob`, `covisible_assign`) are the next lever, before more unreadable data.
- The held-out game is not scored yet; the eight fetched 2026/27 highlights are not in any training set.
- **Open (Tomasz):** Confirm treating PARSeq's scene-text weights under the interim R&D Apache-2.0 assumption ([licence ADR](2026-11-18-open-weights-and-permissive-licence-assumptions.md)), or specify a synthetic-only starting checkpoint.
