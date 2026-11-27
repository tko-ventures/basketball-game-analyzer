# Free throws are shot attempts in their own category

- **Date:** 2026-11-27
- **Status:** Accepted (Tomasz, 2026-11-26, category and gold); the Layer 7 tag stays experimental (`ball-plk-fused-v6`) until the epic #46 decision
- **Area:** Data & Eval / Pipeline (ball, L7)
- **Decider:** Tomasz / Implementer

## Context

Full-game shot gold (Gliwice dev, PGE Start held-out, both 60 fps) added 29 + 22 confirmed field-goal attempts. Highlight clips show almost no free throws, so the first gold left them out. On the full-game dev shots, most of Layer 7's false shots were free throws: precision there was 0.62 to 0.72, against 0.97 to 1.00 on highlights. Tomasz ruled (2026-11-26, #53): "free throws are still throws, we need to capture them", as a separate category of shot.

## Decision

1. **Gold.** A free throw is a 1-point attempt in `data/labels/shot_gold.json` (`points: 1`). A shot whose free throws were confirmed on `/shotgold` is marked `free_throws: "labelled"`. The export lists those shots in `free_throw_sequences`, so a shot without the mark never counts free throws as missed. The implementer reads the first pass from rim strips and overviews (`scripts/shot_gold.py apply-free-throws`), and Tomasz confirms it.
2. **Eval.** `bga eval-ball` scores field goals and free throws apart. Shot F1, outcome and shooter columns count field goals only (gold `points != 1`, and predictions not tagged `free_throw`). A separate FT F1 column (P/R, pred/gt, outcome acc) scores free throws on `free_throw_sequences` only.
3. **Layer 7 contract.** Every shot carries `kind` (`field_goal` | `free_throw`); a free throw scores 1 point. `ball/fused` tags a free throw by whole-scene stillness: median tracked-player movement over 1.5 s before release at most `ft_scene_still_m` (off by default; 0.45 m in `ball-plk-fused-v6`).

## Evidence

- Full-game dev: first-pass free throws measure 0.20 to 0.37 m of scene movement before release; all 28 confirmed field goals measure 0.61 m or more.
- Dev r3, v5 to v6: shot F1 0.844 to 0.898 (P 0.76 to 0.85, R 0.95 unchanged); the other columns are unchanged (`docs/metrics.md`, 2026-11-27).
- Confirmed free throws (Tomasz, 2026-11-27): dev 6, held-out 6. Dev r3 v6: FT F1 0.857 (P 0.75, R 1.00, 8 predicted / 6), outcome acc 1.000; the 2 extra are false shots during line-ups.
- Rejected: shooter standing still within 1.5 m of the free-throw spot. It caught 2 of 7 free throws, both already caught by the scene rule, and fired on 2 false shots during line-ups.

## Consequences

- Fouls (#54) can build on free throws, since they mark shooting fouls.
- A scorebug read (frozen clock) would be a stronger signal than stillness; it is out of scope until #54.
