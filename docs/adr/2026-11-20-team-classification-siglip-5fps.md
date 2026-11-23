# Fit `siglip_kmeans` per shot at 5 fps with fixed seed and required kit metadata

- **Date:** 2026-11-20 *(hardened 2026-11-22)*
- **Status:** Accepted
- **Area:** Pipeline (team)
- **Decider:** Tomasz / Implementer

## Context

Evaluating Skalski's `sports.TeamClassifier` (SigLIP + UMAP + KMeans) on `plk-r3-poss` boxes across 5 random seeds showed that the stock 1 fps fit (~110 crops/shot) was unstable, misclassifying two Dziki tracks in 3 of 5 seeds (0.980 accuracy). In addition, non-gold clips without light/dark kit metadata silently inverted home/away labels.

## Decision

1. **Fit at 5 fps with a fixed seed:** Downstream configs use `fit_fps: 5.0` and an explicit `seed` (scoring **1.000** on 4 of 5 dev seeds, 0.990 on 1, and **1.000** on the 31 held-out tracks; wrong-team boxes drop from ~6% to 1.9–3.5% for +50 ms/frame). Backend default stays `1` so `baseline` reproduces Skalski's settings.
2. **Keep hip/centre crop (`crop_cy: 0.5`):** Moving the 40% crop to the chest (`0.35`) degraded accuracy to 0.878–0.959 and was rejected.
3. **Report separation, don't gate on it:** UMAP silhouette and minor-cluster share are logged in outputs, not used to trigger a fallback (failing 1 fps shots actually had the highest silhouettes, 0.58–0.63). Manual kit-colour fallback is not built unless a future arena fails.
4. **Mandatory kit metadata:** Every clip label file must mark exactly one side `light` and the other `dark` (`team/siglip_kmeans` refuses clips missing it). SigLIP is loaded once in `load()`, while UMAP and KMeans fit fresh per shot.

## Consequences

- Home/away team alignment is deterministic and fails loudly on unannotated clips (5 of 39 PLK clips have the home side in the darker kit).
- Per-game (multi-shot) team fitting is deferred unless single-shot fitting fails on new footage.
