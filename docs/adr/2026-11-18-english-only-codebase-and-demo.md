# English-only codebase and demo site; single analysis view + coach showcase

- **Date:** 2026-11-18 *(supersedes 2026-11-01 Polish demo text & 2026-10-31 dual-video page)*
- **Status:** Accepted
- **Area:** Process / Demo
- **Decider:** Tomasz

## Context

Early demo builds maintained both Polish and English editions and a separate dual-synchronised-video view alongside the interactive analysis view. Maintaining duplicate languages and redundant pages slowed iteration before demos to Dziki Warszawa's coaching staff (head coach Marco Legovich is English-speaking).

## Decision

1. **English everywhere:** Code, comments, docs, URLs, and demo UI text are English only. Polish remains only in source data (team/player names, YouTube titles like `SKRÓT`, plk.pl position scraping).
2. **Demo site layout (`https://basketball-analyzer.demo.tko.ventures`):**
   - `/` — single interactive analysis view (3 shots of `OSp445e0mXU`, broadcast + live 2D court powered by `configs/court-plk-v1-smooth4.yaml`). The dual-video page is removed; legacy `/analysis/`, `/analiza/`, and `/en/*` paths redirect to `/`.
   - `/coach/` — interactive coach prototype (`Watch` → `Insights` → `Feedback`) driven by 8 deterministic simulated possessions (`data/fixtures/synthetic/`, schema `bga.tracking2d/v0`), clearly badged as simulated data.
   - `/roadmap/` — static milestone view (`docs/roadmap.html`).
3. **Showcase boundary:** The demo site is built outside the `bga` pipeline (`scripts/demo_web.py`, `scripts/coach_web.py`) and deployed only from a clean `main` via `scripts/deploy_demo.sh`. Demo heuristics (like `coach_insights.py` or demo roster filters) are not pipeline layers and need a gold metrics row before entering `src/bga/`.

## Consequences

- `scripts/deploy_demo.sh` verifies English `index.html` and clean `main` before publishing.
- Synthetic fixtures never enter gold sets or training manifests (`tests/test_synthetic_plays.py` guards determinism).
