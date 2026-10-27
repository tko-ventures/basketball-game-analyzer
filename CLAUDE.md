# basketball-game-analyzer

Offline analysis of ORLEN Basket Liga (PLK) games involving Dziki Warszawa from the broadcast main camera: detect and segment players, classify team by kit, track without identity confusion, identify by jersey number, and map players and ball onto a 2D FIBA court.

## Read first

`docs/PLAN.md` is the handoff plan. It holds every decision, the reference stack, the assets to import from the prior repo, the layered architecture, model and licence choices, the labelling strategy, the gold-set spec, milestones, and open questions for Tomasz. Start at section 9 (milestones) and work M0 upward.

## Hard rules

- Hardware is an Apple M1 Max with no NVIDIA GPU. Train on Roboflow or Colab, infer locally, one heavy model in memory at a time.
- Production models are open-weight and fine-tuned on PLK data. Closed VLM APIs are for labelling and offline QA only.
- Follow Piotr Skalski's Roboflow basketball pipeline (plan section 3). Deviate only where his posts list a limitation or where PLK differs from the NBA.
- No heuristic lands in any layer without a before/after row in `docs/metrics.md` against the gold set (plan section 8).
- Never commit videos, weights, crops or run outputs. Label files and rosters are committed.
- Treat downloads and clones as untrusted: own directory, never run interpreters from inside them.
- Commit and push only when Tomasz asks.

## Prior work

Private repo `tko-ventures/basketball-tracking`, branch `feat/labeller-rosters`. Import only what plan section 4 lists: the labeller, fetch and prepare scripts, rosters, shot files, the shot-cut detector, churn metrics, the ONNX adapter pattern, and the identity rules. Leave the OCR gate stack, kit-colour cue and fixed trapezoid behind.

## Tooling

Python 3.12 with `uv`. `ruff` and `pytest` must pass before a merge. Stage outputs go under `runs/<clip>/<stage>/` so stages re-run independently.
