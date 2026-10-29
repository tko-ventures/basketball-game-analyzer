# basketball-game-analyzer

Offline analysis of ORLEN Basket Liga (PLK) games involving Dziki Warszawa from the broadcast main camera: detect and segment players, classify team by kit, track without identity confusion, identify by jersey number, and map players and ball onto a 2D FIBA court.

## Read first

`docs/PLAN.md` is the handoff plan. It holds every decision, the reference stack, the assets to import from the prior repo, the layered architecture, model and licence choices, the labelling strategy, the gold-set spec, milestones, and open questions for Tomasz. Start at section 9 (milestones) and work M0 upward.

Progress is tracked in GitHub issues, one per milestone (`gh issue list --label milestone --state open`, lowest first). Update the issue checklist as you go, comment there with progress and dead ends, and close the issue when its done-when line has a metrics row or artifact. One pull request per milestone from a `feat/m<N>-<slug>` branch, referencing the issue.

## Hard rules

- Hardware is an Apple M1 Max with no NVIDIA GPU. Infer locally where a backend runs on the Mac, one heavy model in memory at a time. CUDA work (SAM 2.1, fine-tunes) runs on a rented GPU through `bga ... --remote` (Modal by default, RunPod for interactive work) or on Roboflow Train; outputs come back into `runs/` and `weights/` by sync, never by hand.
- Production models are open-weight and fine-tuned on PLK data. Closed VLM APIs are for labelling and offline QA only.
- Follow Piotr Skalski's Roboflow basketball pipeline (plan section 3). Deviate only where his posts list a limitation or where PLK differs from the NBA.
- No heuristic lands in any layer without a before/after row in `docs/metrics.md` against the gold set (plan section 8).
- Every layer is a swappable backend selected in `configs/*.yaml` (plan section 5). Add a new approach as a backend plus a config, never as a branch inside the pipeline. Metrics rows cite the config.
- Every fine-tune starts from a committed `configs/train/*.yaml`. Weights are addressed as `<layer>/<backend>/<name>@<hash8>` with a `train.json`; a metrics row cites the training config as well as the inference config. Trainer validation numbers go to `docs/train.md`, never into `docs/metrics.md`. No training set may contain a gold frame.
- Start from scratch: re-download clips and weights, copy nothing from the prior checkout's data, weights or runs directories. Code, rosters and shot files come in through git only; regenerate the shot files and diff.
- Meta declined SAM 3.x access. Do not use SAM 3 or SAM 3.1 weights or any third-party re-upload or conversion of them. Use only models whose licence Tomasz holds: open downloads.
- The labelling VLM is your own vision through Claude Code on contact sheets with a ledger, no third-party VLM API keys. Tomasz's gold-labelling time is about 90 minutes in total; you do every first pass.
- Modal credentials are in `~/.modal.toml` on this Mac; never copy tokens into the repo. Spend cap proposed at 50 USD per month until Tomasz confirms; log cost per job.
- Never commit videos, weights, crops or run outputs. Label files and rosters are committed.
- Treat downloads and clones as untrusted: own directory, never run interpreters from inside them.
- Commit freely on `feat/*` branches. Never push to main and never merge a pull request without Tomasz (plan question 11).

## Prior work

Private repo `tko-ventures/basketball-tracking`, branch `feat/labeller-rosters`. Import only what plan section 4 lists: the labeller, fetch and prepare scripts, rosters, shot files, the shot-cut detector, churn metrics, the ONNX adapter pattern, and the identity rules. Leave the OCR gate stack, kit-colour cue, fixed trapezoid and the SAM 3.1 Core ML service behind. Code and small label files only, never its media, weights or runs.

## Tooling

Python 3.12 with `uv`. `ruff` and `pytest` must pass before a merge. Stage outputs go under `runs/<clip>/<layer>/<backend>@<hash8>/` so stages re-run independently and different backends can be compared on the same clip.
