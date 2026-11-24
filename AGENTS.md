# basketball-game-analyzer

Offline analysis of ORLEN Basket Liga (PLK) games involving Dziki Warszawa from the broadcast main camera: detect and segment players, classify team by kit, track without identity confusion, identify by jersey number, and map players and ball onto a 2D FIBA court.

This is the canonical instruction file for all AI coding agents working on this repository. (`CLAUDE.md` points here.)

## Read first

1. [`docs/architecture.md`](docs/architecture.md) — canonical human-oriented overview of what is built and the current active stack (visual companion: [`docs/architecture-c4-view.md`](docs/architecture-c4-view.md)).
2. [`docs/adr/README.md`](docs/adr/README.md) — active Architecture Decision Records (ADRs). Check before proposing or changing a layer, model, licence, or workflow so you do not re-test rejected paths.
3. [`docs/PLAN.md`](docs/PLAN.md) — roadmap, reference stack, layer specs, and milestones (start at section 9).
4. Open milestone issues: `gh issue list --label milestone --state open` (work lowest first). Update issue checklists, comment with progress and dead ends, and close when the done-when line has a metrics row or artifact.

## Documentation and ADR rules (human + parallel agents)

- **Keep `docs/architecture.md` readable for humans and `docs/architecture-c4-view.md` in sync.** `docs/architecture.md` is the canonical architecture doc: update it only when changing the active stack, layer DAG/contracts, core execution/eval/training mechanics, CLI commands, or major subsystems — with ground truth only (no plans or targets). **Every time `docs/architecture.md` changes, update [`docs/architecture-c4-view.md`](docs/architecture-c4-view.md) in the same change** so the C4 diagrams and backend table stay aligned with it. Do **not** enumerate individual experiment configs, per-round manifest counts, or metrics rows in either doc; those belong in `configs/`, `data/`, `docs/metrics.md`, and code.
- **Record architectural decisions as ADRs** under `docs/adr/YYYY-MM-DD-<short-slug>.md` (date-slugs avoid numbering collisions when parallel agents open PRs) and add a row to `docs/adr/README.md`.
- **Proposing vs. accepting:** Write a `Status: Proposed` ADR when a decision needs Tomasz's sign-off; mark replaced ADRs `Superseded by <file>` instead of deleting them. Never append to `docs/decisions.md` or `docs/archive/decisions-log.md`.

## Hard rules

- **Hardware & compute:** Local machine is an Apple M1 Max (MPS, no NVIDIA GPU). Infer locally where practical, keeping at most one heavy model in memory at a time. Fine-tunes (`bga train`) run on Google Colab GPUs (`google-colab-cli`, `--platform colab`, `--gpu`); when Colab refuses a GPU, use another Colab GPU, never fall back silently. Modal trains only while the month's Modal metered usage (`scripts/modal_spend.py`) is below 30 USD ([ADR](docs/adr/2026-11-24-colab-first-training-modal-under-30.md)). Multi-GPU DDP training and parallel remote inference (`bga ... --remote modal`, `--remote-layer`) run on Modal. Outputs sync back into `runs/` and `weights/`, never by hand. RunPod and Roboflow Train are not used.
- **Models & licences:** Production models are open-weight and fine-tuned on PLK data; closed VLM APIs are for offline labelling and QA only. Never use SAM 3 / SAM 3.1 weights (Meta declined access) or YOLO26 / Ultralytics AGPL-3.0. See [`docs/adr/2026-11-18-open-weights-and-permissive-licence-assumptions.md`](docs/adr/2026-11-18-open-weights-and-permissive-licence-assumptions.md).
- **Follow Skalski's pipeline:** Follow Piotr Skalski's Roboflow basketball pipeline (`docs/PLAN.md` section 3); deviate only where his posts note a limitation or where PLK differs from the NBA.
- **Metrics-row gate:** No heuristic or backend change lands in a pipeline layer without a before/after row in `docs/metrics.md` against the gold set (demo and labelling aids are not layers).
- **Swappable backends:** Every layer is a backend registered with `@register` and selected in `configs/*.yaml`. Add a new approach as a backend + config, never as a branch inside the pipeline. Metrics rows cite the config.
- **Training provenance & gold integrity:** Every fine-tune starts from a committed `configs/train/*.yaml`. Weights live at `weights/<layer>/<backend>/<name>@<hash8>/` with `train.json`; metrics rows cite both inference and training configs. Trainer validation numbers go to `docs/train.md`, never `docs/metrics.md`. No training set may contain any gold frame (dev or held-out).
- **Start from scratch:** Copy nothing from the prior repo's `data/`, `weights/`, or `runs/` directories (`tko-ventures/basketball-tracking`, branch `feat/labeller-rosters`).
- **Labelling:** Use your own vision on contact sheets with a ledger (`data/labels/sheet_ledger.tsv`), no third-party VLM API keys. Do every first pass and send Tomasz only geometry or genuinely ambiguous cases.
- **Secrets & spend:** Modal credentials stay in `~/.modal.toml` and Roboflow keys in `~/.config/bga/roboflow.env` / Modal secret `roboflow`; never commit tokens. Spend cap is **75 USD/month metered** (before credits); ask Tomasz before crossing it; log jobs in `docs/costs.md`.
- **Git hygiene:** Never commit videos, weights, crops, or run outputs. Label files and rosters are committed. Commit freely on `feat/*` branches; never push to `main` and never merge a pull request without Tomasz.
- **Untrusted downloads:** Keep third-party downloads and clones in their own directories and never run interpreters from inside them.
- **Language:** Code, docs, paths, commit messages, PRs, and demo UI are English only. Tomasz occasionally dictates in Polish (expect typos like "wersal" for Vercel); always reply in English.

## Working rules and technical gotchas

Each item below closes a real failure mode. Follow them as strictly as the hard rules.

- **Commits go through pre-commit:** Run `uv run pre-commit install` once per checkout and commit with `uv run git commit` so hooks find `pre-commit` across shared worktrees. Stage files before running hooks by hand (`pre-commit run --all-files` skips untracked files). Never chain a commit after hooks with `;` (use `&&`). Never pipe `pytest` into `tail` before committing (the exit code is lost).
- **Squash-merged stacked branches:** Tomasz merges PRs with Squash and merge. After a squash, verify `main`'s tree matches the merged branch's tip and the next branch contains that tip, then record `main` with `git merge -s ours` (no history rewrite or force-push); otherwise do a real merge.
- **Respect Tomasz's live labelling:** Before touching `data/labels/`, run `git status`. Treat uncommitted label edits as Tomasz's work — commit them as his and never overwrite them. Do not restart `tools/labeler/server.py` while he may be using it. Test any new labeller keyboard/mouse instructions on a copy of the queue before sharing them.
- **Spend comes from platform billing, never estimates:** Use `scripts/modal_spend.py` (Modal's billing) for Modal jobs and `colab usage` (Compute Units) for Colab jobs in `docs/costs.md` and any figure reported to Tomasz. Local polling wait time is not remote GPU time.
- **Training progress:** Read the `epoch` column of `metrics.csv` on the Modal volume; "best metric improved" logs do not show current progress.
- **Long background jobs:** Start them in the background and rely on completion notices. When waiting on a condition, print a fresh end marker (`echo EXIT $?`) instead of matching stale text or calling `exit` in a loop condition. Check `pgrep -f` before launching another copy.
- **Re-collecting weights invalidates caches:** `bga train --collect` deletes cached outputs made with the earlier checkpoint of the same weight ID (`invalidate_runs`) and drops the volume's copy. Never evaluate weights while a collect of the same ID may still run.
- **New Modal images:** Start from `bga.remote.modal_runner.image` (`python:3.12-slim-bookworm`; Modal's `debian_slim` is bullseye with dead apt archives, and GitHub commit archives may omit subpackages that a git install includes). Verify imports in a build step (`run_commands("python -c 'import ...'")`) so broken images fail before waiting for a GPU.
- **Checkpoint class names:** Read class names from `model.class_names`, never from dataset COCO categories (which begin with a placeholder class the model drops).
- **Metrics rows come after the commit:** Commit the code that produces a row first so the row cites a commit containing it.
- **`--set` applies to every `--config`:** When comparing configs with different overrides, invoke `bga eval*` separately.
- **Homography sign:** Stored `H` is normalised to `h33 = 1`, which flips sign whenever the image origin is above the floor horizon. Decide "in front of the camera" at a floor point in the image via `bga.court.propagate.court_to_image`, never by the sign of stored `H`.
- **OpenCV override:** `pyproject.toml` overrides `opencv-python` and `opencv-contrib-python` away in favour of `opencv-python-headless`. If `cv2` lacks `VideoCapture`, run `uv sync --reinstall-package opencv-python-headless`.
- **Serving local video:** Python's `http.server` lacks HTTP Range support (breaking seeking); use `scripts/serve_demo.py`.
- **After `ruff format`:** Re-read formatted files before making follow-up string edits.

## Prior work

Private repo `tko-ventures/basketball-tracking`, branch `feat/labeller-rosters`. Only the components listed in `docs/PLAN.md` section 4 were imported (labeller, fetch/prepare scripts, rosters, shot files, shot-cut detector, churn metrics, ONNX adapter pattern, identity rules). The old OCR gate stack, HSV kit-colour cue, fixed trapezoid, and SAM 3.1 Core ML service were intentionally left behind.

## Tooling

Python 3.12 with `uv`. `uv run pre-commit run --all-files` and `uv run pytest` (which also runs doctests in `src/` and `tools/`) must pass before any merge. Stage outputs live under `runs/<clip>/<layer>/<backend>@<hash8>/`.

## Python style

Python follows the style of Jirka Borovec ([Borda](https://github.com/Borda)), the SME Tomasz chose. Sources: [Borda/AI-Rig](https://github.com/Borda/AI-Rig/tree/main/plugins/cc_foundry/rules) (`python-code.md`, `python-testing.md`, read at `30b7bb97`) and the ruff setup of `Borda/vision-synth` and `Borda/lucid-YOLO`. Where this file and his rules differ, this file wins.

- **Hooks, not bare tools:** Lint and format run through the pinned hooks in `.pre-commit-config.yaml`: ruff check and format, docformatter, codespell, and file hygiene. Use `uv run pre-commit run <hook-id> --files <path>` for spot checks, not `ruff` directly. Rule sets and limits are in `pyproject.toml` `[tool.ruff]`. Lines are 120 characters.
- **Docstrings:** Google style on every public module, class, function, and method. The first line says what the object is for in plain words. Use `Args:`, `Returns:`, `Raises:`, and `Examples:` sections. Every public function in `src/` has an `Examples:` doctest run under `pytest`. Mark `# doctest: +SKIP` only when it needs weights, video, the network, or a GPU. Document module-level constants with a `#:` comment above them, not a string after them.
- **Types:** Annotate every public interface. Keep fixed-shape internal records out of bare dicts: use a `@dataclass` (`frozen=True` when immutable, `slots=True` when there are many). Use a `NamedTuple` for tuple-unpacked values. Use `pydantic` only at config and file boundaries, where it is already used. A closed option set is a `class X(str, Enum)` at module boundaries, or a `Literal` for a local annotation. Values are plain strings in YAML, JSON, and on the CLI, converted once at the edge.
- **Complexity limits:** Enforced by ruff: McCabe ≤ 12, branches ≤ 12, statements ≤ 50, returns ≤ 6, arguments ≤ 12. A function needing more than 7 required arguments takes a config dataclass instead. Refactor first. Any `# noqa` needs an inline reason.
- **Code quality:** No mutable default arguments, no `import *`, no bare `except:` that swallows errors, and no global mutable state. Use `__all__` in public `__init__.py` files and never in private `_module.py` files. Prefer composition over deep inheritance. Pass `weights_only=True` to `torch.load`.
- **Library APIs:** Before calling a third-party feature, check the installed version and signature (`uv pip show`, `help()`, or source). Memory is not evidence. Never upgrade a pinned library just to reach an unfamiliar API.
- **Deprecation:** Deprecating a public `bga` function or class uses `pyDeprecate` (`from deprecate import deprecated`), never bare `warnings.warn`.
- **Tests:** Use `pytest` with arrange–act–assert, one scenario per test, and no `if`, `for`, or `while` in test bodies. Use `@pytest.mark.parametrize` with `pytest.param(..., id="slug")` cases. Group related tests in `Test<Subject>` classes so method names state only the expected outcome. Shared setup goes in fixtures (`conftest.py` across files). Use `monkeypatch` or `mock.patch(..., autospec=True)` at the point of use, mocking only process boundaries (network, clock, subprocess, randomness), never our own logic. Never soften a failing test: no `try` in the body, no widened tolerance, no `xfail` without `raises=` and an issue link. Seed RNGs only in an autouse fixture.
- **Scripts:** `scripts/` is exempt from annotation, docstring, naming, and bandit rules (still requires a module docstring). Code in `src/bga/` and `tools/` gets no exemption.
