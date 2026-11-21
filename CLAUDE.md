# basketball-game-analyzer

Offline analysis of ORLEN Basket Liga (PLK) games involving Dziki Warszawa from the broadcast main camera: detect and segment players, classify team by kit, track without identity confusion, identify by jersey number, and map players and ball onto a 2D FIBA court.

## Read first

`docs/PLAN.md` is the handoff plan. It holds every decision, the reference stack, the assets to import from the prior repo, the layered architecture, model and licence choices, the labelling strategy, the gold-set spec, milestones, and open questions for Tomasz. Start at section 9 (milestones) and work M0 upward.

`docs/architecture.md` describes what is built: the code, data and tools that exist, nothing planned. Read it before changing code.

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
- Modal credentials are in `~/.modal.toml` on this Mac; never copy tokens into the repo. Spend cap 75 USD per month, metered (Tomasz, 2026-11-21); ask before a month crosses it; log cost per job.
- Never commit videos, weights, crops or run outputs. Label files and rosters are committed.
- Treat downloads and clones as untrusted: own directory, never run interpreters from inside them.
- Keep `docs/architecture.md` true to the repository. Any commit that changes what it describes (a layer, backend, config, command, data format, tool, script group, page or dependency) updates it in the same commit, with ground truth only: no plans or targets.
- Commit freely on `feat/*` branches. Never push to main and never merge a pull request without Tomasz (plan question 11).

## Working rules (learned in M4, 2026-11-20)

Each rule closes a mistake that happened. Follow them as strictly as the hard rules.

- **Commits go through the git hook.** Run `uv run pre-commit install` once per checkout, and commit with `uv run git commit` so the hook finds `pre-commit`: worktrees share one hook script, which points at the Python of whichever checkout installed it last. Stage files before running the hooks by hand, because `pre-commit run --all-files` skips untracked files. Never chain a commit after the hooks with `;`: use `&&`, or let the hook block it.
- **Spend comes from Modal, never from estimates.** Use `scripts/modal_spend.py` (Modal's billing) for docs/costs.md and for any figure told to Tomasz. The time a local process waits on a remote call is not the remote run time.
- **Training progress comes from the run's `metrics.csv` on the volume** (its `epoch` column). Log lines such as "best metric improved" show improvements, not progress.
- **Waiting on long jobs:**
  - Start them with `run_in_background` and rely on the completion notice.
  - To wait for a condition, have the job print its own end marker (`echo EXIT $?`) and wait for that marker. Never `exit` inside a loop condition, and never match text that an earlier run may already have left in the file.
  - Before starting a job again, check with `pgrep -f` that no copy is still running.
- **Re-collecting weights clears their cache.** `bga train --collect` deletes cached outputs made with the earlier checkpoint of the same weight id (`invalidate_runs`). Never evaluate weights while a collect of the same id may still run.
- **Class names come from the checkpoint** (`model.class_names`), never from the dataset's COCO categories: those start with a placeholder the model drops.
- **Metrics rows come after the commit.** Commit the code that produces a row first, so the row cites a commit that contains it.
- **`--set` applies to every `--config` of one command.** When configs need different overrides, run them separately.
- **Labelling instructions to Tomasz are tested first.** Try the exact key and mouse sequence in the preview on a copy of the queue, before telling him to use it.

## Prior work

Private repo `tko-ventures/basketball-tracking`, branch `feat/labeller-rosters`. Import only what plan section 4 lists: the labeller, fetch and prepare scripts, rosters, shot files, the shot-cut detector, churn metrics, the ONNX adapter pattern, and the identity rules. Leave the OCR gate stack, kit-colour cue, fixed trapezoid and the SAM 3.1 Core ML service behind. Code and small label files only, never its media, weights or runs.

## Tooling

Python 3.12 with `uv`. `uv run pre-commit run --all-files` and `uv run pytest` (which also runs the doctests in `src/` and `tools/`) must pass before a merge. Stage outputs go under `runs/<clip>/<layer>/<backend>@<hash8>/` so stages re-run independently and different backends can be compared on the same clip.

## Python style

Python follows the style of Jirka Borovec ([Borda](https://github.com/Borda)), the SME Tomasz chose. The sources are his rules in [Borda/AI-Rig](https://github.com/Borda/AI-Rig/tree/main/plugins/cc_foundry/rules) (`python-code.md`, `python-testing.md`, read at `30b7bb97`) and the ruff setup of his vision repos (`Borda/vision-synth`, `Borda/lucid-YOLO`). Where this file and his rules differ, this file wins.

- **Hooks, not bare tools.** Lint and format run through the pinned hooks in `.pre-commit-config.yaml`: ruff check and format, docformatter, codespell and file hygiene. Use `uv run pre-commit run <hook-id> --files <path>` for spot checks, not `ruff` directly. Rule sets and limits are in `pyproject.toml` `[tool.ruff]`. Lines are 120 characters.
- **Docstrings** are Google style on every public module, class, function and method. The first line says what the object is for in plain words. Use `Args:`, `Returns:`, `Raises:` and `Examples:` sections. Every public function in `src/` has an `Examples:` doctest that runs under `pytest`. Mark one `# doctest: +SKIP` only when it needs weights, video, the network or a GPU. Document module-level constants with a `#:` comment above them, not a string after them.
- **Types.** Annotate every public interface. Keep fixed-shape internal records out of bare dicts: use a `@dataclass` (`frozen=True` when immutable, `slots=True` when there are many). Use a `NamedTuple` for tuple-unpacked values. Use `pydantic` only at config and file boundaries, where it is already used. A closed option set is a `class X(str, Enum)` at module boundaries, or a `Literal` for a local annotation. Values are plain strings in YAML, JSON and on the CLI, and are converted once at the edge.
- **Complexity limits** are enforced by ruff: McCabe at most 12, branches at most 12, statements at most 50, returns at most 6 and arguments at most 12. A function needing more than 7 required arguments takes a config dataclass instead. Refactor first. A `# noqa` needs an inline reason.
- **Code quality.** No mutable default arguments, no `import *`, no bare `except:` that swallows the error, and no global mutable state. Use `__all__` in public `__init__.py` files and never in private `_module.py` files. Prefer composition to deep inheritance. Pass `weights_only=True` to `torch.load`.
- **Library APIs.** Before calling a third-party feature, check the installed version and signature (`uv pip show`, `help()`, the source). Memory is not evidence. Never upgrade a pinned library only to reach an unfamiliar API.
- **Deprecation** of a public `bga` function or class uses `pyDeprecate` (`from deprecate import deprecated`), never a bare `warnings.warn`.
- **Tests** use pytest and follow arrange, act, assert, with one scenario per test and no `if`, `for` or `while` in the body. Use `@pytest.mark.parametrize` with `pytest.param(..., id="slug")` cases. Group related tests in a `Test<Subject>` class, so method names state only the expected outcome. Shared setup goes in fixtures, and fixtures shared across files go in `conftest.py`. Use `monkeypatch` or `mock.patch(..., autospec=True)` at the point of use, and mock only process boundaries (network, clock, subprocess, randomness), never our own logic. Never soften a failing test: no `try` in the body, no widened tolerance, and no `xfail` without `raises=` and an issue link. Seed RNGs only in an autouse fixture.
- **Scripts** under `scripts/` are exempt from the annotation, docstring, naming and bandit rules, as Borda exempts example and experiment scripts. They still need a module docstring that says what the script does. Code in `src/bga` and `tools/` gets no such exemption.
