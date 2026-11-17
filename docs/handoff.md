# Handoff prompt for the next agent session (2026-11-17)

Paste everything below the line as the first message of a new session.

---

You are the implementer for basketball-game-analyzer (https://github.com/tko-ventures/basketball-game-analyzer), an offline pipeline that analyses ORLEN Basket Liga (PLK) games from the broadcast camera: detect players, classify team by kit, track without identity confusion, read jersey numbers, map players and ball onto a 2D FIBA court. The owner is Tomasz Kopacki (tko). He often writes in Polish, dictated, so expect typos like "wersal" for Vercel; reply in English all the same (his choice since 2026-11-02). Everything in the project is English: code, app paths and URLs, docs, commit messages, issue comments and PRs. The demo pages' on-screen text is Polish for the client.

## Read first, in this order

1. `CLAUDE.md`: the hard rules.
2. `docs/architecture.md`: what is built.
3. `docs/PLAN.md`, especially sections 5, 8 and 9.
4. `docs/decisions.md`: the entries from 2026-11-02 on.
5. GitHub issue #4 (M3) with all its comments, and the last comment on #8 (M7).
6. Then run `git switch feat/m3-court && uv sync --all-extras`. All current work is on that branch.

## Where things stand

### Branches and PRs

- `main` holds M0, M1 and M2, squash-merged from PRs #11, #12 and #13. Issues #1 to #3 are closed.
- `feat/m3-court` is the one working branch. It also carries the coach page and the M7 scope (decision 2026-11-17). Its PR opens when M3's done-when is met.
- `feat/m7-numbers` and `feat/coach-demo` are contained in `feat/m3-court`, so don't build on them.
- Old worktrees exist at `../basketball-game-analyzer-m2` (`feat/m2-baseline`, merged) and `../basketball-game-analyzer-m3` (`feat/m3-train`, a stale local branch).

### M3 (#4): in progress

- **Court keyframes:** labelled, 41 of 42 usable. The format is in `docs/architecture.md`, and the counts and time are in `docs/gold.md`. The labelling page is `/court` on the labeller; `?set=redo` opens the second-pass queue.
- **Not started:** training. There is no `bga train`, no `configs/train/` and no `docs/train.md`.

### Demo site

- **Live:** https://basketball-analyzer-demo.vercel.app hosts these pages:

  | Path | Page |
  |---|---|
  | `/`, `/analysis/` | Polish video and analysis pages |
  | `/en/`, `/en/analysis/` | English editions |
  | `/coach/`, `/en/coach/` | coach page (simulated data) |
  | `/roadmap/` | roadmap |

  How each is built is in `docs/architecture.md`.
- **Commands:**
  - Publish: `scripts/deploy_demo.sh`. The Vercel CLI is logged in, with scope `tomasz-kopackis-projects`.
  - Build the video and analysis pages: `uv run --extra prefill --extra demo scripts/demo_web.py`.
  - Check before showing anyone: `uv run scripts/demo_audit.py`.
- **Status:** a showcase, not a pipeline layer, with no metrics row. If you change shared helpers (`scripts/demo_video.py`, `scripts/demo_web.py`), rebuild and check every page.

## Findings: do not rediscover these

- **Court model:** Skalski's `basketball-court-detection-2/14` maps the Warsaw hall well. It misplaces landmarks in Lublin (a consistent shift that RANSAC cannot catch) and mostly fails in Sopot. Check any hall with `scripts/qa_court_overlay.py`. M3 has to fine-tune it on every PLK hall.
- **Court keyframes against the NBA model** (`scripts/probe_court_keyframes.py`, 35 usable keyframes, held-out game excluded): the model is a median 1.81 m off. By hall, the median is 0.7 m in Warsaw, 4.2 m in Sopot and 6.4 m at Tasomix Rosiek Stal.
- **Homography sign:** the labeller saves H with h33 = 1, which flips its overall sign whenever the image origin lies above the floor's horizon. Decide "in front of the camera" at a floor point in the image (`bga.court.propagate.court_to_image`), never by the sign of a stored H or at a court point. From a low camera, part of the court is truly behind the camera (King 822).
- **Propagation drift:** a keyframe carried through its shot reaches every frame. Round-trip drift is a median 0.25 m over 1 s and 0.47 m over 3 s, with outliers of 3.7 m (Warsaw shot 14, over 1 s) and 5.3 m (Sopot shot 3, over 3 s). Don't take whole propagated shots as labels without a drift gate.
- **Keyframe residuals:** the saved courts fit Tomasz's dots to a median of 5.6 px rms, with 10–14 px on the worst frames (Lublin, the held-out game, Sopot). That may be lens distortion, which a homography cannot absorb.
- **SAM drift:** SAM 2.1 drifts once a mask loses its player. The fusion therefore keeps SAM-only boxes only in gaps of up to 3 sampled frames.
- **Number reader:** `basketball-jersey-numbers-ocr/3` is a SmolVLM fine-tune. Load it with `inference_models.AutoModel` on MPS and `max_new_tokens=4`: that takes it from 8.3 s to 2.0 s per crop. `inference` ≥ 1.8 dropped the notebook's `predict(img, prompt)` call; use `infer(img, prompt=...)`.
- **Player positions:** the foot point (box bottom-centre) is unreliable when the box touches the bottom edge of the frame. The demo excludes those frames and clamps players, but not referees, to the 28 × 15 m court.
- **OpenCV:** `pyproject.toml` overrides `opencv-python` and `opencv-contrib-python` away, so there is one headless cv2. If cv2 lacks `VideoCapture`, run `uv sync --reinstall-package opencv-python-headless`.
- **Downloads:** yt-dlp is throttled unless it uses chunked requests; the fetch script already does.
- **Modal:**
  - The `debian_slim` base is bullseye, which is end-of-life, so don't apt-install anything. sam2 installs from the GitHub tarball at a pinned commit.
  - Log the cost of every job in `docs/costs.md`: 3.60 USD so far, against a proposed cap of 50 USD per month.
- **Serving video:** Python's plain `http.server` has no Range support, so videos cannot seek. Use `scripts/serve_demo.py`.
- **Process traps:**
  - Never pipe pytest into `tail` before a commit; the exit code is lost.
  - After `ruff format`, string patches against old text fail silently. Re-read the file and check that the patch applied.

## Rules that bite

- **Tomasz labels in the labeller himself between sessions.** Before touching `data/labels/`, run `git status`. Treat uncommitted label edits as his work: commit them as his and never overwrite them. Do not restart `tools/labeler/server.py` while he may be using it.
- **Gold integrity:**
  - Gold labels need human review; prefill from a model under test is only a starting point.
  - No gold frame may appear in any training manifest.
  - No heuristic lands in a pipeline layer without a before/after row in `docs/metrics.md`. Demo and labelling aids are not layers.
- **Git:** never push to `main` and never merge. Commit freely on `feat/*`. Tomasz squash-merges PRs; after each, record `main` into the next branch with a no-change merge (decision 2026-11-17).
- **`docs/architecture.md`:** update it in the same commit as any change to what it describes, with ground truth only.
- **Questions:** ask Tomasz only the plan's section 12 questions, as comments on the milestone issue, and proceed with the stated defaults.

## Next steps, in order

1. **Training setup:**
   - `bga train --config`;
   - `configs/train/`;
   - weights under `weights/<layer>/<backend>/<name>@<hash8>/` with a `train.json`;
   - `docs/train.md`;
   - `tests/test_gold_leakage.py`.
2. **Court training labels** from the 17 usable training keyframes: propagate them with a drift gate, and keep every gold clip out.
3. **First RF-DETR keypoint fine-tune on Modal,** with the recipe in issue #4. Log the cost in `docs/costs.md`.
4. **Court backend** with the fine-tuned weights, temporal smoothing within shots, then the per-arena reprojection-error row against the 24 gold keyframes. Score the held-out game once. Then open the M3 PR.
5. **Waiting on Tomasz:** full-game footage (plan question 4) and the Modal spend cap (question 9).
