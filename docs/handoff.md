# Handoff prompt for the next agent session (2026-11-01)

Paste everything below the line as the first message of a new session.

---

You are the implementer for basketball-game-analyzer (https://github.com/tko-ventures/basketball-game-analyzer), an offline pipeline that analyses ORLEN Basket Liga (PLK) games from the broadcast camera: detect players, classify team by kit, track without identity confusion, read jersey numbers, map players and ball onto a 2D FIBA court. The owner is Tomasz Kopacki (tko). He writes to you in Polish, often dictated, so expect typos like "wersal" for Vercel. Answer him in Polish. Keep code, docs, commit messages, issue comments and PRs in English.

## Read first, in this order

1. `CLAUDE.md`: the hard rules.
2. `docs/PLAN.md`, especially sections 2, 5, 7, 8, 9 and 12.
3. `docs/decisions.md`: every dated entry up to 2026-10-31. Several of them change the plan (answers to questions 1, 11, 12, 13 and 14, the demo, and the court-model finding).
4. GitHub issues #1 and #2 with all comments, and the last comment on #4.
5. Then run `git switch feat/m1-gold-set && uv sync`. All current work is on that branch.

## Where things stand

### Branches and PRs

- `main` holds the plan only.
- `feat/m0-bootstrap` is PR #11. M0's done-when is met, and the PR waits for Tomasz to merge.
- `feat/m1-gold-set` is stacked on M0. It holds all M1 work so far plus the client demo, and has no PR yet. After #11 merges, rebase it onto `main` and open the M1 PR ("Closes #2").

### M0 (#1): done

- **Clips:** 31, not 32; the extra prior label file was a duplicate download. All are H.264, mostly 1080p50. Re-download with `scripts/fetch_highlights.sh` into `data/raw/` (gitignored).
- **Shot files:** regenerated and cross-checked against the imported ones (`docs/m0-bootstrap.md`).

### M1 (#2): in progress

- **Decisions:** questions 1, 11, 12 and 14 are answered. The gold games are in `data/labels/gold.json`:
  - `4XMD9FWmQws` PGE Start – Dziki, 2024/25;
  - `OSp445e0mXU` Dziki – Legia, 2025/26;
  - `1jDaJBkUGmc` Energa Trefl – Dziki, 2025/26;
  - `Tnuscrm9RyU` Dziki – Zastal, 2026/27, held out.

  That's 12 central shots, 121.7 s in total. All 198 shots of these clips have camera tags.
- **Rosters:** 19 clubs × 3 seasons in `data/rosters/` (from `scripts/fetch_rosters.py`). Every clip side resolves to a roster file.
- **Prefill, two stages:**
  - Skalski's player model (`basketball-player-detection-3-ycjdo/4`) runs locally through Roboflow `inference` (`scripts/prefill_tracks.py`). ByteTrack on its boxes fragments badly: 752 tracklets for about 156 people.
  - SAM 2.1 hiera-large on Modal fixes most of that, down to 190 tracks. The pieces are `src/bga/remote/sam2_prefill.py`, `scripts/extract_shot_frames.py` and `scripts/sam_to_tracks.py`.

  The SAM tracks are imported into the gold label files with `scripts/import_prefill.py`.
- **Labeller:** has track-level review. A filmstrip of 8 crops per track, plus these actions:

  | Key | Action |
  |---|---|
  | K | confirm |
  | P | split |
  | M | merge |
  | ⇧Del | delete track |
  | ⇧D | accept shot |

  Clicking a track jumps to it, and active time is logged per day in `meta.review_seconds`. The implementer's first pass is an auditable ops file per clip (`data/labels/review/<id>.ops.txt`, applied by `scripts/review_ops.py`; sheets from `scripts/track_sheets.py`). Only Legia shot 10 has had a first pass.
- **Dry run:** Tomasz spent 29.9 min on Legia shot 10 and confirmed 9 of 16 tracks (commit `3d9765e`). The 90-minute budget cannot cover track review of 12 shots. Options A, B and C are on #2. **A is the working default but Tomasz has not confirmed it:** Tomasz reviews the held-out game, and the implementer does the full pass on the other 9 shots.
- **Footage:** Tomasz finds jersey numbers hard to read in the YouTube highlights, which makes plan question 4 (full-game footage) relevant now.

### Client demo

This interrupted M1 on 2026-10-31; the decision is recorded.

- **Live pages:**
  - https://basketball-analyzer-demo.vercel.app shows two synchronised videos.
  - https://basketball-analyzer-demo.vercel.app/analiza/ draws a live 2D analysis in the browser: nearest-defender coverage and the most open attacker, team shape and spacing, and player selection with speed and distance.

  Both use three Warsaw shots of `OSp445e0mXU` (s2, s10, s19), each shot with independent control state.
- **Commands:**
  - Build: `uv run --extra prefill --extra demo scripts/demo_web.py`. It caches number reads and homographies in `runs/demo/cache`, so re-renders take seconds.
  - Publish: `scripts/deploy_demo.sh`. The Vercel CLI is logged in; the scope is `tomasz-kopackis-projects`.
  - Serve locally: `uv run scripts/serve_demo.py`.
  - Page templates: `scripts/demo_page_video.html` and `scripts/demo_page_analysis.html`.
- **Status:** a showcase, not a pipeline layer, with no metrics row. Keep both pages working. If you change shared helpers (`scripts/demo_video.py`, `scripts/demo_web.py`), rebuild and check both pages.

## Findings: do not rediscover these

- **Court model:** Skalski's `basketball-court-detection-2/14` maps the Warsaw hall well. It misplaces landmarks in Lublin (a consistent shift that RANSAC cannot catch) and mostly fails in Sopot. Check any hall with `scripts/qa_court_overlay.py`. M3 has to fine-tune it on every PLK hall.
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
- **Git:** never push to `main` and never merge. Commit freely on `feat/*`.
- **Questions:** ask Tomasz only the plan's section 12 questions, as comments on the milestone issue, and proceed with the stated defaults.

## Next steps, in order

1. **Post one short comment on #2** asking Tomasz three things, then proceed on the defaults:
   - gold-set scope A, B or C (default A);
   - plan question 4: does the club have full-game footage (Hudl Instat or Synergy)?
   - to merge PR #11.
2. **If A stands**, record it in `docs/decisions.md`, and change plan section 8 only through that dated entry.
3. **Finish the implementer's pass on the 9 non-held-out shots.** For each shot:
   - render filmstrip sheets with `scripts/track_sheets.py`;
   - read them;
   - write the ops file: merge, split, delete off-court people, team, number when legible, notes, and confirm when sure;
   - apply it with `scripts/review_ops.py`;
   - fill short gaps between confirmed boxes by interpolation;
   - set `num_visible` where a number is legible.

   Keep a ledger row for every sheet in `data/labels/sheet_ledger.tsv`. Also do a first pass on the 3 held-out shots before Tomasz sees them.
4. **Box gold:** about 100 frames drawn evenly from the 12 shots. Tighten the boxes yourself; Tomasz spot-checks.
5. **Number crops:** about 300, cut from boxes flagged `num_visible`. Label them on contact sheets of about 20 crops, with the ledger. Tomasz spot-checks 30.
6. **Landmarks:** 6 frames per arena, prefilled by the court model, with high-RANSAC-residual points flagged. Tomasz corrects only the flagged points (about 20 min). Expect many flags outside Warsaw.
7. **Tomasz's session or sessions** on the held-out game's track review. Read his time from `meta.review_seconds`.
8. **Export and commit the gold set:** MOT gt, `identities.csv`, crop labels, landmark labels, with the held-out game marked. Then open the M1 PR with the time log, what fell short, and what M2 needs from Tomasz. Then M2.
