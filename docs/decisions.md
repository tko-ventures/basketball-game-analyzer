# Decisions

Dated entries for every change to a decision in `docs/PLAN.md`. Newest last.

## 2026-10-29: answers to plan questions 1, 11, 12 and 14

Tomasz, in the M0 session.

- **Q1, Roboflow.** A workspace exists (`tomasz-kopacki`). The key lives outside the repo: `~/.config/bga/roboflow.env` (sourced from `~/.zshenv`, mode 600) and the Modal secret `roboflow`, both as `ROBOFLOW_API_KEY`. Used for Universe dataset downloads and `inference` prefill during labelling. Labels uploaded to Roboflow go to the project `tomasz-kopacki/basketball-game-analyzer` (instance segmentation, empty on 2026-10-29), not the older `basketball-player-detecion`. Training credits are not visible through the API, so the training default stays Modal.
- **Q11, commit policy.** Default confirmed: commit on `feat/m<N>-<slug>`, one pull request per milestone, never push to main or merge without Tomasz.
- **Q12, gold games.** Confirmed as proposed on #2: `4XMD9FWmQws` (PGE Start – Dziki, 2024/25), `OSp445e0mXU` (Dziki – Legia, 2025/26), `1jDaJBkUGmc` (Energa Trefl – Dziki, 2025/26), and held out `Tnuscrm9RyU` (Dziki – Zastal, Superpuchar, 2026/27).
- **Q14, rosters.** Collect every PLK team's roster now, same schema as `data/rosters/dziki_warszawa.json`.

## 2026-10-31: client demo before the gold set is finished

Tomasz needs a demo for a client first; precision is not required, only the capabilities. Order
changes: an end-to-end demo video (detection, SAM 2.1 tracks, teams, jersey numbers, 2D FIBA court)
is built from the M1 prefill and Skalski's public models before M1 closes. It is a showcase with no
metrics row; `scripts/demo_video.py` states that its output is automatic and uncorrected. M1 resumes
afterwards.

Dry run (Legia shot 10): 29.9 min of Tomasz's time for 9 of 16 tracks, so the 90-minute budget does
not cover track review of 12 shots. Options A/B/C are on issue #2; A (Tomasz reviews the held-out
game, the implementer the rest) is the working default until he decides. Tomasz also finds the
jersey numbers hard to read in the YouTube highlights: better footage (question 4, full games via
Hudl Instat or Synergy) would help the gold set and the number reader.

## 2026-10-31: demo as a web page; NBA court model fails in two of three PLK halls

Tomasz asked for the broadcast view and the 2D court as two separate, synchronised videos on one web
page, each expandable to full screen (`scripts/demo_web.py`, `scripts/serve_demo.py`), and for 2D
positions as key positions joined by interpolation (implemented: one median key position per 0.5 s,
joined by a monotone cubic).

Checked with `scripts/qa_court_overlay.py` (FIBA lines drawn back into the frames): Skalski's
`basketball-court-detection-2/14` maps the Warsaw hall (blue keys) well, but systematically misplaces
landmarks in Lublin (red keys; a consistent shift that RANSAC cannot detect) and mostly fails in
Sopot (orange floor). The demo therefore uses three Warsaw shots, and the 2D panel says so where the
mapping is not verified. This confirms plan layer 1 / M3: the court keypoint model must be fine-tuned
on frames from every PLK hall before 2D positions can be trusted elsewhere.

## 2026-10-31: client demo published on Vercel

At Tomasz's request the web demo is deployed to Vercel production, project
`basketball-analyzer-demo` in the `tomasz-kopackis-projects` scope:
https://basketball-analyzer-demo.vercel.app (public, `noindex`; three Warsaw shots of Dziki – Legia,
automatic and uncorrected). Redeploy with `scripts/deploy_demo.sh` after `scripts/demo_web.py`.

## 2026-10-31: demo footage and hosting (question 13, partly)

Tomasz: the Vercel page is a capability demo, not a product, and any PLK game would do; showing
YouTube highlight clips there is acceptable. The page stays public with `noindex`. Training on the
highlights at scale (question 13 proper) stays on its default. The 2D view now shows players only
(referees remain labelled in the broadcast view), and trails break at gaps and jumps.

## 2026-11-01: English for code, docs and app paths

Tomasz: code, documentation, prompts and paths in the application are English. The analysis page
moved from `/analiza/` to `/analysis/`; Vercel redirects the old path permanently. The demo pages'
on-screen text stays Polish for the client.

## 2026-11-02: gold set v0 by the implementer; Tomasz's review deferred

Tomasz: maximum progress now; more of his time goes into the gold set later if the client is
interested. This replaces options A/B/C on issue #2 and amends plan section 8 as follows:

- **Gold v0** is the implementer's full review of all 12 gold shots (tracks, teams, numbers where
  legible) plus about 100 box-gold frames, with Tomasz's dry-run labels on Legia shot 10 kept as
  they are. The held-out game stays held out and is reviewed with the same care, never trained on.
- **Gold v1** follows Tomasz's review, at least of the held-out game. Every metrics row then gets
  re-run from its configs; nothing else changes.
- Every row in `docs/metrics.md` names the gold version it was scored on (`gold=v0` or `gold=v1`).
  v0 was prefilled by the same public models the baseline uses (Skalski's player model, SAM 2.1),
  so v0 absolute scores are biased upwards; they serve to compare backends, and any number shown
  outside the repo says "gold v0".
- Gold is labelled just in time: landmark frames with M3 (court fine-tune), number crops with M7,
  on better footage if the club has full games (question 4, still open; Tomasz is asking).

## 2026-11-02: court landmarks for M3, Tomasz corrects keyframes only

Tomasz corrects only the court points the implementer flags, in one labeller session of about 20 minutes. Flags come from the implementer's visual check of FIBA lines drawn back into each frame, not from RANSAC residuals: in Lublin the NBA model is consistently shifted, so residuals stay low (issue #4).

To fit the budget, he corrects one keyframe per selected shot, moving only the 4 to 6 points needed for a correct court fit; the overlay re-fits as he drags. A labelling aid then propagates that court mapping through the rest of the shot, using frame-to-frame floor matches with players masked. This yields the training frames (plan: 300 to 1,000). Propagation drift is measured on shots corrected at both ends before any propagated label is used. Landmark gold for evaluation is made the same way on the gold games, whose frames never enter training.

## 2026-11-02: correction, two gold games share the Sosnowiec hall

Seen on the court keyframe sheets for M3: `4XMD9FWmQws` ("PGE Start – Dziki", recorded since M1 as Lublin) is a Polish Cup game ("Puchar Polski" on the scoreboard, "Sosnowiec" on the floor). The held-out `Tnuscrm9RyU` (Superpuchar semi-final) was played in the same Sosnowiec arena: same regional sponsor boards, "…owiec" on the floor. The floor paint differs (2024/25 against 2026/27), so the held-out game still has a floor no training clip shows. It is not a hall the gold set has never seen, though, and M1's claim that its floor "appears in no other clip" holds for the paint only. Court results are reported per hall: Warsaw, Sosnowiec 2024/25, Sopot and Sosnowiec 2026/27 (held out). The NBA court model's failures recorded on 2026-10-31 as "Lublin" are Sosnowiec. No PLK clip in the set was filmed in Lublin.

## 2026-11-10: appearance embeddings (Roboflow `reid`) join the scope

Tomasz pointed to https://reid.roboflow.com and asked for the scope to follow. `reid` 0.1.0 (Apache 2.0, released 2026-11-03) turns a player crop into an appearance embedding. Roboflow `trackers` 2.7.0 fuses those embeddings into BoT-SORT, and Roboflow reports that this raises SportsMOT HOTA from 74.1 to 77.8 with the SportsMOT-trained encoder. Since the plan follows Skalski and Roboflow, the scope changes as follows (plan sections 2, 3, 5, 7, 8, 9, 11, 12 and 13):

- **New layer 3b, appearance.** One embedding per box. The backends are `none` and `reid`. The default encoder is `osnet_x1_0_sportsmot` as published, followed by a PLK fine-tune.
- **Layer 4.** `botsort` and `botsort_reid` (Roboflow `trackers`) become tracker candidates, with `mcbyte` at low priority. The mean tracklet embedding becomes one term of the linking cost.
- **Layer 5.** New game-level identity: tracklets are clustered across the shots of one game by appearance, and number votes are pooled per cluster.
- **Milestones.** M6 lands the layer, the `trackers` configs and the encoder fine-tune. M7 adds the game-level pooling. Each has before/after rows, as every layer change does. No new milestone; no milestone moves.
- **Eval.** `bga eval` gains appearance metrics on gold boxes, so they are independent of detector and tracker.
- **Licences.** Question 15 is new: the encoder weights were trained on research-only data. The default is to use them for measurement and as a fine-tune start, and to train a PLK encoder alongside from a backbone with an explicit permissive weight licence (for example DINOv2, Apache 2.0). Plain ImageNet weights are not clean either: ImageNet's terms are research-only.

**Evidence.** The probe `scripts/probe_reid.py` ran on gold v0, on the 9 dev shots, with the held-out game not read. It is a measurement behind this decision, not a metrics row. Each cell is rank-1: the share of queries whose nearest candidate is the right player.

| Encoder | Teammates 1 s later | 3 s | 5 s | Shot halves | Across cuts | ms per crop |
|---|---|---|---|---|---|---|
| Chance | 0.229 | 0.228 | 0.230 | 0.197 | 0.268 | |
| Box centre in pixels, no motion model | 0.777 | 0.509 | 0.397 | n/a | n/a | |
| `osnet_x1_0_sportsmot` | **0.857** | **0.738** | **0.646** | **0.847** | **0.865** | 2.5 |
| `osnet_x1_0_dancetrack` | 0.746 | 0.534 | 0.447 | 0.635 | 0.662 | 3.0 |
| `osnet_x1_0_soccernet` | 0.727 | 0.500 | 0.417 | 0.447 | 0.459 | 2.5 |
| `fastreid_mot17_sbs50` | 0.692 | 0.467 | 0.421 | 0.647 | 0.486 | 5.3 |
| `osnet_x1_0_msmt17_combineall` | 0.672 | 0.503 | 0.423 | 0.647 | 0.608 | 2.5 |
| `osnet_x1_0_mot17` | 0.627 | 0.427 | 0.365 | 0.518 | 0.608 | 2.5 |

How each column was scored:

- **Teammates 1, 3 and 5 s later.** One crop, against the crops of the player's teammates on the floor that much later in the same shot (712 to 1,325 queries, 4.5 candidates on average).
- **Shot halves.** The mean embedding of a player over the first half of a shot, against his teammates' means over the second half (85 queries).
- **Across cuts.** A numbered player's mean embedding in one shot, against his numbered teammates in another shot of the same game (74 queries). By hall: Sopot 0.962, Warsaw 0.909, Sosnowiec 0.731.
- **Speed.** Milliseconds per crop on the M1 Max (MPS).

**Caveats.** Gold v0 boxes come from the same prefill as the tracks. The pixel reference has no motion model or camera compensation, so a real tracker would likely do better at 1 s. The cross-cut samples are small. These numbers justify the scope change; they do not stand in for M6's metrics rows.

## 2026-11-11: demo for Dziki's head coach; English edition and a number check

Tomasz is showing the demo to Marco Legovich, Dziki Warszawa's head coach in 2025/26 and 2026/27 (`data/rosters/dziki_warszawa.json`). The three demo shots come from his own game against Legia (March 2026), so he will judge every name on his players. Changes, all demo aids and not pipeline layers:

- **English edition.** `/en/` and `/en/analysis/` with the same tracks, teams and numbers as the Polish pages; only the text differs (`scripts/demo_strings.py`). Polish stays the default at `/`, as decided on 2026-11-01. Every page links to the other language.
- **Audit against gold.** `scripts/demo_audit.py` compares what the pages show with gold v0 on matched boxes. Before today's change, teams agreed on 1,513 of 1,515 boxes. Numbers were right on 307 boxes and wrong on 195, and two Dziki players carried wrong labels: Horton (#1) shown as "#32" and Edge (#2) shown as "#8 Kempa".
- **Roster check.** A number that is not on the team's roster for that season is no longer shown. This removed the Dziki "#32" (60 boxes); 307 right and 135 wrong remain. The remaining errors are 2 read as 8 (Edge, shot 1), 25 read as 8 (Oguama, shot 2) and Legia's Thompson shown as "#3 Pluta" (shot 2). Shot 3 has no wrong number.
- **Names.** Labels show the name after the first given name ("#0 Vander Plas", not "#0 Plas").

The demo still reads numbers with Skalski's public SmolVLM model; the fix for the 2/8 confusion is M7's fine-tune on PLK crops. Deployed to Vercel production at Tomasz's request on 2026-11-11 (https://basketball-analyzer-demo.vercel.app/en/).

## 2026-11-11: coach page on simulated 2D data

Tomasz asked for synthetic possessions so the demo can show the head coach (Marco Legovich) what accurate 2D tracking gives him, before the camera pipeline can deliver it. Marco watches the game first and then turns to statistics and insights, so the page follows that order:
1. **Watch:** the court with no overlays, possession by possession.
2. **Insights:** the detected play and coverage, a verdict on the shot, and the shot quality, defender distances, spacing and passes, plus clickable key moments (open team-mates found or missed, screens, helps, rotations) and per-player distance and speed.
3. **Feedback:** drawing on the court (pen, arrow, "this player should have been here"), and notes addressed to the team or a player and tagged "well done" or "to improve". Notes are kept in the browser and exported as text grouped by player.

A Game tab adds totals.

- **Data:** eight possessions in `data/fixtures/synthetic/`, written by `scripts/synth_plays.py` (format and method in `docs/fixtures.md`). Offence is choreographed from common set plays; defence follows man-to-man or zone rules with reaction delay and speed limits, plus the scripted decision each play turns on. Dziki's players are the 2026/27 roster. The opponent is anonymous, so no real team is shown losing a made-up possession.
- **Label:** the page carries a "Simulated tracking data" badge and says in its footer that the possessions are simulated. The fixtures are never gold or training data.
- **Format:** the fixture format (`bga.tracking2d/v0`) is the shape the pipeline's 2D output should take, so the page will take real possessions once M3 and M4 deliver them.
- **Status:** a showcase, not a pipeline layer, so no metrics row. The insight rules (`scripts/coach_insights.py`) are demo rules on simulated data. Before any of them becomes a pipeline layer it needs a before/after row against gold like any other heuristic.
- **Built:** with the other demo pages, at `/coach/` and `/en/coach/`, linked from both video pages. Deployed to Vercel production at Tomasz's request on 2026-11-11, after a coach-friendliness pass: every number is coloured from Dziki's side and explains itself when tapped, timeline legend, feedback steps, and layouts for a laptop and a tablet.


## 2026-11-17: number reading, what M7 takes from an outside review

Tomasz shared a jersey-number pipeline guide he worked out with Google's Gemini assistant (roster-constrained PARSeq scoring, kit-colour clustering, shear and CLAHE preprocessing, per-frame uniqueness). Most of it was already in layer 5. The implementer reviewed it against the plan and the demo's reads, and Tomasz agreed to the following.

- **Adopted, as backends with their own configs and metrics rows in M7** (plan section 5, layer 5):
  - `roster_logprob`: score roster numbers plus `unreadable` from the reader's probabilities and sum them per track, instead of decoding free text, filtering by roster afterwards and waiting for 3 equal reads. The sequence probability includes the end token and is renormalised over the candidates; the guide's division by length is not used.
  - `covisible_assign`: teammates' tracks visible in the same frame cannot share a number, solved as an assignment per team and shot with an abstain option for each track. The guide's version runs per frame on already-aggregated scores and always assigns something; a wrong unique number is worse for the coach than none.
  - `parseq` joins the reader bake-off, after Koshkina and Elder's jersey pipeline (CVPR Workshops 2024), fine-tuned from `baudm` weights on PLK crops. Their released models are trained on SoccerNet and hockey data and are not used.
- **Not adopted:** kit colour clustering with k-means (the HSV colour cue was left behind with the prior repo; `siglip_kmeans` agrees with gold v0 on 1,513 of 1,515 demo boxes), test-time shear sweeps and CLAHE (the prior repo's crop preprocessing; slant, blur and contrast go into training augmentation), and the guide's football specifics (goalkeepers, 11 players on the floor).
- **Check on the demo's reads, before building anything.** The demo already applies a greedy one-owner rule per frame (`scripts/demo_web.py`, `numbers_at`). A stricter uniqueness rule fixes none of its three wrong numbers, because the true owner of the wrong number is never confirmed in the same shot:
  - Edge (#2) read as 8, shot 2: Kempa (#8) is not in the shot. Edge's reads over the game's three demo shots are 8 eleven times and 2 eleven times (shot 2: 8×4, 2×1; shot 10: 2×7, 8×1; shot 19: 8×6, 2×3), so pooling across shots ties as well. The 2/8 confusion is systematic for this kit and font.
  - Oguama (#25) read as 8, shot 10: his reads are 8×3, 11, 27; 25 is never read. Kempa's track in that shot has no reads at all.
  - Thompson (#25) shown as #3, shot 10: Pluta's track (#3) has one read, "4". Game-level pooling would fix this one: in shot 2 Thompson's track reads 25 five times out of nine, which the demo's 60% purity bar hid.
  - In shot 19 two Dziki tracks tie at three reads of 4 each, and the owner rule gave #4 to the player gold has no number for. Per-read probabilities, not counts, are what would break such ties.

  So the main fix stays a reader fine-tuned on PLK crops, with 2/8 well represented in the crop set. The roster vote and the assignment are worth measuring for confirmed fraction and wrong-confirmed count, not as the fix for these errors. This is an inspection of demo output, not a metrics row.

## 2026-11-17: squash merges, one working branch, and a second pass on court keyframes

- **Squash merges.** Tomasz merges milestone PRs with "Squash and merge" (#11, #12 and #13 so far). The feature branches are stacked, so each squash makes the next PR conflict even though no file differs. The fix after each merge:
  1. Check that `main`'s tree equals the merged branch's tip.
  2. Check that the next branch contains that tip.
  3. Record `main` into the next branch with `git merge -s ours`, which changes no files.

  If either check fails, do a real merge instead. There is no history rewrite and no force-push.
- **One working branch.** `feat/m7-numbers`, which holds the coach page and the M7 scope, has been merged into `feat/m3-court`. A deploy from `feat/m3-court` would otherwise have dropped the live coach page, and two decision entries were missing on it. The M3 PR will therefore carry the coach page and the M7 scope documents as well.
- **Court keyframes, second pass.** The court labelling page could not fit two kinds of frame:
  - frames whose only visible landmarks lie on one or two lines (Sopot 2310);
  - frames from a low camera, where part of the court is behind the camera (King 822).

  Two fixes followed: pins on painted lines, and a fold check limited to the dots and pins. Tomasz then redid 24 keyframes, and 41 of 42 are usable. These are labelling-aid changes, not a layer, so they need no metrics row.
- **`docs/architecture.md`** records what is built, ground truth only. Every commit that changes what it describes updates it (`CLAUDE.md`).

## 2026-11-17: M3 lands in two pull requests

Tomasz asked to tidy every branch other than `main`. M3's done-when (a per-arena reprojection-error row) is several days away, so `feat/m3-court` goes to `main` now as **M3 part 1**. That PR references #4 but does not close it. It carries:
- the court keyframes and the court labelling page;
- the propagation fixes;
- the coach page and the M7 scope;
- the docs, including `docs/architecture.md`.

After the squash, the court-model work continues on a fresh branch from `main`, and its PR closes #4. This bends the one-pull-request-per-milestone rule for M3 only.

## 2026-11-17: training runs on Modal

Tomasz chose Modal (modal.com) for training. Every fine-tune runs through `bga train --remote modal`, starting with the M3 court keypoint model; RunPod and Roboflow Train are not used for training. The rules from `CLAUDE.md` still apply:
- each run starts from a committed `configs/train/*.yaml`;
- weights and `train.json` sync back into `weights/`;
- the cost of every job goes in `docs/costs.md`;
- spend stays within the proposed cap of 50 USD per month until Tomasz confirms a figure.

## 2026-11-18: re-ID licences re-checked; question 15 stays open

Tomasz noted that https://github.com/roboflow/re-ID is Apache 2.0 and asked for a check. Confirmed: the repository's LICENSE, its `pyproject.toml` and the PyPI metadata of `reid` 0.1.0 all say Apache 2.0, and so do those of `trackers` 2.7.0. The `reid` README limits the grant to the code ("The code is released under the Apache 2.0 license"), and the plan has recorded the code as Apache 2.0 since 2026-11-10. Nothing in M6 changes for the libraries.

The weights are a separate matter, and the check found nothing new for them:
- the torchreid OSNet weights on Hugging Face are tagged MIT;
- `fastreid_mot17_sbs50` comes from the BoT-SORT model zoo, whose repository is MIT;
- Roboflow's four tracking encoders, `osnet_x1_0_sportsmot` among them, are served from a Roboflow storage bucket with no licence file, and their `reid_config.json` has no licence field;
- the training data is still research-only. SportsMOT is CC BY-NC 4.0 per its own repository and dataset card. The `trackers` README lists it as CC BY 4.0; that table is wrong, and the plan follows the dataset's own statement.

So question 15 stays open with its default. Use the pretrained encoders for measurement and as one starting point of the PLK fine-tune. Train a second PLK encoder from a backbone with an explicit permissive weight licence. Settle the question before anything is sold. The `reid` trainer is still unreleased (draft pull requests 2 and 31), so the torchreid fallback stands. Plan sections 2, 3, 11, 12 and 13 are updated.

## 2026-11-18: question 15 answered, re-ID weights treated as Apache 2.0

Tomasz answered question 15: assume Apache 2.0. The project treats the pretrained re-ID weights, `osnet_x1_0_sportsmot` among them, as licensed like the `reid` code. This is his ruling, not a published licence: Roboflow's tracking encoders still carry none, and SportsMOT is still CC BY-NC 4.0 (entry above).

What changes in M6 (plan sections 2, 5, 9, 11 and 12):
- `osnet_x1_0_sportsmot` is a production encoder for layer 3b, not only a measurement;
- the PLK encoder fine-tune has one starting point, `osnet_x1_0_sportsmot`. The second encoder from a backbone with a permissive weight licence (for example DINOv2) is dropped, since its only purpose was a licence-clean alternative;
- the done-when line is unchanged: appearance metrics for the pretrained and the PLK encoder.

If Roboflow publishes different terms for its encoders, question 15 reopens. Any other backbone would come in as a further training config behind the same layer 3b interface.

## 2026-11-18: Python follows Borda's style

Tomasz asked that the project's Python follow the style of Jirka Borovec ([Borda](https://github.com/Borda)), whom he names as the SME. The rules come from Borda's own published sources: his coding and testing rules in `Borda/AI-Rig` (`plugins/cc_foundry/rules/python-code.md` and `python-testing.md`, read at commit `30b7bb97`) and the ruff setups of `Borda/vision-synth` and `Borda/lucid-YOLO`. `CLAUDE.md` now has a "Python style" section that binds all new and changed code.

What changed:
- lint and format run through pinned pre-commit hooks;
- ruff selects his rule set, with 120-character lines and Google docstrings;
- every public function in `src/` has an `Examples:` doctest, and pytest runs them;
- complexity limits are enforced;
- the existing code was brought into line in one pass, with no change in behaviour.

Three choices are ours, not his:
- `scripts/` gets the exemptions he gives example and experiment scripts;
- codespell only reports, because `--write-changes` rewrote `bga.eval.mot` to `bga.eval.not`;
- the en dash and the multiplication sign are allowed in strings, for the on-screen text.

## 2026-11-18: the codebase and the demo site are English only

Tomasz: "just stick to English language in the codebase". Code, comments, docs and paths were already English, so what changes is the demo site. The Polish edition is dropped, and with it the language switch. `scripts/demo_strings.py` and `coach_strings.py` keep only the English text, which moves from `/en/` to `/`. `deploy_demo.sh` redirects `/en/` and `/en/...` to the same page without the prefix.

Polish stays only where it is data, not wording:
- team and player names;
- YouTube titles (the labeller parses "SKRÓT");
- the plk.pl position labels that `fetch_rosters.py` maps to PG, SG and the rest.

This replaces the 2026-11-01 choice of Polish on-screen text for the client.

## 2026-11-18: one demo page, with the PLK court model

Tomasz: "Drop two synchronized videos from the demo website, also, update the analysis view to use fine tuned court detection model."

- The page with two synchronised videos (the broadcast and a rendered 2D court video) is gone, with its template and the court video renders. The analysis page moves from `/analysis/` to `/`; `deploy_demo.sh` redirects `/analysis/` and the older `/analiza/` there.
- `scripts/demo_web.py` no longer runs the NBA Roboflow court model with a Savitzky-Golay smoothing of its own. It runs the court layer of a bga run config, by default `configs/court-plk-v1-smooth4.yaml`, which is M3's measured court setup (0.21 m in the paint on the dev games against 0.89 m for the NBA model, `docs/metrics.md`). The demo renders every 2nd frame, so the smoothing window of 4 frames each way spans 8 source frames, against 20 at the eval's stride of 5.
- The demo stays a showcase with no metrics row.

## 2026-11-18: M4 answers: XL licence, footage, spend cap, labelling time

Tomasz answered the questions M4 needs (plan sections 2, 5, 11 and 12):
- **Question 5, RF-DETR XL licence:** "let's assume it's apache 2.0 — I want to try make things work first, and then make them right". RF-DETR XL and 2XL may enter the M4 bake-off as training configs. They remain under the Roboflow Platform Model Licence 1.0 in fact, so the PML is re-read before anything built on them ships.
- **Question 13, footage rights:** train on the YouTube highlights. For production the project needs high-quality video under a proper licence.
- **Question 9, spend cap:** 50 USD per month, which may be raised to 100 USD when needed. The implementer asks before a month crosses 50 USD.
- **Placement:** what runs better on Modal runs on Modal; what makes sense on the M1 Max stays local.
- **Labelling time:** "don't worry about my time for now". The 90-minute cap of 2026-10-29 is lifted. The implementer still does every first pass, sends Tomasz only what needs a human (pixel geometry, flagged doubts, conflicts), and logs his time per session.

What this changes first: gold v1 box geometry, which M4's AP50:95 needs, is no longer blocked on his budget. The implementer prefills and flags the 100 box-gold frames, and Tomasz tightens them in a dedicated labeller page.

## 2026-11-18: M4 refocused on balls and numbers; YOLO26 dropped

Tomasz asked whether the planned M4 steps were the best focus. The implementer measured the NBA prefill detector against gold v1 on the 76 dev frames: the share of gold boxes found at IoU 0.5 and score ≥ 0.3. These come from version 4 of Skalski's model at a 0.1 cut, not the baseline config, so the figures are indicative.

| Class | Found | False boxes |
|---|---|---|
| people | 89% (818 of 916) | 347, mostly bench, coaches and crowd |
| rims | 82% (59 of 72) | 13 |
| numbers | 66% (142 of 215) | 62 |
| balls | 19% (11 of 58) | 43 |

Tomasz agreed to these changes (plan sections 5, 6 and 9):
- **What M4 is judged on.** Per-class AP for `ball` and `number` comes first. People are already found. Their false boxes are mostly off court, which the court-polygon filter on the M4 checklist removes without training.
- **First experiment.** Input resolution: a higher training size, or tiling. A 15 to 25 px ball shrinks to about 6 to 8 px at a 560 to 640 px model input. This comes before any architecture comparison.
- **Bake-off.** It runs after the data and the resolution are settled, and compares RF-DETR L, RF-DETR XL and DEIM-L.
- **YOLO26 is dropped** (Tomasz: "let's drop yolo26"). It is AGPL-3.0, and shipping it in a closed product would need a commercial Ultralytics licence.
- **Footage.** Question 4, full-game footage, is raised with Tomasz now rather than later. Highlights cut every few seconds, which caps tracking and identity across a game. Production also needs licensed, high-quality video (question 13). Getting it is a lead-time item outside M4.

Open: how Skalski's own pipeline handles the ball. It is to be read in his posts before the resolution experiment, per the follow-Skalski rule.

## 2026-11-18: full games will be needed; YouTube highlights until then

Tomasz, on question 4: "I think we will have to have access to the full game recordings, but for now, let's keep pushing with what we've got already: YT videos." M4 and later milestones continue on the 31 highlight clips. The pipeline stays clip-agnostic, so full games can come in as new clips with their own shot files, without a layer change. The source and licence of full games stay open (questions 4 and 13).

## 2026-11-18: no public NBA detector version finds PLK balls; v16 reads numbers better

Before the resolution experiment, the implementer checked Skalski's post and the dataset metadata, per the follow-Skalski rule.
- **Skalski's post** gives no ball accuracy, limitations or training resolution.
- **The dataset:** the baseline detector (`basketball-player-detection-3-ycjdo/6`) was trained on 654 NBA frames, each 1920×1080 frame stretched to 640×640. That squashes a 20 px ball to about 7×12 px.
- **Other versions:** Roboflow has trained the same dataset several times. Versions 8 and 11 to 16 download for local inference; 17 and 18 return 404, 18 being the one trained without the resize.

Scored on gold v1 dev at confidence 0.05 (`docs/metrics.md`, detection table), AP50:95 / AP50:

| Version | ball | number | people |
|---|---|---|---|
| 6 | 0.176 / 0.289 | 0.202 / 0.559 | 0.642 / 0.934 |
| 13 | 0.159 / 0.259 | 0.203 / 0.581 | 0.652 / 0.926 |
| 16 | 0.112 / 0.207 | 0.372 / 0.695 | 0.662 / 0.912 |

Consequences:
- No public version fixes the ball, which confirms the refocus above: PLK training data and input resolution have to.
- Round-1 auto-labelling, a labelling aid, takes numbers, people and rims from version 16 and balls from version 6.
- `configs/baseline.yaml` stays on version 6 as the M2 reference.

## 2026-11-19: resolution helps balls and numbers; the next fine-tune runs at 1280 px with PLK data

The M4 resolution experiment trained RF-DETR Large on Skalski's NBA set alone, so the three arms differ only in input scale. They were scored on gold v1 dev (`docs/metrics.md`, detection table). AP50:95 / AP50:

| Arm | Input | ball (58) | number (215) | people (916) | ms per frame on the M1 Max |
|---|---|---|---|---|---|
| A | 704 px, stretched to a square (stock) | 0.165 / 0.258 | 0.263 / 0.680 | 0.670 / 0.937 | 70 |
| B | 1280 px, stretched | 0.234 / 0.373 | 0.423 / 0.796 | 0.694 / 0.938 | 201 |
| C | 704 px on two square tiles of the full frame height | 0.233 / 0.340 | 0.382 / 0.748 | 0.680 / 0.928 | 130 |

Findings:
- Input scale is a real lever: from A to B, ball AP50 goes from 0.26 to 0.37 and numbers from 0.68 to 0.80, with people unchanged.
- B and C are close on the ball. With 58 gold balls, their difference is within noise.
- B is better on numbers and rims; C is 35% faster on the Mac.
- On NBA data alone, no arm passes 0.4 AP50 on PLK balls. The domain gap remains, and the round-1 PLK labels are meant to close it: Tomasz placed 199 balls on 264 frames, 120 of them by hand where the detector had none.

Decisions:
- The next fine-tune adds the round-1 PLK frames at 1280 px (arm B's setting).
- Tiling stays as the fallback if Mac inference speed becomes the constraint. The production config is chosen on the held-out game at M4 close.

(Corrected the same day, see the next entry. The sentence that stood here claimed a 13-fold speed-up from more CPUs and a cost of about 22 USD; both came from local wait times, not from the remote runs.)

## 2026-11-19: correction: Modal costs and the CPU speed-up came from local wait times

Tomasz pointed out that Modal's dashboard showed 10.70 USD used of the month's 30 USD credits, while `docs/costs.md` said 27.80 USD. The implementer read the per-app figures with `modal billing report`.
- **The cut-short runs were short.** On 2026-11-18 the three resolution-experiment runs trained for about 6 to 10 minutes of GPU time each and stopped: arm A reached epoch 29, B epoch 9, C epoch 8. The local `bga train` processes kept waiting until their Modal token expired at 22:07, three hours later. The implementer took that wait for training time and logged about 20 USD; the billed cost was 0.99 USD.
- **No 13-fold speed-up.** The "about 360 s per epoch" came from the same wait. The billed GPU time gives about 20 s per epoch for arm A with Modal's default CPU reservation, against about 16 s with 8 CPUs and 8 loader workers. The 8 CPUs and 32 GB stay, for about 20% more speed at about 30% more cost (CPU and memory are billed on top of the GPU), and can be cut if cost matters more than wall time.
- **Unknown:** why the first runs stopped after a few minutes. The spawn-and-poll launcher has finished every run since.

What changes:
- `docs/costs.md` cites Modal's billing for those runs and for the month. November: 10.86 USD metered, all covered by credits, 0 billed.
- Spend is checked against `modal billing summary` rather than the sum of estimated rows.

## 2026-11-19: no court-polygon filter for detections; the Mac keeps PyTorch on MPS

Two M4 checklist items were measured on `plk-r2@b13bfb19`, with gold v1 dev and the M1 Max. Both are probes, not pipeline changes, so neither has a metrics row.

**Court-polygon filter.** People whose foot point the PLK court model (`court-plk-v1-smooth4`) maps more than a margin outside the court were dropped, and AP was recomputed. People AP50:95 / AP50:

| Margin | people | referee |
|---|---|---|
| none | 0.732 / 0.962 | 0.629 / 0.859 |
| 2 m | 0.732 / 0.962 | 0.629 / 0.859 |
| 1 m | 0.712 / 0.934 | 0.540 / 0.734 |
| 0.5 m | 0.688 / 0.898 | 0.446 / 0.608 |

The filter never helps. The PLK-trained detector already leaves bench, coaches and crowd unboxed (the gold set boxes none of them), and the referees on the sideline are what a tight margin cuts. Per the plan's rule, the heuristic is not added. The filter could still serve 2D positions later (M8), where an off-court foot point is a different question.

**Mac speed.** `plk-r2` at 1280 px, one frame, after warm-up:

| Runtime | ms per frame |
|---|---|
| PyTorch on MPS (the `rfdetr_torch` backend) | 192 |
| ONNX Runtime, CPU | 2,108 |
| ONNX Runtime, CoreML provider | fails: the CoreML runtime cannot execute part of the 1280 px graph |
| Native Core ML export, fp16, CPU and GPU | 134 |
| Native Core ML export, fp16, all units including the Neural Engine | 274 |

Decisions:
- The plan's "ONNX export, CoreML provider on the Mac" does not work at 1280 px. The Mac keeps PyTorch on MPS.
- A native Core ML backend would save about 30% but needs its own decoder and an fp16 accuracy row. It waits until Mac speed matters.
- Full-game runs can use a cloud GPU (decision of 2026-11-19 on fast feedback).

## 2026-11-19: faster training settings adopted; RF-DETR Large stays, XL and DEIM dropped

Tomasz asked to shorten Modal training. Five options were tested on `plk-r2`'s data. The speed probes are 5 epochs on an H100, with Modal's billed seconds, all at an effective batch of 16 images:

| Setup | Seconds | USD |
|---|---|---|
| batch 2 x 8 steps (the recipe so far) | 288 | 0.32 |
| batch 8 x 2 | 178 | 0.19 |
| batch 16 x 1 | 140 | 0.15 |
| batch 8 x 2, validating every 5 epochs | 141 | 0.16 |
| 4 x H100, batch 4 x 1, 32 CPUs | 235 | 1.03 |

Two full runs then checked accuracy against `plk-r2` on gold v1 dev (`docs/metrics.md`). Ball AP50:95 / AP50:
- `plk-r2`, 53 min: 0.487 / 0.798.
- `plk-r2-fast`, batch 16 and validation every 5 epochs, 18 min: 0.482 / 0.769.
- `plk-r2-fast-es`, the same plus early stopping, which stopped at epoch 44, 18 min: 0.480 / 0.768.

Numbers, people and rims are within noise as well. The ball difference is one or two of the 58 gold balls.

Decisions:
- **Every detection fine-tune from here on** uses batch 16 without accumulation, validation every 5 epochs, and early stopping after 2 validations without improvement: 3 times faster for the same accuracy. Early stopping saved little here, but it is a free stop for runs that plateau earlier.
- **Multi-GPU is dropped for now.** At about 865 training images, four processes cost more in start-up and syncing than they save. The `devices` option stays in the code for full-game datasets.
- **Warm start** (`init_weights`) is in place and is first tested at round 3, the first new data to start from `plk-r2`.

**Bake-off.** RF-DETR XL (`plk-r2-xl`, same data and resolution as `plk-r2`, 64 min, 4.18 USD) scored 0.486 / 0.783 on balls and 0.733 / 0.955 on people, the same as Large within noise. It is several times slower on the Mac and carries the Platform Model Licence. RF-DETR Large stays. DEIM-L is not run: it needs its own trainer integration, and the Large against XL result says model size is not the lever on this data; labelled PLK frames are.

**Spend:** Modal billing for November is 28.08 USD metered, against 30 USD of monthly credits.

## 2026-11-20: PR #22 review: the plan's text brought in line with what M4 did

A two-axis review of PR #22 (coding standards, and spec against issue #5 and the plan) found places where the plan still described work M4 did differently. The code findings are fixed in the same change; these are the plan's.

- **Section 7 review was done for the ball only.** The manifests of rounds 1 and 2 (231 and 238 frames) carry Tomasz's ball decisions and the draft's other boxes unreviewed: people, numbers, rims and action classes as the detector left them at the round's keep threshold. Round 2 drafted with `plk-r1` alone, not with the NBA model as a second opinion, and added hundreds of frames, not the few thousand of the bootstrap loop. The gold v1 rows in `docs/metrics.md` are what shows the shortcut was good enough for M4: people AP50 0.93 to 0.96, numbers 0.82 dev and 0.90 held-out. Section 7 now records the deviation; its rules are unchanged.
- **Pose is not part of M4.** Layer 2 lists 17-point COCO pose as an output, but nothing in M4 consumes it and M4's done-when does not score it. It lands with the ankle ground point in M8.
- **The court-polygon filter** stays out per the decision of 2026-11-19. Layer 2 and the M4 paragraph still prescribed it, and now say it was measured and not adopted.
- **ms per frame for M4 is the metrics row: 351 ms for `detect-plk-r2` on gold v1 dev, 331 ms on the held-out game.** That is the mean of the timed predict calls over every frame of the shots, including the first, cold calls after loading. The 192 ms of 2026-11-19 is a one-frame probe after warm-up, used only to compare runtimes. The two numbers measure different things and the row is the one the done-when cites.
- **Training-speed options are in M4 but the shipped weights predate them.** `recipe.devices`, `eval_interval`, `early_stopping_patience` and `init_weights`, `plk-r2-fast`, `plk-r2-fast-es` and `scripts/probe_train_speed.py` came from Tomasz's request of 2026-11-19 to shorten training. They are adopted for every later fine-tune (decision of 2026-11-19), while the weights M4 ships, `plk-r2@b13bfb19`, were trained before them with batch 2 and 8 accumulation steps.
- **`bga eval`, `eval-court` and `eval-detect` now refuse unproven weights**, as section 5 requires: a layer whose weights are neither a `pretrained:` id nor a weight id with `weights/<id>/train.json` stops the command. Before, such weights were written to the metrics table as pretrained. Every committed config passes.

## 2026-11-20: possession labels split into the holder box and a dead-ball flag

Tomasz labelled the 643-frame possession queue (50.5 min, 626 done, 17 skipped) with the rule "who has possession in live play": an inbounder behind the end line and a ball with the referee are nobody. The detector's `player-in-possession` class sees one frame and cannot tell a throw-in from live play, so one label cannot serve both. Tomasz chose to split them:

- **Box class, visual:** whoever holds or dribbles the ball is `player-in-possession`, an inbounder too. A shooter with the ball still in his hands keeps the shot class and is the holder.
- **`dead_ball` per frame:** the ball is out of live play (a throw-in from behind a line, the referee or a coach has it, after a score until the throw-in). The M9 possession metric reads it; the detector's training set does not.
- **`holder` per frame is explicit.** The page had stored "nobody" and "the shooter holds it" the same way (no `player-in-possession` box); 12 frames were ambiguous.

The implementer's pass read the 5-frame strips of all 218 "nobody" frames and the 7 holders not nearest the ball (ledger row of 2026-11-20). Tomasz's own calls (#192, #201, #206, #226, #248, #283) and 6 shooters holding the ball were applied directly; 31 frames go back to him with the proposed answer prefilled: 22 live dribbles or holds, 2 wrong holders, 6 holders that may be throw-ins, and #306. The ops are in `data/labels/review/possession.ops.txt`.

## 2026-11-20: detector classes: six to label, four kept for the M9 comparison only

Tomasz approved the class review of 2026-11-20. Three levels, each decided when it is needed:

- **Boxes (detector, one frame), decided now:** `player`, `referee`, `ball`, `rim`, `number` and `player-in-possession` are labelled in every round. Each is visible in a single frame; the first five score AP50 0.80 to 0.94 on gold v1 dev with `plk-r2`.
- **Kept in the schema, no longer labelled:** `player-jump-shot`, `player-layup-dunk`, `player-shot-block` and `ball-in-basket`. They are moments, not objects: rounds 1 and 3 hold 1 to 3 boxes of each, and `plk-r2` scores AP50 0 to 0.25 on them. Skalski's `ShotEventTracker` reads them, so they stay in the schema (no relabelling, and earlier weights stay comparable). M9 builds shot detection as two backends, his class-based one and one from the ball's path to the rim with the release point on the court (FIBA three-point line 6.75 m), and keeps whichever scores better on gold.
- **Game state and events, at the start of M9:** live or dead ball (the `dead_ball` flag exists), team in possession, stoppage type (free throw, throw-in, jump ball); shots made or missed and 2 or 3 points, offensive and defensive rebounds (in FIBA an offensive rebound resets the shot clock to 14), passes, turnovers. Fouls are not read from video.
- **Plays, at the start of the play-extraction milestone:** pick and roll and its coverages (drop, hedge, switch, ice, blitz), hand-off, off-ball screen, isolation, post-up, man against zone. Marco picks the 5 to 8 that matter to him.

## 2026-11-20: possession rule clarified by Tomasz

- **A dribbler is the holder**, mid-bounce included: "he only drops the ball to move around the court while controlling the ball". Holding also covers a shooter with the ball still in his hands.
- **A ball in flight is nobody's**, a pass to a player behind the end line included (#248: nobody, dead ball).
- **Dead ball is independent of the holder.** A player can hold or dribble a dead ball (#22, #100, #108: the shot clock stopped at 0.0 or a static 24).

His review of the 31 frames had answered nobody on 11 dribbles; `possession.r3.ops.txt` applies the rule to them and shows each on a 5-frame strip for him to veto.

## 2026-11-20: plk-r3-poss is the detector for the work that follows

Tomasz approved the recommendation. Later layers (M5 team, M6 tracking, M9 ball and possession) build on `configs/detect-plk-r3-poss.yaml`, weights `detect/rfdetr_torch/plk-r3-poss@c62744fe`, in place of `detect-plk-r2`.

- **Gold v1 dev, AP50, against `plk-r2`:** in possession 0.256 to 0.559, people 0.942 to 0.966, number 0.822 to 0.848, referee 0.859 to 0.866, rim 0.926 to 0.915, ball 0.798 to 0.766.
- **The ball is not worse in substance.** AP50:95 is 0.487 against 0.478, and the AP50 gap is about 2 of 58 balls. It comes with the training settings adopted on 2026-11-19: `plk-r2-fast-es`, `plk-r2`'s data with those settings, scored 0.768.
- **The held-out game stays unscored** until M9, whose done-when is possession accuracy on gold; this detector is one input to it.
- **A round 4 of auto-labelling** would prefill with `plk-r3-poss`, after its ball auto-accept threshold is calibrated on gold v1 dev, as for rounds 2 and 3.
- **More possession labels wait for M9's possession score.** If it falls short, another queue prefilled by this detector is the cheapest next step.
