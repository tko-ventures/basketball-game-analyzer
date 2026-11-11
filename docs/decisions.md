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

