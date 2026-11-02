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
