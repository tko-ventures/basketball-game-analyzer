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
