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
