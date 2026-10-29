# Decisions

Dated entries for every change to a decision in `docs/PLAN.md`. Newest last.

## 2026-10-29: answers to plan questions 1, 11, 12 and 14

Tomasz, in the M0 session.

- **Q1, Roboflow.** A workspace exists (`tomasz-kopacki`). The key lives outside the repo: `~/.config/bga/roboflow.env` (sourced from `~/.zshenv`, mode 600) and the Modal secret `roboflow`, both as `ROBOFLOW_API_KEY`. Used for Universe dataset downloads and `inference` prefill during labelling. Training credits are not visible through the API, so the training default stays Modal.
- **Q11, commit policy.** Default confirmed: commit on `feat/m<N>-<slug>`, one pull request per milestone, never push to main or merge without Tomasz.
- **Q12, gold games.** Confirmed as proposed on #2: `4XMD9FWmQws` (PGE Start – Dziki, 2024/25), `OSp445e0mXU` (Dziki – Legia, 2025/26), `1jDaJBkUGmc` (Energa Trefl – Dziki, 2025/26), and held out `Tnuscrm9RyU` (Dziki – Zastal, Superpuchar, 2026/27).
- **Q14, rosters.** Collect every PLK team's roster now, same schema as `data/rosters/dziki_warszawa.json`.
