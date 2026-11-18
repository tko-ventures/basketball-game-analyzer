# Remote compute costs

One row per remote job (CLAUDE.md: log cost per job; proposed cap 50 USD per month). Cost is GPU
seconds inside the function times Modal's list price for that GPU on 2026-10-29; container start and
image builds are not included, so the Modal invoice is the authority.

| Date | Platform | GPU | Job | GPU seconds | USD |
|---|---|---|---|---|---|
| 2026-10-29 | modal | L40S | sam2_prefill OSp445e0mXU shot 10 (first test) | 442.0 | 0.2394 |
| 2026-10-29 | modal | L40S | sam2_prefill 4XMD9FWmQws shot 4 | 605.3 | 0.3278 |
| 2026-10-29 | modal | L40S | sam2_prefill 4XMD9FWmQws shot 42 | 344.5 | 0.1866 |
| 2026-10-29 | modal | L40S | sam2_prefill 4XMD9FWmQws shot 78 | 270.0 | 0.1462 |
| 2026-10-29 | modal | L40S | sam2_prefill OSp445e0mXU shot 10 | 381.6 | 0.2067 |
| 2026-10-29 | modal | L40S | sam2_prefill OSp445e0mXU shot 19 | 355.9 | 0.1928 |
| 2026-10-29 | modal | L40S | sam2_prefill OSp445e0mXU shot 2 | 777.1 | 0.4209 |
| 2026-10-29 | modal | L40S | sam2_prefill 1jDaJBkUGmc shot 16 | 962.7 | 0.5215 |
| 2026-10-29 | modal | L40S | sam2_prefill 1jDaJBkUGmc shot 3 | 876.1 | 0.4745 |
| 2026-10-29 | modal | L40S | sam2_prefill 1jDaJBkUGmc shot 7 | 604.6 | 0.3275 |
| 2026-10-29 | modal | L40S | sam2_prefill Tnuscrm9RyU shot 18 | 259.9 | 0.1408 |
| 2026-10-29 | modal | L40S | sam2_prefill Tnuscrm9RyU shot 2 | 515.1 | 0.279 |
| 2026-10-29 | modal | L40S | sam2_prefill Tnuscrm9RyU shot 33 | 252.3 | 0.1367 |
| 2026-11-02 | modal | cpu | bga run track/bytetrack OSp445e0mXU_s19 | 3.1 | 0.0 |
| 2026-11-02 | modal | cpu | bga run track/bytetrack OSp445e0mXU_s19 | 1.5 | 0.0 |
| 2026-11-18 | modal | L40S | bga train court/rfdetr_keypoint/plk-v1@c26b7841 | 5510.3 | 2.9848 |
| 2026-11-18 | modal | L40S | sam2_boxmask box gold v1 (100 frames) | 32.7 | 0.0177 |
| 2026-11-18 | modal | L40S | sam2_boxmask box gold v1 (100 frames) | 27.2 | 0.0147 |
| 2026-11-18 | modal | L40S | bga train detect/rfdetr_torch/nba-704@9039f787, part 1 (19:08-22:07, cut short at epoch 29; time from the log, no result returned) | 10746 | 5.8207 |
| 2026-11-18 | modal | H100 | bga train detect/rfdetr_torch/nba-1280@37e7c0f7, part 1 (19:11-22:07, cut short at epoch 8; time from the log) | 10577 | 11.6053 |
| 2026-11-18 | modal | L40S | bga train detect/rfdetr_torch/nba-tile2@bcf7b985, part 1 (19:12-20:31, cut short at epoch 8; time from the log) | 4706 | 2.5491 |
| 2026-11-18 | modal | L40S + H100 + L40S | the three runs resumed 22:25, stopped 22:28 to stay inside the spend cap (estimate) | 450 | 0.4 |

Month to date (2026-11): 23.39 USD of GPU time. October 2026: 3.60 USD.
