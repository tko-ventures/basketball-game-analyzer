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

Month to date (2026-10): 3.60 USD of GPU time.
