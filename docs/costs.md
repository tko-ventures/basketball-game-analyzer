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
| 2026-11-18 | modal | L40S | bga train detect/rfdetr_torch/nba-704@9039f787, part 1 (stopped at epoch 29; Modal billing for app ap-bW2Pi4, GPU + CPU + memory) | 576 | 0.331 |
| 2026-11-18 | modal | H100 | bga train detect/rfdetr_torch/nba-1280@37e7c0f7, part 1 (stopped at epoch 9; Modal billing, app ap-eWkDAe) | 410 | 0.465 |
| 2026-11-18 | modal | L40S | bga train detect/rfdetr_torch/nba-tile2@bcf7b985, part 1 (stopped at epoch 8; Modal billing, app ap-r4UEdm) | 343 | 0.197 |
| 2026-11-18 | modal | L40S + H100 + L40S | the three runs resumed 22:25 and stopped 22:28 (Modal billing, apps ap-MQWc3J, ap-g4V3ZA, ap-yHcHRO) | 177 | 0.275 |
| 2026-11-19 | modal | L40S | probe: nba-704 2 epochs, 8 workers, 8 CPUs (28 s per epoch incl. start-up) | 55.7 | 0.0302 |
| 2026-11-19 | modal | L40S | bga train detect/rfdetr_torch/nba-704@9039f787 | 318.7 | 0.1726 |
| 2026-11-19 | modal | L40S | bga train detect/rfdetr_torch/nba-tile2@bcf7b985 | 460.2 | 0.2493 |
| 2026-11-19 | modal | H100 | bga train detect/rfdetr_torch/nba-1280@37e7c0f7 | 1512.5 | 1.6595 |
| 2026-11-19 | modal | H100 | bga train detect/rfdetr_torch/plk-r1@7ab3f6fa | 2099.1 | 2.3031 |
| 2026-11-19 | modal | H100 | bga train detect/rfdetr_torch/plk-r2@b13bfb19 | 3201.1 | 3.5123 |
| 2026-11-19 | modal | H100 | probe: plk-r2 3 epochs, 1 GPU, batch 2x8, 8 workers, 8 CPUs (80 s per epoch incl. start-up) | 239.3 | 0.2626 |
| 2026-11-19 | modal | H100:4 | probe: plk-r2 3 epochs, 4 GPU, batch 4x1, 8 workers, 8 CPUs (90 s per epoch incl. start-up) | 271.4 | 1.1911 |
| 2026-11-19 | modal | H100 | probe: plk-r2 5 epochs, 1 GPU, batch 8x2, 8 workers, 8 CPUs (36 s per epoch incl. start-up) | 177.5 | 0.1947 |
| 2026-11-19 | modal | H100:4 | probe: plk-r2 5 epochs, 4 GPU, batch 4x1, 8 workers, 8 CPUs (47 s per epoch incl. start-up) | 235.1 | 1.0317 |
| 2026-11-19 | modal | H100 | probe: plk-r2 5 epochs, 1 GPU, batch 2x8, 8 workers, 8 CPUs (58 s per epoch incl. start-up) | 287.8 | 0.3158 |
| 2026-11-19 | modal | H100 | probe: plk-r2 5 epochs, 1 GPU, batch 16x1, 8 workers, 8 CPUs (28 s per epoch incl. start-up) | 139.5 | 0.153 |
| 2026-11-19 | modal | H100 | probe: plk-r2 5 epochs, 1 GPU, batch 8x2, 8 workers, 8 CPUs (28 s per epoch incl. start-up) | 141.4 | 0.1552 |
| 2026-11-19 | modal | H100 | bga train detect/rfdetr_torch/plk-r2-fast-es@3888eea6 | 1056.0 | 1.1586 |
| 2026-11-19 | modal | H100 | bga train detect/rfdetr_torch/plk-r2-fast@7eae6b7c | 1080.2 | 1.1852 |
| 2026-11-19 | modal | H100 | bga train detect/rfdetr_torch/plk-r2-xl@646266cd | 3814.1 | 4.1849 |
| 2026-11-20 | modal | H100 | bga train detect/rfdetr_torch/plk-r3-poss@c62744fe | 654.1 | 0.7177 |
| 2026-11-20 | modal | H100 | bga train detect/rfdetr_torch/plk-r3@f72eba80 | 798.1 | 0.8757 |
| 2026-11-20 | modal | L4 | bga run numbers/smolvlm_ocr OSp445e0mXU_s10 | 180.6 | 0.0401 |

Month to date (2026-11), from Modal's own billing (`modal billing summary`, 2026-11-19 night): 28.08 USD metered, 27.99 covered by the workspace's 30 USD monthly credits, 0.00 USD billed so far; further November runs are billed. The rows above are per-job estimates at list price for the GPU alone, except where they cite Modal billing. On 2026-11-18 four rows estimated from local log times overstated the cut-short training runs about 25-fold (the remote runs had stopped within minutes while the local process went on waiting); they now carry the billed figures. October 2026: 4.18 USD metered per Modal (3.60 USD in the rows).
