# M0: repository bootstrap (2026-10-29)

Artifact for [#1](https://github.com/tko-ventures/basketball-game-analyzer/issues/1). Done-when: the labeller opens every clip from fresh downloads; `ruff` and `pytest` pass.

## Footage as downloaded

31 clips (not 32: the prior repo's `DJ6xFaC0Neo_1080p50.json` was a second download of `20260612_DJ6xFaC0Neo`), 3.4 GB, all H.264, from the ids in `data/clip_ids.txt` with `scripts/fetch_highlights.sh`.

| Resolution and rate | Clips |
|---|---|
| 1920x1080, 50 fps | 26 |
| 1920x1080, 25 fps | 3 (`d5lCDxjy1R4`, `OmvWaQKu3rU`, `hFWL47plFQI`) |
| 1280x720, 25 fps | 1 (`s0Awx_WZCt0`) |
| 1920x1080, 60 fps | 1 (`SswWgXybNB8`) |

Plan section 2 assumes 1080p50 throughout. Four of the five exceptions are three of the four 2024/25 uploads plus `hFWL47plFQI`, a differently titled 2025/26 upload; the fifth, `SswWgXybNB8` (60 fps), is a regular 2025/26 highlight. The pipeline must read fps per clip, never assume 50 (the labeller's stride already does: `round(fps / 10)`). `4XMD9FWmQws` (PGE Start, 2024/25) is a 10-minute cut, not a 3-minute highlight. `d5lCDxjy1R4` and `s0Awx_WZCt0` share the title "Anwil Włocławek - Dziki Warszawa 79:70", so they are probably the same game uploaded twice; M1 must not put them on both sides of the held-out split.

## Shot files: regenerated against imported

`uv run tools/labeler/prepare_clips.py --jobs 8` from scratch (5 min 21 s for all 31 clips on the M1 Max), compared with the shot files imported from the prior repo (commit `c8ef83f`) by `uv run scripts/compare_shots.py --git-rev c8ef83f data/labels`. A cut matches if it is within 2 frames.

| clip | frames old/new | shots old | shots new | cuts matched (±2) | only old | only new |
|---|---|---|---|---|---|---|
| 20241013_d5lCDxjy1R4_Skrot_meczu_Anwil_W | 3965/3965 | 18 | 18 | 17 | 0 | 0 |
| 20241028_s0Awx_WZCt0_skrot_meczu_Anwil_W | 2323/2323 | 28 | 27 | 26 | 1 | 0 |
| 20241219_OmvWaQKu3rU_MKS_D_browa_Gornicz | 1994/1994 | 40 | 41 | 39 | 0 | 1 |
| 20250213_4XMD9FWmQws_PGE_Start_Dziki_Skr | 30160/30160 | 95 | 95 | 94 | 0 | 0 |
| 20251023_MbcSmlYfB-8_Skrot_spotkania_Gor | 5884/5884 | 28 | 30 | 27 | 0 | 2 |
| 20251113_kW4HG-5MLNM_SKROT_Dziki_-_King_ | 9281/9281 | 29 | 29 | 28 | 0 | 0 |
| 20251123_jXhvXy9iQeQ_SKROT_Legia_-_Dziki | 9651/9651 | 35 | 36 | 34 | 0 | 1 |
| 20251219_vevXM24gp3U_SKROT_Energa_Czarni | 9436/9436 | 30 | 30 | 29 | 0 | 0 |
| 20251223_Y4TmeMchTh0_SKROT_WKS_l_sk_-_Dz | 9795/9795 | 33 | 33 | 32 | 0 | 0 |
| 20251228_TUtBwzsraKs_SKROT_Dziki_-_Tasom | 9427/9427 | 28 | 28 | 27 | 0 | 0 |
| 20260117_SswWgXybNB8_SKROT_Dziki_-_WKS_l | 12608/12608 | 30 | 30 | 29 | 0 | 0 |
| 20260125_lwxZaHLNDmU_SKROT_Dziki_-_Miast | 9265/9265 | 24 | 24 | 23 | 0 | 0 |
| 20260308_sZ5Rab-32uk_SKROT_Dziki_-_AMW_A | 9369/9369 | 29 | 29 | 28 | 0 | 0 |
| 20260313_cG4oqryldrw_SKROT_MKS_-_Dziki_2 | 8928/8928 | 26 | 26 | 25 | 0 | 0 |
| 20260317_hFWL47plFQI_SKROT_MECZU_13.03.2 | 4591/4591 | 58 | 58 | 57 | 0 | 0 |
| 20260321_hUoD7ptXzCE_SKROT_King_-_Dziki_ | 9103/9103 | 28 | 28 | 27 | 0 | 0 |
| 20260328_OSp445e0mXU_SKROT_Dziki_-_Legia | 9377/9377 | 30 | 30 | 29 | 0 | 0 |
| 20260404_-MurBVifFFQ_SKROT_Dziki_-_PGE_S | 9089/9089 | 31 | 31 | 30 | 0 | 0 |
| 20260410_zZqAUytpmgc_SKROT_Arriva_LOTTO_ | 9718/9718 | 29 | 29 | 28 | 0 | 0 |
| 20260418_xTfkhQ0wO20_SKROT_Dziki_-_Energ | 9359/9359 | 30 | 30 | 29 | 0 | 0 |
| 20260425_enJCkGVWlHI_SKROT_Tasomix_Rosie | 9417/9417 | 28 | 28 | 27 | 0 | 0 |
| 20260430_FCaYkdygkYI_SKROT_Dziki_-_Anwil | 9253/9253 | 25 | 25 | 24 | 0 | 0 |
| 20260507_yN_-DqS3Y_s_SKROT_ORLEN_Zastal_ | 9176/9176 | 29 | 29 | 28 | 0 | 0 |
| 20260514_1jDaJBkUGmc_SKROT_Energa_Trefl_ | 9714/9714 | 28 | 28 | 27 | 0 | 0 |
| 20260516_EExZbLxgez4_SKROT_Energa_Trefl_ | 10492/10492 | 37 | 37 | 36 | 0 | 0 |
| 20260521_BEFym3rFSOw_SKROT_Dziki_-_Energ | 10384/10384 | 41 | 41 | 39 | 1 | 1 |
| 20260529_l7VkOarkFZk_SKROT_Legia_-_Dziki | 12415/12415 | 42 | 42 | 41 | 0 | 0 |
| 20260601_qb1y4Jz5AkA_SKROT_Dziki_-_Legia | 10144/10144 | 32 | 32 | 31 | 0 | 0 |
| 20260609_FbwQDVu5Tw8_SKROT_AMW_Arka_-_Dz | 9291/9291 | 31 | 31 | 30 | 0 | 0 |
| 20260612_DJ6xFaC0Neo_SKROT_Dziki_-_AMW_A | 9375/9375 | 45 | 44 | 43 | 1 | 0 |
| 20260926_Tnuscrm9RyU_SKROT_Dziki_-_Zasta | 10253/10253 | 44 | 45 | 43 | 0 | 1 |
| **total** | | | | **1027** | **3** | **6** |

Frame counts agree in every clip, so the fresh downloads decode to the same frames. The 9 disagreeing cuts were checked by eye on one contact sheet (`scripts/shot_diff_sheet.py`, ledger row in `data/labels/sheet_ledger.tsv`):

| Clip | Frame | Side | Verdict |
|---|---|---|---|
| s0Awx_WZCt0 | 90 | old only | no cut (motion blur): old false positive |
| OmvWaQKu3rU | 1389 | new only | real cut, basket close-up to play: old missed it |
| MbcSmlYfB-8 | 993 | new only | real cut, close-up to wide: old missed it |
| MbcSmlYfB-8 | 1984 | new only | no cut: new false positive |
| jXhvXy9iQeQ | 5596 | new only | real cut, close-up to wide: old missed it |
| BEFym3rFSOw | 6562 / 6570 | old / new | no visible cut in either window; ambiguous |
| DJ6xFaC0Neo | 8642 | old only | no cut: old false positive |
| Tnuscrm9RyU | 4738 | new only | no cut: new false positive |

The regenerated files are right on 5, wrong on 2, one pair is unclear. The detector is the same code; the differences are borderline frames where decoder output differs. Both versions over-split a little; camera tags are set by hand in the labeller and the L0 detector gets a gold-set row when the gold shots exist (M1/M2). The regenerated files are the ones committed.

## Labeller opens every clip

`uv run tools/labeler/server.py`, then `uv run scripts/check_labeller.py`: each clip's labels load as version 2 with home and away teams and a season, and the middle frame of the first and last shot decodes through `/api/frame`.

| clip | fps | frames | shots | home | away | season | frames decode | open s |
|---|---|---|---|---|---|---|---|---|
| 20241013_d5lCDxjy1R4_Skrot_meczu_Anwil_W_oc_ | 25 | 3965 | 18 | Anwil Włocławek | Dziki Warszawa | 2024/25 | 2/2 | 0.0 |
| 20241028_s0Awx_WZCt0_skrot_meczu_Anwil_W_oc_ | 25 | 2323 | 27 | Anwil Włocławek | Dziki Warszawa | 2024/25 | 2/2 | 0.0 |
| 20241219_OmvWaQKu3rU_MKS_D_browa_Gornicza_-_ | 25 | 1994 | 41 | MKS Dąbrowa Górnicza | Dziki Warszawa | 2024/25 | 2/2 | 0.0 |
| 20250213_4XMD9FWmQws_PGE_Start_Dziki_Skrot_m | 50 | 30160 | 95 | PGE Start | Dziki Warszawa | 2024/25 | 2/2 | 0.0 |
| 20251023_MbcSmlYfB-8_Skrot_spotkania_Gornik_ | 50 | 5884 | 30 | Górnik Zamek Książ Wałbrzych | Dziki Warszawa | 2025/26 | 2/2 | 0.0 |
| 20251113_kW4HG-5MLNM_SKROT_Dziki_-_King_8._k | 50 | 9281 | 29 | Dziki Warszawa | King | 2025/26 | 2/2 | 0.0 |
| 20251123_jXhvXy9iQeQ_SKROT_Legia_-_Dziki_9._ | 50 | 9651 | 36 | Legia | Dziki Warszawa | 2025/26 | 2/2 | 0.0 |
| 20251219_vevXM24gp3U_SKROT_Energa_Czarni_-_D | 50 | 9436 | 30 | Energa Czarni | Dziki Warszawa | 2025/26 | 2/2 | 0.0 |
| 20251223_Y4TmeMchTh0_SKROT_WKS_l_sk_-_Dziki_ | 50 | 9795 | 33 | WKS Śląsk | Dziki Warszawa | 2025/26 | 2/2 | 0.0 |
| 20251228_TUtBwzsraKs_SKROT_Dziki_-_Tasomix_R | 50 | 9427 | 28 | Dziki Warszawa | Tasomix Rosiek Stal | 2025/26 | 2/2 | 0.0 |
| 20260117_SswWgXybNB8_SKROT_Dziki_-_WKS_l_sk_ | 60 | 12608 | 30 | Dziki Warszawa | WKS Śląsk | 2025/26 | 2/2 | 0.0 |
| 20260125_lwxZaHLNDmU_SKROT_Dziki_-_Miasto_Sz | 50 | 9265 | 24 | Dziki Warszawa | Miasto Szkła | 2025/26 | 2/2 | 0.0 |
| 20260308_sZ5Rab-32uk_SKROT_Dziki_-_AMW_Arka_ | 50 | 9369 | 29 | Dziki Warszawa | AMW Arka | 2025/26 | 2/2 | 0.0 |
| 20260313_cG4oqryldrw_SKROT_MKS_-_Dziki_22._k | 50 | 8928 | 26 | MKS | Dziki Warszawa | 2025/26 | 2/2 | 0.0 |
| 20260317_hFWL47plFQI_SKROT_MECZU_13.03.2026_ | 25 | 4591 | 58 | MKS DG | Dziki Warszawa | 2025/26 | 2/2 | 0.0 |
| 20260321_hUoD7ptXzCE_SKROT_King_-_Dziki_23._ | 50 | 9103 | 28 | King | Dziki Warszawa | 2025/26 | 2/2 | 0.0 |
| 20260328_OSp445e0mXU_SKROT_Dziki_-_Legia_24. | 50 | 9377 | 30 | Dziki Warszawa | Legia | 2025/26 | 2/2 | 0.0 |
| 20260404_-MurBVifFFQ_SKROT_Dziki_-_PGE_Start | 50 | 9089 | 31 | Dziki Warszawa | PGE Start | 2025/26 | 2/2 | 0.0 |
| 20260410_zZqAUytpmgc_SKROT_Arriva_LOTTO_Twar | 50 | 9718 | 29 | Arriva LOTTO Twarde Pierniki | Dziki Warszawa | 2025/26 | 2/2 | 0.0 |
| 20260418_xTfkhQ0wO20_SKROT_Dziki_-_Energa_Cz | 50 | 9359 | 30 | Dziki Warszawa | Energa Czarni | 2025/26 | 2/2 | 0.0 |
| 20260425_enJCkGVWlHI_SKROT_Tasomix_Rosiek_St | 50 | 9417 | 28 | Tasomix Rosiek Stal | Dziki Warszawa | 2025/26 | 2/2 | 0.0 |
| 20260430_FCaYkdygkYI_SKROT_Dziki_-_Anwil_29. | 50 | 9253 | 25 | Dziki Warszawa | Anwil | 2025/26 | 2/2 | 0.0 |
| 20260507_yN_-DqS3Y_s_SKROT_ORLEN_Zastal_-_Dz | 50 | 9176 | 29 | ORLEN Zastal | Dziki Warszawa | 2025/26 | 2/2 | 0.0 |
| 20260514_1jDaJBkUGmc_SKROT_Energa_Trefl_-_Dz | 50 | 9714 | 28 | Energa Trefl | Dziki Warszawa | 2025/26 | 2/2 | 0.0 |
| 20260516_EExZbLxgez4_SKROT_Energa_Trefl_-_Dz | 50 | 10492 | 37 | Energa Trefl | Dziki Warszawa | 2025/26 | 2/2 | 0.0 |
| 20260521_BEFym3rFSOw_SKROT_Dziki_-_Energa_Tr | 50 | 10384 | 41 | Dziki Warszawa | Energa Trefl | 2025/26 | 2/2 | 0.0 |
| 20260529_l7VkOarkFZk_SKROT_Legia_-_Dziki_2._ | 50 | 12415 | 42 | Legia | Dziki Warszawa | 2025/26 | 2/2 | 0.0 |
| 20260601_qb1y4Jz5AkA_SKROT_Dziki_-_Legia_3._ | 50 | 10144 | 32 | Dziki Warszawa | Legia | 2025/26 | 2/2 | 0.0 |
| 20260609_FbwQDVu5Tw8_SKROT_AMW_Arka_-_Dziki_ | 50 | 9291 | 31 | AMW Arka | Dziki Warszawa | 2025/26 | 2/2 | 0.0 |
| 20260612_DJ6xFaC0Neo_SKROT_Dziki_-_AMW_Arka_ | 50 | 9375 | 44 | Dziki Warszawa | AMW Arka | 2025/26 | 2/2 | 0.0 |
| 20260926_Tnuscrm9RyU_SKROT_Dziki_-_Zastal_po | 50 | 10253 | 45 | Dziki Warszawa | Zastal | 2026/27 | 2/2 | 0.0 |

31/31 clips open with shots ready to tag

A browser round trip on `kW4HG-5MLNM` (tag shot central, draw a box, team home, number 7, mark done) saved `frame 411, class 1` to the MOT file and `1,home,7,Grzegorz Grochowski,Dziki Warszawa` to `identities.csv`; the test edit was reverted.

## Checks

`uv run ruff check .`: all checks passed. `uv run pytest`: 22 passed.
