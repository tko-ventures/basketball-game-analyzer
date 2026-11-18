# Metrics

Gold-set scores from `bga eval` (plan section 8). One row per config and split, appended by `bga eval --write`; never edited by hand except for this header.

How to read a row:
- **config:** the config's name and the hash of its layer hashes. The same committed config gives the same hash on every split.
- **gold:** the gold version. **v0 is the implementer's labels**, prefilled by the same public models the baseline uses, so v0 scores are biased upwards; use them to compare rows. On v0, team accuracy is close to meaningless, because the gold teams were proposed by the same kit classifier. See `docs/gold.md`.
- **split:** dev is the 9 non-held-out shots. Held-out is the Dziki – Zastal game, scored once per milestone.
- **HOTA, DetA, AssA, IDF1, MOTA, IDSW:** TrackEval over all people (players and referees as one class), on every 5th frame.
- **AP50:** class-agnostic person AP on the 100 box-gold frames. AP50:95 waits for gold v1 geometry.
- **team acc:** gold player tracks whose matched detections are mostly given the right team. The track count is in brackets.
- **number acc:** gold tracks with a legible number whose best-matched predicted track got that number confirmed. In brackets: the fraction that got any number confirmed, the count confirmed wrong, and the number of tracks.
- **ms/frame:** mean time per processed frame on the M1 Max, per layer. Numbers is the jersey-number reader.


| date | commit | config | gold | split | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | AP50 | team acc | number acc (confirmed, wrong) | ms/frame detect, team, court, numbers | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-02 | eabd4ea | `baseline`@70498c00 | v0 | dev (9 shots) | 0.602 | 0.675 | 0.539 | 0.664 | 0.719 | 273 | 0.910 | 1.000 (98) | 0.478 (0.797, 22; 69 tracks) | 56, 258, 71, 1240 | first baseline: Skalski's public models, no PLK training |
| 2026-11-02 | dd3384f | `baseline`@70498c00 | v0 | held-out (3 shots) | 0.648 | 0.717 | 0.587 | 0.710 | 0.768 | 69 | 0.911 | 1.000 (31) | 0.577 (0.846, 7; 26 tracks) | 55, 273, 69, 1113 | held-out game, scored once for M2 |

## Court layer

Scored by `bga eval-court` on the gold court keyframes (`data/labels/court/queue.json`, purpose gold, labelled by Tomasz): 6 per gold game, the held-out game only with `--held-out`. Each keyframe is mapped by the config's court backend, and the court vertices visible under Tomasz's mapping are sent through the predicted mapping. **median error:** median over keyframes of the per-frame median distance in metres to their true positions. **paint:** the same over the key and free-throw line vertices; **paint < 0.5 m:** share of keyframes meeting the plan's provisional target there. Unmapped keyframes count as failures and have no error.

| date | commit | config | court weights (training config) | split | keyframes (unmapped) | median error m | paint m | paint < 0.5 m | paint m per arena | ms/frame | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-18 | 837925b | `baseline`@70498c00 | `pretrained:roboflow/basketball-court-detection-2/14` (pretrained) | dev | 18 (0) | 1.13 | 0.89 | 0.33 | Dziki Warszawa 0.36; Energa Trefl 5.46; PGE Start 0.89 | 70 |  |
| 2026-11-18 | 837925b | `court-plk-v1`@35006bb3 | `court/rfdetr_keypoint/plk-v1@c26b7841` (`configs/train/court-rfdetr_keypoint-plk-v1.yaml`) | dev | 18 (0) | 0.29 | 0.23 | 0.78 | Dziki Warszawa 0.25; Energa Trefl 0.52; PGE Start 0.18 | 170 |  |
| 2026-11-18 | 2b43da0 | `court-plk-v1-smooth2`@ebf32c64 | `court/rfdetr_keypoint/plk-v1@c26b7841` (`configs/train/court-rfdetr_keypoint-plk-v1.yaml`) | dev | 18 (0) | 0.28 | 0.23 | 0.89 | Dziki Warszawa 0.23; Energa Trefl 0.40; PGE Start 0.19 | 232 | median smoothing within the shot; against court-plk-v1 |
| 2026-11-18 | 2b43da0 | `court-plk-v1-smooth4`@c6538344 | `court/rfdetr_keypoint/plk-v1@c26b7841` (`configs/train/court-rfdetr_keypoint-plk-v1.yaml`) | dev | 18 (0) | 0.28 | 0.21 | 0.89 | Dziki Warszawa 0.21; Energa Trefl 0.37; PGE Start 0.19 | 232 | median smoothing within the shot; against court-plk-v1 |
| 2026-11-18 | 2b43da0 | `baseline`@70498c00 | `pretrained:roboflow/basketball-court-detection-2/14` (pretrained) | held-out | 6 (0) | 0.35 | 0.25 | 0.83 | Superpuchar venue (2026) 0.25 | 168 | held-out game, scored once at M3 close |
| 2026-11-18 | 2b43da0 | `court-plk-v1-smooth4`@c6538344 | `court/rfdetr_keypoint/plk-v1@c26b7841` (`configs/train/court-rfdetr_keypoint-plk-v1.yaml`) | held-out | 6 (0) | 0.21 | 0.20 | 0.83 | Superpuchar venue (2026) 0.20 | 244 | held-out game, scored once at M3 close |

## Detection layer

Scored by `bga eval-detect` on the box-gold frames of gold v1 (`data/gold/v1/boxes.json`: every object of the 10-class schema boxed and tightened by Tomasz), dev frames unless `--held-out`. Each cell is COCO **AP50:95 / AP50 (gold boxes)** for one class. **people** merges the six person classes on both sides, so it measures finding people whatever their action class. **rare classes:** jump shot, layup or dunk, shot block and ball in basket, each too small for a stable AP on its own. **ms/frame:** the detect layer on the M1 Max.

| date | commit | config | detect weights (training config) | gold | split (frames) | people | ball | number | rim | player | referee | in possession | rare classes | ms/frame | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-18 | f62e213 | `baseline`@70498c00 | `pretrained:roboflow/basketball-player-detection-3-ycjdo/6` (pretrained) | v1 | dev (76) | 0.626 / 0.905 (916) | 0.162 / 0.258 (58) | 0.195 / 0.534 (215) | 0.415 / 0.929 (72) | 0.620 / 0.893 (681) | 0.556 / 0.780 (183) | 0.116 / 0.154 (37) | jump 0.000 / 0.000 (6); layup 0.000 / 0.000 (1); block 0.077 / 0.129 (8); in basket 0.000 / 0.000 (3) | 55 | first per-class row: Skalski's NBA detector as configured (confidence 0.4) |
| 2026-11-18 | f62e213 | `baseline[detect.params.confidence=0.05]`@097cd56d | `pretrained:roboflow/basketball-player-detection-3-ycjdo/6` (pretrained) | v1 | dev (76) | 0.642 / 0.934 (916) | 0.176 / 0.289 (58) | 0.202 / 0.559 (215) | 0.421 / 0.946 (72) | 0.642 / 0.932 (681) | 0.573 / 0.808 (183) | 0.135 / 0.184 (37) | jump 0.000 / 0.000 (6); layup 0.000 / 0.000 (1); block 0.096 / 0.186 (8); in basket 0.000 / 0.000 (3) | 54 | same model at confidence 0.05, so AP sees the low-score tail; against the row above |
