# Training runs

One row per `bga train` run, appended by the command. Columns: the training config, the weight id it
produced (`weights/<id>/` with `train.json`), the number of training and validation images, the
trainer's own validation number at the selected epoch, and the GPU, time and cost. These are trainer
numbers on its validation split. Gold-set scores of the weights go to `docs/metrics.md` instead, in a
row that cites the training config and the inference config.

Data credits: `basketball-court-detection-2` by Roboflow (roboflow-jvuqo on Roboflow Universe), CC BY 4.0,
used in the court keypoint runs. Base weights: RF-DETR Keypoint preview by Roboflow, Apache 2.0.

| date | commit | training config | weight id | train / valid images | trainer validation (selected epoch) | GPU, time, cost | notes |
|---|---|---|---|---|---|---|---|
| 2026-11-18 | 2195c4c | `configs/train/court-rfdetr_keypoint-plk-v1.yaml` | `court/rfdetr_keypoint/plk-v1@c26b7841` | 681 / 130 | val/keypoint_map_50_95 0.979 (epoch 46) | L40S, 5510.3 s, 2.9848 USD | first run; 610 NBA + 71 PLK train images, 124 NBA + 6 PLK valid |
| 2026-11-19 | 0bd7ba4 | `configs/train/detect-rfdetr_torch-nba-704.yaml` | `detect/rfdetr_torch/nba-704@9039f787` | 464 / 96 | val/mAP_50_95 0.620 (epoch 28) | cut short; cost in docs/costs.md | partial: best of epochs 0 to 29 of 50 |
| 2026-11-19 | 0bd7ba4 | `configs/train/detect-rfdetr_torch-nba-1280.yaml` | `detect/rfdetr_torch/nba-1280@37e7c0f7` | 464 / 96 | val/mAP_50_95 0.564 (epoch 9) | cut short; cost in docs/costs.md | partial: best of epochs 0 to 10 of 50 |
| 2026-11-19 | 0bd7ba4 | `configs/train/detect-rfdetr_torch-nba-tile2.yaml` | `detect/rfdetr_torch/nba-tile2@bcf7b985` | 928 / 192 | val/mAP_50_95 0.579 (epoch 7) | cut short; cost in docs/costs.md | partial: best of epochs 0 to 9 of 50 |
| 2026-11-19 | 0bd7ba4 | `configs/train/detect-rfdetr_torch-nba-704.yaml` | `detect/rfdetr_torch/nba-704@9039f787` | 464 / 96 | val/mAP_50_95 0.653 (epoch 45) | L40S, 318.7 s, 0.1726 USD |  |
| 2026-11-19 | 0bd7ba4 | `configs/train/detect-rfdetr_torch-nba-tile2.yaml` | `detect/rfdetr_torch/nba-tile2@bcf7b985` | 928 / 192 | val/mAP_50_95 0.661 (epoch 30) | L40S, 460.2 s, 0.2493 USD |  |
| 2026-11-19 | 0bd7ba4 | `configs/train/detect-rfdetr_torch-nba-1280.yaml` | `detect/rfdetr_torch/nba-1280@37e7c0f7` | 464 / 96 | val/mAP_50_95 0.671 (epoch 45) | H100, 1512.5 s, 1.6595 USD |  |
| 2026-11-19 | 8e9129e | `configs/train/detect-rfdetr_torch-plk-r1.yaml` | `detect/rfdetr_torch/plk-r1@7ab3f6fa` | 662 / 129 | val/mAP_50_95 0.592 (epoch 28) | H100, 2099.1 s, 2.3031 USD |  |
| 2026-11-19 | de9d970 | `configs/train/detect-rfdetr_torch-plk-r2.yaml` | `detect/rfdetr_torch/plk-r2@b13bfb19` | 865 / 164 | val/mAP_50_95 0.609 (epoch 43) | H100, 3201.1 s, 3.5123 USD |  |
| 2026-11-19 | 667db51 | `configs/train/detect-rfdetr_torch-plk-r2-fast-es.yaml` | `detect/rfdetr_torch/plk-r2-fast-es@3888eea6` | 865 / 164 | val/mAP_50_95 0.603 (epoch 34) | H100, 1056.0 s, 1.1586 USD |  |
| 2026-11-19 | 667db51 | `configs/train/detect-rfdetr_torch-plk-r2-fast.yaml` | `detect/rfdetr_torch/plk-r2-fast@7eae6b7c` | 865 / 164 | val/mAP_50_95 0.604 (epoch 49) | H100, 1080.2 s, 1.1852 USD |  |
| 2026-11-19 | b6b9d2e | `configs/train/detect-rfdetr_torch-plk-r2-xl.yaml` | `detect/rfdetr_torch/plk-r2-xl@646266cd` | 865 / 164 | val/mAP_50_95 0.640 (epoch 49) | H100, 3814.1 s, 4.1849 USD |  |
