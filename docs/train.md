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
| 2026-11-20 | 7c2f54a | `configs/train/detect-rfdetr_torch-plk-r3-poss.yaml` | `detect/rfdetr_torch/plk-r3-poss@c62744fe` | 1077 / 196 | val/mAP_50_95 0.645 (epoch 14) | H100, 654.1 s, 0.7177 USD |  |
| 2026-11-20 | 7c2f54a | `configs/train/detect-rfdetr_torch-plk-r3.yaml` | `detect/rfdetr_torch/plk-r3@f72eba80` | 1077 / 196 | val/mAP_50_95 0.634 (epoch 19) | H100, 798.1 s, 0.8757 USD |  |
| 2026-11-21 | aaeea5a | `configs/train/appearance-reid-plk-r1-team85.yaml` | `appearance/reid/plk-r1-team85@7f62ed2c` | 16477 / 5487 | val/map 39.369 (epoch 12) | L4, 3094.3 s, 0.6876 USD |  |
| 2026-11-21 | aaeea5a | `configs/train/appearance-reid-plk-r1.yaml` | `appearance/reid/plk-r1@553936e9` | 19144 / 5894 | val/map 39.592 (epoch 9) | L4, 2572.0 s, 0.5716 USD |  |
| 2026-11-21 | ddc883f | `configs/train/appearance-reid-plk-r1-shot.yaml` | `appearance/reid/plk-r1-shot@1954bf32` | 19144 / 5894 | val/rank1 78.400 (epoch 0) | L4, 675.0 s, 0.15 USD |  |
| 2026-11-21 | f175d12 | `configs/train/appearance-reid-plk-r1-shot-last.yaml` | `appearance/reid/plk-r1-shot-last@bef4a309` | 19144 / 5894 | epoch 15.000 (epoch 15) | L4, 539.9 s, 0.12 USD |  |
| 2026-11-22 | b96dfac | `configs/train/numbers-crop_cls-resnet18-r1.yaml` | `numbers/crop_cls/resnet18-r1@092ff76e` | 4439 / 1646 | val/top1 0.517 (epoch 10) | L4, 302.7 s, 0.0673 USD |  |
| 2026-11-22 | b96dfac | `configs/train/numbers-crop_cls-dinov2s-r1.yaml` | `numbers/crop_cls/dinov2s-r1@89b68b3b` | 4439 / 1646 | val/top1 0.527 (epoch 20) | L4, 167.1 s, 0.0371 USD |  |
| 2026-11-23 | 7300717 | `configs/train/numbers-crop_cls-resnet18-r1-fact.yaml` | `numbers/crop_cls/resnet18-r1-fact@62907b32` | 4439 / 1646 | val/top1 0.518 (epoch 24) | L4, 164.1 s, 0.0365 USD |  |
| 2026-11-23 | 0d466a1 | `configs/train/numbers-crop_cls-dinov2s-r1-fact.yaml` | `numbers/crop_cls/dinov2s-r1-fact@64ba1381` | 4439 / 1646 | val/top1 0.531 (epoch 12) | L4, 164.4 s, 0.0365 USD |  |
| 2026-11-23 | 4d80cb3 | `configs/train/numbers-parseq-r1.yaml` | `numbers/parseq/r1@c384dce4` | 4439 / 1646 | val/top1 0.657 (epoch 5) | L4, 366.3 s, 0.0814 USD |  |
| 2026-11-23 | b63a87b | `configs/train/numbers-parseq-r2.yaml` | `numbers/parseq/r2@c4b3dde8` | 34517 / 1646 | val/top1 0.701 (epoch 2) | colab T4, 2515.9 s, 0.75 CCU |  |
