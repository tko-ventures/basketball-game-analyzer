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
