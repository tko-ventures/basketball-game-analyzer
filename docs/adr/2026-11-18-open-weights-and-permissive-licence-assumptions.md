# Open-weight policy and interim permissive licence assumptions

- **Date:** 2026-11-18 *(extended 2026-11-22 classifiers, 2026-11-23 PARSeq)*
- **Status:** Accepted *(pre-ship audit required before commercial use)*
- **Area:** Process / Licences
- **Decider:** Tomasz

## Context

Production models must be open-weight and fine-tuned on PLK data (closed VLM APIs are for offline labelling/QA only). Several strong open-weight checkpoints carry ambiguous or research-only data provenance, while others have copyleft licences.

## Decision

1. **Hard exclusions (do not use):**
   - **SAM 3 / SAM 3.1:** Meta declined access; never use SAM 3.x weights or third-party re-uploads.
   - **YOLO26 / Ultralytics AGPL-3.0:** Dropped to avoid commercial-licence lock-in.
2. **Interim R&D ruling ("make things work first, then make them right"):**
   - Treat the following weights as Apache 2.0 during R&D to unblock measurement and PLK fine-tuning:
     - **RF-DETR XL / 2XL** (Roboflow Platform Model Licence 1.0)
     - **Roboflow `reid` pretrained encoders** (`osnet_x1_0_sportsmot` — code is Apache 2.0, SportsMOT dataset is CC BY-NC 4.0)
     - **ImageNet-pretrained ResNet-18** (`crop_cls`)
     - **PARSeq released scene-text weights** (`parseq-bb5792a6.pt`, trained on mixed real scene-text datasets) — *open confirmation from Tomasz on 2026-11-23*.
   - **Permissive baselines kept where cheap:** DINOv2 ViT-S/14 (Apache 2.0) is trained alongside ResNet-18 for `crop_cls`; RF-DETR Large (Apache 2.0) remains the active detector.

## Consequences

- M6 dropped training a separate DINOv2-backed re-ID encoder solely for licence cleanliness; `osnet_x1_0_sportsmot` is the single fine-tune starting point.
- **Pre-ship gate:** Before selling or deploying any product commercially, re-audit every non-Apache weight (re-ID, PARSeq, any PML/ImageNet model) and re-train from a strictly permissive backbone/dataset if required.
