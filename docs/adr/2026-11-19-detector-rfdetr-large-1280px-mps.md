# RF-DETR Large at 1280 px on PyTorch/MPS; drop XL, DEIM, YOLO26, and court filter

- **Date:** 2026-11-19 *(incorporates 2026-11-18 refocus & 2026-11-20 PR #22 alignment)*
- **Status:** Accepted
- **Area:** Pipeline (detect)
- **Decider:** Tomasz / Implementer

## Context

On gold v1 dev, the pretrained NBA detector found 89% of people but only 19% of balls and 66% of numbers because 1080p frames stretched to 640 px shrink a ~20 px ball to ~7 px. M4 needed higher recall on `ball` and `number`, faster Modal training, and a verified local Mac runtime.

## Decision

1. **Resolution over architecture:** Train and infer **RF-DETR Large at 1280 px** (`detect/rfdetr_torch`). On NBA data alone, raising resolution from 704 px to 1280 px lifted ball AP50 from 0.258 to 0.373 and number AP50 from 0.680 to 0.796; adding PLK frames lifted ball AP50 to ~0.77–0.80. Two-tile 704 px inference stays in code (`bga.detect.tiles`) as a faster Mac fallback if needed.
2. **Model bake-off outcome:**
   - **RF-DETR Large stays** (Apache 2.0).
   - **RF-DETR XL dropped:** Matched Large within noise (ball 0.486/0.783 vs 0.487/0.798), much slower on Mac, and under PML 1.0.
   - **DEIM-L not run:** Large vs XL showed labelled PLK data—not model capacity—is the bottleneck.
   - **YOLO26 dropped:** AGPL-3.0 licence.
3. **Runtime on Mac:** Keep **PyTorch on MPS** (`rfdetr_torch`). ONNX Runtime with the CoreML provider fails on part of the 1280 px graph. Native Core ML (fp16, ~30% faster) waits until Mac speed becomes a bottleneck.
4. **No court-polygon detection filter:** Measured on `plk-r2` (2 m, 1 m, 0.5 m margins). The PLK detector already ignores bench/crowd; tight margins only hurt sideline referees (AP50 0.859 → 0.608).
5. **Standard training recipe:** Batch 16 (no grad accumulation), validate every 5 epochs, early stopping after 2 validations without improvement — **3× faster** (18 min vs 53 min on H100) for identical accuracy. Multi-GPU DDP stays in code for full-game datasets but is not used at ~1k images.

## Consequences

- Downstream milestones run `rfdetr_torch` at 1280 px on MPS (~330–350 ms/frame cold-included average on M1 Max).
- `bga.detect.onnx_rfdetr` remains unregistered legacy code.
- Pose estimation (17-point COCO) is deferred out of M4 to M8 (ankle ground point).
