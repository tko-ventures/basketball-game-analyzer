"""SAM 2.1 box-prompted masks on Modal for the gold v1 box prefill (labelling aid, plan section 7).

    uv run scripts/box_gold_v1.py prefill                                   # writes runs/box_gold/v1/sam_input.json
    uv run --extra remote modal run src/bga/remote/sam2_boxmask.py           # SAM 2.1 on Modal
    uv run scripts/box_gold_v1.py prefill                                   # merges the masks into the queue

Each prefill box becomes a SAM 2.1 box prompt on its still frame (image mode, one mask per box). The mask's
bounding box is a second geometry, independent of the detector's, that the box page offers next to the prefill:
where the two agree the box is likely tight, where they differ the reviewer looks. Frames are decoded locally,
sent as JPEG (quality 95) in batches, and results come back to ``runs/box_gold/v1/sam_boxes.json`` with GPU, wall
time and cost, each mask next to its prompt box; a row goes to ``docs/costs.md``. SAM 2.1 weights: facebook/sam2.1-hiera-large, Apache 2.0, baked
into the image. SAM 3.x is not used anywhere (plan section 2).

"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

#: facebookresearch/sam2 commit installed in the image (the same as the track prefill).
SAM2_SHA = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
#: Hugging Face id of the SAM 2.1 weights (Apache 2.0).
MODEL_ID = "facebook/sam2.1-hiera-large"
#: Modal GPU type.
GPU = "L40S"
#: Modal list price per hour by GPU, 2026-10-29.
USD_PER_HOUR = {"L40S": 1.95, "A100-80GB": 2.50, "H100": 3.95, "L4": 0.80}
#: Frames per remote call.
BATCH = 10
#: Input and output files, relative to the repository root.
INPUT, OUTPUT = Path("runs/box_gold/v1/sam_input.json"), Path("runs/box_gold/v1/sam_boxes.json")

#: The Modal app.
app = modal.App("bga-sam2-boxmask")
#: Container image with SAM 2.1 and its weights baked in.
image = (
    modal.Image
    .debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy<2.3", "huggingface_hub", "pillow", "tqdm")
    .env({"SAM2_BUILD_CUDA": "0", "HF_HOME": "/root/hf"})
    .pip_install(
        f"https://github.com/facebookresearch/sam2/archive/{SAM2_SHA}.tar.gz",
        extra_options="--no-build-isolation",
    )
    .run_commands(
        f'python -c "from huggingface_hub import hf_hub_download; '
        f"hf_hub_download('{MODEL_ID}', 'sam2.1_hiera_large.pt')\""
    )
)


@app.function(image=image, gpu=GPU, timeout=1800)
def box_masks(items: list[tuple[str, bytes, list[list[float]]]]) -> dict:
    """Mask every box prompt on its frame and return the masks' bounding boxes.

    Args:
        items: ``(key, jpeg bytes, boxes)`` per frame, boxes as ``[x1, y1, x2, y2]`` in pixels.

    Returns:
        ``{"seconds", "usd", "gpu", "frames": {key: [[x1, y1, x2, y2, mask pixels, predicted IoU] or None, ...]}}``,
        one entry per input box in input order; None where the mask is empty.

    Examples:
        >>> box_masks.remote([("OSp445e0mXU_1200", jpeg, [[100, 200, 180, 420]])])["frames"]  # doctest: +SKIP
        {'OSp445e0mXU_1200': [[103, 201, 177, 418, 9120, 0.94]]}

    """
    import io

    import numpy as np
    import torch
    from PIL import Image
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    t0 = time.time()
    pred = SAM2ImagePredictor.from_pretrained(MODEL_ID, device="cuda")
    out: dict[str, list] = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for key, jpeg, boxes in items:
            if not boxes:
                out[key] = []
                continue
            pred.set_image(np.array(Image.open(io.BytesIO(jpeg)).convert("RGB")))
            masks, scores, _ = pred.predict(box=np.array(boxes, dtype=np.float32), multimask_output=False)
            masks = masks.reshape(len(boxes), *masks.shape[-2:]) > 0
            scores = np.asarray(scores).reshape(len(boxes))
            res = []
            for m, s in zip(masks, scores, strict=True):
                rows, cols = np.nonzero(m.any(axis=1))[0], np.nonzero(m.any(axis=0))[0]
                if not len(rows):
                    res.append(None)
                    continue
                res.append([
                    int(cols[0]),
                    int(rows[0]),
                    int(cols[-1]) + 1,
                    int(rows[-1]) + 1,
                    int(m.sum()),
                    round(float(s), 3),
                ])
            out[key] = res
    secs = time.time() - t0
    return {"seconds": round(secs, 1), "usd": round(secs / 3600 * USD_PER_HOUR[GPU], 4), "gpu": GPU, "frames": out}


@app.local_entrypoint()
def main() -> None:
    """Decode the queued frames locally, mask their boxes on Modal and write the results and a cost row.

    Examples:
        >>> main()  # doctest: +SKIP

    """
    import cv2

    from bga.remote.modal_runner import append_cost_rows

    req = json.loads(INPUT.read_text())
    items, caps = [], {}
    for f in req["frames"]:
        cap = caps.setdefault(f["video"], cv2.VideoCapture(f["video"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f["frame"])
        ok, img = cap.read()
        if not ok:
            raise RuntimeError(f"cannot decode {f['video']} frame {f['frame']}")
        jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
        items.append((f["key"], jpeg, f["boxes"]))
    batches = [items[i : i + BATCH] for i in range(0, len(items), BATCH)]
    print(f"{len(items)} frames, {sum(len(i[2]) for i in items)} boxes, {len(batches)} calls on {GPU}")
    frames, secs, usd = {}, 0.0, 0.0
    prompts = {k: boxes for k, _, boxes in items}
    for res in box_masks.map(batches):
        frames.update({k: {"boxes": prompts[k], "masks": m} for k, m in res["frames"].items()})
        secs += res["seconds"]
        usd += res["usd"]
    OUTPUT.write_text(json.dumps({"model": MODEL_ID, "sam2_sha": SAM2_SHA, "gpu": GPU, "frames": frames}))
    append_cost_rows([
        f"| {time.strftime('%Y-%m-%d')} | modal | {GPU} | sam2_boxmask box gold v1 ({len(items)} frames) "
        f"| {round(secs, 1)} | {round(usd, 4)} |"
    ])
    print(f"{OUTPUT}: {round(secs, 1)} GPU s, {round(usd, 4)} USD")
