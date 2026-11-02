"""SAM 2.1 video propagation on Modal for the gold-set track prefill (labelling aid, plan section 7).

    uv run scripts/extract_shot_frames.py
    uv run --extra remote modal run src/bga/remote/sam2_prefill.py                 # all gold shots
    uv run --extra remote modal run src/bga/remote/sam2_prefill.py --clip-id OSp445e0mXU --shots 10

Per shot: the detector's person boxes on the first sampled frame become SAM 2.1 box prompts; the masks
are propagated through every frame. A detection that stays uncovered on two consecutive sampled
frames is added as a new object where it first appears and propagation continues from there (at most
MAX_ROUNDS times). Mask bounding boxes on the sampled frames come back to
runs/<clip>/prefill/sam21_hiera_large/shot_<n>.json with GPU, wall time and cost; scripts/sam_to_tracks.py
turns them into a labeller tracks.json. SAM 2.1 weights: facebook/sam2.1-hiera-large, Apache 2.0,
baked into the image. SAM 3.x is not used anywhere (plan section 2).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

SAM2_SHA = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
MODEL_ID = "facebook/sam2.1-hiera-large"
GPU = "L40S"
USD_PER_HOUR = {"L40S": 1.95, "A100-80GB": 2.50, "H100": 3.95, "L4": 0.80}  # Modal list, 2026-10-29
MAX_ROUNDS = 12
COVER_IOU = 0.3
OUT_NAME = "sam21_hiera_large"

app = modal.App("bga-sam2-prefill")
vol = modal.Volume.from_name("bga-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
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


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def first_persistent_uncovered(prompts, results, sampled, start):
    """First sampled frame >= start with detections no mask covers that are still uncovered on the next
    sampled frame (so one-frame junk does not become an object)."""

    def uncovered(f):
        masks = [b for b in results.get(f, {}).values() if b]
        return [p for p in prompts.get(f, []) if max((_iou(p, m) for m in masks), default=0.0) < COVER_IOU]

    for i, f in enumerate(sampled[:-1]):
        if f < start:
            continue
        nxt = uncovered(sampled[i + 1])
        keep = [p for p in uncovered(f) if any(_iou(p, q) > 0.3 for q in nxt)]
        if keep:
            return f, keep
    return None


@app.function(image=image, gpu=GPU, volumes={"/data": vol}, timeout=3600)
def track_shot(shot_dir: str, prompts: dict[str, list]) -> dict:
    import numpy as np
    import torch
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    prompts = {int(k): v for k, v in prompts.items()}
    sampled = sorted(prompts)
    sampled_set = set(sampled)
    pred = SAM2VideoPredictor.from_pretrained(MODEL_ID, device="cuda")
    results: dict[int, dict[int, list | None]] = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = pred.init_state(video_path=f"/data/{shot_dir}", offload_video_to_cpu=True)
        next_id, start, rounds = 1, sampled[0], 0
        new = (start, prompts[start])
        while new:
            f, boxes = new
            for b in boxes:
                pred.add_new_points_or_box(
                    state, frame_idx=f, obj_id=next_id, box=np.array(b, dtype=np.float32)
                )
                next_id += 1
            for fidx, obj_ids, masks in pred.propagate_in_video(state, start_frame_idx=f):
                if fidx not in sampled_set:
                    continue
                m = masks[:, 0] > 0
                rows, cols = m.any(dim=2), m.any(dim=1)
                frame = {}
                for i, oid in enumerate(obj_ids):
                    if not rows[i].any():
                        frame[int(oid)] = None
                        continue
                    ys, xs = torch.nonzero(rows[i])[:, 0], torch.nonzero(cols[i])[:, 0]
                    frame[int(oid)] = [
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max()) + 1,
                        int(ys.max()) + 1,
                        int(m[i].sum()),
                    ]
                results[fidx] = frame
            rounds += 1
            new = (
                first_persistent_uncovered(prompts, results, sampled, f + 1) if rounds < MAX_ROUNDS else None
            )
    secs = time.time() - t0
    return {
        "shot_dir": shot_dir,
        "model": MODEL_ID,
        "sam2_sha": SAM2_SHA,
        "gpu": GPU,
        "rounds": rounds,
        "objects": next_id - 1,
        "seconds": round(secs, 1),
        "usd": round(secs / 3600 * USD_PER_HOUR[GPU], 4),
        "frames": {str(k): {str(o): b for o, b in v.items()} for k, v in sorted(results.items())},
    }


@app.local_entrypoint()
def main(clip_id: str = "", shots: str = ""):
    gold = json.loads(Path("data/labels/gold.json").read_text())
    want = {int(s) for s in shots.split(",") if s}
    jobs = []
    with vol.batch_upload(force=True) as up:
        for g in gold["games"]:
            if clip_id and g["youtube_id"] != clip_id:
                continue
            for s in g["shots"]:
                if want and s["shot"] not in want:
                    continue
                if (Path("runs") / g["clip"] / "prefill" / OUT_NAME / f"shot_{s['shot']}.json").exists():
                    continue  # already tracked; delete the file to rerun
                local = Path("runs") / g["clip"] / "sam_frames" / f"shot_{s['shot']}"
                meta = json.loads((local / "prompts.json").read_text())
                remote = f"frames/{g['youtube_id']}/shot_{s['shot']}"
                up.put_directory(str(local), remote)
                jobs.append((g, s, meta, remote))
    print(f"uploaded {len(jobs)} shots; tracking on {GPU}")
    t0 = time.time()
    for (g, s, meta, _), res in zip(
        jobs, track_shot.starmap([(r, m["prompts"]) for _, _, m, r in jobs]), strict=True
    ):
        res.update({k: meta[k] for k in ("clip", "shot", "start", "end", "scale", "size")})
        out = Path("runs") / g["clip"] / "prefill" / OUT_NAME
        out.mkdir(parents=True, exist_ok=True)
        (out / f"shot_{s['shot']}.json").write_text(json.dumps(res))
        with Path("docs/costs.md").open("a") as log:
            log.write(
                f"| {time.strftime('%Y-%m-%d')} | modal | {GPU} "
                f"| sam2_prefill {g['youtube_id']} shot {s['shot']} | {res['seconds']} | {res['usd']} |\n"
            )
        print(
            f"{g['youtube_id']} shot {s['shot']}: {res['objects']} objects, {res['rounds']} rounds, "
            f"{res['seconds']} s, {res['usd']} USD",
            flush=True,
        )
    print(f"wall {time.time() - t0:.0f} s")
