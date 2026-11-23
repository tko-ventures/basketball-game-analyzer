#!/usr/bin/env python3
"""Probe, not a metrics row: PARSeq's released scene-text weights read zero-shot on the gold dev number crops.

    uv run --with "strhub @ git+https://github.com/baudm/parseq@1902db043c029a7e03a3818c616c06600af574be" \
        scripts/probe_parseq.py weights/pretrained/parseq/parseq-bb5792a6.pt

Answers whether a text recogniser that has never seen a jersey is worth fine-tuning for the M7 bake-off. The
checkpoint is the state dict from the v1.0.0 release (https://github.com/baudm/parseq/releases), loaded with
``weights_only=True`` into the model built from the repo's ``configs/model/parseq.yaml`` values. Crops are cut as
the numbers backends cut them (10 px padding) and stretched to 32 x 128.

A number's probability is the product of its digits' probabilities at their positions and of the end token after
them; ``unreadable`` takes the rest (letters, longer strings, the empty string). The reads then go through
``bga.numbers.base.roster_choice`` and ``bga.eval.numbers.Tally``, as in ``bga eval-numbers``.

"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bga.config import load_config  # noqa: E402
from bga.eval.numbers import Tally, load_gold_crops  # noqa: E402
from bga.numbers.base import NUMBER_CLASSES, UNREADABLE, CropRead, number_crop, roster_choice, roster_numbers  # noqa: E402
from bga.pipeline import sequences  # noqa: E402
from bga.stage import FrameReader  # noqa: E402

CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
HPARAMS = {
    "charset_train": CHARSET,
    "charset_test": CHARSET,
    "max_label_length": 25,
    "batch_size": 384,
    "lr": 7e-4,
    "warmup_pct": 0.075,
    "weight_decay": 0.0,
    "img_size": [32, 128],
    "patch_size": [4, 8],
    "embed_dim": 384,
    "enc_num_heads": 6,
    "enc_mlp_ratio": 4,
    "enc_depth": 12,
    "dec_num_heads": 12,
    "dec_mlp_ratio": 4,
    "dec_depth": 1,
    "perm_num": 6,
    "perm_forward": True,
    "perm_mirrored": True,
    "decode_ar": True,
    "refine_iters": 1,
    "dropout": 0.1,
}
EOS = 0  # token ids: the end token, then the charset in order, so digit d is token d + 1


def number_probs(p):
    """Probabilities over NUMBER_CLASSES from per-position token probabilities, shape (positions, tokens)."""
    out = np.zeros(len(NUMBER_CLASSES), np.float64)
    for n in range(100):
        s = str(n)
        prob = p[len(s), EOS]
        for i, ch in enumerate(s):
            prob *= p[i, int(ch) + 1]
        out[n + 1] = prob
    out[0] = max(0.0, 1.0 - out[1:].sum())
    return out


def to_input(crop):
    rgb = cv2.cvtColor(cv2.resize(crop, (128, 32), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    return (torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0 - 0.5) / 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("weights", type=Path)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/numbers-smolvlm-nba.yaml", help="shots and pad")
    a = ap.parse_args()

    from strhub.models.parseq.system import PARSeq

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    system = PARSeq(**HPARAMS)
    system.model.load_state_dict(torch.load(a.weights, map_location="cpu", weights_only=True))
    system = system.eval().to(device)

    cfg = load_config(a.config)
    seqs = {s.name: s for s in sequences(cfg)}
    by_seq = defaultdict(list)
    for c in load_gold_crops():
        if c.sequence in seqs:
            by_seq[c.sequence].append(c)
    tally, secs, n = Tally(), 0.0, 0
    decoded = []
    for name, crops in by_seq.items():
        seq = seqs[name]
        rosters = {side: roster_numbers(seq.meta, side) for side in ("home", "away")}
        read = FrameReader(seq.video)
        try:
            images = [number_crop(read(c.frame), c.xyxy, 10) for c in crops]
        finally:
            read.close()
        t0 = time.perf_counter()
        with torch.no_grad():
            probs = system(torch.stack([to_input(i) for i in images]).to(device)).softmax(-1).cpu()
        secs, n = secs + time.perf_counter() - t0, n + len(crops)
        labels, _ = system.tokenizer.decode(probs)
        for c, row, text in zip(crops, probs.numpy(), labels, strict=True):
            dist = number_probs(row)
            best = NUMBER_CLASSES[int(dist.argmax())]
            r = CropRead(None if best == UNREADABLE else best, dict(zip(NUMBER_CLASSES, dist.tolist(), strict=True)))
            tally.add(c, r, roster_choice(r, rosters.get(c.team, set())))
            decoded.append((c.label, text, r.value))
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in tally.summary().items()})
    print(f"{n} crops on {len(by_seq)} shots, {1000 * secs / max(1, n):.1f} ms per crop on {device}")
    print("gold, decoded text, number answered (first 40):", decoded[:40])


if __name__ == "__main__":
    main()
