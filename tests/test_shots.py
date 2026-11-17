import cv2
import numpy as np

from bga.ingest.hist_diff import detect_shots

FPS = 25


def _scene(seed: int, n: int) -> list[np.ndarray]:
    """A textured scene panning slowly, so there is frame-to-frame motion but no cut."""
    rng = np.random.default_rng(seed)
    base = cv2.resize(rng.integers(0, 255, (18, 32, 3), dtype=np.uint8), (400, 160), cv2.INTER_CUBIC)
    base = cv2.add(base, np.full_like(base, seed * 40 % 120))
    return [base[:, i : i + 192].copy() for i in range(n)]


def test_detects_hard_cuts_between_scenes(tmp_path):
    path = tmp_path / "cuts.avi"
    lengths = [40, 30, 50]  # cuts at 40 and 70
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (192, 160))
    for seed, n in enumerate(lengths):
        for f in _scene(seed + 1, n):
            w.write(f)
    w.release()
    shots = detect_shots(path)
    assert [(s["start"], s["end"]) for s in shots] == [(0, 39), (40, 69), (70, 119)]
    assert {s["camera"] for s in shots} == {"unset"}
