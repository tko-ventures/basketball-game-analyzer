"""Shot-cut detector ``hist_diff`` (layer 0 default), lifted from the prior labeller's ``ShotJobs``.

A cut is a mean pixel change far above a 25-frame rolling median baseline (same-court cuts
barely move the colour histogram), or a very large HSV histogram jump (court to crowd or
graphics). Shots shorter than ``min_len_s`` are merged into the previous one.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

DEFAULT_THRESHOLD = 0.25
DEFAULT_MIN_LEN_S = 0.4


class ShotCutDetector:
    """Streaming detector: feed frames in order, read ``cuts`` (frame indices that start a shot)."""

    def __init__(
        self, fps: float, threshold: float = DEFAULT_THRESHOLD, min_len_s: float = DEFAULT_MIN_LEN_S
    ):
        self.threshold = threshold
        self.min_len = max(1, int(min_len_s * fps))
        self.cuts = [0]
        self.n = 0
        self._prev_hist = None
        self._prev_gray = None
        self._recent: list[float] = []

    def update(self, frame_bgr: np.ndarray) -> None:
        small = cv2.resize(frame_bgr, (96, 54), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self._prev_hist is not None:
            d = cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            pix = float(np.abs(gray - self._prev_gray).mean())
            base = float(np.median(self._recent)) if self._recent else 0.0
            is_cut = (pix > 25.0 and pix > 4 * max(base, 3.0) and d > 0.04) or d > 2 * self.threshold
            if is_cut and self.n - self.cuts[-1] >= self.min_len:
                self.cuts.append(self.n)
            self._recent.append(pix)
            if len(self._recent) > 25:
                self._recent.pop(0)
        self._prev_hist, self._prev_gray = hist, gray
        self.n += 1

    def shots(self) -> list[dict]:
        """Shots as ``{"start", "end", "camera": "unset"}`` with inclusive frame bounds."""
        out = []
        for i, s in enumerate(self.cuts):
            e = (self.cuts[i + 1] - 1) if i + 1 < len(self.cuts) else self.n - 1
            out.append({"start": s, "end": e, "camera": "unset"})
        return out


def detect_shots(
    video: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
    min_len_s: float = DEFAULT_MIN_LEN_S,
    progress: Callable[[float], None] | None = None,
) -> list[dict]:
    """Decode every frame of ``video`` and return its shots."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    det = ShotCutDetector(fps, threshold, min_len_s)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            det.update(frame)
            if progress and total and det.n % 250 == 0:
                progress(min(0.99, det.n / total))
    finally:
        cap.release()
    if progress:
        progress(1.0)
    return det.shots()
