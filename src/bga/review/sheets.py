"""Contact sheets: many small images with captions on one picture, so one image read covers ~20 items.

Every sheet that is read gets a row in ``data/labels/sheet_ledger.tsv`` (plan section 7).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

LEDGER = Path("data/labels/sheet_ledger.tsv")
LEDGER_HEADER = "date\tsheet\tpurpose\titems\tresult\n"


def fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Letterbox ``img`` into a w x h tile."""
    s = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))), cv2.INTER_AREA)
    tile = np.zeros((h, w, 3), np.uint8)
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    tile[y : y + r.shape[0], x : x + r.shape[1]] = r
    return tile


def contact_sheet(
    items: Sequence[tuple[np.ndarray, str]],
    cols: int,
    tile_wh: tuple[int, int] = (320, 180),
    caption_h: int = 22,
) -> np.ndarray:
    """Grid of (image, caption) tiles, captions drawn above each tile."""
    w, h = tile_wh
    rows = max(1, -(-len(items) // cols))
    sheet = np.full((rows * (h + caption_h), cols * w, 3), 32, np.uint8)
    for i, (img, cap) in enumerate(items):
        r, c = divmod(i, cols)
        y, x = r * (h + caption_h), c * w
        sheet[y + caption_h : y + caption_h + h, x : x + w] = fit(img, w, h)
        cv2.putText(sheet, cap, (x + 4, y + caption_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return sheet


def log_sheet(sheet: str | Path, purpose: str, items: int, result: str, ledger: Path = LEDGER) -> None:
    if not ledger.exists():
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text(LEDGER_HEADER)
    row = [dt.date.today().isoformat(), Path(sheet).name, purpose, str(items), result.replace("\t", " ")]
    with ledger.open("a") as f:
        f.write("\t".join(row) + "\n")
