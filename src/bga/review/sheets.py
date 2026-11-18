"""Contact sheets: many small images with captions on one picture, so one image read covers ~20 items.

Every sheet that is read gets a row in ``data/labels/sheet_ledger.tsv`` (plan section 7).

"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

#: Ledger of every contact sheet read, one TSV row each.
LEDGER = Path("data/labels/sheet_ledger.tsv")
#: Header row written when the ledger is created.
LEDGER_HEADER = "date\tsheet\tpurpose\titems\tresult\n"


def fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Letterbox ``img`` into a w x h tile.

    Args:
        img: BGR image.
        w: Tile width.
        h: Tile height.

    Returns:
        A black ``(h, w, 3)`` tile with ``img`` scaled to fit and centred.

    Examples:
        >>> tile = fit(np.full((100, 100, 3), 255, np.uint8), 40, 20)
        >>> tile.shape, int(tile[:, :10].max()), int(tile[:, 10:30].min())
        ((20, 40, 3), 0, 255)

    """
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
    """Grid of (image, caption) tiles, captions drawn above each tile.

    Args:
        items: Images with their captions, filled row by row.
        cols: Tiles per row.
        tile_wh: Tile width and height; images are letterboxed into it.
        caption_h: Height of the caption band above each tile.

    Returns:
        The sheet as a BGR image on a dark grey background.

    Examples:
        >>> img = np.zeros((90, 160, 3), np.uint8)
        >>> contact_sheet([(img, "a"), (img, "b"), (img, "c")], cols=2).shape
        (404, 640, 3)

    """
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
    """Append a row for a sheet that was read to the ledger, creating it with its header if needed.

    Args:
        sheet: Path of the sheet; only its file name is logged.
        purpose: What the sheet was read for.
        items: Number of items on the sheet.
        result: What the read found; tabs become spaces.
        ledger: Ledger file.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     log_sheet("runs/x/sheet_01.jpg", "qa", 20, "ok", ledger=Path(d) / "ledger.tsv")
        ...     (Path(d) / "ledger.tsv").read_text().splitlines()[1].split()[1:]
        ['sheet_01.jpg', 'qa', '20', 'ok']

    """
    if not ledger.exists():
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text(LEDGER_HEADER)
    row = [dt.date.today().isoformat(), Path(sheet).name, purpose, str(items), result.replace("\t", " ")]
    with ledger.open("a") as f:
        f.write("\t".join(row) + "\n")
