"""Square tiles of a wide frame, for detectors that resize their input to a square (plan section 5, layer 2).

RF-DETR stretches every image to a square input. A 1920x1080 frame at 704 px shrinks a 20 px ball to about 7x13 px.
Two overlapping square crops of the full frame height keep the aspect ratio and scale objects by 704/1080 instead of
704/1920 across. Training crops the boxes with the tiles; inference runs on each tile and merges the detections back.

"""

from __future__ import annotations

from bga.detect.nms import iou
from bga.types import BBox

#: A tile as x1, y1, x2, y2 in frame pixels.
Tile = tuple[int, int, int, int]


def square_tiles(width: int, height: int, n: int = 2) -> list[Tile]:
    """Split a frame into ``n`` square tiles of its full short side, spread evenly and overlapping.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        n: Number of tiles; 1 returns the whole frame.

    Returns:
        The tiles, left to right (top to bottom for a portrait frame).

    Examples:
        >>> square_tiles(1920, 1080)
        [(0, 0, 1080, 1080), (840, 0, 1920, 1080)]
        >>> square_tiles(1920, 1080, 1)
        [(0, 0, 1920, 1080)]

    """
    if n <= 1:
        return [(0, 0, width, height)]
    side, long = min(width, height), max(width, height)
    starts = [round(i * (long - side) / (n - 1)) for i in range(n)]
    return [(s, 0, s + side, side) if width >= height else (0, s, side, s + side) for s in starts]


def crop_box(box: BBox, tile: Tile, min_visible: float = 0.5) -> BBox | None:
    """Clip a frame box to a tile and shift it into tile coordinates.

    Args:
        box: Box in frame pixels, x1, y1, x2, y2.
        tile: The tile.
        min_visible: Smallest share of the box's area that must lie inside the tile.

    Returns:
        The clipped box relative to the tile's corner, or None when too little of it is inside.

    Examples:
        >>> crop_box((900, 100, 1000, 300), (840, 0, 1920, 1080))
        (60, 100, 160, 300)
        >>> crop_box((1050, 100, 1150, 300), (0, 0, 1080, 1080)) is None
        True

    """
    x1, y1, x2, y2 = max(box[0], tile[0]), max(box[1], tile[1]), min(box[2], tile[2]), min(box[3], tile[3])
    area = (box[2] - box[0]) * (box[3] - box[1])
    if x2 <= x1 or y2 <= y1 or area <= 0 or (x2 - x1) * (y2 - y1) < min_visible * area:
        return None
    return (x1 - tile[0], y1 - tile[1], x2 - tile[0], y2 - tile[1])


def merge(per_tile: list[list[list]], tiles: list[Tile], iou_thr: float = 0.5, edge_px: float = 2.0) -> list[list]:
    """Merge per-tile detections ``[x1, y1, x2, y2, score, class]`` into frame coordinates.

    A box that touches a tile edge lying inside the frame is probably cut by the tile, so its score is halved before
    a greedy per-class suppression at ``iou_thr``; where two tiles overlap, the whole box from the other tile wins.

    Args:
        per_tile: Detections of each tile, in tile coordinates.
        tiles: The tiles, in the same order.
        iou_thr: Same-class boxes overlapping more than this are one object.
        edge_px: Distance from an inner tile edge that counts as touching it.

    Returns:
        The kept detections in frame pixels, highest score first.

    Examples:
        >>> tiles = square_tiles(1920, 1080)
        >>> cut = [[1000, 100, 1080, 300, 0.9, "player"]]  # cut by the right edge of the left tile
        >>> whole = [[160, 100, 260, 300, 0.8, "player"]]  # the same player in the right tile
        >>> merge([cut, whole], tiles)
        [[1000, 100, 1100, 300, 0.8, 'player']]

    """
    frame_w = max(t[2] for t in tiles)
    frame_h = max(t[3] for t in tiles)
    cands = []
    for dets, (tx, ty, tx2, ty2) in zip(per_tile, tiles, strict=True):
        for x1, y1, x2, y2, s, c in dets:
            fx1, fy1, fx2, fy2 = x1 + tx, y1 + ty, x2 + tx, y2 + ty
            cut = (
                (tx > 0 and fx1 - tx <= edge_px)
                or (tx2 < frame_w and tx2 - fx2 <= edge_px)
                or (ty > 0 and fy1 - ty <= edge_px)
                or (ty2 < frame_h and ty2 - fy2 <= edge_px)
            )
            cands.append(([fx1, fy1, fx2, fy2], s * (0.5 if cut else 1.0), s, c))
    cands.sort(key=lambda d: -d[1])
    kept: list[tuple] = []
    for box, rank, s, c in cands:
        if all(k[3] != c or iou(tuple(k[0]), tuple(box)) <= iou_thr for k in kept):
            kept.append((box, rank, s, c))
    return [[*box, s, c] for box, _, s, c in kept]
