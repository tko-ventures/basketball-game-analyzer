"""Layer 3b (appearance).

Output per sequence:
    {"boxes": [[frame, det_index], ...], "dim": d, "model": "<encoder>", "ms_per_crop": ms,
     "arrays": {"embeddings": <(len(boxes), d) float16 array, L2-normalised>}}

Row ``i`` of the embeddings belongs to the detect box ``det_index`` of ``frame``; the array is stored next to the JSON
as ``<sequence>.embeddings.npy`` (``bga.pipeline.write_output``). ``none`` writes no boxes and no array.

"""

from __future__ import annotations

import numpy as np

from bga.stage import Backend as AppearanceBackend  # noqa: F401  # re-exported: the appearance layer's backend protocol


def l2_normalise(emb: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, so a dot product is the cosine similarity; zero rows stay zero.

    Args:
        emb: Embeddings as an ``(n, d)`` array.

    Returns:
        A float32 array of the same shape.

    Examples:
        >>> l2_normalise(np.array([[3.0, 4.0], [0.0, 0.0]])).tolist()
        [[0.6000000238418579, 0.800000011920929], [0.0, 0.0]]

    """
    emb = np.asarray(emb, dtype=np.float32)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.where(norm > 0, norm, 1.0)


def box_rows(out: dict) -> dict[tuple[int, int], int]:
    """Index an appearance output by box: ``(frame, det_index)`` to its row in the embeddings.

    Args:
        out: An appearance layer output.

    Returns:
        The row of each embedded box; boxes the layer skipped are absent.

    Examples:
        >>> box_rows({"boxes": [[100, 0], [100, 2], [105, 1]]})
        {(100, 0): 0, (100, 2): 1, (105, 1): 2}

    """
    return {(int(f), int(i)): row for row, (f, i) in enumerate(out["boxes"])}
