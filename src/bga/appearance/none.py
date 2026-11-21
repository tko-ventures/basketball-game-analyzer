"""``appearance/none``: no appearance embeddings, for configs that measure a layer without the appearance term."""

from __future__ import annotations

from typing import Any, ClassVar

from bga.registry import register
from bga.stage import BaseBackend, FrameReader, Sequence


@register("appearance", "none")
class NoAppearance(BaseBackend):
    """Appearance backend that embeds nothing: its output lists no boxes and carries no array.

    Examples:
        >>> NoAppearance({}).run(None, {}, None)
        {'boxes': [], 'dim': 0, 'model': None, 'ms_per_crop': None}

    """

    defaults: ClassVar[dict[str, Any]] = {}

    def run(self, seq: Sequence, upstream: dict, read: FrameReader) -> dict:
        """Return the empty appearance output.

        Args:
            seq: Unused.
            upstream: Unused.
            read: Unused.

        Returns:
            ``{"boxes": [], "dim": 0, "model": None, "ms_per_crop": None}``.

        """
        return {"boxes": [], "dim": 0, "model": None, "ms_per_crop": None}
