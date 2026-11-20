"""supervision 0.30.9's ``ByteTrack``, vendored unchanged so the ``track/bytetrack`` backend keeps its outputs.

supervision removed ``ByteTrack`` in 0.31 (moved to the ``trackers`` package, a different implementation), and M6 needs
``trackers`` 2.7, which needs supervision 0.31. The five modules here are supervision 0.30.9's
``supervision/tracker/byte_tracker/`` with three changes: package-relative imports, the deprecation decorator removed
from ``ByteTrack``, and this file. Licence: MIT, Copyright (c) 2022 Roboflow, in ``LICENSE`` next to this file.
Kept out of ruff, docformatter and codespell, so the code stays byte-comparable with the original.

"""

from .core import ByteTrack

__all__ = ["ByteTrack"]
