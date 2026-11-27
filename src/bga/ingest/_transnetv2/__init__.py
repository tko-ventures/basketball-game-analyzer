"""TransNet V2's PyTorch model (soCzech/TransNetV2 at ``85cef72af9a9``), vendored for the ``cuts/transnetv2`` backend.

``model.py`` is ``inference-pytorch/transnetv2_pytorch.py`` of that commit, unchanged. The upstream project ships it
as a script, not a package. Its weights come from the official TensorFlow SavedModel of the same commit, converted
by ``scripts/convert_transnetv2.py``. Licence: MIT, Copyright (c) 2020 Tomáš Souček, in ``LICENSE`` next to this
file. Kept out of ruff, docformatter and codespell, so the code stays byte-comparable with the original.

"""

from .model import TransNetV2

__all__ = ["TransNetV2"]
