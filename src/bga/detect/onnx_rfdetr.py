"""RF-DETR through ONNX Runtime with the CoreML provider on the Mac (pattern from the prior repo).

Becomes the ``rfdetr_onnx`` backend of layer 2 once the registry lands in M2. Class names come from
``<weights>.meta.json`` (``{"classes": [...]}``) when present, else the export's 10-class schema
with its leading background logit.

"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from bga.detect.nms import nested_box_nms
from bga.types import DETECTION_CLASSES, Detection

if TYPE_CHECKING:
    import onnxruntime as ort

log = logging.getLogger(__name__)

#: Class names of a Roboflow RF-DETR ONNX export: a leading background logit, then the 10-class schema.
ROBOFLOW_EXPORT_CLASSES = ("background", *DETECTION_CLASSES)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _session(path: Path, device: str) -> ort.InferenceSession:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    available = set(ort.get_available_providers())
    providers = ["CPUExecutionProvider"]
    if device in {"coreml", "mps", "auto"} and "CoreMLExecutionProvider" in available:
        providers.insert(0, "CoreMLExecutionProvider")
    elif device == "cuda" and "CUDAExecutionProvider" in available:
        providers.insert(0, "CUDAExecutionProvider")
    try:
        return ort.InferenceSession(str(path), sess_options=opts, providers=providers)
    except Exception as exc:
        if len(providers) == 1:
            raise
        log.warning("ONNX providers %s failed (%s); falling back to CPU", providers, exc)
        return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))


def decode(
    boxes_cxcywh: np.ndarray,
    logits: np.ndarray,
    class_names: tuple[str, ...],
    image_wh: tuple[int, int],
    confidence: float,
    top_k: int = 100,
) -> list[Detection]:
    """Decode one image's DETR output: ``boxes`` (Q, 4) normalised cxcywh, ``logits`` (Q, C).

    The ``top_k`` highest sigmoid scores over all (query, class) pairs are kept down to ``confidence``; a leading
    ``background`` class is dropped and the class ids shifted past it.

    Args:
        boxes_cxcywh: Boxes per query, normalised to the image.
        logits: Class logits per query.
        class_names: Name per logit column, optionally led by ``background``.
        image_wh: Image width and height in pixels.
        confidence: Lowest score kept.
        top_k: Most (query, class) pairs considered.

    Returns:
        Detections in pixel xyxy, clipped to the image, highest score first.

    Examples:
        >>> boxes = np.array([[0.5, 0.5, 0.2, 0.4]], dtype=np.float32)
        >>> logits = np.full((1, len(ROBOFLOW_EXPORT_CLASSES)), -10.0, dtype=np.float32)
        >>> logits[0, 4] = 5.0
        >>> dets = decode(boxes, logits, ROBOFLOW_EXPORT_CLASSES, (1000, 500), confidence=0.5)
        >>> [(d.class_name, d.class_id, tuple(round(v, 1) for v in d.xyxy)) for d in dets]
        [('player', 3, (400.0, 150.0, 600.0, 350.0))]

    """
    scores = _sigmoid(logits)
    n_cls = scores.shape[-1]
    names = list(class_names)
    has_bg = bool(names) and names[0].startswith("background")
    flat = scores.reshape(-1)
    k = min(top_k, flat.size)
    order = np.argpartition(-flat, k - 1)[:k]
    order = order[np.argsort(-flat[order])]
    w, h = image_wh
    out: list[Detection] = []
    for index in order:
        score = float(flat[index])
        if score < confidence:
            break
        query, label = divmod(int(index), n_cls)
        if has_bg:
            if label == 0:
                continue
            label -= 1
        names_fg = names[1:] if has_bg else names
        if label >= len(names_fg):
            continue
        cx, cy, bw, bh = (float(v) for v in boxes_cxcywh[query])
        xyxy = (
            max(0.0, (cx - bw / 2) * w),
            max(0.0, (cy - bh / 2) * h),
            min(float(w), (cx + bw / 2) * w),
            min(float(h), (cy + bh / 2) * h),
        )
        out.append(Detection(xyxy=xyxy, class_name=names_fg[label], score=score, class_id=label))
    return out


class OnnxRFDETR:
    """RF-DETR detector on an ONNX export, decoded and filtered by :func:`nested_box_nms`.

    Examples:
        >>> det = OnnxRFDETR("weights/detect/rfdetr.onnx", device="coreml")  # doctest: +SKIP
        >>> det.predict(frame_bgr)  # doctest: +SKIP
        [Detection(xyxy=(...), class_name='player', score=..., class_id=3), ...]

    """

    def __init__(
        self,
        weights: str | Path,
        device: str = "auto",
        confidence: float = 0.35,
        nms_iou: float = 0.7,
        nms_containment: float = 0.85,
    ) -> None:
        """Configure the detector; the ONNX session opens on :meth:`load`.

        Args:
            weights: Path of the ``.onnx`` file; a sibling ``.meta.json`` may list its class names.
            device: ``coreml``, ``mps`` or ``auto`` try the CoreML provider, ``cuda`` the CUDA one; CPU otherwise.
            confidence: Lowest score kept by :func:`decode`.
            nms_iou: ``iou_threshold`` of :func:`nested_box_nms`.
            nms_containment: ``containment_threshold`` of :func:`nested_box_nms`.

        """
        self.weights = Path(weights)
        self.device = device
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.nms_containment = nms_containment
        self._sess = None
        self.class_names: tuple[str, ...] = ROBOFLOW_EXPORT_CLASSES
        self.input_size = 0

    def load(self) -> None:
        """Open the ONNX session and read class names and input size.

        Raises:
            FileNotFoundError: The weights file does not exist.

        """
        if not self.weights.is_file():
            raise FileNotFoundError(self.weights)
        meta = self.weights.with_suffix(".meta.json")
        if meta.is_file():
            self.class_names = tuple(json.loads(meta.read_text())["classes"])
        self._sess = _session(self.weights, self.device)
        shape = self._sess.get_inputs()[0].shape
        self.input_size = int(shape[-1]) if isinstance(shape[-1], int) else 640
        log.info("RF-DETR ONNX %s providers=%s", self.weights, self._sess.get_providers())

    def unload(self) -> None:
        """Drop the ONNX session."""
        self._sess = None

    def predict(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Detect on one frame, loading the session first if needed.

        Args:
            frame_bgr: Frame as an OpenCV BGR image.

        Returns:
            Detections in the frame's pixel coordinates, after nested-box suppression, highest score first.

        """
        if self._sess is None:
            self.load()
        s = self.input_size
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (s, s)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.transpose((rgb - _MEAN) / _STD, (2, 0, 1))[None].astype(np.float32)
        boxes, logits = self._sess.run(None, {self._sess.get_inputs()[0].name: x})[:2]
        h, w = frame_bgr.shape[:2]
        dets = decode(boxes[0], logits[0], self.class_names, (w, h), self.confidence)
        return nested_box_nms(dets, iou_threshold=self.nms_iou, containment_threshold=self.nms_containment)
