"""Cut detectors: TransNet V2's streamed windows match upstream's, and transition flags become shots."""

import numpy as np
import pytest

from bga.ingest.cuts import transitions_to_shots, use_2d_convs, windows


def upstream_windows(frames: np.ndarray) -> list[np.ndarray]:
    """Upstream ``TransNetV2.predict_frames`` input iterator (soCzech/TransNetV2 ``inference/transnetv2.py``)."""
    no_padded_frames_start = 25
    no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)
    padded = np.concatenate([frames[:1]] * no_padded_frames_start + [frames] + [frames[-1:]] * no_padded_frames_end, 0)
    return [padded[ptr : ptr + 100] for ptr in range(0, len(padded) - 99, 50)]


def frames_of(n: int) -> np.ndarray:
    """``n`` frames whose pixels hold their index modulo 256 (and the index // 256 in channel 1)."""
    out = np.zeros((n, 27, 48, 3), np.uint8)
    out[:, :, :, 0] = (np.arange(n) % 256)[:, None, None]
    out[:, :, :, 1] = (np.arange(n) // 256)[:, None, None]
    return out


LENGTHS = [1, 49, 50, 51, 74, 75, 99, 100, 101, 149, 150, 151, 600, 613]


class TestWindows:
    @pytest.mark.parametrize("n", [pytest.param(n, id=f"n{n}") for n in LENGTHS])
    def test_match_upstream(self, n):
        frames = frames_of(n)
        ours = [w for w, _ in windows(iter(frames))]
        assert np.array_equal(np.stack(ours), np.stack(upstream_windows(frames)))

    @pytest.mark.parametrize("n", [pytest.param(n, id=f"n{n}") for n in LENGTHS])
    def test_keep_one_prediction_per_frame(self, n):
        assert sum(k for _, k in windows(iter(frames_of(n)))) == n

    def test_empty_stream_yields_nothing(self):
        assert list(windows(iter([]))) == []


class TestTransitionsToShots:
    def test_cover_every_frame(self):
        flags = np.array([1, 0, 0, 1, 1, 0], bool)
        shots = transitions_to_shots(flags)
        assert [(s["start"], s["end"]) for s in shots] == [(0, 0), (1, 4), (5, 5)]

    def test_no_transition_is_one_shot(self):
        assert transitions_to_shots(np.zeros(7, bool)) == [{"start": 0, "end": 6}]


class TestUse2dConvs:
    def test_same_output_as_3d_convs(self):
        torch = pytest.importorskip("torch")
        from bga.ingest._transnetv2 import TransNetV2

        torch.manual_seed(0)
        model = TransNetV2().eval()
        x = torch.from_numpy(frames_of(100)[None])
        with torch.no_grad():
            before = model(x)[0]
            after = use_2d_convs(model)(x)[0]
        assert torch.allclose(after, before, atol=1e-5)
