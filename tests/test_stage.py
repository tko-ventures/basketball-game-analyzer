"""Frame reader: the frames asked for come back whatever the step, and short steps ahead do not seek."""

import itertools

import cv2
import numpy as np
import pytest

from bga.stage import FrameReader

#: Grey level added per frame of the synthetic clip, enough for the codec to keep neighbours apart.
LEVEL = 3
#: Frames in the synthetic clip.
FRAMES = 80


class CountingCapture:
    """Wraps a ``cv2.VideoCapture`` and counts its seeks."""

    def __init__(self, cap):
        self.cap, self.seeks = cap, 0

    def set(self, prop, value):
        self.seeks += 1
        return self.cap.set(prop, value)

    def __getattr__(self, name):
        return getattr(self.cap, name)


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> tuple[str, list[np.ndarray]]:
    """A clip of flat grey frames, brighter with every frame, and its frames decoded one after another."""
    path = str(tmp_path_factory.mktemp("clip") / "grey.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 50.0, (64, 64))
    for i in range(FRAMES):
        writer.write(np.full((64, 64, 3), i * LEVEL, np.uint8))
    writer.release()
    cap = cv2.VideoCapture(path)
    decoded = [cap.read()[1] for _ in range(FRAMES)]
    cap.release()
    return path, decoded


@pytest.fixture
def truth(clip) -> list[np.ndarray]:
    """The clip's frames by index, decoded one after another."""
    return clip[1]


@pytest.fixture
def read(clip):
    """A reader of the synthetic clip whose seeks are counted."""
    reader = FrameReader(clip[0], cache=4)
    reader.cap = CountingCapture(reader.cap)
    yield reader
    reader.close()


class TestFrameReader:
    @pytest.mark.parametrize(
        "step",
        [
            pytest.param(1, id="every-frame"),
            pytest.param(5, id="gold-stride"),
            pytest.param(FrameReader.MAX_SKIP, id="longest-skip"),
            pytest.param(FrameReader.MAX_SKIP + 1, id="seek"),
        ],
    )
    def test_returns_the_frames_asked_for(self, read, truth, step):
        frames = list(range(3, FRAMES, step))
        assert [np.array_equal(read(f), truth[f]) for f in frames] == [True] * len(frames)

    def test_clip_frames_differ_from_their_neighbours(self, truth):
        assert [np.array_equal(a, b) for a, b in itertools.pairwise(truth)] == [False] * (FRAMES - 1)

    def test_short_steps_ahead_do_not_seek(self, read):
        frames = [read(f) for f in range(0, 40, 5)]
        assert (len(frames), read.cap.seeks) == (8, 0)

    def test_first_read_inside_the_clip_seeks_once(self, read):
        frames = [read(f) for f in range(70, FRAMES, 5)]
        assert (len(frames), read.cap.seeks) == (2, 1)

    def test_step_back_returns_the_earlier_frame(self, read, truth):
        read(40)
        assert np.array_equal(read(10), truth[10])

    def test_cached_frame_is_returned_without_decoding(self, read):
        first = read(20)
        assert read(20) is first

    def test_frame_past_the_end_is_unreadable(self, read):
        with pytest.raises(IndexError, match="frame 500 unreadable"):
            read(500)

    def test_reader_recovers_after_an_unreadable_frame(self, read, truth):
        with pytest.raises(IndexError):
            read(FRAMES + 2)
        assert np.array_equal(read(7), truth[7])
