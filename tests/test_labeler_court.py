import pytest

from tools.labeler.server import REPO, court_queue_path, load_court_labels, safe_path, save_court_record


@pytest.fixture
def two_records(tmp_path) -> dict:
    """Court labels of clip c1 after saving an ok keyframe (12.5 s) and an unusable one (3 s)."""
    rec = {"H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "fixed": {"3": [10, 20]}, "status": "ok"}
    save_court_record(tmp_path, {"clip": "c1", "frame": 120, "record": rec, "active_seconds": 12.5})
    save_court_record(tmp_path, {"clip": "c1", "frame": 480, "record": {"status": "unusable"}, "active_seconds": 3})
    return load_court_labels(tmp_path, "c1")


class TestCourtRecord:
    def test_merges_frames(self, two_records):
        assert set(two_records["frames"]) == {"120", "480"}

    def test_keeps_fixed_points(self, two_records):
        assert two_records["frames"]["120"]["fixed"] == {"3": [10, 20]}

    def test_logs_time(self, two_records):
        assert sum(two_records["meta"]["review_seconds"].values()) == 15.5

    @pytest.mark.parametrize(
        ("body", "match"),
        [
            pytest.param({"clip": "../x", "frame": 1, "record": {"status": "ok"}}, "bad clip stem", id="clip-escape"),
            pytest.param({"clip": "c", "frame": 1, "record": {"status": "maybe"}}, "status must be", id="bad-status"),
        ],
    )
    def test_rejects_bad_input(self, tmp_path, body, match):
        with pytest.raises(ValueError, match=match):
            save_court_record(tmp_path, body)


class TestSafePath:
    def test_allows_symlinked_data(self):
        assert safe_path("data/raw/x.mp4") == REPO / "data/raw/x.mp4"

    @pytest.mark.parametrize(
        "rel", [pytest.param("../outside.mp4", id="parent"), pytest.param("data/../../outside.mp4", id="via-data")]
    )
    def test_refuses_escapes(self, rel):
        with pytest.raises(PermissionError):
            safe_path(rel)


class TestCourtQueuePath:
    @pytest.mark.parametrize(
        ("args", "expected"),
        [pytest.param((), "queue.json", id="main"), pytest.param(("redo",), "queue_redo.json", id="named")],
    )
    def test_names(self, tmp_path, args, expected):
        assert court_queue_path(tmp_path, *args).name == expected

    def test_refuses_path_characters(self, tmp_path):
        with pytest.raises(ValueError, match="bad queue name"):
            court_queue_path(tmp_path, "../x")
