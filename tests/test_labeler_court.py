import pytest

from tools.labeler.server import load_court_labels, save_court_record


def test_court_record_merges_and_logs_time(tmp_path):
    rec = {"H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "fixed": {"3": [10, 20]}, "status": "ok"}
    save_court_record(tmp_path, {"clip": "c1", "frame": 120, "record": rec, "active_seconds": 12.5})
    save_court_record(
        tmp_path, {"clip": "c1", "frame": 480, "record": {"status": "unusable"}, "active_seconds": 3}
    )
    d = load_court_labels(tmp_path, "c1")
    assert set(d["frames"]) == {"120", "480"} and d["frames"]["120"]["fixed"] == {"3": [10, 20]}
    assert sum(d["meta"]["review_seconds"].values()) == 15.5


def test_court_record_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError):
        save_court_record(tmp_path, {"clip": "../x", "frame": 1, "record": {"status": "ok"}})
    with pytest.raises(ValueError):
        save_court_record(tmp_path, {"clip": "c", "frame": 1, "record": {"status": "maybe"}})
