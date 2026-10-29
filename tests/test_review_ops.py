from scripts.import_prefill import import_tracks
from scripts.review_ops import apply_ops


def _labels():
    labels = {"identities": {}, "frames": {}, "meta": {}}
    tracks = {
        "frames": [
            {
                "frame_idx": 0,
                "tracks": [
                    {"track_id": 1, "xyxy": [0, 0, 10, 20]},
                    {"track_id": 2, "xyxy": [50, 0, 60, 20], "team": "ref"},
                ],
            },
            {
                "frame_idx": 5,
                "tracks": [
                    {"track_id": 3, "xyxy": [1, 0, 11, 20]},
                    {"track_id": 2, "xyxy": [51, 0, 61, 20], "team": "ref"},
                ],
            },
            {
                "frame_idx": 10,
                "tracks": [{"track_id": 3, "xyxy": [2, 0, 12, 20]}, {"track_id": 4, "xyxy": [90, 0, 99, 20]}],
            },
        ]
    }
    assert import_tracks(labels, tracks) == 3
    return labels


def test_import_marks_prefill_and_referees():
    labels = _labels()
    assert labels["frames"]["0"]["prefill"] and not labels["frames"]["0"]["done"]
    assert labels["identities"]["1002"]["team"] == "ref"
    assert labels["identities"]["1001"]["team"] == ""


def test_merge_split_delete_assign_confirm():
    labels = _labels()
    counts = apply_ops(
        labels,
        [
            "clip x",
            "merge 1003 1001  # fragment of the same player",
            "team 1001 home",
            "number 1001 23",
            "split 1001 10 1005",
            "delete 1004",
            "confirm 1001",
        ],
    )
    assert counts == {"merge": 1, "team": 1, "number": 1, "split": 1, "delete": 1, "confirm": 1}
    tids = {k: [b["tid"] for b in f["boxes"]] for k, f in labels["frames"].items()}
    assert tids == {"0": [1001, 1002], "5": [1001, 1002], "10": [1005]}
    assert labels["identities"]["1001"] == {
        "team": "home",
        "number": "23",
        "name": "",
        "source": "prefill",
        "reviewed": True,
    }
    assert labels["identities"]["1005"]["team"] == "home" and not labels["identities"]["1005"]["reviewed"]
    assert "1003" not in labels["identities"] and "1004" not in labels["identities"]


def test_note_keeps_hash_and_inline_comments_are_stripped():
    labels = _labels()
    apply_ops(
        labels,
        ["# whole-line comment", "note 1001 possibly #7, not legible   # my comment", "team 1001 away  # x"],
    )
    assert labels["identities"]["1001"]["note"] == "possibly #7, not legible"
    assert labels["identities"]["1001"]["team"] == "away"
