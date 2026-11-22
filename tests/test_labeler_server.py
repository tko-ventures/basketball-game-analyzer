import json

import pytest

from tools.labeler import server
from tools.labeler.prepare_clips import season_for


def _labels():
    return {
        "version": 2,
        "clip": "data/clips/x.mp4",
        "fps": 50,
        "frame_count": 100,
        "width": 1920,
        "height": 1080,
        "stride": 5,
        "meta": {
            "season": "2025/26",
            "home": {"team": "Dziki Warszawa", "roster": "dziki_warszawa", "kit": "white"},
            "away": {"team": "Legia", "roster": "", "kit": "green"},
        },
        "shots": [],
        "identities": {
            "1": {"team": "home", "number": "5", "name": "A"},
            "2": {"team": "away", "number": "12"},
            "3": {"team": "ref", "number": ""},
        },
        "frames": {
            "0": {
                "done": True,
                "boxes": [
                    {"tid": 1, "x1": 10, "y1": 20, "x2": 40, "y2": 100},
                    {"tid": 3, "x1": 0, "y1": 0, "x2": 5, "y2": 5, "occluded": True},
                ],
            },
            "5": {"done": False, "interp": True, "boxes": [{"tid": 2, "x1": 1, "y1": 2, "x2": 3, "y2": 4}]},
            "10": {"done": False, "boxes": [{"tid": 1, "x1": 1, "y1": 1, "x2": 2, "y2": 2}]},
        },
    }


def test_save_writes_json_mot_and_identities(tmp_path):
    server.save_labels(tmp_path, _labels())
    assert json.loads((tmp_path / "x.json").read_text())["saved_at"]
    mot = (tmp_path / "x.mot.txt").read_text().strip().splitlines()
    # frame 10 is neither done nor interpolated, so it is excluded; frames are 1-based
    assert mot == ["1,1,10,20,30,80,1,1,1.0", "1,3,0,0,5,5,1,3,0.5", "6,2,1,2,2,2,1,2,1.0"]
    ids = (tmp_path / "x.identities.csv").read_text().splitlines()
    assert ids[0] == "tid,team,number,name,team_name,note"
    assert ids[1] == "1,home,5,A,Dziki Warszawa,"
    assert ids[2] == "2,away,12,,Legia,"
    assert ids[3] == "3,ref,,,,"


@pytest.mark.parametrize(
    ("clip", "expected"),
    [
        pytest.param("data/clips/x.mp4", {"done": 1, "boxes": 2}, id="labelled"),
        pytest.param("data/clips/missing.mp4", {"done": 0, "boxes": 0}, id="missing"),
    ],
)
def test_label_progress_counts_done_only(tmp_path, clip, expected):
    server.save_labels(tmp_path, _labels())
    assert server.label_progress(tmp_path, clip) == expected


def test_safe_path_rejects_escape():
    with pytest.raises(PermissionError):
        server.safe_path("../../etc/passwd")


@pytest.mark.parametrize(
    "title,home,away",
    [
        pytest.param("SKRÓT | Dziki - Legia | 24. kolejka", "Dziki Warszawa", "Legia", id="short-dziki-home"),
        pytest.param(
            "SKRÓT | Energa Trefl - Dziki | 1. mecz 1/4 play-off", "Energa Trefl", "Dziki Warszawa", id="playoff"
        ),
        pytest.param(
            "Skrót meczu | Anwil Włocławek - Dziki Warszawa 79:70",
            "Anwil Włocławek",
            "Dziki Warszawa",
            id="with-score",
        ),
        pytest.param(
            "MKS Dąbrowa Górnicza - Dziki Warszawa | Skrót meczu 🎞️",
            "MKS Dąbrowa Górnicza",
            "Dziki Warszawa",
            id="teams-first-emoji",
        ),
        pytest.param("PGE Start 🆚 Dziki | Skrót meczu", "PGE Start", "Dziki Warszawa", id="vs-emoji"),
        pytest.param(
            "Skrót spotkania Górnik Zamek Książ Wałbrzych-Dziki Warszawa",
            "Górnik Zamek Książ Wałbrzych",
            "Dziki Warszawa",
            id="unspaced-dash",
        ),
        pytest.param(
            "SKRÓT MECZU 13.03.2026 I MKS DG vs DZIKI WARSZAWA", "MKS DG", "Dziki Warszawa", id="upper-case-vs"
        ),
        pytest.param("SKRÓT | Dziki - Legia\xa0| 3. mecz półfinału", "Dziki Warszawa", "Legia", id="nbsp"),
        pytest.param("Season preview", "", "", id="no-teams"),
    ],
)
def test_teams_from_title(title, home, away):
    assert server.teams_from_title(title) == (home, away)


def test_migrate_v1_maps_dziki_and_opp_to_sides():
    v1 = {
        "version": 1,
        "meta": {
            "season": "2025/26",
            "opponent": "Legia",
            "dziki_kit": "dark",
            "notes": "",
            "title": "SKRÓT | Legia - Dziki | 9. kolejka",
        },
        "identities": {"1": {"team": "dziki"}, "2": {"team": "opp"}, "3": {"team": "ref"}},
    }
    v2 = server.migrate_labels(v1)
    assert v2["version"] == 2
    assert v2["meta"]["home"] == {"team": "Legia", "roster": "legia_warszawa", "kit": ""}
    assert v2["meta"]["away"] == {"team": "Dziki Warszawa", "roster": "dziki_warszawa", "kit": "dark"}
    assert "opponent" not in v2["meta"]
    assert "dziki_kit" not in v2["meta"]
    assert [v2["identities"][k]["team"] for k in "123"] == ["away", "home", "ref"]


@pytest.mark.parametrize(
    ("date", "season"),
    [
        pytest.param("20251023", "2025/26", id="autumn"),
        pytest.param("20260514", "2025/26", id="spring"),
        pytest.param("20260926", "2026/27", id="next-autumn"),
        pytest.param(None, "", id="no-date"),
    ],
)
def test_season_for(date, season):
    assert season_for(date) == season


@pytest.fixture
def box_queue(tmp_path):
    rec = {
        "youtube_id": "y",
        "frame": 5,
        "status": "todo",
        "boxes": [{"id": 3, "xyxy": [1, 2, 30, 40], "cls": "player"}],
    }
    (tmp_path / server.BOX_QUEUE).write_text(json.dumps({"frames": [rec], "meta": {}}))
    return tmp_path, rec


class TestSaveBoxFrame:
    def test_replaces_the_frame_and_logs_time(self, box_queue):
        root, rec = box_queue
        server.save_box_frame(root, {"index": 0, "frame": {**rec, "status": "done"}, "active_seconds": 4})
        q = server.load_box_queue(root)
        assert (q["frames"][0]["status"], sum(q["meta"]["review_seconds"].values())) == ("done", 4.0)

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            pytest.param({"status": "maybe"}, "status must be", id="bad-status"),
            pytest.param({"boxes": [{"xyxy": [1, 2, 30, 40], "cls": "coach"}]}, "unknown class", id="unknown-class"),
            pytest.param({"boxes": [{"xyxy": [30, 2, 1, 40], "cls": "ball"}]}, "empty box", id="empty-box"),
            pytest.param({"frame": 6}, "does not match", id="other-frame"),
            pytest.param({"holder": 9}, "not a box of this frame", id="holder-not-a-box"),
            pytest.param({"dead_ball": "yes"}, "dead_ball must be", id="dead-ball-not-bool"),
        ],
    )
    def test_rejects_bad_records(self, box_queue, change, message):
        root, rec = box_queue
        with pytest.raises(ValueError, match=message):
            server.save_box_frame(root, {"index": 0, "frame": {**rec, **change}})

    def test_keeps_the_possession_answer(self, box_queue):
        root, rec = box_queue
        server.save_box_frame(root, {"index": 0, "frame": {**rec, "status": "done", "holder": 3, "dead_ball": True}})
        f = server.load_box_queue(root)["frames"][0]
        assert (f["holder"], f["dead_ball"]) == (3, True)

    def test_rejects_a_record_from_a_stale_page(self, box_queue):
        root, rec = box_queue
        server.save_box_frame(root, {"index": 0, "frame": {**rec, "holder": None, "dead_ball": False}})
        with pytest.raises(ValueError, match="lacks dead_ball, holder"):
            server.save_box_frame(root, {"index": 0, "frame": {**rec, "status": "done"}})

    def test_rejects_a_frame_changed_after_the_page_loaded_it(self, box_queue):
        root, rec = box_queue
        server.save_box_frame(root, {"index": 0, "frame": {**rec, "status": "done"}})
        with pytest.raises(ValueError, match="changed after this page loaded it"):
            server.save_box_frame(root, {"index": 0, "frame": {**rec, "status": "skip"}})

    def test_accepts_a_save_carrying_the_stored_stamp(self, box_queue):
        root, rec = box_queue
        first = server.save_box_frame(root, {"index": 0, "frame": {**rec, "status": "done"}})
        server.save_box_frame(root, {"index": 0, "frame": {**rec, "status": "skip", "saved_at": first["saved_at"]}})
        assert server.load_box_queue(root)["frames"][0]["status"] == "skip"

    def test_rejected_record_leaves_queue_unchanged(self, box_queue):
        root, rec = box_queue
        before = (root / server.BOX_QUEUE).read_text()
        with pytest.raises(ValueError, match="does not match"):
            server.save_box_frame(root, {"index": 0, "frame": {**rec, "frame": 6}})
        assert (root / server.BOX_QUEUE).read_text() == before


class TestBoxQueueSets:
    def test_named_queue_is_separate_from_box_gold(self, tmp_path):
        rec = {"youtube_id": "y", "frame": 5, "status": "todo", "boxes": []}
        (tmp_path / "box_queue_r1balls.json").write_text(json.dumps({"frames": [rec], "meta": {}}))
        server.save_box_frame(tmp_path, {"index": 0, "frame": {**rec, "status": "skip"}}, "r1balls")
        assert server.load_box_queue(tmp_path, "r1balls")["frames"][0]["status"] == "skip"

    @pytest.mark.parametrize("name", [pytest.param("../x", id="path"), pytest.param("R1", id="upper-case")])
    def test_rejects_bad_names(self, tmp_path, name):
        with pytest.raises(ValueError, match="bad queue name"):
            server.box_queue_path(tmp_path, name)


@pytest.fixture
def number_crops(tmp_path):
    p = tmp_path / server.NUMBER_CROPS
    p.parent.mkdir()
    crops = [{"id": "n001", "clip": "c", "label": "23"}, {"id": "n002", "clip": "c", "label": "illegible"}]
    p.write_text(json.dumps({"crops": crops, "spotcheck": {"ids": ["n002"]}}))
    return tmp_path


class TestNumberQueue:
    @pytest.mark.parametrize(
        ("name", "ids"),
        [pytest.param("spotcheck", ["n002"], id="spotcheck"), pytest.param("all", ["n001", "n002"], id="all")],
    )
    def test_selects_the_set(self, number_crops, name, ids):
        assert [c["id"] for c in server.load_number_queue(number_crops, name)["crops"]] == ids

    def test_gives_each_crop_its_video(self, number_crops):
        assert server.load_number_queue(number_crops, "all")["crops"][0]["video"] == "data/raw/c.mp4"

    def test_rejects_an_unknown_set(self, number_crops):
        with pytest.raises(ValueError, match="unknown set"):
            server.load_number_queue(number_crops, "mine")


class TestSaveNumberReview:
    def test_stores_a_fix_beside_the_label(self, number_crops):
        server.save_number_review(number_crops, {"id": "n001", "verdict": "fix", "label": "28", "active_seconds": 3})
        crop = server.load_number_queue(number_crops, "reviewed")["crops"][0]
        assert (crop["label"], crop["review"]["verdict"], crop["review"]["label"]) == ("23", "fix", "28")

    def test_a_fix_to_the_same_label_is_ok(self, number_crops):
        res = server.save_number_review(number_crops, {"id": "n001", "verdict": "fix", "label": "23"})
        assert res["review"] == {"verdict": "ok", "at": res["review"]["at"]}

    def test_null_verdict_clears_the_review(self, number_crops):
        first = server.save_number_review(number_crops, {"id": "n001", "verdict": "ok"})
        server.save_number_review(number_crops, {"id": "n001", "verdict": None, "prev_at": first["review"]["at"]})
        assert server.load_number_queue(number_crops, "reviewed")["crops"] == []

    @pytest.mark.parametrize(
        ("body", "message"),
        [
            pytest.param({"id": "n009", "verdict": "ok"}, "unknown crop", id="unknown-crop"),
            pytest.param({"id": "n001", "verdict": "maybe"}, "verdict must be", id="bad-verdict"),
            pytest.param({"id": "n001", "verdict": "fix", "label": "123"}, "bad label", id="bad-label"),
        ],
    )
    def test_rejects_bad_answers(self, number_crops, body, message):
        with pytest.raises(ValueError, match=message):
            server.save_number_review(number_crops, body)

    def test_rejects_an_answer_from_a_stale_page(self, number_crops):
        server.save_number_review(number_crops, {"id": "n001", "verdict": "ok"})
        with pytest.raises(ValueError, match="changed after this page loaded it"):
            server.save_number_review(number_crops, {"id": "n001", "verdict": "fix", "label": "8", "prev_at": None})
