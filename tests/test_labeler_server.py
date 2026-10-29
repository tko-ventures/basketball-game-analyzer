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


def test_label_progress_counts_done_only(tmp_path):
    server.save_labels(tmp_path, _labels())
    assert server.label_progress(tmp_path, "data/clips/x.mp4") == {"done": 1, "boxes": 2}
    assert server.label_progress(tmp_path, "data/clips/missing.mp4") == {"done": 0, "boxes": 0}


def test_safe_path_rejects_escape():
    with pytest.raises(PermissionError):
        server.safe_path("../../etc/passwd")


@pytest.mark.parametrize(
    "title,home,away",
    [
        ("SKRÓT | Dziki - Legia | 24. kolejka", "Dziki Warszawa", "Legia"),
        ("SKRÓT | Energa Trefl - Dziki | 1. mecz 1/4 play-off", "Energa Trefl", "Dziki Warszawa"),
        ("Skrót meczu | Anwil Włocławek - Dziki Warszawa 79:70", "Anwil Włocławek", "Dziki Warszawa"),
        ("MKS Dąbrowa Górnicza - Dziki Warszawa | Skrót meczu 🎞️", "MKS Dąbrowa Górnicza", "Dziki Warszawa"),
        ("PGE Start 🆚 Dziki | Skrót meczu", "PGE Start", "Dziki Warszawa"),
        (
            "Skrót spotkania Górnik Zamek Książ Wałbrzych-Dziki Warszawa",
            "Górnik Zamek Książ Wałbrzych",
            "Dziki Warszawa",
        ),
        ("SKRÓT MECZU 13.03.2026 I MKS DG vs DZIKI WARSZAWA", "MKS DG", "Dziki Warszawa"),
        ("Season preview", "", ""),
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
    assert v2["meta"]["home"] == {"team": "Legia", "roster": "", "kit": ""}
    assert v2["meta"]["away"] == {"team": "Dziki Warszawa", "roster": "dziki_warszawa", "kit": "dark"}
    assert "opponent" not in v2["meta"] and "dziki_kit" not in v2["meta"]
    assert [v2["identities"][k]["team"] for k in "123"] == ["away", "home", "ref"]


def test_season_for():
    assert season_for("20251023") == "2025/26"
    assert season_for("20260514") == "2025/26"
    assert season_for("20260926") == "2026/27"
    assert season_for(None) == ""
