"""Shot gold: picking shots and frames, validating answers, the shots backend, and the labeller's /shotgold API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bga.config import RunConfig
from bga.eval.shot_gold import (
    Status,
    clean_event,
    clean_person,
    load_queue,
    pick_shots,
    progress,
    sample_frames,
    save_queue,
    set_frame_label,
    set_shot_events,
)
from bga.ingest.shot_gold_shots import ShotGoldShots
from tools.labeler.shotgold_api import load_boxes, load_shotgold_state, save_frame_answer, save_shot_events

#: A label stem in the style of the gold clips.
CLIP = "20260101_abc123XYZ_k_Test"
#: One valid shot attempt inside the fixture's first shot.
ATTEMPT = {"release_frame": 200, "rim_frame": 240, "outcome": "made", "points": 2, "shooter": None}


def _frame(frame: int) -> dict:
    return {
        "frame": frame,
        "status": "todo",
        "origin": "todo",
        "proposal": None,
        "ball": None,
        "holder": None,
        "note": "",
    }


def _shot(shot: int, start: int, end: int, held_out: bool) -> dict:
    return {
        "sequence": f"abc123XYZ_s{shot}",
        "youtube_id": "abc123XYZ",
        "clip": CLIP,
        "shot": shot,
        "start": start,
        "end": end,
        "fps": 50.0,
        "held_out": held_out,
        "events_status": "todo",
        "events": [],
        "frames": [_frame(start + 50), _frame(start + 100)],
    }


@pytest.fixture
def labels(tmp_path: Path) -> Path:
    meta = {"home": {"team": "Dziki", "kit": "light"}, "away": {"team": "Legia", "kit": "dark"}}
    (tmp_path / f"{CLIP}.json").write_text(json.dumps({"clip": "data/raw/v.mp4", "fps": 50.0, "meta": meta}))
    return tmp_path


@pytest.fixture
def queue_path(tmp_path: Path) -> Path:
    path = tmp_path / "shot_gold.json"
    save_queue({"version": 1, "shots": [_shot(1, 101, 499, False), _shot(4, 1000, 1400, True)]}, path)
    return path


class TestPickShots:
    @pytest.mark.parametrize(
        ("change", "expected"),
        [
            pytest.param({}, [0, 1, 2, 3], id="all-eligible"),
            pytest.param({"camera": "other"}, [0, 2, 3], id="skips-other-cameras"),
            pytest.param({"end": 700}, [0, 2, 3], id="skips-short-shots"),
        ],
    )
    def test_keeps_only_long_central_shots(self, change: dict, expected: list[int]) -> None:
        shots = [{"start": 500 * i, "end": 500 * i + 399, "camera": "central"} for i in range(4)]
        shots[1] = {**shots[1], **change}
        assert pick_shots(shots, set(), 50.0, 10) == expected

    def test_skips_shots_already_in_gold(self) -> None:
        shots = [{"start": 500 * i, "end": 500 * i + 399, "camera": "central"} for i in range(4)]
        assert pick_shots(shots, {0, 3}, 50.0, 10) == [1, 2]

    def test_spreads_the_picks_over_the_clip(self) -> None:
        shots = [{"start": 500 * i, "end": 500 * i + 399, "camera": "central"} for i in range(12)]
        assert pick_shots(shots, set(), 50.0, 3) == [2, 6, 10]


class TestSampleFrames:
    def test_frames_sit_on_the_grid_away_from_the_cuts(self) -> None:
        frames = sample_frames(1002, 1300, 50.0)
        assert frames == [1030, 1080, 1130, 1180, 1230]


class TestCleanEvent:
    @pytest.mark.parametrize(
        ("change", "message"),
        [
            pytest.param({"rim_frame": 150}, "after rim frame", id="release-after-rim"),
            pytest.param({"rim_frame": 900}, "outside the shot", id="rim-outside-shot"),
            pytest.param({"outcome": "blocked"}, "outcome", id="unknown-outcome"),
            pytest.param({"points": 1}, "points", id="free-throw-points"),
            pytest.param({"shooter": {"xyxy": [0, 0, 5, 5], "team": "ref"}}, "team", id="shooter-without-side"),
            pytest.param({"shooter": {"xyxy": [9, 0, 5, 5], "team": "home"}}, "positive size", id="inverted-box"),
        ],
    )
    def test_rejects_invalid_attempt(self, change: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            clean_event({**ATTEMPT, **change}, _shot(1, 101, 499, False))

    def test_nobody_is_a_valid_person(self) -> None:
        assert clean_person(None) is None


class TestAnswers:
    def test_first_pass_label_records_its_origin(self, queue_path: Path) -> None:
        queue = load_queue(queue_path)
        body = {"sequence": "abc123XYZ_s1", "frame": 151, "ball": [5, 5, 25, 25], "holder": None}
        rec = set_frame_label(queue, body, Status.FIRST_PASS)
        assert (rec["status"], rec["origin"]) == ("first_pass", "first_pass")

    def test_tomasz_answer_keeps_the_first_pass_origin(self, queue_path: Path) -> None:
        queue = load_queue(queue_path)
        body = {"sequence": "abc123XYZ_s1", "frame": 151, "ball": None, "holder": None}
        set_frame_label(queue, body, Status.FLAGGED)
        rec = set_frame_label(queue, body)
        assert (rec["status"], rec["origin"]) == ("done", "flagged")

    def test_attempts_are_stored_in_rim_order(self, queue_path: Path) -> None:
        queue = load_queue(queue_path)
        later = {**ATTEMPT, "release_frame": 400, "rim_frame": 440}
        shot = set_shot_events(queue, {"sequence": "abc123XYZ_s1", "events": [later, ATTEMPT]})
        assert [ev["rim_frame"] for ev in shot["events"]] == [240, 440]

    def test_unknown_frame_is_refused(self, queue_path: Path) -> None:
        with pytest.raises(ValueError, match="no labelled frame"):
            set_frame_label(load_queue(queue_path), {"sequence": "abc123XYZ_s1", "frame": 7})


class TestShotGoldShots:
    @pytest.mark.parametrize(
        ("held_out", "names"),
        [
            pytest.param("exclude", ["abc123XYZ_s1"], id="dev-only"),
            pytest.param("only", ["abc123XYZ_s4"], id="held-out-only"),
            pytest.param("all", ["abc123XYZ_s1", "abc123XYZ_s4"], id="both"),
        ],
    )
    def test_selects_by_split(self, queue_path: Path, labels: Path, held_out: str, names: list[str]) -> None:
        backend = ShotGoldShots({"queue": str(queue_path), "labels": str(labels), "held_out": held_out})
        assert [s.name for s in backend.sequences()] == names

    def test_frames_cover_the_shot_on_the_stride_grid(self, queue_path: Path, labels: Path) -> None:
        seq = ShotGoldShots({"queue": str(queue_path), "labels": str(labels)}).sequences()[0]
        assert (seq.frames[0], seq.frames[-1], seq.video) == (105, 495, "data/raw/v.mp4")

    def test_queue_path_does_not_change_the_hash(self) -> None:
        def cfg(queue: str) -> RunConfig:
            return RunConfig(name="c", layers={"shots": {"backend": "shot_gold", "params": {"queue": queue}}})

        assert cfg("a.json").layer_hash("shots") == cfg("b.json").layer_hash("shots")


class TestShotGoldPage:
    def test_state_gives_each_frame_its_neighbours_inside_the_shot(self, queue_path: Path, labels: Path) -> None:
        shot = load_shotgold_state(queue_path, labels)["shots"][0]
        assert shot["frames"][0]["strip"] == [121, 131, 141, 146, 151, 156, 161, 171, 181]

    def test_state_names_the_sides_and_kits(self, queue_path: Path, labels: Path) -> None:
        shot = load_shotgold_state(queue_path, labels)["shots"][0]
        assert (shot["home"], shot["away"]["kit"]) == ({"team": "Dziki", "kit": "light"}, "dark")

    def test_frame_answer_is_written_to_the_file(self, queue_path: Path) -> None:
        holder = {"xyxy": [10, 10, 60, 160], "team": "away"}
        counts = save_frame_answer(
            {"sequence": "abc123XYZ_s4", "frame": 1050, "ball": None, "holder": holder}, queue_path
        )
        saved = load_queue(queue_path)["shots"][1]["frames"][0]
        assert (counts["frames_done"], saved["holder"]["team"], saved["status"]) == (1, "away", "done")

    def test_shot_answer_is_written_to_the_file(self, queue_path: Path) -> None:
        save_shot_events({"sequence": "abc123XYZ_s1", "events": [ATTEMPT]}, queue_path)
        assert progress(load_queue(queue_path))["shots_done"] == 1

    def test_invalid_answer_leaves_the_file_unchanged(self, queue_path: Path) -> None:
        before = queue_path.read_text()
        bad = {"xyxy": [10, 10, 60, 160], "team": "referee"}
        with pytest.raises(ValueError, match="team"):
            save_frame_answer({"sequence": "abc123XYZ_s4", "frame": 1050, "ball": None, "holder": bad}, queue_path)
        assert queue_path.read_text() == before

    @pytest.mark.parametrize(
        "name", [pytest.param("../x", id="parent"), pytest.param("a/b", id="slash"), pytest.param("", id="empty")]
    )
    def test_box_cache_refuses_names_that_are_not_file_stems(self, tmp_path: Path, name: str) -> None:
        with pytest.raises(ValueError, match="bad sequence name"):
            load_boxes(name, tmp_path)
