"""Tests for holder gold v1 (``bga.eval.holders``) and the labeller's ``/holders`` page API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bga.eval.holders import BOXES_PATH, build_holders, parse_op, read_op_history, read_ops
from tools.labeler.holders_api import load_holders_state, save_holder_answer

#: A box-gold frame (Energa Trefl - Dziki, shot 3) where box 8 is a player and box gold has no holder.
FLAGGED = ("1jDaJBkUGmc", 1160)
#: Ops lines on real box-gold frames, independent of the decisions in the committed ops file.
OPS_TEXT = (
    "# test ops\nholder 1jDaJBkUGmc 975 9 implementer  # shooter holds\nholder 1jDaJBkUGmc 1160 8 flagged  # held?\n"
)


@pytest.fixture
def ops_copy(tmp_path: Path) -> Path:
    """A small ops file with one implementer line and one open question, so answers never touch the real one."""
    path = tmp_path / "ops.txt"
    path.write_text(OPS_TEXT, encoding="utf-8")
    return path


class TestParseOp:
    """Ops line parsing."""

    @pytest.mark.parametrize(
        "line",
        [
            pytest.param("holder abc 12 4", id="no-status"),
            pytest.param("holder abc 12 4 maybe", id="unknown-status"),
            pytest.param("hold abc 12 4 flagged", id="unknown-verb"),
        ],
    )
    def test_rejects_malformed_line(self, line: str) -> None:
        with pytest.raises(ValueError, match="bad ops line"):
            parse_op(line)


class TestBuildHolders:
    """Holder gold from box gold and the ops file."""

    def test_flagged_frame_keeps_box_gold_and_records_proposal(self, ops_copy: Path, tmp_path: Path) -> None:
        out = tmp_path / "holders.json"
        build_holders(BOXES_PATH, ops_copy, out)
        rows = {(r["youtube_id"], r["frame"]): r for r in json.loads(out.read_text(encoding="utf-8"))["frames"]}
        assert (rows[FLAGGED]["status"], rows[FLAGGED]["holder"], rows[FLAGGED]["proposed"]) == ("flagged", None, 8)

    def test_rejects_holder_that_is_not_a_player(self, tmp_path: Path) -> None:
        ops = tmp_path / "ops.txt"
        frame = next(
            fr
            for fr in json.loads(BOXES_PATH.read_text(encoding="utf-8"))["frames"]
            if (fr["youtube_id"], fr["frame"]) == FLAGGED
        )
        ball = next(i for i, b in enumerate(frame["boxes"]) if b["cls"] == "ball")
        ops.write_text(f"holder 1jDaJBkUGmc 1160 {ball} implementer  # x\n", encoding="utf-8")
        with pytest.raises(ValueError, match="not a player"):
            build_holders(BOXES_PATH, ops, tmp_path / "holders.json")


class TestHoldersPage:
    """The ``/holders`` page API: state and answers."""

    def test_state_lists_flagged_frames_first(self, ops_copy: Path) -> None:
        items = load_holders_state(ops_path=ops_copy)["items"]
        assert [(it["origin"], it["frame"]) for it in items] == [("flagged", 1160), ("implementer", 975)]

    def test_answer_keeps_original_line_and_applies_decision(self, ops_copy: Path, tmp_path: Path) -> None:
        out = tmp_path / "holders.json"
        save_holder_answer(
            {"youtube_id": FLAGGED[0], "frame": FLAGGED[1], "holder": None}, ops_path=ops_copy, out_path=out
        )
        history = read_op_history(ops_copy)[FLAGGED]
        assert [(op.status.value, op.holder) for op in history] == [("flagged", 8), ("tomasz", None)]

    def test_second_answer_replaces_first(self, ops_copy: Path, tmp_path: Path) -> None:
        out = tmp_path / "holders.json"
        body = {"youtube_id": FLAGGED[0], "frame": FLAGGED[1]}
        save_holder_answer({**body, "holder": None}, ops_path=ops_copy, out_path=out)
        counts = save_holder_answer({**body, "holder": 8}, ops_path=ops_copy, out_path=out)
        assert (len(read_op_history(ops_copy)[FLAGGED]), read_ops(ops_copy)[FLAGGED].holder, counts["tomasz"]) == (
            2,
            8,
            1,
        )

    def test_second_answer_keeps_note_written_on_first(self, ops_copy: Path, tmp_path: Path) -> None:
        out = tmp_path / "holders.json"
        body = {"youtube_id": FLAGGED[0], "frame": FLAGGED[1], "holder": 8}
        save_holder_answer(body, ops_path=ops_copy, out_path=out)
        text = ops_copy.read_text(encoding="utf-8").replace("8 tomasz  # held?", "8 tomasz  # Tomasz: a pass")
        ops_copy.write_text(text, encoding="utf-8")
        save_holder_answer(body, ops_path=ops_copy, out_path=out)
        assert read_ops(ops_copy)[FLAGGED].note == "Tomasz: a pass"

    def test_answer_rejects_frame_outside_box_gold(self, ops_copy: Path, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not a box-gold frame"):
            save_holder_answer(
                {"youtube_id": "nope", "frame": 1, "holder": None}, ops_path=ops_copy, out_path=tmp_path / "h.json"
            )
