#!/usr/bin/env python3
"""Local ground-truth labeller for basketball tracking.

Run from the repo root:

    python tools/labeler/server.py            # http://127.0.0.1:8765
    python tools/labeler/server.py --port 9000 --clips-dir data/clips --clips-dir data/raw

Stdlib HTTP server + OpenCV. Labels are written to ``data/labels/<clip-stem>.json`` (source of
truth) plus ``<clip-stem>.mot.txt`` (MOTChallenge gt) and ``<clip-stem>.identities.csv`` on every save.

Teams are ``home`` / ``away`` / ``ref`` / ``other``; ``meta.home`` and ``meta.away`` map each side to a
team name, a roster file in ``data/rosters/`` and a kit description, per clip.

"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import mimetypes
import os
import re
import sys
import threading
import time
from collections.abc import Callable
from enum import StrEnum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from bga.ingest.hist_diff import DEFAULT_MIN_LEN_S, DEFAULT_THRESHOLD, detect_shots
from bga.types import DETECTION_CLASSES

#: Repository root; clip, label and roster paths in requests are relative to it.
REPO = Path(__file__).resolve().parents[2]
#: Folder with the labeller's HTML, JS and CSS.
STATIC = Path(__file__).resolve().parent / "static"
#: Video file suffixes listed as clips.
VIDEO_EXT = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
#: Version of the per-clip label JSON written by this server.
LABEL_VERSION = 2
#: Identity teams the labeller knows.
TEAMS = ("home", "away", "ref", "other")
#: MOTChallenge class per identity team; anything else is 4 (other).
MOT_CLASS = {"home": 1, "away": 2, "ref": 3}


# --------------------------------------------------------------------------- #
# Video access
# --------------------------------------------------------------------------- #
class Clip:
    """One VideoCapture per clip, guarded by a lock.

    Sequential reads are used when
    the requested frame is shortly after the last decoded one (much faster than seek).

    """

    def __init__(self, path: Path) -> None:
        """Open the video and read its frame rate, frame count and size.

        Args:
            path: Video file to open.

        Raises:
            RuntimeError: When OpenCV cannot open the file.

        """
        self.path = path
        self.lock = threading.Lock()
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.next_idx = 0  # index of the frame the next cap.read() returns

    def read(self, idx: int) -> np.ndarray | None:
        """Decode one frame, grabbing forward over gaps of up to 60 frames and seeking otherwise.

        Args:
            idx: Zero-based frame index.

        Returns:
            The BGR frame, or ``None`` when the video ends or the frame cannot be decoded.

        """
        with self.lock:
            gap = idx - self.next_idx
            if gap < 0 or gap > 60:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                self.next_idx = idx
                gap = 0
            for _ in range(gap):
                if not self.cap.grab():
                    return None
                self.next_idx += 1
            ok, frame = self.cap.read()
            if not ok:
                return None
            self.next_idx += 1
            return frame


class ClipRegistry:
    """Thread-safe cache of open clips, keyed by repo-relative path."""

    def __init__(self) -> None:
        """Start with no clips open."""
        self._clips: dict[str, Clip] = {}
        self._lock = threading.Lock()

    def get(self, rel: str) -> Clip:
        """Return the open clip for a path, opening it on first use.

        Args:
            rel: Video path relative to the repo root.

        Returns:
            The cached :class:`Clip`.

        Raises:
            PermissionError: When the path climbs out of the repo.
            RuntimeError: When the video cannot be opened.

        """
        with self._lock:
            if rel not in self._clips:
                self._clips[rel] = Clip(safe_path(rel))
            return self._clips[rel]


def safe_path(rel: str) -> Path:
    """Resolve a repo-relative path from a request, refusing paths outside the repo.

    Args:
        rel: Path relative to the repo root.

    Returns:
        The absolute, normalised path.

    Raises:
        PermissionError: When the normalised path is not inside the repo.

    Examples:
        >>> safe_path("data/raw/x.mp4") == REPO / "data/raw/x.mp4"
        True
        >>> safe_path("data/../../outside.mp4")
        Traceback (most recent call last):
        ...
        PermissionError: data/../../outside.mp4

    """
    # Normalise ".." without following symlinks, so a worktree's data/raw link to the main checkout
    # is accepted while a path that climbs out of the repo is not.
    p = Path(os.path.normpath(REPO / rel))
    if REPO not in p.parents and p != REPO:
        raise PermissionError(rel)
    return p


def stem_for(rel: str) -> str:
    """Return the clip stem that names its label files.

    Args:
        rel: Video path.

    Returns:
        The file name without folder and suffix.

    Examples:
        >>> stem_for("data/raw/dziki_legia.mp4")
        'dziki_legia'

    """
    return Path(rel).stem


# --------------------------------------------------------------------------- #
# Shot (camera cut) detection — background job
# --------------------------------------------------------------------------- #
class ShotJobs:
    """Background shot detection per clip, using the layer-0 ``hist_diff`` detector."""

    def __init__(self, labels_dir: Path) -> None:
        """Set up empty progress and result tables.

        Args:
            labels_dir: Label folder of the app.

        """
        self.labels_dir = labels_dir
        self.progress: dict[str, float] = {}
        self.results: dict[str, list] = {}
        self._lock = threading.Lock()

    def start(self, rel: str, threshold: float = DEFAULT_THRESHOLD, min_len_s: float = DEFAULT_MIN_LEN_S) -> None:
        """Start detection on a daemon thread unless a run for the clip is still in progress.

        Progress (0 to 1) appears in :attr:`progress` and the shot list in :attr:`results` when done.

        Args:
            rel: Video path relative to the repo root.
            threshold: Histogram-difference cut threshold.
            min_len_s: Shortest shot in seconds.

        """
        with self._lock:
            if rel in self.progress and self.progress[rel] < 1.0:
                return
            self.progress[rel] = 0.0
        threading.Thread(target=self._run, args=(rel, threshold, min_len_s), daemon=True).start()

    def _run(self, rel: str, threshold: float, min_len_s: float) -> None:
        def progress(p: float) -> None:
            self.progress[rel] = p

        shots = detect_shots(safe_path(rel), threshold, min_len_s, progress)
        self.results[rel] = shots
        self.progress[rel] = 1.0


# --------------------------------------------------------------------------- #
# Label storage
# --------------------------------------------------------------------------- #
def default_labels(rel: str, clip: Clip) -> dict:
    """Build the empty label document for a clip that has no label file yet.

    The stride targets about 10 labelled frames per second, and the whole clip starts as one untagged shot.

    Args:
        rel: Video path relative to the repo root.
        clip: Open clip; only ``fps``, ``frame_count``, ``width`` and ``height`` are read.

    Returns:
        A version-2 label dict.

    Examples:
        >>> from types import SimpleNamespace
        >>> clip = SimpleNamespace(fps=50.0, frame_count=100, width=1920, height=1080)
        >>> labels = default_labels("data/raw/x.mp4", clip)
        >>> labels["stride"], labels["shots"]
        (5, [{'start': 0, 'end': 99, 'camera': 'unset'}])

    """
    return {
        "version": LABEL_VERSION,
        "clip": rel,
        "fps": clip.fps,
        "frame_count": clip.frame_count,
        "width": clip.width,
        "height": clip.height,
        "stride": max(1, round(clip.fps / 10)),  # ~10 labelled frames per second
        "meta": default_meta(),
        "shots": [{"start": 0, "end": max(0, clip.frame_count - 1), "camera": "unset"}],
        "identities": {},
        "frames": {},
    }


def default_meta() -> dict:
    """Build empty clip metadata: season, home and away sides, and notes.

    Returns:
        A fresh dict; the two sides are independent copies.

    Examples:
        >>> default_meta()
        {'season': '', 'home': {'team': '', 'roster': '', 'kit': ''}, 'away': {'team': '', 'roster': '', 'kit': ''},
         'notes': ''}

    """
    side = {"team": "", "roster": "", "kit": ""}
    return {"season": "", "home": dict(side), "away": dict(side), "notes": ""}


#: Canonical team name and roster stem of Dziki Warszawa.
DZIKI = {"team": "Dziki Warszawa", "roster": "dziki_warszawa"}
#: Title fragments dropped before splitting a title into teams: "skrót", dates, scores and the film emoji.
_TITLE_NOISE = r"(?i)skr[oó]t( meczu| spotkania)?|\d{1,2}\.\d{1,2}\.\d{4}|\d+:\d+|🎞️"


def teams_from_title(title: str) -> tuple[str, str]:
    """Home and away team names from a highlight title ("SKRÓT | Dziki - Legia | 24. kolejka").

    PLK titles name the home team first. Dziki is returned as "Dziki Warszawa", other names are cut to 40 characters;
    unparsable titles give ("", "").

    Args:
        title: YouTube title of the highlight.

    Returns:
        ``(home, away)`` team names.

    Examples:
        >>> teams_from_title("SKRÓT | Dziki - Legia | 24. kolejka")
        ('Dziki Warszawa', 'Legia')
        >>> teams_from_title("PGE Start 🆚 Dziki | Skrót meczu")
        ('PGE Start', 'Dziki Warszawa')
        >>> teams_from_title("Season preview")
        ('', '')

    """
    t = re.sub(r"\s+", " ", re.sub(_TITLE_NOISE, " ", title))  # also folds non-breaking spaces
    t = re.sub(r" I ", "|", t)  # a lone capital I stands in for "|" in some titles
    segs = [x for x in t.split("|") if re.search(r"(?i)dzik", x)] or [t]
    parts = [p.strip(" .") for p in re.split(r"\s*[-–]\s*|\s+vs\.?\s+|\s*🆚\s*", segs[0], flags=re.I)]
    parts = [p for p in parts if p]
    if len(parts) != 2:
        return "", ""
    return tuple(DZIKI["team"] if re.search(r"(?i)dzik", p) else p[:40] for p in parts)  # type: ignore[return-value]


def _words(name: str) -> set[str]:
    return set(re.findall(r"\w+", name.casefold()))


def roster_key_for(team: str, rosters_dir: Path | None = None) -> str:
    """Stem of the roster file for a team name as written in a title ("Legia", "Energa Trefl").

    Matches the official name or an alias exactly, else a unique roster whose name across seasons contains every word of
    ``team``. Returns "" when nothing or more than one roster matches.

    Args:
        team: Team name, any case.
        rosters_dir: Folder of roster JSON files; defaults to ``data/rosters`` in the repo.

    Returns:
        The roster file stem, or "".

    Examples:
        >>> roster_key_for("Legia")
        'legia_warszawa'
        >>> roster_key_for("Energa Trefl")
        'energa_trefl_sopot'
        >>> roster_key_for("Warszawa")  # Dziki and Legia both match
        ''

    """
    if not team:
        return ""
    rosters_dir = rosters_dir or REPO / "data/rosters"
    exact, partial = [], []
    for p in sorted(rosters_dir.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        names = {r.get("team", ""), *r.get("aliases", [])}
        names |= {v.get("team_name", "") for v in r.get("seasons", {}).values()}
        names.discard("")
        if any(n.casefold() == team.casefold() for n in names):
            exact.append(p.stem)
        elif any(_words(team) <= _words(n) for n in names):
            partial.append(p.stem)
    hits = exact or partial
    return hits[0] if len(hits) == 1 else ""


def side_meta(team: str, kit: str = "", rosters_dir: Path | None = None) -> dict:
    """Build one side's metadata (team, roster stem, kit), looking up the roster by team name.

    Args:
        team: Team name.
        kit: Free-text kit description.
        rosters_dir: Folder of roster JSON files; defaults to ``data/rosters`` in the repo.

    Returns:
        ``{"team", "roster", "kit"}``.

    Examples:
        >>> side_meta("Legia", "green")
        {'team': 'Legia', 'roster': 'legia_warszawa', 'kit': 'green'}

    """
    return {"team": team, "roster": roster_key_for(team, rosters_dir), "kit": kit}


def migrate_labels(data: dict) -> dict:
    """v1 (team = dziki / opp, meta.opponent, meta.dziki_kit) to v2 (home / away per clip).

    Sides come from ``meta.title`` when it names Dziki, else Dziki is taken as home. Version 2 data is returned
    unchanged; older data is migrated in place.

    Args:
        data: Label dict as read from disk.

    Returns:
        The same dict, at version 2.

    Examples:
        >>> meta = {"opponent": "Legia", "title": "SKRÓT | Legia - Dziki"}
        >>> v1 = {"meta": meta, "identities": {"1": {"team": "dziki"}}}
        >>> v2 = migrate_labels(v1)
        >>> v2["version"], v2["meta"]["home"]["team"], v2["identities"]["1"]["team"]
        (2, 'Legia', 'away')

    """
    if data.get("version", 1) >= 2:
        return data
    meta = data.setdefault("meta", {})
    opponent = meta.pop("opponent", "")
    kit = meta.pop("dziki_kit", "")
    home, away = teams_from_title(meta.get("title", ""))
    if DZIKI["team"] not in (home, away):
        home, away = DZIKI["team"], opponent
    meta["home"] = side_meta(home, kit if home == DZIKI["team"] else "")
    meta["away"] = side_meta(away, kit if away == DZIKI["team"] else "")
    meta.setdefault("season", "")
    meta.setdefault("notes", "")
    dziki_side = "home" if home == DZIKI["team"] else "away"
    side = {"dziki": dziki_side, "opp": "away" if dziki_side == "home" else "home"}
    for ident in data.get("identities", {}).values():
        ident["team"] = side.get(ident.get("team", ""), ident.get("team", ""))
    data["version"] = 2
    return data


def load_labels(labels_dir: Path, rel: str, clip: Clip) -> dict:
    """Read a clip's labels, migrated to version 2, or build defaults when there is no label file.

    Args:
        labels_dir: Folder holding ``<stem>.json``.
        rel: Video path relative to the repo root.
        clip: Open clip, used only for the defaults.

    Returns:
        The label dict.

    Examples:
        >>> import tempfile
        >>> from types import SimpleNamespace
        >>> clip = SimpleNamespace(fps=25.0, frame_count=10, width=1280, height=720)
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     load_labels(Path(tmp), "data/raw/x.mp4", clip)["stride"]
        2

    """
    p = labels_dir / f"{stem_for(rel)}.json"
    if p.exists():
        return migrate_labels(json.loads(p.read_text(encoding="utf-8")))
    return default_labels(rel, clip)


def write_exports(labels_dir: Path, data: dict) -> None:
    """Write the MOTChallenge ground truth and the identity table of a clip.

    ``<stem>.mot.txt`` holds the boxes of done or interpolated frames (1-based frames, class 1 home, 2 away,
    3 referee, 4 other; visibility 0.5 when occluded). ``<stem>.identities.csv`` lists every identity with its
    side's team name.

    Args:
        labels_dir: Output folder.
        data: Version-2 label dict.

    Examples:
        >>> import tempfile
        >>> boxes = [{"tid": 1, "x1": 10, "y1": 20, "x2": 40, "y2": 100}]
        >>> frames = {"0": {"done": True, "boxes": boxes}}
        >>> data = {"clip": "data/raw/x.mp4", "identities": {"1": {"team": "home"}}, "frames": frames}
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     write_exports(Path(tmp), data)
        ...     print((Path(tmp) / "x.mot.txt").read_text().strip())
        1,1,10,20,30,80,1,1,1.0

    """
    stem = stem_for(data["clip"])
    ids = data.get("identities", {})
    meta = data.get("meta", {})
    # MOTChallenge gt: frame(1-based),id,x,y,w,h,conf,class,visibility
    # class: 1 = home player, 2 = away player, 3 = referee, 4 = other
    rows = []
    for fidx, fr in data.get("frames", {}).items():
        if not (fr.get("done") or fr.get("interp")):
            continue
        for b in fr.get("boxes", []):
            ident = ids.get(str(b["tid"]), {})
            x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
            vis = 0.5 if b.get("occluded") else 1.0
            rows.append((
                int(fidx) + 1,
                int(b["tid"]),
                round(x1, 1),
                round(y1, 1),
                round(x2 - x1, 1),
                round(y2 - y1, 1),
                1,
                MOT_CLASS.get(ident.get("team", ""), 4),
                vis,
            ))
    rows.sort()
    with open(labels_dir / f"{stem}.mot.txt", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    with open(labels_dir / f"{stem}.identities.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tid", "team", "number", "name", "team_name", "note"])
        for tid, ident in sorted(ids.items(), key=lambda kv: int(kv[0])):
            team = ident.get("team", "")
            team_name = (meta.get(team) or {}).get("team", "") if team in ("home", "away") else ""
            w.writerow([tid, team, ident.get("number", ""), ident.get("name", ""), team_name, ident.get("note", "")])


def save_labels(labels_dir: Path, data: dict) -> None:
    """Stamp ``saved_at`` and write ``<stem>.json`` atomically, plus the exports once there are labels.

    Args:
        labels_dir: Output folder, created if missing.
        data: Version-2 label dict; ``saved_at`` is set on it.

    Examples:
        >>> import tempfile
        >>> data = {"clip": "data/raw/x.mp4", "identities": {}, "frames": {"0": {"done": True, "boxes": []}}}
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     save_labels(Path(tmp), data)
        ...     sorted(p.name for p in Path(tmp).iterdir())
        ['x.identities.csv', 'x.json', 'x.mot.txt']

    """
    labels_dir.mkdir(parents=True, exist_ok=True)
    stem = stem_for(data["clip"])
    p = labels_dir / f"{stem}.json"
    tmp = p.with_suffix(".json.tmp")
    data["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    if data.get("identities") or data.get("frames"):
        write_exports(labels_dir, data)


def _court_path(labels_dir: Path, stem: str) -> Path:
    if not re.fullmatch(r"[\w.\-]+", stem):
        raise ValueError(f"bad clip stem {stem!r}")
    return labels_dir / "court" / f"{stem}.json"


def court_queue_path(labels_dir: Path, name: str = "") -> Path:
    """queue.json, or queue_<name>.json for a named re-labelling set (/court?set=<name>).

    Args:
        labels_dir: Label folder; queues live in its ``court`` subfolder.
        name: Set name of lower-case letters, digits and underscores, or "" for the main queue.

    Returns:
        Path of the queue file (it may not exist).

    Raises:
        ValueError: When the name has other characters.

    Examples:
        >>> court_queue_path(Path("data/labels")).as_posix()
        'data/labels/court/queue.json'
        >>> court_queue_path(Path("data/labels"), "redo").name
        'queue_redo.json'

    """
    if name and not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError(f"bad queue name {name!r}")
    return labels_dir / "court" / (f"queue_{name}.json" if name else "queue.json")


def load_court_labels(labels_dir: Path, stem: str) -> dict:
    """Court keyframe labels of one clip: {"clip", "frames": {frame: {H, fixed, status}}, "meta"}.

    Args:
        labels_dir: Label folder; court labels live in its ``court`` subfolder.
        stem: Clip stem of word characters, dots and dashes.

    Returns:
        The stored labels, or an empty record when the clip has none.

    Raises:
        ValueError: When the stem has other characters.

    Examples:
        >>> load_court_labels(Path("/nonexistent"), "c1")
        {'clip': 'c1', 'frames': {}, 'meta': {}}

    """
    p = _court_path(labels_dir, stem)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"clip": stem, "frames": {}, "meta": {}}


def save_court_record(labels_dir: Path, body: dict) -> None:
    """Merge one keyframe record and add the reviewer's active seconds to today's total.

    Args:
        labels_dir: Label folder; the record goes to ``court/<clip>.json``, written atomically.
        body: ``{"clip", "frame", "record", "active_seconds"}`` as posted by the court page.

    Raises:
        ValueError: When the record status is not ``ok`` or ``unusable``, or the clip stem is malformed.
        KeyError: When ``clip``, ``frame`` or ``record`` is missing.

    Examples:
        >>> import tempfile
        >>> body = {"clip": "c1", "frame": 120, "record": {"status": "ok"}, "active_seconds": 12.5}
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     save_court_record(Path(tmp), body)
        ...     labels = load_court_labels(Path(tmp), "c1")
        >>> list(labels["frames"]), sum(labels["meta"]["review_seconds"].values())
        (['120'], 12.5)

    """
    stem, frame, rec = body["clip"], str(int(body["frame"])), body["record"]
    if rec.get("status") not in ("ok", "unusable"):
        raise ValueError("status must be ok or unusable")
    data = load_court_labels(labels_dir, stem)
    data["frames"][frame] = {**rec, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    secs = data["meta"].setdefault("review_seconds", {})
    day = time.strftime("%Y-%m-%d")
    secs[day] = round(secs.get(day, 0) + float(body.get("active_seconds") or 0), 1)
    p = _court_path(labels_dir, stem)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)


#: Box gold v1 queue, edited by the /boxes page and written by ``scripts/box_gold_v1.py``.
BOX_QUEUE = "box_gold_v1.json"


class BoxStatus(StrEnum):
    """Status of a box-queue frame; ``skip`` keeps a frame the reviewer could not decide out of every export.

    Examples:
        >>> BoxStatus("skip") is BoxStatus.SKIP, BoxStatus.DONE == "done"
        (True, True)

    """

    TODO = "todo"
    DONE = "done"
    SKIP = "skip"


def box_queue_path(labels_dir: Path, name: str = "") -> Path:
    """The box gold v1 queue, or ``box_queue_<name>.json`` for another queue on the same page (/boxes?set=<name>).

    Args:
        labels_dir: Label folder.
        name: Queue name of lower-case letters, digits and underscores, or "" for box gold v1.

    Returns:
        Path of the queue file (it may not exist).

    Raises:
        ValueError: When the name has other characters.

    Examples:
        >>> box_queue_path(Path("data/labels")).name, box_queue_path(Path("data/labels"), "r1balls").name
        ('box_gold_v1.json', 'box_queue_r1balls.json')

    """
    if name and not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError(f"bad queue name {name!r}")
    return labels_dir / (f"box_queue_{name}.json" if name else BOX_QUEUE)


#: Serialises read-modify-write of the box queue across request threads.
_BOX_LOCK = threading.Lock()


def load_box_queue(labels_dir: Path, name: str = "") -> dict:
    """A box queue: ``{"frames": [...], "classes", "meta"}``, box gold v1 unless ``name`` is given.

    Args:
        labels_dir: Label folder holding the queue.
        name: Queue name (:func:`box_queue_path`).

    Returns:
        The stored queue, or an empty one when the file does not exist.

    Examples:
        >>> load_box_queue(Path("/nonexistent"))
        {'frames': [], 'classes': [], 'meta': {}}

    """
    p = box_queue_path(labels_dir, name)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"frames": [], "classes": [], "meta": {}}


def save_box_frame(labels_dir: Path, body: dict, name: str = "") -> dict:
    """Replace one frame of the box queue and add the reviewer's active seconds to today's total.

    Args:
        labels_dir: Label folder holding the queue, rewritten atomically.
        body: ``{"index", "frame", "active_seconds"}`` as posted by the /boxes page; ``frame`` is the whole record.
        name: Queue name (:func:`box_queue_path`); "" for box gold v1.

    Returns:
        ``{"done": frames marked done, "total": frames}``.

    Raises:
        ValueError: When the record is for another frame than the stored one at ``index``, its status is not
            ``todo``, ``done`` or ``skip``, or a box has a class outside the 10-class schema or a malformed ``xyxy``.
        KeyError: When ``index`` or ``frame`` is missing.

    Examples:
        >>> import tempfile
        >>> rec = {"youtube_id": "y", "frame": 5, "status": "todo", "boxes": []}
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     _ = (Path(tmp) / BOX_QUEUE).write_text(json.dumps({"frames": [rec], "meta": {}}))
        ...     box = {"xyxy": [1, 2, 30, 40], "cls": "ball"}
        ...     done = {**rec, "status": "done", "boxes": [box]}
        ...     res = save_box_frame(Path(tmp), {"index": 0, "frame": done, "active_seconds": 9})
        ...     q = load_box_queue(Path(tmp))
        >>> res, q["frames"][0]["boxes"][0]["cls"], sum(q["meta"]["review_seconds"].values())
        ({'done': 1, 'total': 1}, 'ball', 9.0)

    """
    i, rec = int(body["index"]), body["frame"]
    try:
        status = BoxStatus(rec.get("status"))
    except ValueError:
        raise ValueError("status must be todo, done or skip") from None
    rec = {**rec, "status": status.value}
    for b in rec.get("boxes", []):
        if b.get("cls") not in DETECTION_CLASSES:
            raise ValueError(f"unknown class {b.get('cls')!r}")
        x1, y1, x2, y2 = (float(v) for v in b["xyxy"])
        if not (x2 > x1 and y2 > y1):
            raise ValueError(f"empty box {b['xyxy']}")
    with _BOX_LOCK:
        q = load_box_queue(labels_dir, name)
        old = q["frames"][i]
        if (old["youtube_id"], old["frame"]) != (rec.get("youtube_id"), rec.get("frame")):
            raise ValueError("frame record does not match the queue at this index")
        q["frames"][i] = {**rec, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        secs = q.setdefault("meta", {}).setdefault("review_seconds", {})
        day = time.strftime("%Y-%m-%d")
        secs[day] = round(secs.get(day, 0) + float(body.get("active_seconds") or 0), 1)
        p = box_queue_path(labels_dir, name)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(q, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, p)
    return {"done": sum(f["status"] == BoxStatus.DONE for f in q["frames"]), "total": len(q["frames"])}


def label_progress(labels_dir: Path, rel: str) -> dict:
    """Count the done frames of a clip and the boxes on them.

    Args:
        labels_dir: Folder holding ``<stem>.json``.
        rel: Video path.

    Returns:
        ``{"done", "boxes"}``; zeros when the label file is missing or unreadable.

    Examples:
        >>> label_progress(Path("/nonexistent"), "data/raw/x.mp4")
        {'done': 0, 'boxes': 0}

    """
    p = labels_dir / f"{stem_for(rel)}.json"
    if not p.exists():
        return {"done": 0, "boxes": 0}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"done": 0, "boxes": 0}
    frames = d.get("frames", {}).values()
    return {
        "done": sum(1 for f in frames if f.get("done")),
        "boxes": sum(len(f.get("boxes", [])) for f in frames if f.get("done")),
    }


# --------------------------------------------------------------------------- #
# Pipeline prefill (outputs/**/tracks.json from run_pipeline / demo overlay)
# --------------------------------------------------------------------------- #
def pipeline_boxes(tracks_json: Path) -> dict:
    """Read a pipeline ``tracks.json`` into prefill boxes per frame.

    Tracks without ``xyxy`` or ``bbox`` are dropped, and teams outside :data:`TEAMS` become "".

    Args:
        tracks_json: Pipeline output file.

    Returns:
        ``{"clip_path", "frames": {frame_idx: [{"track_id", "xyxy", "team", "number"}]}}``.

    Examples:
        >>> import tempfile
        >>> track = {"track_id": 7, "xyxy": [1, 2, 3, 4], "team": "home", "jersey_number": "5"}
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     p = Path(tmp) / "tracks.json"
        ...     _ = p.write_text(json.dumps({"clip_path": "x.mp4", "frames": [{"frame_idx": 0, "tracks": [track]}]}))
        ...     pipeline_boxes(p)
        {'clip_path': 'x.mp4', 'frames': {'0': [{'track_id': 7, 'xyxy': [1, 2, 3, 4], 'team': 'home', 'number': '5'}]}}

    """
    d = json.loads(tracks_json.read_text(encoding="utf-8"))
    out: dict[str, list] = {}
    for fr in d.get("frames", []):
        boxes = []
        for t in fr.get("tracks", []):
            xyxy = t.get("xyxy") or t.get("bbox")
            if not xyxy:
                continue
            boxes.append({
                "track_id": t.get("track_id"),
                "xyxy": xyxy,
                "team": t.get("team") if t.get("team") in TEAMS else "",
                "number": t.get("jersey_number"),
            })
        out[str(fr["frame_idx"])] = boxes
    return {"clip_path": d.get("clip_path"), "frames": out}


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class App:
    """State shared by all requests: folders, open clips and shot-detection jobs."""

    def __init__(self, clips_dirs: list[Path], labels_dir: Path, rosters_dir: Path) -> None:
        """Hold the folders; frames are cached as JPEG under ``labels_dir/.cache``.

        Args:
            clips_dirs: Folders searched recursively for videos.
            labels_dir: Folder of label files.
            rosters_dir: Folder of roster JSON files.

        """
        self.clips_dirs = clips_dirs
        self.labels_dir = labels_dir
        self.rosters_dir = rosters_dir
        self.cache_dir = labels_dir / ".cache"
        self.clips = ClipRegistry()
        self.shots = ShotJobs(labels_dir)

    def list_clips(self) -> list[dict]:
        """List the videos in the clip folders, skipping ``.part`` downloads.

        Returns:
            One ``{"path", "size_mb", "done", "boxes"}`` per video, paths relative to the repo.

        """
        out = []
        for d in self.clips_dirs:
            if not d.exists():
                continue
            for p in sorted(d.rglob("*")):
                if p.suffix.lower() in VIDEO_EXT and not p.name.endswith(".part"):
                    rel = str(p.relative_to(REPO))
                    out.append({
                        "path": rel,
                        "size_mb": round(p.stat().st_size / 1e6, 1),
                        **label_progress(self.labels_dir, rel),
                    })
        return out

    def rosters(self) -> dict:
        """Load every roster file.

        Returns:
            Roster JSON keyed by file stem; empty when the folder is missing.

        """
        out = {}
        if self.rosters_dir.exists():
            for p in sorted(self.rosters_dir.glob("*.json")):
                out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        return out

    def pipeline_outputs(self) -> list[str]:
        """Find pipeline ``tracks.json`` files under ``outputs/`` and ``runs/`` for prefill.

        Returns:
            Sorted paths relative to the repo.

        """
        res = []
        for root in ("outputs", "runs"):
            r = REPO / root
            if r.exists():
                res += [str(p.relative_to(REPO)) for p in r.rglob("tracks.json")]
        return sorted(res)

    def frame_jpeg(self, rel: str, idx: int) -> bytes | None:
        """Return one frame as JPEG (quality 92), from the disk cache or decoded and cached.

        Args:
            rel: Video path relative to the repo root.
            idx: Zero-based frame index.

        Returns:
            JPEG bytes, or ``None`` when the frame cannot be decoded or encoded.

        """
        cdir = self.cache_dir / stem_for(rel)
        cpath = cdir / f"{idx:07d}.jpg"
        if cpath.exists():
            return cpath.read_bytes()
        frame = self.clips.get(rel).read(idx)
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            return None
        cdir.mkdir(parents=True, exist_ok=True)
        cpath.write_bytes(buf.tobytes())
        return buf.tobytes()


type _GetRoute = Callable[[LabelerHandler, dict[str, str]], None]
type _PostRoute = Callable[[LabelerHandler, dict, dict[str, str]], None]


class LabelerHandler(BaseHTTPRequestHandler):
    """Request handler of the labeller, court and box pages and their JSON API.

    The :class:`App` is set as the ``app`` class attribute by :func:`make_handler`. Bad query or body values answer 400
    on GET, paths outside the repo 403, unknown routes 404, and other errors 500, all as ``{"error": ...}``.

    """

    server_version = "bga-labeler/2"
    app: App

    def log_message(self, fmt: str, *args: object) -> None:
        """Log requests to stderr, except the frequent frame fetches."""
        if "/api/frame" not in (args[0] if args else ""):
            sys.stderr.write(f"{self.log_date_time_string()} - {fmt % args}\n")

    # helpers
    def _json(self, obj: object, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, data: bytes, ctype: str, cache: bool = False) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _err(self, status: int, msg: str) -> None:
        self._json({"error": msg}, status)

    # GET routes
    def _get_index(self, q: dict[str, str]) -> None:
        self._bytes((STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")

    def _get_static(self, path: str) -> None:
        p = (STATIC / path[len("/static/") :]).resolve()
        if STATIC not in p.parents or not p.exists():
            self._err(404, "not found")
            return
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        self._bytes(p.read_bytes(), ctype)

    def _get_court_page(self, q: dict[str, str]) -> None:
        self._bytes((STATIC / "court.html").read_bytes(), "text/html; charset=utf-8")

    def _get_court_queue(self, q: dict[str, str]) -> None:
        qp = court_queue_path(self.app.labels_dir, q.get("set", ""))
        self._json(json.loads(qp.read_text()) if qp.exists() else {"keyframes": []})

    def _get_court_labels(self, q: dict[str, str]) -> None:
        self._json(load_court_labels(self.app.labels_dir, q["clip"]))

    def _get_boxes_page(self, q: dict[str, str]) -> None:
        self._bytes((STATIC / "boxes.html").read_bytes(), "text/html; charset=utf-8")

    def _get_box_queue(self, q: dict[str, str]) -> None:
        self._json(load_box_queue(self.app.labels_dir, q.get("set", "")))

    def _get_clips(self, q: dict[str, str]) -> None:
        self._json({
            "clips": self.app.list_clips(),
            "rosters": self.app.rosters(),
            "pipeline_outputs": self.app.pipeline_outputs(),
        })

    def _get_clip(self, q: dict[str, str]) -> None:
        rel = q["path"]
        clip = self.app.clips.get(rel)
        self._json({"labels": load_labels(self.app.labels_dir, rel, clip)})

    def _get_frame(self, q: dict[str, str]) -> None:
        data = self.app.frame_jpeg(q["path"], int(q["idx"]))
        if data is None:
            self._err(404, "frame not decodable")
            return
        self._bytes(data, "image/jpeg", cache=True)

    def _get_shots(self, q: dict[str, str]) -> None:
        rel = q["path"]
        self._json({"progress": self.app.shots.progress.get(rel), "shots": self.app.shots.results.get(rel)})

    def _get_pipeline(self, q: dict[str, str]) -> None:
        self._json(pipeline_boxes(safe_path(q["tracks"])))

    _GET_ROUTES: ClassVar[dict[str, _GetRoute]] = {
        "/": _get_index,
        "/index.html": _get_index,
        "/court": _get_court_page,
        "/court/": _get_court_page,
        "/api/court/queue": _get_court_queue,
        "/api/court/labels": _get_court_labels,
        "/boxes": _get_boxes_page,
        "/boxes/": _get_boxes_page,
        "/api/boxes/queue": _get_box_queue,
        "/api/clips": _get_clips,
        "/api/clip": _get_clip,
        "/api/frame": _get_frame,
        "/api/shots": _get_shots,
        "/api/pipeline": _get_pipeline,
    }

    # POST routes
    def _post_labels(self, body: dict, q: dict[str, str]) -> None:
        if not isinstance(body, dict) or "clip" not in body:
            self._err(400, "labels must include clip")
            return
        safe_path(body["clip"])
        save_labels(self.app.labels_dir, body)
        self._json({"ok": True, "saved_at": body.get("saved_at")})

    def _post_court_labels(self, body: dict, q: dict[str, str]) -> None:
        save_court_record(self.app.labels_dir, body)
        self._json({"ok": True})

    def _post_box_frame(self, body: dict, q: dict[str, str]) -> None:
        self._json({"ok": True, **save_box_frame(self.app.labels_dir, body, q.get("set", ""))})

    def _post_shots(self, body: dict, q: dict[str, str]) -> None:
        self.app.shots.start(q["path"], float(body.get("threshold", DEFAULT_THRESHOLD)))
        self._json({"started": True})

    _POST_ROUTES: ClassVar[dict[str, _PostRoute]] = {
        "/api/labels": _post_labels,
        "/api/court/labels": _post_court_labels,
        "/api/boxes/frame": _post_box_frame,
        "/api/shots": _post_shots,
    }

    def do_GET(self) -> None:
        """Serve the pages, static files, frames and the read API."""
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        route = self._GET_ROUTES.get(u.path)
        try:
            if route is not None:
                route(self, q)
            elif u.path.startswith("/static/"):
                self._get_static(u.path)
            else:
                self._err(404, "not found")
        except (KeyError, ValueError) as e:
            self._err(400, f"bad request: {e}")
        except PermissionError:
            self._err(403, "path outside repo")
        except Exception as e:
            self._err(500, repr(e))

    def do_POST(self) -> None:
        """Save clip labels, a court keyframe record or a box gold frame, or start shot detection, from a JSON body."""
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            route = self._POST_ROUTES.get(u.path)
            if route is None:
                self._err(404, "not found")
                return
            route(self, body, q)
        except PermissionError:
            self._err(403, "path outside repo")
        except Exception as e:
            self._err(500, repr(e))


def make_handler(app: App) -> type[LabelerHandler]:
    """Build the request handler class bound to one app.

    Args:
        app: Shared state for every request.

    Returns:
        A :class:`LabelerHandler` subclass named ``Handler`` with ``app`` set.

    Examples:
        >>> handler = make_handler(App([], Path("data/labels"), Path("data/rosters")))
        >>> handler.__name__, issubclass(handler, LabelerHandler), handler.app.cache_dir.as_posix()
        ('Handler', True, 'data/labels/.cache')

    """
    return type("Handler", (LabelerHandler,), {"app": app})


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and serve the labeller until interrupted.

    Args:
        argv: Arguments without the program name; ``None`` reads ``sys.argv``.

    Examples:
        >>> main(["--port", "9000"])  # doctest: +SKIP
        Labeller on http://127.0.0.1:9000  (repo ...)

    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--clips-dir",
        action="append",
        default=None,
        help="Folder(s) with videos, relative to repo (default: data/clips, data/raw)",
    )
    ap.add_argument("--labels-dir", default="data/labels")
    ap.add_argument("--rosters-dir", default="data/rosters")
    a = ap.parse_args(argv)
    clips_dirs = [REPO / d for d in (a.clips_dir or ["data/clips", "data/raw"])]
    app = App(clips_dirs, REPO / a.labels_dir, REPO / a.rosters_dir)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(app))
    print(f"Labeller on http://{a.host}:{a.port}  (repo {REPO})")
    with contextlib.suppress(KeyboardInterrupt):
        srv.serve_forever()


if __name__ == "__main__":
    main()
