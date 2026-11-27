"""``bga`` command line (plan section 5): run configs, score them on the gold set, list backends.

Usage:
    bga run --config configs/baseline.yaml [--set track.params.lost_track_buffer_s=2] [--seq OSp445e0mXU_s2]
    bga eval --config configs/baseline.yaml [--config other.yaml] [--gold v0] [--held-out] [--write]
    bga eval-court --config configs/baseline.yaml [--config other.yaml] [--held-out] [--write]
    bga eval-detect --config configs/detect-plk-r2.yaml [--set ...] [--gold v1] [--held-out] [--write]
    bga eval-appearance --config configs/appearance-reid-sportsmot.yaml [--gold v0] [--held-out] [--write]
    bga eval-numbers --config configs/numbers-smolvlm-nba.yaml [--held-out] [--write]
    bga eval-positions --config configs/positions-plk-ankles.yaml [--held-out] [--write]
    bga export --config configs/positions-plk-ankles.yaml [--held-out] [--all-splits] [--out all.parquet]
    bga radar --config configs/positions-plk-ankles.yaml [--held-out] [--all-splits]
    bga train --config configs/train/<layer>-<backend>-<name>.yaml [--build-only] [--collect]
    bga backends

"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
from pathlib import Path
from typing import Annotated

import typer

from bga import pipeline, registry
from bga.config import RunConfig, load_config

#: The ``bga`` Typer application, installed as the ``bga`` console script.
app = typer.Typer(add_completion=False, no_args_is_help=True)
#: Metrics file that ``--write`` appends rows to.
METRICS_MD = Path("docs/metrics.md")
#: Header of the gold-set table in ``docs/metrics.md`` written by ``bga eval``.
HEADER = (
    "| date | commit | config | gold | split | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | AP50 | team acc "
    "| number acc (confirmed, wrong) | ms/frame detect, team, court, numbers | notes |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)


#: Where trained weights live, one directory per weight id with its ``train.json``.
WEIGHTS = Path("weights")


def _require_provenance(cfg: object, root: Path = WEIGHTS) -> None:
    """Refuse a config whose layer weights have neither a ``pretrained:`` declaration nor a ``train.json`` (plan 5).

    Args:
        cfg: A run config (anything with ``layers``, each with ``weights``).
        root: Weights directory.

    Raises:
        typer.BadParameter: When a trained weight id has no ``<root>/<id>/train.json``.

    Examples:
        >>> from types import SimpleNamespace as N
        >>> _require_provenance(N(layers={"detect": N(weights="pretrained:roboflow/x/6"), "track": N(weights=None)}))
        >>> _require_provenance(N(layers={"detect": N(weights="detect/rfdetr_torch/x@00000000")}), Path("/nonexistent"))
        Traceback (most recent call last):
        ...
        click.exceptions.BadParameter: detect weights detect/rfdetr_torch/x@00000000 have no train.json ...

    """
    for name, entry in cfg.layers.items():
        w = entry.weights
        if w and not w.startswith("pretrained:") and not (root / w / "train.json").exists():
            raise typer.BadParameter(
                f"{name} weights {w} have no train.json under {root}/ and no pretrained: declaration;"
                " collect them with bga train --collect"
            )


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")
    logging.getLogger("bga").setLevel(logging.INFO)


@app.command(help="Run every layer of a config over its sequences into runs/<clip>/<layer>/<backend>@<hash8>/.")
def run(
    config: Annotated[Path, typer.Option(help="configs/<name>.yaml")],
    set_: Annotated[list[str], typer.Option("--set", help="override, e.g. detect.params.confidence=0.3")] = [],  # noqa: B006  # typer reads list defaults once and never mutates them
    seq: Annotated[list[str], typer.Option(help="only these sequences")] = [],  # noqa: B006  # typer reads list defaults once and never mutates them
    layer: Annotated[list[str], typer.Option(help="only these layers")] = [],  # noqa: B006  # typer reads list defaults once and never mutates them
    force: Annotated[list[str], typer.Option(help="recompute these layers")] = [],  # noqa: B006  # typer reads list defaults once and never mutates them
    remote: Annotated[str | None, typer.Option(help="modal: run CUDA backends on Modal")] = None,
    remote_layer: Annotated[list[str], typer.Option(help="also run these layers remotely")] = [],  # noqa: B006  # typer reads list defaults once and never mutates them
    remote_gpu: Annotated[str, typer.Option(help="Modal GPU for --remote-layer layers; cpu for none")] = "L4",
    verbose: bool = False,
) -> None:
    """Run every layer of a config over its sequences into runs/<clip>/<layer>/<backend>@<hash8>/.

    Args:
        config: Run config.
        set_: ``layer.params.key=value`` overrides.
        seq: Only these sequences.
        layer: Only these layers.
        force: Recompute these layers instead of reusing their cached outputs.
        remote: ``modal`` to run CUDA backends on Modal.
        remote_layer: Also run these layers remotely.
        remote_gpu: Modal GPU type for the ``remote_layer`` layers, or ``cpu``.
        verbose: Log progress.

    Examples:
        >>> run(Path("configs/baseline.yaml"), seq=["OSp445e0mXU_s2"])  # doctest: +SKIP
        baseline@70498c00: 1 sequences

    """
    _setup_logging(verbose)
    cfg = load_config(config, set_)
    seqs = pipeline.run(
        cfg,
        only=seq or None,
        layers=layer or None,
        force=force or None,
        remote=remote,
        remote_layers=remote_layer,
        remote_gpu=None if remote_gpu == "cpu" else remote_gpu,
    )
    typer.echo(f"{cfg.name}@{cfg.config_hash()}: {len(seqs)} sequences")


def _fmt(v: object, nd: int = 3) -> str:
    """Format a metrics cell: ``–`` for missing or NaN, ``nd`` decimals for floats, ``str`` otherwise.

    Examples:
        >>> _fmt(None), _fmt(float("nan")), _fmt(0.12345), _fmt(0.5, 0), _fmt(7)
        ('–', '–', '0.123', '0', '7')

    """
    return (
        "–"
        if v is None or (isinstance(v, float) and math.isnan(v))
        else f"{v:.{nd}f}"
        if isinstance(v, float)
        else str(v)
    )


def _insert_rows(text: str, header: str, rows: list[str], intro: str = "") -> str:
    r"""Insert rows at the end of the table with this header; add the table (after ``intro``) if absent.

    Examples:
        >>> header = "| a |\n|---|\n"
        >>> _insert_rows("# M\n\n| a |\n|---|\n| 1 |\n\nafter\n", header, ["| 2 |"])
        '# M\n\n| a |\n|---|\n| 1 |\n| 2 |\n\nafter\n'

    """
    lines = text.rstrip("\n").split("\n")
    head = header.splitlines()[0]
    if head not in lines:
        return "\n".join(lines) + "\n\n" + intro + header + "\n".join(rows) + "\n"
    i = lines.index(head) + 2
    while i < len(lines) and lines[i].startswith("|"):
        i += 1
    return "\n".join(lines[:i] + rows + lines[i:]) + "\n"


def _remote_note(cfg: RunConfig, seqs: list) -> str:
    """Name the layers whose outputs were made on Modal, with their GPU, for a metrics row's notes.

    Their ms per frame was measured there, not on the Mac.

    Args:
        cfg: The scored config.
        seqs: Its sequences.

    Returns:
        ``"ms on Modal: numbers (L4)"`` style text, or ``""`` when every layer ran on the Mac.

    Examples:
        >>> _remote_note(cfg, seqs)  # doctest: +SKIP
        'ms on Modal: numbers (L4)'

    """
    remote = {}
    for layer in cfg.layers:
        if layer == "shots" or not seqs:
            continue
        path = pipeline.output_path(cfg, Path("runs"), seqs[0], layer)
        rec = pipeline.load_output(cfg, Path("runs"), seqs[0], layer).get("remote") if path.exists() else None
        if rec:
            remote[layer] = rec.get("gpu") or "cpu"
    return ("ms on Modal: " + ", ".join(f"{k} ({v})" for k, v in remote.items())) if remote else ""


@app.command("eval", help="Run (reusing cached layer outputs) and score each config on the gold set.")
def eval_(
    config: Annotated[list[Path], typer.Option(help="one or more configs, scored side by side")],
    set_: Annotated[list[str], typer.Option("--set", help="override applied to every config")] = [],  # noqa: B006  # typer reads list defaults once and never mutates them
    gold: str = "v0",
    held_out: Annotated[bool, typer.Option(help="score the held-out game instead (once per milestone)")] = False,
    write: Annotated[bool, typer.Option(help="append rows to docs/metrics.md")] = False,
    note: str = "",
    remote: Annotated[str | None, typer.Option(help="modal: run CUDA backends on Modal")] = None,
    remote_layer: Annotated[list[str], typer.Option(help="also run these layers remotely")] = [],  # noqa: B006  # typer reads list defaults once and never mutates them
    remote_gpu: Annotated[str, typer.Option(help="Modal GPU for --remote-layer layers; cpu for none")] = "L4",
    verbose: bool = False,
) -> None:
    """Run (reusing cached layer outputs) and score each config on the gold set.

    Args:
        config: Run configs, scored side by side.
        set_: Overrides applied to every config.
        gold: Gold set version.
        held_out: Score the held-out game instead of the dev shots (once per milestone).
        write: Add the rows to the tracking table in ``docs/metrics.md``.
        note: Text for the notes column; layers whose outputs were made on Modal are named there, since their
            ms per frame is not the Mac's.
        remote: ``modal`` to run CUDA backends and ``remote_layer`` layers on Modal.
        remote_layer: Also run these layers remotely.
        remote_gpu: Modal GPU type for the ``remote_layer`` layers, or ``cpu``.
        verbose: Log progress.

    Raises:
        typer.BadParameter: When a layer's trained weights have no ``train.json``.

    Examples:
        >>> eval_([Path("configs/baseline.yaml")])  # doctest: +SKIP

    """
    from bga.eval.scores import score

    _setup_logging(verbose)
    rows = []
    for path in config:
        cfg = load_config(path, set_)
        _require_provenance(cfg)
        cfg.layers["shots"].params["held_out"] = "only" if held_out else "exclude"
        cfg.layers["shots"].params["version"] = gold
        seqs = pipeline.run(
            cfg, remote=remote, remote_layers=remote_layer, remote_gpu=None if remote_gpu == "cpu" else remote_gpu
        )
        s = score(cfg, seqs, gold=gold)
        row_note = "; ".join(x for x in (note, _remote_note(cfg, seqs)) if x)
        ms = s["ms"]
        row = (
            f"| {dt.date.today().isoformat()} | {pipeline.git_commit()} | `{path.stem}`@{cfg.config_hash()} "
            f"| {gold} | {'held-out' if held_out else 'dev'} ({s['sequences']} shots) "
            f"| {_fmt(s['HOTA'])} | {_fmt(s['DetA'])} | {_fmt(s['AssA'])} "
            f"| {_fmt(s['IDF1'])} | {_fmt(s['MOTA'])} "
            f"| {s['IDSW']} | {_fmt(s.get('AP50'))} | {_fmt(s['team_acc'])} ({s['team_tracks']}) "
            f"| {_fmt(s.get('id_acc'))} ({_fmt(s.get('id_confirmed'))}, {s.get('id_wrong', '–')}; "
            f"{s.get('id_tracks', 0)} tracks) "
            f"| {_fmt(ms.get('detect'), 0)}, {_fmt(ms.get('team'), 0)}, {_fmt(ms.get('court'), 0)}, "
            f"{_fmt(ms.get('numbers'), 0)} | {row_note} |"
        )
        rows.append(row)
        typer.echo(
            f"{cfg.name}@{cfg.config_hash()}: HOTA {s['HOTA']:.3f} IDF1 {s['IDF1']:.3f} MOTA {s['MOTA']:.3f} "
            f"AP50 {_fmt(s.get('AP50'))} team {_fmt(s['team_acc'])} ids/seq {s['pred_ids_per_seq']:.1f} "
            f"short-lived {s['churn_short_lived']:.2f} ms {ms}"
        )
    typer.echo(HEADER + "\n".join(rows))
    if write:
        text = METRICS_MD.read_text() if METRICS_MD.exists() else "# Metrics\n\nGold-set scores from `bga eval`.\n\n"
        METRICS_MD.write_text(_insert_rows(text, HEADER, rows))


def _weights_cell(weights: str | None) -> str:
    """Metrics cell naming a layer's weights and, for trained weights, the training config from their train.json.

    Examples:
        >>> _weights_cell("pretrained:roboflow/x/6"), _weights_cell(None)
        ('`pretrained:roboflow/x/6` (pretrained)', '`None` (pretrained)')

    """
    train_cfg = "pretrained"
    tj = Path("weights") / (weights or "") / "train.json"
    if weights and not weights.startswith("pretrained:") and tj.exists():
        c = json.loads(tj.read_text()).get("config")
        if c:
            train_cfg = f"`configs/train/{c['layer']}-{c['backend']}-{c['name']}.yaml`"
    return f"`{weights}` ({train_cfg})"


#: Header of the detection table in ``docs/metrics.md`` written by ``bga eval-detect``.
DETECT_HEADER = (
    "| date | commit | config | detect weights (training config) | gold | split (frames) | people | ball | number "
    "| rim | player | referee | in possession | rare classes | ms/frame | notes |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)
#: Section heading and explanation placed above the detection table when it is first added.
DETECT_INTRO = (
    "## Detection layer\n\n"
    "Scored by `bga eval-detect` on the box-gold frames of gold v1 (`data/gold/v1/boxes.json`: every object of the"
    " 10-class schema boxed and tightened by Tomasz), dev frames unless `--held-out`. Each cell is COCO **AP50:95 /"
    " AP50 (gold boxes)** for one class. **people** merges the six person classes on both sides, so it measures"
    " finding people whatever their action class. **rare classes:** jump shot, layup or dunk, shot block and"
    " ball in basket, each too small for a stable AP on its own. **ms/frame:** the detect layer on the M1 Max.\n\n"
)
#: Rare classes, shown together in one cell.
RARE = {
    "player-jump-shot": "jump",
    "player-layup-dunk": "layup",
    "player-shot-block": "block",
    "ball-in-basket": "in basket",
}


def _ap_cell(c: object) -> str:
    """Metrics cell ``AP50:95 / AP50 (n)`` of one :class:`bga.eval.detect.ClassAP`.

    Examples:
        >>> from bga.eval.detect import ClassAP
        >>> _ap_cell(ClassAP(0.41234, 0.7, 58)), _ap_cell(ClassAP(float("nan"), float("nan"), 0))
        ('0.412 / 0.700 (58)', '– / – (0)')

    """
    return f"{_fmt(c.ap)} / {_fmt(c.ap50)} ({c.n})"


@app.command(
    "eval-detect", help="Run the detect layer (reusing cached outputs) and score per-class AP on the box-gold frames."
)
def eval_detect(
    config: Annotated[list[Path], typer.Option(help="one or more run configs; their detect layer is scored")],
    set_: Annotated[list[str], typer.Option("--set", help="override applied to every config")] = [],  # noqa: B006  # typer reads list defaults once and never mutates them
    gold: str = "v1",
    held_out: Annotated[bool, typer.Option(help="score the held-out game instead (once per milestone)")] = False,
    write: Annotated[bool, typer.Option(help="append rows to docs/metrics.md")] = False,
    note: str = "",
    verbose: bool = False,
) -> None:
    """Run the detect layer (reusing cached outputs) and score per-class AP on the box-gold frames.

    Args:
        config: Run configs; only their shots and detect layers run.
        set_: Overrides applied to every config.
        gold: Gold set version holding ``boxes.json``.
        held_out: Score the held-out game instead of the dev frames (once per milestone).
        write: Add the rows to the detection table in ``docs/metrics.md``.
        note: Text for the notes column.
        verbose: Log progress.

    Raises:
        typer.BadParameter: When a layer's trained weights have no ``train.json``.

    Examples:
        >>> eval_detect([Path("configs/detect-plk-r2.yaml")])  # doctest: +SKIP

    """
    from bga.eval.detect import PEOPLE
    from bga.eval.detect import score as detect_score

    _setup_logging(verbose)
    rows = []
    for path in config:
        cfg = load_config(path, set_)
        _require_provenance(cfg)
        cfg.layers["shots"].params["held_out"] = "only" if held_out else "exclude"
        seqs = pipeline.run(cfg, layers=["shots", "detect"])
        s = detect_score(cfg, seqs, Path("runs"), Path("data/gold") / gold / "boxes.json")
        c = s["classes"]
        rare = "; ".join(f"{short} {_ap_cell(c[k])}" for k, short in RARE.items())
        rows.append(
            f"| {dt.date.today().isoformat()} | {pipeline.git_commit()} | `{cfg.name}`@{cfg.config_hash()} "
            f"| {_weights_cell(cfg.layers['detect'].weights)} | {gold} "
            f"| {'held-out' if held_out else 'dev'} ({s['frames']}) "
            + "".join(f"| {_ap_cell(c[k])} " for k in (PEOPLE, "ball", "number", "rim", "player", "referee"))
            + f"| {_ap_cell(c['player-in-possession'])} | {rare} | {_fmt(s['ms_per_frame'], 0)} | {note} |"
        )
        typer.echo(f"{cfg.name}@{cfg.config_hash()}: " + ", ".join(f"{k} {_ap_cell(v)}" for k, v in c.items()))
    typer.echo(DETECT_HEADER + "\n".join(rows))
    if write:
        METRICS_MD.write_text(_insert_rows(METRICS_MD.read_text(), DETECT_HEADER, rows, DETECT_INTRO))


#: Header of the court table in ``docs/metrics.md`` written by ``bga eval-court``.
COURT_HEADER = (
    "| date | commit | config | court weights (training config) | split | keyframes (unmapped) "
    "| median error m | paint m | paint < 0.5 m | paint m per arena | ms/frame | notes |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)
#: Section heading and explanation placed above the court table when it is first added.
COURT_INTRO = (
    "## Court layer\n\n"
    "Scored by `bga eval-court` on the gold court keyframes (`data/labels/court/queue.json`, purpose gold,"
    " labelled by Tomasz): 6 per gold game, the held-out game only with `--held-out`. Each keyframe is"
    " mapped by the config's court backend, and the court vertices visible under Tomasz's mapping are"
    " sent through the predicted mapping. **median error:** median over keyframes of the per-frame median"
    " distance in metres to their true positions. **paint:** the same over the key and free-throw line"
    " vertices; **paint < 0.5 m:** share of keyframes meeting the plan's provisional target there."
    " Unmapped keyframes count as failures and have no error.\n\n"
)


@app.command(
    "eval-court", help="Score each config's court layer on the gold court keyframes (reprojection error per arena)."
)
def eval_court(
    config: Annotated[list[Path], typer.Option(help="one or more run configs; their court layer is scored")],
    held_out: Annotated[bool, typer.Option(help="score the held-out game (once per milestone)")] = False,
    write: Annotated[bool, typer.Option(help="append rows to docs/metrics.md")] = False,
    note: str = "",
) -> None:
    """Score each config's court layer on the gold court keyframes (reprojection error per arena).

    Args:
        config: Run configs; only their court layers are scored.
        held_out: Score the held-out game (once per milestone).
        write: Add the rows to the court table in ``docs/metrics.md``.
        note: Text for the notes column.

    Raises:
        typer.BadParameter: When a layer's trained weights have no ``train.json``.

    Examples:
        >>> eval_court([Path("configs/court-plk-v1-smooth4.yaml")])  # doctest: +SKIP

    """
    from bga.eval.court import score as court_score

    rows = []
    for path in config:
        cfg = load_config(path)
        _require_provenance(cfg)
        entry = cfg.layers["court"]
        s = court_score(cfg, held_out=held_out)
        a = s["summary"]["all"]
        arenas = "; ".join(f"{h} {_fmt(v['paint_m'], 2)}" for h, v in s["summary"].items() if h != "all")
        rows.append(
            f"| {dt.date.today().isoformat()} | {pipeline.git_commit()} | `{path.stem}`@{cfg.config_hash()} "
            f"| {_weights_cell(entry.weights)} | {'held-out' if held_out else 'dev'} "
            f"| {a['n']} ({a['unmapped']}) "
            f"| {_fmt(a['median_m'], 2)} | {_fmt(a['paint_m'], 2)} | {_fmt(a['paint_under_0_5'], 2)} "
            f"| {arenas} | {s['ms_per_frame']:.0f} | {note} |"
        )
        typer.echo(json.dumps(s["summary"], indent=1))
    typer.echo(COURT_HEADER + "\n".join(rows))
    if write:
        METRICS_MD.write_text(_insert_rows(METRICS_MD.read_text(), COURT_HEADER, rows, COURT_INTRO))


#: Header of the appearance table in ``docs/metrics.md`` written by ``bga eval-appearance``.
APPEARANCE_HEADER = (
    "| date | commit | config | appearance weights (training config) | gold | split (shots, boxes) "
    "| 1 s rank-1 / MRR | 3 s rank-1 / MRR | cross-shot rank-1 / MRR | queries (chance) 1 s, 3 s, cross-shot "
    "| ms/crop | notes |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)
#: Section heading and explanation placed above the appearance table when it is first added.
APPEARANCE_INTRO = (
    "## Appearance layer\n\n"
    "Scored by `bga eval-appearance` on the clean gold boxes of gold v0 (visibility 1: neither interpolated nor"
    " occluded), dev shots unless `--held-out`, so the scores depend on neither the detector nor the tracker. The"
    " config's appearance backend embeds every box, and candidates are ranked by cosine distance."
    " **1 s, 3 s:** one crop of a player, queried every 0.5 s, against the crops of his teammates 1 s or 3 s later in"
    " the same shot. **cross-shot:** the mean embedding of a numbered player in one shot against those of his numbered"
    " teammates in another shot of the same game. **chance:** the mean rank-1 of a random ranking."
    " **ms/crop:** the encoder on the M1 Max, crops of one frame in one batch.\n\n"
)


def _rank_cell(r: object) -> str:
    """Metrics cell ``rank-1 / MRR`` of one :class:`bga.eval.appearance.RankSummary`.

    Examples:
        >>> from bga.eval.appearance import RankSummary
        >>> _rank_cell(RankSummary(74, 3.7, 0.27, 0.8649, 0.92)), _rank_cell(RankSummary(0, 0.0, 0.0, 0.0, 0.0))
        ('0.865 / 0.920', '–')

    """
    return f"{_fmt(r.rank1)} / {_fmt(r.mrr)}" if r.n else "–"


@app.command(
    "eval-appearance",
    help="Score each config's appearance backend on the gold boxes: re-association and cross-shot rank-1 and MRR.",
)
def eval_appearance(
    config: Annotated[list[Path], typer.Option(help="one or more run configs; their appearance layer is scored")],
    gold: str = "v0",
    held_out: Annotated[bool, typer.Option(help="score the held-out game instead (once per milestone)")] = False,
    write: Annotated[bool, typer.Option(help="append rows to docs/metrics.md")] = False,
    note: str = "",
    verbose: bool = False,
) -> None:
    """Score each config's appearance backend on the gold boxes: re-association and cross-shot rank-1 and MRR.

    Args:
        config: Run configs; only their shots and appearance layers are used.
        gold: Gold set version.
        held_out: Score the held-out game instead of the dev shots (once per milestone).
        write: Add the rows to the appearance table in ``docs/metrics.md``.
        note: Text for the notes column.
        verbose: Log progress.

    Raises:
        typer.BadParameter: When a layer's trained weights have no ``train.json``.

    Examples:
        >>> eval_appearance([Path("configs/appearance-reid-sportsmot.yaml")])  # doctest: +SKIP

    """
    from bga.eval.appearance import GAPS_S
    from bga.eval.appearance import score as appearance_score

    _setup_logging(verbose)
    rows = []
    for path in config:
        cfg = load_config(path)
        _require_provenance(cfg)
        cfg.layers["shots"].params["held_out"] = "only" if held_out else "exclude"
        cfg.layers["shots"].params["version"] = gold
        s = appearance_score(cfg, gold=gold)
        re, cs = s["reassoc"], s["cross_shot"]
        counts = ", ".join(f"{r.n} ({_fmt(r.chance)})" for r in (*(re[g] for g in GAPS_S), cs))
        rows.append(
            f"| {dt.date.today().isoformat()} | {pipeline.git_commit()} | `{path.stem}`@{cfg.config_hash()} "
            f"| {_weights_cell(cfg.layers['appearance'].weights)} | {gold} "
            f"| {'held-out' if held_out else 'dev'} ({s['sequences']}, {s['boxes']}) "
            + "".join(f"| {_rank_cell(re[g])} " for g in GAPS_S)
            + f"| {_rank_cell(cs)} | {counts} | {_fmt(s['ms_per_crop'], 1)} | {note} |"
        )
        by_game = "; ".join(f"{k}: {_rank_cell(v)} ({v.n})" for k, v in s["cross_shot_by_game"].items())
        typer.echo(f"{cfg.name}@{cfg.config_hash()}: cross-shot by game: {by_game}")
    typer.echo(APPEARANCE_HEADER + "\n".join(rows))
    if write:
        METRICS_MD.write_text(_insert_rows(METRICS_MD.read_text(), APPEARANCE_HEADER, rows, APPEARANCE_INTRO))


#: Header of the number-reader table in ``docs/metrics.md`` written by ``bga eval-numbers``.
NUMBERS_HEADER = (
    "| date | commit | config | reader weights (training config) | gold | split (shots, crops) "
    "| top-1 roster (legible crops) | top-1 free | answered | refused (crops) | ECE | ms/crop | notes |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)
#: Section heading and explanation placed above the number-reader table when it is first added.
NUMBERS_INTRO = (
    "## Number readers\n\n"
    "Scored by `bga eval-numbers` on the gold number crops (`data/gold/v1/numbers.json`, `docs/gold.md`), dev shots"
    " unless `--held-out`, so the scores depend on neither the detector nor the tracker. The config's numbers backend"
    " reads each crop cut with its own padding. **top-1 roster:** share of the legible crops whose answer,"
    " restricted to the roster of the crop's team, is the right number. **top-1 free:** the same without the roster."
    " **answered:**"
    " share of the legible crops given any number under the roster. **refused:** share of the illegible and"
    " not-a-number crops the reader declines under the roster, with their count. **ECE:** expected calibration error of"
    " the roster answer's probability (10 bins), for readers that report one. **ms/crop:** on the M1 Max.\n\n"
)


@app.command("eval-numbers", help="Score each config's number reader on the gold number crops.")
def eval_numbers(
    config: Annotated[list[Path], typer.Option(help="one or more run configs; their numbers layer is scored")],
    held_out: Annotated[bool, typer.Option(help="score the held-out game instead (once per milestone)")] = False,
    write: Annotated[bool, typer.Option(help="append rows to docs/metrics.md")] = False,
    note: str = "",
    verbose: bool = False,
) -> None:
    """Score each config's number reader on the gold number crops: top-1 under the roster, refusal, calibration.

    Args:
        config: Run configs; only their shots and numbers layers are used.
        held_out: Score the held-out game instead of the dev shots (once per milestone).
        write: Add the rows to the number-reader table in ``docs/metrics.md``.
        note: Text for the notes column.
        verbose: Log progress and list every wrong answer.

    Raises:
        typer.BadParameter: When a layer's trained weights have no ``train.json``.

    Examples:
        >>> eval_numbers([Path("configs/numbers-smolvlm-nba.yaml")])  # doctest: +SKIP

    """
    from bga.eval.numbers import score as numbers_score

    _setup_logging(verbose)
    rows = []
    for path in config:
        cfg = load_config(path)
        _require_provenance(cfg)
        cfg.layers["shots"].params["held_out"] = "only" if held_out else "exclude"
        s = numbers_score(cfg)
        rows.append(
            f"| {dt.date.today().isoformat()} | {pipeline.git_commit()} | `{path.stem}`@{cfg.config_hash()} "
            f"| {_weights_cell(cfg.layers['numbers'].weights)} | v1 "
            f"| {'held-out' if held_out else 'dev'} ({s['sequences']}, {s['crops']}) "
            f"| {_fmt(s['top1_roster'])} ({s['legible']}) | {_fmt(s['top1_free'])} | {_fmt(s['answered'])} "
            f"| {_fmt(s['refused'])} ({s['refuse']}) | {_fmt(s['ece'])} | {_fmt(s['ms_per_crop'], 0)} | {note} |"
        )
        if verbose:
            typer.echo("wrong (id, label, answer): " + ", ".join(f"{i} {lab}→{a}" for i, lab, a in s["wrong"]))
    typer.echo(NUMBERS_HEADER + "\n".join(rows))
    if write:
        METRICS_MD.write_text(_insert_rows(METRICS_MD.read_text(), NUMBERS_HEADER, rows, NUMBERS_INTRO))


#: Header of the 2D positions table in ``docs/metrics.md`` written by ``bga eval-positions``.
POSITIONS_HEADER = (
    "| date | commit | config | positions backend | court backend | split (shots) "
    "| kf median m (pts) | kf p90 m | on-court | cov | speed p95 m/s "
    "| >10 m/s | pose % | airborne | ms/frame | notes |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)
#: Section heading and explanation placed above the 2D positions table when it is first added.
POSITIONS_INTRO = (
    "## 2D positions\n\n"
    "Scored by `bga eval-positions` on the gold shots, dev unless `--held-out`. **kf median / p90 m:** error in"
    " metres on the gold court keyframe inside each shot against Tomasz's labelled homography `H_gt` applied to the"
    " foot point. **on-court:** share of non-referee player positions inside the 28 x 15 m FIBA court (+0.5 m"
    " sideline margin). **cov:** share of tracked person-frames with a finite 2D position. **speed p95 / >10 m/s:**"
    " 95th-percentile frame-to-frame speed and share of unphysical >10 m/s jumps. **pose % / airborne:** ankle"
    " keypoint match share and held-airborne frames for pose-based backends. "
    "**ms/frame:** positions stage on M1 Max.\n\n"
)


@app.command("eval-positions", help="Score each config's 2D positions layer on the gold shots.")
def eval_positions(
    config: Annotated[list[Path], typer.Option(help="one or more run configs with a positions layer")],
    set_: Annotated[list[str], typer.Option("--set", help="override applied to every config")] = [],  # noqa: B006  # typer reads list defaults once and never mutates them
    held_out: Annotated[bool, typer.Option(help="score the held-out game instead (once per milestone)")] = False,
    write: Annotated[bool, typer.Option(help="append rows to docs/metrics.md")] = False,
    note: str = "",
    verbose: bool = False,
) -> None:
    """Score each config's 2D positions layer on gold court keyframes, on-court rate, and trajectory smoothness.

    Args:
        config: Run configs, scored side by side.
        set_: Overrides applied to every config.
        held_out: Score the held-out game instead of dev (once per milestone).
        write: Add the rows to the 2D positions table in ``docs/metrics.md``.
        note: Text for the notes column.
        verbose: Log progress.

    Raises:
        typer.BadParameter: When a layer's trained weights have no ``train.json``.

    Examples:
        >>> eval_positions([Path("configs/positions-plk-ankles.yaml")])  # doctest: +SKIP

    """
    from bga.eval.positions import score as positions_score

    _setup_logging(verbose)
    rows = []
    for path in config:
        cfg = load_config(path, set_)
        _require_provenance(cfg)
        cfg.layers["shots"].params["held_out"] = "only" if held_out else "exclude"
        s = positions_score(cfg, held_out=held_out)
        pos_b = cfg.layers["positions"].backend
        crt_b = cfg.layers["court"].backend if "court" in cfg.layers else "–"
        rows.append(
            f"| {dt.date.today().isoformat()} | {pipeline.git_commit()} | `{cfg.name}`@{cfg.config_hash()} "
            f"| {pos_b} | {crt_b} | {'held-out' if held_out else 'dev'} ({s['sequences']}) "
            f"| {_fmt(s['kf_med_m'])} ({s['kf_points']}) | {_fmt(s['kf_p90_m'])} | {_fmt(s['on_court'])} "
            f"| {_fmt(s['cov'])} | {_fmt(s['speed_p95'], 2)} | {_fmt(s['jump_pct'])} | {_fmt(s['pose_pct'])} "
            f"| {s['airborne_frames']} | {_fmt(s['ms_per_frame'], 1)} | {note} |"
        )
    typer.echo(POSITIONS_HEADER + "\n".join(rows))
    if write:
        METRICS_MD.write_text(_insert_rows(METRICS_MD.read_text(), POSITIONS_HEADER, rows, POSITIONS_INTRO))


#: Header of the ball, possession, and shots table in ``docs/metrics.md`` written by ``bga eval-ball``.
BALL_HEADER = (
    "| date | commit | config | ball backend | detect weights | split (shots, frames) "
    "| ball F1 (R, med px) | poss acc | holder F1 | team acc | shot F1 (P/R, pred/gt) "
    "| outcome acc (made R / missed R) | shooter acc | ms/frame | notes | FT F1 (P/R, pred/gt), outcome acc "
    "| attempt F1, any kind (P/R, pred/gt) |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)
#: Section heading and explanation placed above the ball table when it is first added.
BALL_INTRO = (
    "## Ball, possession, and shots\n\n"
    "Scored by `bga eval-ball` on the gold shots, dev unless `--held-out` (both gold `v1` and exported"
    " `data/gold/v1/shot_gold.json` unless `--no-shot-gold`). **ball F1 (R, med px):** tracked `ball_px`"
    " matched to gold `ball` boxes within `max(25 px, box_side)`. **poss acc / holder F1 / team acc:** per-frame"
    " holder track matched to the gold holder box at IoU >= 0.5 (and exact `None` when unheld): from"
    " `data/gold/v1/holders.json` + `shot_gold.json` (2026-11-20 possession definition) from 2026-11-24,"
    " `player-in-possession` boxes before."
    " **shot F1 / outcome acc / shooter acc:** shot attempts matched to `events.json` + `shot_gold.json` within 0.8 s"
    " on `rim_frame`, made/missed accuracy on matched shots (with the recall of gold-made and gold-missed attempts"
    " from 2026-11-26), and shooter track match against gold `v0` tracks or"
    " shot-gold `shooter_xyxy` boxes. From 2026-11-26 those columns count field goals only, and **FT F1** scores"
    " free throws (1-point gold attempts) on the shot-gold shots whose free throws were labelled. From 2026-11-27"
    " **attempt F1, any kind** scores every attempt on those shots whatever its kind, so a free throw the tag"
    " misses is a found attempt there and a false field goal in the shot columns."
    " **ms/frame:** Layer 7 stage on M1 Max.\n\n"
)


@app.command("eval-ball", help="Score each config's ball, possession, and shot-event layer on the gold shots.")
def eval_ball(
    config: Annotated[list[Path], typer.Option(help="one or more run configs with a ball layer")],
    set_: Annotated[list[str], typer.Option("--set", help="override applied to every config")] = [],  # noqa: B006  # typer reads list defaults once and never mutates them
    held_out: Annotated[bool, typer.Option(help="score the held-out game instead (once per milestone)")] = False,
    shot_gold: Annotated[
        bool,
        typer.Option("--shot-gold/--no-shot-gold", help="include exported shot gold (data/gold/v1/shot_gold.json)"),
    ] = True,
    write: Annotated[bool, typer.Option(help="append rows to docs/metrics.md")] = False,
    note: str = "",
    verbose: bool = False,
) -> None:
    """Score each config's Layer 7 ``ball`` backend on gold ``v1`` and exported shot gold.

    Args:
        config: Run configs, scored side by side.
        set_: Overrides applied to every config.
        held_out: Score the held-out game instead of dev (once per milestone).
        shot_gold: Include exported ``data/gold/v1/shot_gold.json`` alongside gold ``v1``.
        write: Add the rows to the ball, possession, and shots table in ``docs/metrics.md``.
        note: Text for the notes column.
        verbose: Log progress.

    Raises:
        typer.BadParameter: When a layer's trained weights have no ``train.json``.

    Examples:
        >>> eval_ball([Path("configs/ball-plk-fused.yaml")])  # doctest: +SKIP

    """
    from bga.eval.ball import GOLD_V1_SHOT_GOLD
    from bga.eval.ball import score as ball_score

    _setup_logging(verbose)
    rows = []
    sg_path = GOLD_V1_SHOT_GOLD if shot_gold else None
    for path in config:
        cfg = load_config(path, set_)
        _require_provenance(cfg)
        cfg.layers["shots"].params["held_out"] = "only" if held_out else "exclude"
        s = ball_score(cfg, shot_gold_path=sg_path)
        ball_b = cfg.layers["ball"].backend
        det_w = _weights_cell(cfg.layers["detect"].weights)
        shot_cell = (
            f"{_fmt(s['shot_f1'])} ({_fmt(s['shot_p'], 2)}/{_fmt(s['shot_r'], 2)}, {s['shot_pred']}/{s['shot_gt']})"
            if s["shot_gt"]
            else f"– ({s['shot_pred']}/0)"
        )
        ft_cell = (
            f"{_fmt(s['ft_f1'])} ({_fmt(s['ft_p'], 2)}/{_fmt(s['ft_r'], 2)}, {s['ft_pred']}/{s['ft_gt']}), "
            f"{_fmt(s['ft_outcome_acc'])}"
            if s.get("ft_gt")
            else "–"
        )
        att_cell = (
            f"{_fmt(s['att_f1'])} ({_fmt(s['att_p'], 2)}/{_fmt(s['att_r'], 2)}, {s['att_pred']}/{s['att_gt']})"
            if s.get("att_gt")
            else "–"
        )
        rows.append(
            f"| {dt.date.today().isoformat()} | {pipeline.git_commit()} | `{cfg.name}`@{cfg.config_hash()} "
            f"| {ball_b} | {det_w} | {'held-out' if held_out else 'dev'} ({s['sequences']}, {s['box_frames']}) "
            f"| {_fmt(s['ball_f1'])} ({_fmt(s['ball_r'], 2)}, {_fmt(s['ball_med_px'], 1)}) "
            f"| {_fmt(s['poss_acc'])} | {_fmt(s['holder_f1'])} | {_fmt(s['team_acc'])} "
            f"| {shot_cell} | {_fmt(s['shot_outcome_acc'])} "
            f"({_fmt(s['shot_made_r'], 2)}/{_fmt(s['shot_missed_r'], 2)}) | {_fmt(s['shot_shooter_acc'])} "
            f"| {_fmt(s['ms_per_frame'], 2)} | {note} | {ft_cell} | {att_cell} |"
        )
    typer.echo(BALL_HEADER + "\n".join(rows))
    if write:
        METRICS_MD.write_text(_insert_rows(METRICS_MD.read_text(), BALL_HEADER, rows, BALL_INTRO))


#: Header of the Layer 0 camera table in ``docs/metrics.md`` written by ``bga eval-ingest``.
INGEST_HEADER = (
    "| date | commit | config | camera backend | samples | split (games, shots) | camera acc | central P | central R "
    "| acc per game | ms/shot | notes |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)
#: Section heading and explanation placed above the Layer 0 camera table when it is first added.
INGEST_INTRO = (
    "## Ingest (Layer 0)\n\n"
    "Scored by `bga eval-ingest` on the gold games of `data/labels/gold.json`, dev unless `--held-out`. **camera"
    " acc:** share of label-file shots tagged `central` or `other` (shots of at least 6 s) that the `shots/video`"
    " config's camera backend tags the same, on the label file's own shot bounds. **central P / R:** precision and"
    " recall of `central`. **ms/shot:** the classifier on M1 Max, frame reading excluded.\n\n"
)


@app.command("eval-ingest", help="Score each shots/video config's camera classifier on the gold games' camera tags.")
def eval_ingest(
    config: Annotated[list[Path], typer.Option(help="one or more run configs whose shots backend is video")],
    set_: Annotated[list[str], typer.Option("--set", help="override applied to every config")] = [],  # noqa: B006  # typer reads list defaults once and never mutates them
    held_out: Annotated[bool, typer.Option(help="score the held-out games instead (once per milestone)")] = False,
    write: Annotated[bool, typer.Option(help="append rows to docs/metrics.md")] = False,
    note: str = "",
    verbose: bool = False,
) -> None:
    """Score each config's Layer 0 camera classifier on the camera tags of the gold games.

    Args:
        config: Run configs with ``shots.backend: video``, scored side by side.
        set_: Overrides applied to every config.
        held_out: Score the held-out games instead of dev (once per milestone).
        write: Add the rows to the ingest table in ``docs/metrics.md``.
        note: Text for the notes column.
        verbose: Log progress.

    Raises:
        typer.BadParameter: When a config's shots backend is not ``video``.

    Examples:
        >>> eval_ingest([Path("configs/ingest-siglip-zeroshot.yaml")])  # doctest: +SKIP

    """
    from bga.eval.ingest import score_camera
    from bga.ingest.video import VideoShots

    _setup_logging(verbose)
    rows = []
    for path in config:
        cfg = load_config(path, set_)
        if cfg.layers["shots"].backend != "video":
            raise typer.BadParameter(f"{path}: shots backend is {cfg.layers['shots'].backend!r}, not 'video'")
        p = VideoShots(cfg.layers["shots"].params).p
        s = score_camera(p["camera"], p["samples"], held_out=held_out)
        games = ", ".join(f"{k} {v:.3f}" for k, v in s.per_game.items())
        rows.append(
            f"| {dt.date.today().isoformat()} | {pipeline.git_commit()} | `{cfg.name}`@{cfg.config_hash()} "
            f"| {p['camera']['backend']} | {p['samples']} | {'held-out' if held_out else 'dev'} ({s.games}, {s.shots}) "
            f"| {_fmt(s.acc)} | {_fmt(s.central_p)} | {_fmt(s.central_r)} | {games} | {_fmt(s.ms_per_shot, 0)} "
            f"| {note} |"
        )
    typer.echo(INGEST_HEADER + "\n".join(rows))
    if write:
        METRICS_MD.write_text(_insert_rows(METRICS_MD.read_text(), INGEST_HEADER, rows, INGEST_INTRO))


def _shots_split(held_out: bool, all_splits: bool) -> str:
    """Resolve ``shots.params.held_out`` from ``--held-out`` and ``--all-splits`` flags."""
    if all_splits:
        return "all"
    return "only" if held_out else "exclude"


@app.command("export", help="Export per-frame player parquet tables (plan section 5) for each sequence of a run.")
def export_(
    config: Annotated[Path, typer.Option(help="configs/<name>.yaml")],
    set_: Annotated[list[str], typer.Option("--set")] = [],  # noqa: B006  # typer reads list defaults once and never mutates them
    seq: Annotated[list[str], typer.Option(help="only these sequences")] = [],  # noqa: B006  # typer reads list defaults once and never mutates them
    held_out: Annotated[bool, typer.Option(help="export the held-out game's sequences")] = False,
    all_splits: Annotated[bool, typer.Option(help="export both dev and held-out gold sequences")] = False,
    out: Annotated[
        Path | None, typer.Option(help="optional combined parquet file across all selected sequences")
    ] = None,
    verbose: bool = False,
) -> None:
    """Export per-frame player parquet files for every sequence of a run config.

    Args:
        config: Run config.
        set_: ``layer.params.key=value`` overrides.
        seq: Only these sequences.
        held_out: Export only the held-out game's sequences.
        all_splits: Export both dev and held-out gold sequences.
        out: Optional path for a single combined parquet file across all exported sequences.
        verbose: Log progress.

    Examples:
        >>> export_(Path("configs/positions-plk-ankles.yaml"), seq=["OSp445e0mXU_s2"])  # doctest: +SKIP

    """
    import pyarrow.parquet as pq

    from bga.export.parquet import export_ball_sequence, export_sequence, rows_to_table, sequence_rows

    _setup_logging(verbose)
    cfg = load_config(config, set_)
    _require_provenance(cfg)
    cfg.layers["shots"].params["held_out"] = _shots_split(held_out, all_splits)
    seqs = pipeline.run(cfg, only=seq or None)
    all_rows = []
    for s in seqs:
        path = export_sequence(cfg, s)
        typer.echo(path)
        if "ball" in cfg.layers:
            typer.echo(export_ball_sequence(cfg, s))
        if out is not None:
            all_rows.extend(sequence_rows(cfg, s))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(rows_to_table(all_rows), out)
        typer.echo(f"combined ({len(all_rows)} rows): {out}")


@app.command(help="Render the radar video (broadcast above, 2D court below) for each sequence of a run.")
def radar(
    config: Annotated[Path, typer.Option(help="configs/<name>.yaml")],
    set_: Annotated[list[str], typer.Option("--set")] = [],  # noqa: B006  # typer reads list defaults once and never mutates them
    seq: Annotated[list[str], typer.Option(help="only these sequences")] = [],  # noqa: B006  # typer reads list defaults once and never mutates them
    held_out: Annotated[bool, typer.Option(help="render the held-out game's sequences")] = False,
    all_splits: Annotated[bool, typer.Option(help="render both dev and held-out gold sequences")] = False,
) -> None:
    """Render the radar video (broadcast above, 2D court below) for each sequence of a run.

    Args:
        config: Run config.
        set_: ``layer.params.key=value`` overrides.
        seq: Only these sequences.
        held_out: Render only the held-out game's sequences.
        all_splits: Render both dev and held-out gold sequences.

    Examples:
        >>> radar(Path("configs/court-plk-v1-smooth4.yaml"), seq=["OSp445e0mXU_s2"])  # doctest: +SKIP

    """
    from bga.export.radar import render

    cfg = load_config(config, set_)
    _require_provenance(cfg)
    cfg.layers["shots"].params["held_out"] = _shots_split(held_out, all_splits)
    for s in pipeline.run(cfg, only=seq or None):
        typer.echo(render(cfg, s))


@app.command(help="Fine-tune on Colab or Modal; weights land in weights/<layer>/<backend>/<name>@<hash8>/.")
def train(
    config: Annotated[Path, typer.Option(help="configs/train/<layer>-<backend>-<name>.yaml")],
    platform: Annotated[
        str | None,
        typer.Option(
            envvar="BGA_TRAIN_PLATFORM",
            help="override training platform (colab or modal) without changing the weight id",
        ),
    ] = None,
    gpu: Annotated[
        str | None,
        typer.Option(
            envvar="BGA_TRAIN_GPU",
            help="override training GPU (e.g. T4, L4, A100, L40S) without changing the weight id",
        ),
    ] = None,
    build_only: Annotated[bool, typer.Option(help="build and check the dataset, do not train")] = False,
    collect: Annotated[bool, typer.Option(help="only pull a finished run's outputs into weights/")] = False,
    wait: Annotated[bool, typer.Option(help="wait for the run already spawned (call.json), then collect")] = False,
    partial: Annotated[
        bool, typer.Option(help="with --collect: take a cut-short run's best checkpoint so far")
    ] = False,
    resume: Annotated[
        bool, typer.Option(help="Modal only: continue a run cut short from its last.ckpt on the volume")
    ] = True,
    verbose: bool = False,
) -> None:
    """Fine-tune on Colab or Modal; weights land in weights/<layer>/<backend>/<name>@<hash8>/.

    Args:
        config: Training config, named ``<layer>-<backend>-<name>.yaml``.
        platform: Optional platform override (``colab`` or ``modal``); defaults to ``BGA_TRAIN_PLATFORM`` or config.
        gpu: Optional GPU override (e.g. ``L4``, ``A100``, ``L40S``); defaults to ``BGA_TRAIN_GPU`` or config.
        build_only: Build and check the dataset, then stop.
        collect: Only pull a finished run's outputs into ``weights/``.
        wait: Wait for the run already spawned (``call.json``), then collect.
        partial: With ``collect``, take a cut-short run's best checkpoint so far.
        resume: Modal only: continue a run cut short from its ``last.ckpt`` on the volume.
        verbose: Log progress.

    Raises:
        typer.BadParameter: When the file name does not match the config's layer, backend and name, or platform/GPU
            is invalid.

    Examples:
        >>> train(Path("configs/train/detect-rfdetr_torch-plk-r2.yaml"), build_only=True)  # doctest: +SKIP

    """
    from bga.train import run as train_run
    from bga.train.config import load_train_config

    _setup_logging(verbose)
    cfg = load_train_config(config)
    if config.name != f"{cfg.layer}-{cfg.backend}-{cfg.name}.yaml":
        raise typer.BadParameter(f"file name must be {cfg.layer}-{cfg.backend}-{cfg.name}.yaml")
    typer.echo(f"weight id {cfg.weight_id()}")
    if wait:
        typer.echo(json.dumps(train_run.wait(cfg), indent=1))
    elif not collect:
        try:
            resolved_platform, resolved_gpu = train_run.resolve_platform(cfg, platform=platform, gpu=gpu)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(f"platform {resolved_platform} ({resolved_gpu})")
        info = train_run.build(cfg)
        typer.echo(json.dumps(info["counts"], indent=1))
        if build_only:
            return
        result = train_run.launch(cfg, resume=resume, platform=resolved_platform, gpu=resolved_gpu)
        typer.echo(json.dumps(result, indent=1))
    typer.echo(f"weights in {train_run.collect(cfg, partial=partial)}")


@app.command(help="List registered backends per layer.")
def backends() -> None:
    """List registered backends per layer.

    Examples:
        >>> backends()  # doctest: +SKIP
        shots: gold

    """
    for layer, names in registry.available().items():
        typer.echo(f"{layer}: {', '.join(names)}")


if __name__ == "__main__":
    app()
