"""``bga`` command line (plan section 5): run configs, score them on the gold set, list backends.

Usage:
    bga run --config configs/baseline.yaml [--set track.params.lost_track_buffer_s=2] [--seq OSp445e0mXU_s2]
    bga eval --config configs/baseline.yaml [--config other.yaml] [--gold v0] [--held-out] [--write]
    bga eval-court --config configs/baseline.yaml [--config other.yaml] [--held-out] [--write]
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
from bga.config import load_config

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


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")
    logging.getLogger("bga").setLevel(logging.INFO)


@app.command()
def run(
    config: Annotated[Path, typer.Option(help="configs/<name>.yaml")],
    set_: Annotated[list[str], typer.Option("--set", help="override, e.g. detect.params.confidence=0.3")] = [],  # noqa: B006
    seq: Annotated[list[str], typer.Option(help="only these sequences")] = [],  # noqa: B006
    layer: Annotated[list[str], typer.Option(help="only these layers")] = [],  # noqa: B006
    force: Annotated[list[str], typer.Option(help="recompute these layers")] = [],  # noqa: B006
    remote: Annotated[str | None, typer.Option(help="modal: run CUDA backends on Modal")] = None,
    remote_layer: Annotated[list[str], typer.Option(help="also run these layers remotely")] = [],  # noqa: B006
    verbose: bool = False,
) -> None:
    """Run every layer of a config over its sequences into runs/<clip>/<layer>/<backend>@<hash8>/."""
    _setup_logging(verbose)
    cfg = load_config(config, set_)
    seqs = pipeline.run(
        cfg,
        only=seq or None,
        layers=layer or None,
        force=force or None,
        remote=remote,
        remote_layers=remote_layer,
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


@app.command("eval")
def eval_(
    config: Annotated[list[Path], typer.Option(help="one or more configs, scored side by side")],
    set_: Annotated[list[str], typer.Option("--set", help="override applied to every config")] = [],  # noqa: B006
    gold: str = "v0",
    held_out: Annotated[bool, typer.Option(help="score the held-out game instead (once per milestone)")] = False,
    write: Annotated[bool, typer.Option(help="append rows to docs/metrics.md")] = False,
    note: str = "",
    verbose: bool = False,
) -> None:
    """Run (reusing cached layer outputs) and score each config on the gold set."""
    from bga.eval.scores import score

    _setup_logging(verbose)
    rows = []
    for path in config:
        cfg = load_config(path, set_)
        cfg.layers["shots"].params["held_out"] = "only" if held_out else "exclude"
        cfg.layers["shots"].params["version"] = gold
        seqs = pipeline.run(cfg)
        s = score(cfg, seqs, gold=gold)
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
            f"{_fmt(ms.get('numbers'), 0)} | {note} |"
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


@app.command("eval-detect")
def eval_detect(
    config: Annotated[list[Path], typer.Option(help="one or more run configs; their detect layer is scored")],
    set_: Annotated[list[str], typer.Option("--set", help="override applied to every config")] = [],  # noqa: B006
    gold: str = "v1",
    held_out: Annotated[bool, typer.Option(help="score the held-out game instead (once per milestone)")] = False,
    write: Annotated[bool, typer.Option(help="append rows to docs/metrics.md")] = False,
    note: str = "",
    verbose: bool = False,
) -> None:
    """Run the detect layer (reusing cached outputs) and score per-class AP on the box-gold frames."""
    from bga.eval.detect import PEOPLE
    from bga.eval.detect import score as detect_score

    _setup_logging(verbose)
    rows = []
    for path in config:
        cfg = load_config(path, set_)
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


@app.command("eval-court")
def eval_court(
    config: Annotated[list[Path], typer.Option(help="one or more run configs; their court layer is scored")],
    held_out: Annotated[bool, typer.Option(help="score the held-out game (once per milestone)")] = False,
    write: Annotated[bool, typer.Option(help="append rows to docs/metrics.md")] = False,
    note: str = "",
) -> None:
    """Score each config's court layer on the gold court keyframes (reprojection error per arena)."""
    from bga.eval.court import score as court_score

    rows = []
    for path in config:
        cfg = load_config(path)
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


@app.command()
def radar(
    config: Annotated[Path, typer.Option(help="configs/<name>.yaml")],
    set_: Annotated[list[str], typer.Option("--set")] = [],  # noqa: B006
    seq: Annotated[list[str], typer.Option(help="only these sequences")] = [],  # noqa: B006
    held_out: bool = False,
) -> None:
    """Render the radar video (broadcast above, 2D court below) for each sequence of a run."""
    from bga.export.radar import render

    cfg = load_config(config, set_)
    cfg.layers["shots"].params["held_out"] = "only" if held_out else "exclude"
    for s in pipeline.run(cfg, only=seq or None):
        typer.echo(render(cfg, s))


@app.command()
def train(
    config: Annotated[Path, typer.Option(help="configs/train/<layer>-<backend>-<name>.yaml")],
    build_only: Annotated[bool, typer.Option(help="build and check the dataset, do not train")] = False,
    collect: Annotated[bool, typer.Option(help="only pull a finished run's outputs into weights/")] = False,
    verbose: bool = False,
) -> None:
    """Fine-tune on the config's platform; weights land in weights/<layer>/<backend>/<name>@<hash8>/."""
    from bga.train import run as train_run
    from bga.train.config import load_train_config

    _setup_logging(verbose)
    cfg = load_train_config(config)
    if config.name != f"{cfg.layer}-{cfg.backend}-{cfg.name}.yaml":
        raise typer.BadParameter(f"file name must be {cfg.layer}-{cfg.backend}-{cfg.name}.yaml")
    typer.echo(f"weight id {cfg.weight_id()}")
    if not collect:
        info = train_run.build(cfg)
        typer.echo(json.dumps(info["counts"], indent=1))
        if build_only:
            return
        result = train_run.launch(cfg)
        typer.echo(json.dumps(result, indent=1))
    typer.echo(f"weights in {train_run.collect(cfg)}")


@app.command()
def backends() -> None:
    """List registered backends per layer."""
    for layer, names in registry.available().items():
        typer.echo(f"{layer}: {', '.join(names)}")


if __name__ == "__main__":
    app()
