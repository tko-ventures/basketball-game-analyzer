"""``bga`` command line (plan section 5): run configs, score them on the gold set, list backends.

bga run  --config configs/baseline.yaml [--set track.params.lost_track_buffer_s=2] [--seq OSp445e0mXU_s2]
bga eval --config configs/baseline.yaml [--config other.yaml] [--gold v0] [--held-out] [--write]
bga backends
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from pathlib import Path
from typing import Annotated

import typer

from bga import pipeline, registry
from bga.config import load_config

app = typer.Typer(add_completion=False, no_args_is_help=True)
METRICS_MD = Path("docs/metrics.md")
HEADER = (
    "| date | commit | config | gold | split | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | AP50 | team acc "
    "| ms/frame detect, team, court | notes |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")
    logging.getLogger("bga").setLevel(logging.INFO)


@app.command()
def run(
    config: Annotated[Path, typer.Option(help="configs/<name>.yaml")],
    set_: Annotated[
        list[str], typer.Option("--set", help="override, e.g. detect.params.confidence=0.3")
    ] = [],  # noqa: B006
    seq: Annotated[list[str], typer.Option(help="only these sequences")] = [],  # noqa: B006
    layer: Annotated[list[str], typer.Option(help="only these layers")] = [],  # noqa: B006
    force: Annotated[list[str], typer.Option(help="recompute these layers")] = [],  # noqa: B006
    verbose: bool = False,
) -> None:
    """Run every layer of a config over its sequences into runs/<clip>/<layer>/<backend>@<hash8>/."""
    _setup_logging(verbose)
    cfg = load_config(config, set_)
    seqs = pipeline.run(cfg, only=seq or None, layers=layer or None, force=force or None)
    typer.echo(f"{cfg.name}@{cfg.config_hash()}: {len(seqs)} sequences")


def _fmt(v, nd=3) -> str:
    return (
        "–"
        if v is None or (isinstance(v, float) and math.isnan(v))
        else f"{v:.{nd}f}"
        if isinstance(v, float)
        else str(v)
    )


@app.command("eval")
def eval_(
    config: Annotated[list[Path], typer.Option(help="one or more configs, scored side by side")],
    set_: Annotated[list[str], typer.Option("--set", help="override applied to every config")] = [],  # noqa: B006
    gold: str = "v0",
    held_out: Annotated[
        bool, typer.Option(help="score the held-out game instead (once per milestone)")
    ] = False,
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
            f"| {_fmt(ms.get('detect'), 0)}, {_fmt(ms.get('team'), 0)}, {_fmt(ms.get('court'), 0)} | {note} |"
        )
        rows.append(row)
        typer.echo(
            f"{cfg.name}@{cfg.config_hash()}: HOTA {s['HOTA']:.3f} IDF1 {s['IDF1']:.3f} MOTA {s['MOTA']:.3f} "
            f"AP50 {_fmt(s.get('AP50'))} team {_fmt(s['team_acc'])} ids/seq {s['pred_ids_per_seq']:.1f} "
            f"short-lived {s['churn_short_lived']:.2f} ms {ms}"
        )
    typer.echo(HEADER + "\n".join(rows))
    if write:
        text = (
            METRICS_MD.read_text()
            if METRICS_MD.exists()
            else "# Metrics\n\nGold-set scores from `bga eval`.\n\n"
        )
        if HEADER.splitlines()[0] not in text:
            text += "\n" + HEADER
        METRICS_MD.write_text(text.rstrip("\n") + "\n" + "\n".join(rows) + "\n")


@app.command()
def backends() -> None:
    """List registered backends per layer."""
    for layer, names in registry.available().items():
        typer.echo(f"{layer}: {', '.join(names)}")


if __name__ == "__main__":
    app()
