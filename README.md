<img src="assets/icon.svg" width="64" height="64" alt="">

# basketball-game-analyzer

Offline analysis of Polish top-league (ORLEN Basket Liga) basketball games from the broadcast main camera: player detection and segmentation, team classification by kit, multi-object tracking, jersey-number identification, and 2D court mapping of players and ball.

- Architecture (what is built): [docs/architecture.md](docs/architecture.md)
- Architectural decisions (ADRs): [docs/adr/README.md](docs/adr/README.md)
- Plan & milestones: [docs/PLAN.md](docs/PLAN.md); scores: [docs/metrics.md](docs/metrics.md)
- Agent instructions: [AGENTS.md](AGENTS.md) ([CLAUDE.md](CLAUDE.md) points here)
- Progress: [milestone issues](https://github.com/tko-ventures/basketball-game-analyzer/issues?q=is%3Aissue+label%3Amilestone)

Status (2026-11-23): M0–M6 are built (gold sets, PLK court model, PLK detector with possession, 5 fps team classifier, BoT-SORT with PLK re-ID + greedy linking). M7 (PLK jersey-number reading with PARSeq and roster scoring) is in progress.

## Setup

```bash
uv sync --all-extras                             # Python 3.12, ruff, pytest, all optional backends
scripts/fetch_highlights.sh                      # download clips in data/clip_ids.txt -> data/raw/ (gitignored)
uv run tools/labeler/prepare_clips.py --jobs 6   # shot cuts and teams -> data/labels/ (skips existing files)
uv run tools/labeler/server.py                   # labeller on http://127.0.0.1:8765 (/court, /boxes, /numbers)
uv run bga run --config configs/baseline.yaml    # run pipeline on gold shots -> runs/ (needs ROBOFLOW_API_KEY)
uv run bga eval --config configs/baseline.yaml   # score against gold v0
uv run pre-commit run --all-files && uv run pytest
```

Created in [T3 Code](https://t3.codes).
