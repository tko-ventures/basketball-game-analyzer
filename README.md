<img src="assets/icon.svg" width="64" height="64" alt="">

# basketball-game-analyzer

Offline analysis of Polish top-league (ORLEN Basket Liga) basketball games from the broadcast main camera: player detection and segmentation, team classification by kit, multi-object tracking, jersey-number identification, and 2D court mapping of players and ball.

- What is built: [docs/architecture.md](docs/architecture.md)
- Plan: [docs/PLAN.md](docs/PLAN.md); decisions: [docs/decisions.md](docs/decisions.md); scores: [docs/metrics.md](docs/metrics.md)
- Agent instructions: [CLAUDE.md](CLAUDE.md)
- Progress: [milestone issues](https://github.com/tko-ventures/basketball-game-analyzer/issues?q=is%3Aissue+label%3Amilestone)

Status (2026-11-17): M0 to M2 and M3 part 1 are on `main`: repository, gold set v0, the baseline pipeline with its first metrics rows, and all 42 court keyframes. The M3 court model (0.2 m error near the paint, against 0.9 m for the NBA model) is in review on `feat/m3-court-model`.

## Setup

```bash
uv sync --all-extras                             # Python 3.12, ruff, pytest, all optional backends
scripts/fetch_highlights.sh                      # the 31 clips in data/clip_ids.txt -> data/raw/ (gitignored)
uv run tools/labeler/prepare_clips.py --jobs 6   # shot cuts and teams -> data/labels/ (skips existing files)
uv run tools/labeler/server.py                   # labeller on http://127.0.0.1:8765 (court keyframes at /court)
uv run bga run --config configs/baseline.yaml    # the pipeline on the gold shots -> runs/ (needs ROBOFLOW_API_KEY)
uv run bga eval --config configs/baseline.yaml   # scores against gold v0
uv run ruff check . && uv run pytest
```

Created in [T3 Code](https://t3.codes).
