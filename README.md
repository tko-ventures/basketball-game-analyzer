<img src="assets/icon.svg" width="64" height="64" alt="">

# basketball-game-analyzer

Offline analysis of Polish top-league (ORLEN Basket Liga) basketball games from the broadcast main camera: player detection and segmentation, team classification by kit, multi-object tracking, jersey-number identification, and 2D court mapping of players and ball.

- Plan and handoff document: [docs/PLAN.md](docs/PLAN.md)
- Agent instructions: [CLAUDE.md](CLAUDE.md)
- Progress: [milestone issues](https://github.com/tko-ventures/basketball-game-analyzer/issues?q=is%3Aissue+label%3Amilestone)

Status (2026-10-29): M0 bootstrap. Each layer is a swappable backend chosen in `configs/*.yaml`, so alternative approaches are compared through the eval harness. Training setups are configs too, so any fine-tune is evaluated against the same gold set. CUDA-only backends such as SAM 2.1 run on Modal.

## Setup

```bash
uv sync                                          # Python 3.12, ruff, pytest
scripts/fetch_highlights.sh                      # the 31 clips in data/clip_ids.txt -> data/raw/ (gitignored)
uv run tools/labeler/prepare_clips.py --jobs 6   # shot cuts and teams -> data/labels/ (skips existing files)
uv run tools/labeler/server.py                   # labeller on http://127.0.0.1:8765
uv run ruff check . && uv run pytest
```

Created in [T3 Code](https://t3.codes).
