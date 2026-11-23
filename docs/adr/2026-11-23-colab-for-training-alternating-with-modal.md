# Google Colab for training, alternating with Modal by credit availability

- **Date:** 2026-11-23
- **Status:** Accepted
- **Area:** Infra
- **Decider:** Tomasz

## Context

The Google AI subscription includes monthly Colab Compute Units (200 CCU) accessible headlessly via `google-colab-cli` (`colab`), while Modal compute counts against the 75 USD/month metered cap. Single-GPU fine-tunes (`bga train`) fit Colab's ephemeral GPU runtimes (`T4`, `L4`, `A100`, `H100`), whereas parallel per-sequence remote inference (`--remote modal`, `--remote-layer`) relies on Modal's multi-container fan-out and persistent `bga-data` volume.

## Decision

1. **Colab + Modal alternation for training:** Support both `colab` and `modal` in `bga train` (via `platform.name: colab|modal` in `configs/train/*.yaml`, or runtime overrides `--platform colab|modal` / `--gpu <GPU>` and env vars `BGA_TRAIN_PLATFORM` / `BGA_TRAIN_GPU`). Prefer Colab for single-GPU fine-tunes while CCUs are available; switch to Modal anytime via CLI flag or env var without modifying configs or changing `weight_id()` hashes.
2. **Modal remains the inference & multi-GPU platform:** Remote layer execution (`--remote modal`, `--remote-layer`) and multi-GPU DDP training (`recipe.devices > 1`) stay on Modal (`bga-runner`, `bga-train`). RunPod and Roboflow Train remain unused.
3. **Single trainer implementation:** Shared trainer logic lives in `bga.train.dispatch` so Colab (`bga.remote.colab_train`) and Modal (`bga.remote.modal_train`) run identical training code.
4. **Spend & session hygiene:** Colab sessions always stop in `finally` and log CCU spend from `colab usage` into `train.json`, `docs/train.md`, and `docs/costs.md`. Modal billing truth (`scripts/modal_spend.py`) and the **75 USD/month metered cap** remain unchanged.

## Consequences

- Existing committed `configs/train/*.yaml` keep their exact `config_hash()` and `weight_id()` values.
- `bga train` uploads code/datasets and pulls checkpoints over chunked Jupyter Contents API transfers, preserving Colab's preinstalled CUDA PyTorch and installing only missing setup packages.
- `bga train --collect` reads local Colab downloads or Modal volume outputs based on `call.json`, and still drops stale weights from Modal's `bga-data` volume so remote inference stays in sync.
