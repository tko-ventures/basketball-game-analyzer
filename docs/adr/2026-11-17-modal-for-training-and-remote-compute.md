# Modal as sole cloud GPU platform, billing truth, and 75 USD/month cap

- **Date:** 2026-11-17 *(updated 2026-11-19 billing, 2026-11-21 cap)*
- **Status:** Accepted
- **Area:** Infra
- **Decider:** Tomasz

## Context

The local dev machine is an Apple M1 Max with no NVIDIA GPU. Fine-tuning models and running CUDA-only or frame-heavy layers (SAM 2.1, BoT-SORT with re-ID) require cloud GPUs, and early cost logs drifted from actual spend because local polling wait time was mistaken for remote GPU runtime.

## Decision

1. **Modal only:** Every fine-tune (`bga train`) and remote layer execution (`--remote modal`, `--remote-layer`) runs on Modal (`bga-train`, `bga-runner`). RunPod and Roboflow Train are not used.
2. **Placement rule:** Run on the M1 Max (MPS) when local speed is practical; offload CUDA-only or heavy multi-shot layers to Modal. Keep one heavy model resident in local memory at a time.
3. **Billing truth:** Spend figures come strictly from Modal's own billing (`scripts/modal_spend.py` / `modal billing report`), never from local elapsed wait times. Every job appends an estimate row to `docs/costs.md`, reconciled against Modal's summary.
4. **Spend cap:** **75 USD/month metered** (before credits, raised from 50 USD on 2026-11-21). Agents must ask Tomasz before crossing the cap.

## Consequences

- `bga train` spawns detached Modal jobs and polls `call.json`, surviving auth-token expiration and supporting resume from `last.ckpt`.
- Remote inference streams outputs back per sequence into local `runs/` with platform/cost recorded in `meta.json`.
- Modal credentials stay in `~/.modal.toml` (never in git); Roboflow API key lives in Modal secret `roboflow` and `~/.config/bga/roboflow.env`.
