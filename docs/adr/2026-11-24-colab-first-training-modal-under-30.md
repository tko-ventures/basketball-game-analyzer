# Training on Colab first; Modal only while the month's Modal usage is below 30 USD

- **Date:** 2026-11-24
- **Status:** Accepted
- **Area:** Compute
- **Decider:** Tomasz
- **Supersedes:** [2026-11-23](2026-11-23-colab-for-training-alternating-with-modal.md) (decision 1 only; its
  decisions 2 to 4 stand)

## Context

The 2026-11-23 ADR let `bga train` alternate between Colab and Modal, "preferring Colab while Compute Units are
available". On 2026-11-24 Colab refused an H100 for `plk-r4` (no entitlement on the account), and the implementer
moved the run to a Modal H100 without asking, while Modal's metered usage for November already stood at 53.24 USD.
Tomasz stopped it: "no modal training, you must use colab!", then set the rule: "we prefer colab, modal.com only
if the modal usage is below $30 for the given month".

Verified directly against Colab's `/tun/m/ccu-info` endpoint on 2026-11-24 for Tomasz's account:

- **Eligible GPUs:** `G4` (NVIDIA RTX PRO 6000 Blackwell, 96 GB), `A100` (40/80 GB, `--high-mem`), `L4` (24 GB), `T4`
  (16 GB).
- **Ineligible GPU:** `H100` (do not target `--platform colab --gpu H100`).
- **Eligible TPUs:** `V6E1`, `V5E1` (unused by `bga train`, which requires CUDA GPUs).

## Decision

1. **Colab first.** Every fine-tune runs with `--platform colab`. Target eligible Colab GPUs (`G4`, `A100`, `L4`, `T4`
   — not `H100`); when Colab refuses a GPU, pick another eligible Colab GPU and, if memory needs it, a new training
   config with a smaller batch and gradient accumulation.
2. **Modal for training only under 30 USD.** A training run may go to Modal only while Modal's metered usage for the
   calendar month, read from Modal's billing with `scripts/modal_spend.py` ("Metered Cost"), is below 30 USD, the
   size of the monthly credits. Check it right before launching, and never let an estimate stand in for it.
3. Remote inference (`--remote modal`, `--remote-layer`) and multi-GPU training stay on Modal, under the 75 USD/month
   metered cap, as before.

## Consequences

- Once the month's Modal usage passes 30 USD, training is Colab-only until the next month.
- `--platform` and `--gpu` still override where a run executes without changing its weight id, so a run moved
  between Colab GPUs keeps comparable rows as long as its recipe is unchanged.
- The cancelled Modal start of `plk-r4@14b28a52` (2026-11-24) is logged in `docs/costs.md` from Modal's billing.
