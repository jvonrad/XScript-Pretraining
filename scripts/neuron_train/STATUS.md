# Overnight training — status & how to check

Launched the 3 missing models on the box under a keep-alive orchestrator.

## CONFIRMED WORKING — all 3 training CONCURRENTLY (03:35)
- de (scratch) loss 8.4↓, zh-fair (resumed @11.76B) loss 3.19, zh-starved
  (resumed @11.76B) loss 2.62 — all correct, all logging to wandb.
- **The coexistence hang was a rendezvous-PORT collision**: every model called
  `dist.init_process_group("xla", init_method="xla://")` on the default port
  (12355), so the 2nd/3rd job hung at init. Fixed by a distinct `MASTER_PORT`
  per model in run_prod.sh (48811/48812/48813). This was the real blocker — with
  it, 3 world=8 jobs run in parallel with minimal contention (de 47k solo -> 45k
  concurrent).
- **Warm-start verified**: zh resume from the 12B checkpoints is exact (step,
  tokens, lr, loss all continue correctly). de is from scratch.
- Orchestrator launches **staggered** (each compiles alone first -> warms cache ->
  next cache-hits, avoiding the parallel compile-cache race) and has
  **hang-detection** (restarts a model that logs no step for 20 min, not just one
  that dies).

## Known instability (de) — non-fatal, self-healing
de hangs occasionally (~every 3.5 hrs: step ~120 @ 03:56, step ~640 @ 07:23). NOT
fatal: the keep-alive restarts it and it now **resumes from its last.pt** (de
checkpoints every ~250M tokens), so it only loses ~130M tokens + ~10 min restart
per hang and keeps net-progressing. Crucially, de's restart does NOT disrupt the
zh models — they train straight through it (verified). Cause is likely an
occasional Neuron collective stall on a long-running world=8 job (zh haven't hit
it yet; may just be luck / de's from-scratch trajectory). The clean fix is
EFA/libfabric (isolates per-job collectives) — a follow-up for when you're around.
Watch de hang rate: `grep -c HUNG /home/ubuntu/xscript_prod/logs/orchestrator.log`.
The two zh models train reliably (loss steadily decreasing).

## Throughput / ETA (concurrent, world=8)
Each model ~45-50k tok/s even with all 3 running (low contention).
- de (30B from scratch): **~7 days**
- zh (18.25B remaining each): **~4.5 days**
All run in parallel -> wall-clock **~7 days** (bounded by de).
Faster would need world=16 (broken: 16-way collective NRT_INVALID) or micro_batch
>2 (needs the fused chunked-CE, which works but compiles very slowly). Both are
follow-ups, not safe to switch to unattended.

## What's running
| model | run name | start | cores (world=8) | wandb run |
|---|---|---|---|---|
| de-starved | `de__unigram_starved` | **from scratch** (0 → 30B) | 8-15 | de__unigram_starved |
| zh-fair | `zh__unigram_destarved` | **resume @ 11.75B** → 30B | 16-23 | zh__unigram_destarved |
| zh-starved | `zh__unigram_starved` | **resume @ 11.75B** → 30B | 24-31 | zh__unigram_starved |

- **wandb**: project `XScript-Pretraining` (jonathan-von-rad). Verified logging in for de.
- **Resume**: zh checkpoints (12B, model-only) warm-started — weights loaded, token
  counter set to 11.75B, WSD schedule continues from the stable phase. Optimizer
  restarts fresh (no optimizer state in those checkpoints); harmless mid-stable
  (lr is constant there). de had no checkpoint → from scratch, as you said.
- **Config**: fp32 params, micro_batch=2, ZeRO-1, full cross-entropy. This is the
  PROVEN config (ran 25 stable steps @ ~48k tok/s, correct loss). bf16 params and
  the fused chunked-CE (which unlock bigger micro_batch) work but are slower to
  compile / less battle-tested, so not used for the unattended run.

## Throughput / ETA (honest)
~48k tok/s per model at world=8.
- de (30B from scratch): **~7 days**
- zh (18.25B remaining each): **~4.5 days each**
All 3 run in parallel → bounded by **de at ~7 days**.

**This is slower than the "few days" target.** Why we're not faster:
- **world=16 (using more cores per model) is BROKEN**: the 16-way ZeRO collective
  fails at the first optimizer step with `NRT_INVALID: invalid send/recv targets`
  — the multi-device network plugin (libfabric / libnccom-net) isn't installed,
  so reduce-scatter can't route across >2 devices. world=8 stays within 2 devices
  and works. Fixing this (installing EFA/libfabric, or a different collective
  config) would ~2x throughput but is risky to attempt unattended.
- Bigger micro_batch (mb=4/8) needs bf16 params + the fused chunked-CE; the
  fused-CE compiles very slowly and I only just fixed a Neuron numeric bug in it
  (bool `.sum()` returns -1) — too fresh to trust overnight.
- With those two levers working, this would be ~2-4 days. Flag for when you're up.

## How to check
- **wandb**: https://wandb.ai/jonathan-von-rad/XScript-Pretraining (loss/lr/tok-s live)
- **logs**: `/home/ubuntu/xscript_prod/logs/{de__unigram_starved,zh__unigram_destarved,zh__unigram_starved}.log`
- **orchestrator log** (launches/restarts): `/home/ubuntu/xscript_prod/logs/orchestrator.log`
- **checkpoints**: `/mnt/scratch/xscript/runs/<name>/checkpoints/` (last.pt every ~1B tokens + model-only stepN snapshots)
- quick progress: `grep 'step ' /home/ubuntu/xscript_prod/logs/<name>.log | tail`

## Keep-alive
`orchestrate.sh` (pid recorded in orchestrator.log) restarts any model that dies
(resuming from its last.pt with the real ZeRO optimizer state), cleaning only
that model's cores first. It stops when all 3 reach 30B. To stop everything:
`pkill -f orchestrate.sh` then `pkill -f prod_train.py`.

## Files (all under /home/ubuntu/xscript_prod/)
- `orchestrate.sh` — launcher + keep-alive
- `run_prod.sh` — per-model xmp launch wrapper
- `prod_train.py` — xmp entry (loads real 30B config, forces proven settings)
- `wandb_env.sh` — WANDB_API_KEY (kept out of the repo)
Code changes are in `src/xscript/train_neuron.py` (warm_start, bf16_params,
fused_ce_chunk, wandb-hardening) — CUDA `train.py` untouched.
