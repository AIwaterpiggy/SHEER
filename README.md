# SHEER

Compact research code for missing deep-layer K/V restoration with Native FREE.
Only these two inference/evaluation configurations are supported:

| Configuration | Fixed source layer | Restored blocks (0-based) | Existing execution path |
|---|---:|---|---|
| FREE / T5-large / SAMSum | 6 | 6–23 | Batched Task C2: restore K/V at FREE's flush point |
| FREE / LongT5-base / Multi-News | 3 | 3–11 | Task C1: exact replay, then overwrite restored K/V |

The first missing block uses native RMSNorm and K/V projection; learned maps
apply only to deeper blocks. Both settings use threshold 0.9, fixed thresholds,
batch/beam 1 and no Official CALM. **LongT5 retains exact replay; this example
is not a direct-insertion or inference-speedup implementation.**

This package copies 42 existing Python files without changes. It is not the
complete paper experiment suite, fitting pipeline or performance benchmark.

## Current availability

The server-validation snapshot is on branch `work/compact-free-extraction`.
Clone that branch explicitly; `main` is not the validated release:

```bash
git clone --branch work/compact-free-extraction --single-branch https://github.com/AIwaterpiggy/SHEER.git SHEER-server-check
cd SHEER-server-check
git rev-parse HEAD
```

Local CPU imports and 141 existing synthetic runtime tests pass. The
maintainer-provided server report records **RUN_AND_RESTORATION_PASS** for
both settings at `a1ac0024fd77cce8ad829e33b39d6adf98b47eb2`: two samples each,
with actual restoration and no failures or fallback. See the
[2026-09-16 server validation record](docs/SERVER_VALIDATION_2026-09-16.md).
This is a bounded smoke result, not full paper reproduction or a speed claim.

The original Multi-News restoration artifact is included in `assets/`. The
SAMSum artifact was verified on the server but has not yet been supplied to
this package and has no verified public download link. The server's cached
SAMSum tokenizer matches the archived identity; fresh-environment preparation
is still unverified. Redistribution/metadata review also remains open.
**The two-setting release is not yet ready to advertise as fully runnable.**

## 1. Environment

The existing runner requires a compatible PyTorch CUDA environment and
Transformers 4.28.1. For the maintainer smoke check, reuse the validated server
interpreter without upgrading it; the Linux/CUDA environment is not newly
locked by this package. `requirements.txt` is the unchanged research list,
not a tested fresh GPU-install recipe.

A separate Windows/Python 3.11 **CPU-only** installation recipe and package
snapshot are documented in [VALIDATION.md](VALIDATION.md). Do not use
`requirements-windows-cpu.lock.txt` for the CUDA runs below. NLTK sentence
resources and the `evaluate` ROUGE metric must be available as well.

## 2. Prepare existing assets

See [Asset sources and checksums](docs/ASSETS.md) for:

- public dataset links and the exact FREE distilled checkpoint names;
- checkpoint/artifact SHA-256 inventories and expected directory layout;
- tokenizer identities, the accepted cached SAMSum load, and remaining
  fresh-environment preparation checks.

Do not substitute plain pretrained T5/LongT5, `weighted_ce` checkpoints or
preliminary restoration artifacts. Existing server assets can be used in
place; there is no requirement to download them to a local PC or redistribute
third-party weights in this repository.

## 3. Run a small evaluation

After the assets and server environment have been checked, use the original
entry point from the repository root:

```bash
python run_summarization.py configs/free_samsum.json
python run_summarization.py configs/free_multinews.json
```

Each example requests eight validation samples. JSON mode accepts one config
path, not additional CLI overrides. Use unused output directories: the
inherited evaluator can overwrite files despite `overwrite_output_dir=false`.

For the first check, follow [Server smoke check](docs/SERVER_SMOKE.md). It
prepares an isolated two-sample config using existing asset paths, preserves
all method settings and avoids overwriting prior outputs. No fitting, paper
quality run or performance measurement is involved.

## Outputs and scope

The runner writes `eval_predictions.jsonl`, `eval_results.json` and
`missing_kv_accounting.json` under the
configured output directory. Inspect restoration counters as well as generated
text: a tiny sample can finish without exercising a restoration flush.
A completed smoke check is not evidence of paper-level quality or speedup.
Per-sample accounting is disabled because these examples do not enable the
research provenance pipeline; aggregate restoration accounting remains on.

Development evidence is in [VALIDATION.md](VALIDATION.md); source identities
and the publication checklist are in [PROVENANCE.md](PROVENANCE.md).
Keep the original attributions and resolve the redistribution questions in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before publication. No blanket
license or release tag is assigned here.
