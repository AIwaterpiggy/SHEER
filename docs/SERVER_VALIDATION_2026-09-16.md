# Reported server smoke validation: 2026-09-16

Both supported settings are **RUN_AND_RESTORATION_PASS**: two samples each,
with real model generation and observed K/V restoration. No extra eight-sample
run was needed. This is not a benchmark or a full paper reproduction.

## Evidence and tested scope

This is a sanitized transcription of the maintainer-provided Korean server
report, created at `2026-09-16T08:44:03.482189+00:00`. The documentation update
did not connect to the server, independently inspect its raw evidence files,
or rerun generation. Personal absolute paths, predictions, dataset text and
raw logs are not published here. The report and supporting evidence remain
in the server checkout's ignored `outputs/` directories.

- Tested SHEER commit: `a1ac0024fd77cce8ad829e33b39d6adf98b47eb2`.
- Branch: `work/compact-free-extraction`.
- Entry point: unchanged `run_summarization.py`, settings run sequentially.
- Existing research virtual environment reused through its explicit Python
  path; the virtual environment prefix was preserved.
- All 43 `SOURCE_MANIFEST.json` files matched size/SHA-256 before and after.
- SHEER tracked/untracked status was clean before and after. All new run
  outputs were isolated under ignored, previously unused directories.
- The original research checkout's HEAD/status were unchanged. Its newer
  HEAD was not required to equal the extraction's source commit.

## Results

| Setting | Status | Requested / evaluated / predictions | Process exit | Effective generated lengths |
|---|---|---|---|---|
| SAMSum / T5-large / source 6 | RUN_AND_RESTORATION_PASS | 2 / 2 / 2 | 0 | 37, 30 |
| Multi-News / LongT5-base / source 3 | RUN_AND_RESTORATION_PASS | 2 / 2 / 2 | 0 | 177, 203 |

### SAMSum: Batched Task C2

- Fixed source 6; restored targets 6..23, learned targets 7..23.
- Policy: `phase3c_fixed_source6_batched_lazy_insertion`.
- Aggregate validation and `task_c2_batched_validation`: `ok`.
- Batched flush attempts/successes: 6/6; failures/fallback flushes: 0/0.
- Restored pending tokens: 7; requested/inserted token-layer units: 126/126.

The batched counters, not the general Task C1 restoration counters, establish
restoration coverage here. The latter can remain zero for this setting.

### Multi-News: Task C1 exact replay, then overwrite

- Fixed source 3; restored targets 3..11, learned targets 4..11.
- Policy: `taskc1_force_restore_all_exact_catchup_then_overwrite`.
- Aggregate validation: `ok`.
- Restoration flushes: 38; layer events: 342.
- Requested/succeeded/overwritten token-layer units: 414/414/414.
- Exact catchup required/executed: 414/414; exact catchup flushes: 38.

Exact replay is the intended existing path. This was not converted to direct
insertion and does not establish an inference speedup.

### Shared checks and counter definitions

Both settings report `restoration_failed_token_layer_units=0`,
`fallback_token_layer_units=0` and `fallback_event_count=0`.
Accounting `generation_count=2` for each. Accounting generated-token counts
are 69 (SAMSum) and 382 (Multi-News). Effective prediction lengths exclude
decoder-start/pad/EOS, so their sums, 67 and 380, use a different definition;
they are not presented as a count mismatch.

Exactly seven JSON keys differed from each public example:
`model_name_or_path`, `tokenizer_name`, `kv_runtime_restoration_artifact`,
`max_eval_samples`, `output_dir`, `eval_predictions_output` and
`kv_runtime_accounting_output`. All other values were unchanged, including
Native FREE shallow/deep mode, threshold 0.9, adaptive off, Official CALM off,
finite validation on, precision/length/decoding/cache settings,
`missing_kv_provenance_enabled=false` and per-sample accounting `null`.
The original `AdditionalArguments` / `update_autoconfig` bootstrap was used.

## Reported environment (observations, not an installation lock)

| Component | Observed version |
|---|---|
| Python | 3.10.12 |
| torch / CUDA | 2.12.0 / 13.0 |
| transformers / tokenizers | 4.28.1 / 0.13.3 |
| datasets / evaluate | 5.0.0 / 0.4.6 |
| nltk / numpy | 3.9.4 / 2.2.6 |
| accelerate / peft | 1.14.0 / 0.3.0 |
| sentencepiece / rouge-score | 0.2.1 / 0.1.2 |
| huggingface-hub / safetensors | 0.36.2 / 0.8.0 |

GPU 0 was an NVIDIA GeForce RTX 5090 (32,607 MiB reported). No other compute
job was present immediately before either run. These observations do not
define a tested fresh-install recipe; the working environment was not rebuilt.

The run cleared `PYTHONPATH`, set `PYTHONDONTWRITEBYTECODE=1` and
`CUDA_VISIBLE_DEVICES=0`, and enabled `HF_HUB_OFFLINE`, `HF_DATASETS_OFFLINE`,
`TRANSFORMERS_OFFLINE`, `HF_EVALUATE_OFFLINE` and `HF_HUB_DISABLE_TELEMETRY`
with value `1`. Existing cache locations were retained.

## Asset and import identity

Both checkpoint configs/weights and both restoration artifacts matched the
sizes/hashes recorded in [ASSET_MANIFEST.json](../assets/ASSET_MANIFEST.json),
except that the SAMSum artifact size was newly observed as 1,919,017 bytes
and is now recorded there. Its already-frozen hash was unchanged:
`8a0a2b7cc1c450673938081d8f02604bbfe4b478926358be0c202add36d357e9`.
The original artifact remains absent from this public assembly.

| Tokenizer setting | Accepted input | Full identity SHA-256 |
|---|---|---|
| SAMSum | Existing cached `t5-large` identifier | `b6c8d1bf050fb14efe4b02a4c4a658728b89d059132d671bbf0cade5b3af275f` |
| Multi-News | Existing local directory for `google/long-t5-tglobal-base`, revision `5093e24fd41835bd08e5817c58d55f060430c478` | `9d15ad27c161d5afad54065a7b0557de3a8850994efaf070a2d0e57a535e58cf` |

Both used `use_fast=True, local_files_only=True` and matched archived
semantics and inventory without an override or rewritten expected hash.
SAMSum's existing cache reported `refs/main` as
`150ebc2c4b72291e770f58e6057481c8d2ed331a` and loaded `model_max_length=512`.
This does not validate the development PC's mismatching local-directory
probe, a fresh download or a different identifier. See
[remaining tokenizer preparation work](ASSETS.md#tokenizers).

The following modules resolved inside the SHEER checkout, with no original
research source path added to `sys.path`: `run_summarization`,
`models.deploying_t5`, `models.deploying_longt5`, `util.additional_args`,
`sum_lib.trainer_sum`, `our_kv_restoration.runtime_kv_restoration`.
Only third-party libraries came from the reused environment.

## Offline resources and integrity

Existing validation caches contained 818 rows for `knkarthick/samsum` and
5,622 for bare `multi_news`; config names remained `null`. Only the first
two rows per setting were generated. These cache counts are not proof of
historical paper-population identity or fresh data preparation.
NLTK `punkt`/`punkt_tab` and the cached `evaluate` ROUGE module passed
resource checks and the actual evaluation calls.

Logs retained preprocessing-function fingerprint serialization warnings
(random fingerprints used), deprecation warnings and offline-cache notices.
The runs completed without changing cache settings or clearing caches.
Ordinary library-cache bookkeeping may have occurred.

The report states that the checked checkpoint/artifact/tokenizer files
(12 files) retained their resolved paths, sizes and modification timestamps;
package versions were unchanged. No source/config/asset/environment edits,
installs, downloads, refits, asset copies/resaves, original-checkout baseline
generation, full evaluation, benchmarking, commits, pushes or uploads were
performed during the server check.

Private evidence includes environment/source/asset/tokenizer/import/resource
checks, initial/final integrity checks, command/config diffs, exit status,
logs, predictions, evaluation results, aggregate accounting and output
validation. This public summary does not substitute for independent access
to those raw files.

## Remaining release gates

There is no reported blocker to these two bounded server runs and no further
model experiment is needed for this smoke task. Before advertising a complete
public two-setting release, still provide the genuine SAMSum artifact or a
verified public link, finalize reproducible tokenizer preparation, and finish
checkpoint-access and redistribution/metadata review. Do not infer that
private server availability is public reader access.

No full paper reproduction, paper-quality ROUGE, extraction-vs-original
output parity, performance/speedup, fresh Linux/CUDA installation or final
release readiness is claimed. Incidental ROUGE/trainer timings are not
reported as research results. No license, release tag or DOI is assigned by
this record.
