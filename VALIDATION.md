# Validation status

Local CPU validation and reported server smoke validation, 2026-09-16.
This is not a release-readiness, real-checkpoint parity or paper-reproduction
certificate. The sections below distinguish the earlier local checks from
the later server result.

## Server smoke result (maintainer-provided report)

Both supported settings are **RUN_AND_RESTORATION_PASS** at
`a1ac0024fd77cce8ad829e33b39d6adf98b47eb2`. Each ran two validation samples
through the original runner with existing server assets on an RTX 5090:

| Setting | Exit / predictions | Actual restoration | Failures / fallback |
|---|---|---|---|
| SAMSum / T5-large / source 6 | 0 / 2 | 6 successful batched flushes; 126 inserted token-layer units | 0 / 0 |
| Multi-News / LongT5-base / source 3 | 0 / 2 | 38 exact-catchup/restoration flushes; 414 restored/overwritten token-layer units | 0 / 0 |

Both accounting validations passed. Checkpoint/artifact hashes, tokenizer
identities and all 43 copied source/dependency files matched. Imports came
from SHEER; no original-repository baseline or additional eight-sample run
was performed. The runtime, example configs and assets were not modified.

The [sanitized server record](docs/SERVER_VALIDATION_2026-09-16.md) records
the supplied report's environment, counters, scope and remaining gates.
This documentation update did not access the server or independently inspect
its raw logs. Existing-cache success does not establish fresh-install
portability, full-model parity, paper quality or performance.

### Local checks for the documentation reconciliation

Only six documentation/inventory files were updated and the sanitized server
record was added. No source, example config, dependency list, checksum file
or artifact bytes changed. Focused local checks passed:

- 11 existing publication-packaging tests (same cases, not new runtime cases).
- Complete runner import, both JSON/production-bootstrap checks and loading
  the real included Multi-News artifact through the original manager.
- All 43 manifest source/dependency hashes and 42 Python AST checks.
- Frozen checkpoint/artifact/tokenizer identities and method fields retained;
  at this documentation-only stage SAMSum was still absent locally, with no
  invented public download URL. The subsequent intake is recorded below.
- JSON and local Markdown links, scoped personal-path/credential checks,
  and `git diff --check`.
- All six server-guide Bash command blocks are unchanged from the tested
  commit and pass `bash -n`; the three embedded Python blocks parse.

No GPU/model generation or download was performed during this documentation
update. The previously recorded 141 runtime cases were not rerun for these
documentation-only changes. Local focused evidence remains Git-ignored.

## SAMSum artifact intake

The user subsequently supplied the original SAMSum restoration parameter
file. Its packaged copy is 1,919,017 bytes and byte-identical to the supplied
file, with SHA-256
`8a0a2b7cc1c450673938081d8f02604bbfe4b478926358be0c202add36d357e9`.
No conversion, resaving, fitting, metadata edit or synthetic substitution
was performed. Both supported restoration artifacts are now included.

Local CPU checks passed:

- Original `load_phase3c_policy_artifact` schema validation: schema 2, genuine
  `fixed_layer` source 6, exact threshold 0.9 and geometry 24/1024/16/64.
- Hidden and K maps cover exactly targets 7..23 (17 maps each); six V gap
  bins cover all gaps 1..17. The 80 float32 tensors contain 468,992 elements
  and are all finite. Target 6 retains the native same-layer projection.
- Original `AdditionalArguments` / `update_autoconfig` and
  `RuntimeKVRestorationManager.from_path` accept the public settings and both
  real artifacts. SAMSum reports `fixed_shallow_layer`, source 6,
  `fixed_layer_runtime` and compatible scopes without CALM or preliminary
  overrides. Multi-News still reports the corresponding fixed-source-3 mode.
- Four focused intake tests plus the existing 11 publication-packaging tests:
  **15 passed**. Tests/helpers and XML evidence remain in ignored
  `.validation/`; no new public validation framework is introduced.

The limited string-metadata scan found two historical model-path fields and
no tested GitHub/Hugging Face/AWS credential-token or private-key patterns.
Paths are retained for frozen-byte identity and disclosed in
[PROVENANCE.md](PROVENANCE.md); this is not comprehensive security or rights
clearance. The source attachment was not edited. This `.pt` is an inference
input, not an evaluation-results file. No checkpoint/data download, model
construction, generation, GPU timing or new server run was performed.

## Environment and source identity

- Windows, CPython 3.11.7, uv 0.10.8; isolated `.venv`, no global-package edits.
- Torch 2.6.0+cpu, Transformers 4.28.1, Accelerate 0.20.3, PEFT 0.3.0,
  Datasets 2.14.7, Evaluate 0.4.1. CUDA unavailable in this environment.
- Original `requirements.txt` remains byte-identical. Additional pins are in
  `requirements-cpu-validation.in`; the observed 83-package version snapshot
  is `requirements-windows-cpu.lock.txt`.
- A second empty environment, `.validation/lock-venv`, installed from that
  snapshot passed dependency checks, imports, config/artifact checks and the
  same 141 tests. This is not a cross-platform or distribution-file-hash lock.
- Initial imports/test collection failed because the unseeded uv environment
  lacked `pkg_resources`, imported by Accelerate. Adding the validation-only
  `setuptools==69.5.1` dependency resolved this; no research code was patched.
- All 42 Python files plus original requirements match both
  `SOURCE_MANIFEST.json` and the original checkout (43/43 SHA-256/size/bytes).
  All 42 Python files parse and pass `py_compile`; the three local evidence
  helpers also pass `py_compile`. No restoration/evaluation code,
  example config, checkpoint or artifact was changed in stage 3.
- Original source HEAD: `860e12ea29dd86f98709a2381c9f372c337fcce6`.
  SHEER base HEAD: `e3ede7e5100b66d74cc68c6067a7ea0e3304e990`.

## Imports, config and artifact

The complete `run_summarization` module imports from the SHEER root without
adding the original checkout to `sys.path`. This imports T5/LongT5 models,
the restoration manager, training support and dependencies.

Both **unchanged** JSON examples parse with the actual Transformers 4.28.1
`HfArgumentParser` and all four original dataclasses. The unchanged
`AdditionalArguments` / `update_autoconfig` path is then called with
synthetic config objects of the correct full model geometry (no weights).

| Resolved field | SAMSum | Multi-News |
|---|---|---|
| `num_layers` / `num_decoder_layers` | 24 / 24 | 12 / 12 |
| `d_model` / `num_heads` / `d_kv` | 1024 / 16 / 64 | 768 / 12 / 64 |
| `shallow_exit_layer` | 6 | 3 |
| `kv_runtime_restoration_batched_insertion_enabled` | true | false |

Both resolve `exit_min_layer=None` (field exists), `use_shallow_deep=true`,
`use_early_exit=false`, `use_adapt_threshold=false`, shallow/deep and
restoration thresholds 0.9, restoration method `phase3c_kv_final`, recent
exact window 0, force-restore-all true and finite validation true.
Direct insertion, CALM, Official FREE CALM and the preliminary source-6
override are all false. Evaluation is enabled, training/prediction disabled,
and batch/beam are 1. Dispatch remains `samsum` / `multi_news`.

The **real Multi-News artifact**, 724,249 bytes, retains SHA-256
`d9cdb61aaf069b9fe76eef4fe546fdaaa3239f0f553af029b2b2e4d4797f7014`.
`RuntimeKVRestorationManager.from_path` loads it and reports:

- calibration source mode `fixed_layer`, fixed source 3;
- runtime source mode `fixed_shallow_layer`, fixed source 3;
- calibration layer/runtime scope matches: true;
- runtime compatible: true; use classification `fixed_layer_runtime`.

At this local CPU stage, SAMSum's artifact was missing; no replacement was
supplied. Neither real distilled checkpoint existed in the expected local
directories. No real-model construction, generation or quality measurement
was performed locally. The later server result is recorded separately above.

## Existing synthetic runtime tests

Three source test files were copied unchanged:

- `tests/test_task_c2_batched_lazy_restoration.py`
- `tests/test_task_c2_batched_deferred_finite_validation.py`
- `tests/test_longt5_source3_taskc1_runtime.py`

Together they pass **141 tests**, with no failures or skips:

| Execution location | Result |
|---|---|
| Isolated extraction probe, initial validation environment | 141 passed |
| Original research checkout, same validation environment | 141 passed |
| Isolated extraction probe, fresh version-snapshot environment | 141 passed |

These are the same 141 cases repeated, not 423 independent cases. Coverage
includes existing tiny-model/fake-block generation and restoration fixtures,
cache/flush behavior, source/target contracts, finite-check atomicity,
generation-local resets and accounting. They use synthetic weights/maps;
passing does not establish full-model output parity or GPU performance.
Dependency deprecation warnings remain; no model-code workaround was added.

To retain the compact public source boundary, `.validation/probe` is ignored.
It contains all 43 manifest files plus 15 byte-identical source test/support
files. Extra fitting/fixture dependencies live only there, not in the 42-file
public runtime. This is a local audit, not a test suite shipped with a clone.

Tests used one Torch/OMP/MKL thread, `HF_HUB_OFFLINE=1`,
`HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and
`PYTHONDONTWRITEBYTECODE=1`. The original checkout used pytest's
`-p no:cacheprovider`; all XML reports went under SHEER `.validation`.
Representative invocation from the prepared probe:

```python
import torch, pytest
torch.set_num_threads(1)
raise SystemExit(pytest.main([
    "-q",
    "tests/test_task_c2_batched_lazy_restoration.py",
    "tests/test_task_c2_batched_deferred_finite_validation.py",
    "tests/test_longt5_source3_taskc1_runtime.py",
]))
```

Local-only evidence helpers: `.validation/check_configs.py`,
`.validation/check_resources.py`, `.validation/check_tokenizers.py`.
Local test evidence: `.validation/runtime-tests.xml`,
`.validation/original-runtime-tests.xml`, `.validation/lock-runtime-tests.xml`.
These helpers/reports and all environments/caches are Git-ignored.

## Data, tokenizers and metrics

- `datasets.load_dataset_builder(name, name=None)` and one example from
  `load_dataset(name, name=None, split="validation", streaming=True)` pass
  for both original names: `knkarthick/samsum` and `multi_news`. Text/summary
  fields are nonempty. This is **not** full non-streaming preparation or
  historical-population verification. The Multi-News builder reports
  756,785,627 download bytes for full data; full preparation was not run.
- NLTK `punkt` and `punkt_tab` downloaded to `.cache/nltk_data`; a two-sentence
  tokenization check passes. `evaluate.load("rouge")` downloads and gives all
  four scores of 1.0 on a synthetic identical sentence. This is resource
  availability testing, not model quality.
- Small tokenizer snapshots only were downloaded to the ignored HF cache;
  no model parameter files were downloaded. Both load locally with
  `AutoTokenizer(..., use_fast=True, local_files_only=True)` and encode a
  synthetic sentence. Neither is staged in the example `tokenizers/` paths.
- Multi-News revision `5093e24fd41835bd08e5817c58d55f060430c478` matches the
  archived N128 identity using the original `tokenizer_identity`:
  `9d15ad27c161d5afad54065a7b0557de3a8850994efaf070a2d0e57a535e58cf`.
- SAMSum probe revision `150ebc2c4b72291e770f58e6057481c8d2ed331a` matches the
  vocabulary/backend serialization of its schema-v2 archive, but its local
  behavior config omits the archived `model_max_length=512`. Asset inventories
  also differ (an archived cache-blob-name entry versus local `spiece.model`
  and `tokenizer.json`). Its full identity does **not** match archived
  `b6c8d1bf050fb14efe4b02a4c4a658728b89d059132d671bbf0cade5b3af275f`.
  No tokenizer normalization, alternate identity rule or config correction
  was introduced. The later server check matched the archive using the
  existing cached `t5-large` identifier; it does not establish equivalence
  of this local-directory probe. Fresh-environment preparation remains open.

## Stage-4 publication preparation

The README now focuses on the two supported execution paths. Asset links,
filenames/checksums and tokenizer caveats are in `docs/ASSETS.md`; server-only
read-only preflight, two-sample execution and counter checks are in
`docs/SERVER_SMOKE.md`. Those instructions are not a server execution result.
The Windows CPU installation recipe is kept below, outside the public GPU
quick-start path. No large weights, server jobs, fitting or benchmark runs
were added by this step.

While checking the actual trainer output contract, the prior public JSON
combination was found to be invalid: provenance off plus a per-sample output
path results in `stable_sample_id is required for per-sample missing-KV
accounting`. Both settings reproduced that failure using the original
trainer methods on CPU without weights. The minimal packaging correction
sets only `missing_kv_per_sample_accounting_output=null` in each example.
The original 42 source files and `requirements.txt` remain byte-identical;
aggregate accounting, predictions and all method settings are unchanged.

The 141-test results above cover source runtime behavior, not every runner
configuration combination. They are not presented as evidence that the old
per-sample output configuration was executable end-to-end.

Stage-4 local checks passed:

- 11 focused tests in ignored `.validation/test_publication_packaging.py`:
  reproduce the old failure, verify the corrected writer is disabled while
  aggregate sidecars remain enabled, exercise the documented config generation
  without overwrites, and distinguish run-only from restoration-covered results
  using synthetic output fixtures (including fallback rejection).
- Both updated configs pass the existing argument/bootstrap checks, and the
  real Multi-News artifact still loads through the original runtime manager.
- All 43 source/dependency files still match the original checkout and manifest;
  all 42 source files pass AST checks. Reconstructing each prior JSON by
  restoring only the old per-sample path reproduces its stage-start SHA-256.
- All five Bash blocks pass `bash -n`; all three embedded Python blocks parse.
  At this stage the preflight and GPU runner had not yet been executed on
  the server; the later reported result is recorded above.
- The publication candidate list has exactly 60 reviewed files: 43 unchanged
  source/dependency files and 17 packaging/artifact files. No environment,
  cache, output, local fixture, checkpoint or private Git history is included.
- Packaging JSON, local Markdown links, whitespace and `git diff --check` pass.
  A limited scan of copied source found no private-key blocks, common GitHub/HF
  token literals or the checked personal absolute-path prefixes. This is not
  comprehensive security/rights clearance; artifact metadata review remains open.

Focused command (from SHEER, using the isolated CPU environment):

```powershell
.venv\Scripts\python.exe -m pytest -q .validation/test_publication_packaging.py --junitxml=.validation/packaging-tests.xml
.venv\Scripts\python.exe .validation/check_configs.py
git diff --check
```

## Optional local CPU installation

Tested with CPython 3.11.7 and uv 0.10.8 on Windows (PowerShell). Use an unused
environment path if `.venv` already exists; do not alter the server environment
with this CPU-only recipe.

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe --torch-backend cpu -r requirements-windows-cpu.lock.txt
uv pip check --python .venv\Scripts\python.exe
$env:HF_HOME = "$PWD\.cache\huggingface"
$env:NLTK_DATA = "$PWD\.cache\nltk_data"
.venv\Scripts\python.exe -m nltk.downloader -d .cache/nltk_data punkt punkt_tab
.venv\Scripts\python.exe -c "import run_summarization; print('Import OK')"
```

The resources are downloaded to ignored local caches; they are not distributed
in Git. Do not install the unbounded original requirements over these pins or
use this environment for the CUDA-targeted example runs.

## Remaining publication checks

1. Finalize a reproducible tokenizer preparation recipe, especially for
   SAMSum. Its accepted cached identifier is verified; a newly prepared
   local directory is not. Keep the existing identity checks fail-closed.
2. Confirm reader-facing checkpoint access and document environment limits.
   Existing server checkpoint hashes are verified; fresh public downloads
   and a fresh Linux/CUDA installation were not tested. Do not rebuild the
   working server environment or download large weights to the PC for this.
3. Complete the rights/metadata review in `PROVENANCE.md` before a release
   tag or `main` promotion. No further model run is required for the bounded
   smoke task, and no original-checkout parity claim is made.

Do not claim paper-quality reproduction, latency/speedup, full-checkpoint CPU
support, LongT5 direct insertion or release readiness from these checks.
The original research worktree remains clean. Packaging whitespace/link checks
and `git diff --check` pass; original source formatting remains untouched.

## Server-handoff staging audit

Before the user-authorized branch push, the actual staged Git blobs for all
43 original files were checked against `SOURCE_MANIFEST.json`, not only the
working-tree copies. All match. Exactly 60 reviewed files are included.
The original artifact hash and the focused 11-test check pass again. The
server guide now includes cloning the handoff branch; its six Bash blocks
pass syntax checks without executing server commands.

The full initial-import `git diff --cached --check` reports the copied
source's original CRLF/trailing-whitespace formatting. A CRLF-aware check
still reports inherited whitespace in source files. These bytes are preserved
deliberately, not reformatted during extraction. The packaging-only staged
whitespace check passes; a clean full-source whitespace result is not claimed.
