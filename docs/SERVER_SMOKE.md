# Maintainer server smoke check

Purpose: check that the compact extraction can use the **existing** server
assets for the two supported settings. This is not a benchmark, refit, training
run or repeat of the paper experiments. The maintainer-provided
[2026-09-16 server record](SERVER_VALIDATION_2026-09-16.md) reports successful
generation and restoration for both settings at
`a1ac0024fd77cce8ad829e33b39d6adf98b47eb2`, with two samples each. This guide
remains a procedure for future checks; that result applies to the recorded
snapshot, environment and existing assets, not every future clone.
Local syntax/fixture checks are recorded in [VALIDATION.md](../VALIDATION.md).

## 1. Use a separate SHEER directory and the existing environment

Use branch `work/compact-free-extraction`, not the research working tree or
the old `main` snapshot. From a parent directory with no existing
`SHEER-server-check` directory:

```bash
git clone --branch work/compact-free-extraction --single-branch https://github.com/AIwaterpiggy/SHEER.git SHEER-server-check
cd SHEER-server-check
git rev-parse HEAD
```

Record the returned commit for the check. The publication file list is in
[PROVENANCE.md](../PROVENANCE.md); environments, caches, local test fixtures,
private research history and outputs are not included. This is a validation
handoff, not a final paper release.

Run one setting at a time in Bash. Replace every `/absolute/...` placeholder.
For `multinews`, change the setting **and all asset values** to its own files.
Choose a tokenizer directory or the exact cached identifier used by the accepted
server run. Do not blindly replace `t5-large` with a new local snapshot.

```bash
set -euo pipefail
cd '/absolute/path/to/separate/SHEER'
export SHEER_PYTHON='/absolute/path/to/validated-server-venv/bin/python'
export SHEER_SETTING='samsum'  # then repeat separately with multinews
export SHEER_MODEL='/absolute/path/to/samsum_t5_large_shallowdeep_kd_dyna'
export SHEER_ARTIFACT='/absolute/path/to/native_free_source6_population_matched_phase3c_candidate.pt'
export SHEER_TOKENIZER='t5-large'  # already cached, or verified absolute directory
export PYTHONPATH=''
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
```

Keep the existing server cache locations. If a tokenizer, dataset, NLTK
resource or ROUGE module is not cached, **stop and identify what is missing**;
do not silently change dataset names, download weights or rebuild the environment.
In particular, do not install the Windows CPU lock into the server environment.
Run on one available GPU without other experimental jobs competing for it.

## 2. Read-only source, asset and environment preflight

The block hashes files in chunks (including the existing large weights),
checks imports and inspects tokenizer identity. It does not construct a model
or call generation. Full checkpoint hashes can take some time to read.

```bash
"$SHEER_PYTHON" - <<'PY'
import hashlib, importlib.metadata, json, os
from pathlib import Path

root = Path.cwd().resolve()
assert (root / 'docs/SERVER_SMOKE.md').is_file(), 'Use the separate SHEER root'
assert not (root / 'outputs').is_symlink(), 'Do not redirect smoke outputs'
setting = os.environ['SHEER_SETTING']
assert setting in ('samsum', 'multinews')
manifest = json.loads((root / 'assets/ASSET_MANIFEST.json').read_text())
assets = manifest[setting]
model = Path(os.environ['SHEER_MODEL'])
artifact = Path(os.environ['SHEER_ARTIFACT'])
assert model.is_absolute() and model.is_dir(), model
assert artifact.is_absolute() and artifact.is_file(), artifact

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

source = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
for entry in source['files']:
    path = root / entry['path']
    assert path.stat().st_size == entry['bytes'] and sha256(path) == entry['sha256'], path
for entry in assets['checkpoint']['files']:
    path = model / entry['path']
    assert path.stat().st_size == entry['bytes'] and sha256(path) == entry['sha256'], path
assert sha256(artifact) == assets['artifact']['sha256'], artifact
print('Source/checkpoint/artifact bytes: PASS')

import torch, run_summarization
from transformers import AutoTokenizer
from our_kv_restoration.missing_kv_dump_provenance import tokenizer_identity
assert Path(run_summarization.__file__).resolve().parent == root
for name in ('torch', 'transformers', 'datasets', 'evaluate', 'peft', 'accelerate'):
    print(name, importlib.metadata.version(name))
assert torch.cuda.is_available(), 'Use the existing working CUDA environment'
print('GPU:', torch.cuda.get_device_name(0))
identifier = os.environ['SHEER_TOKENIZER']
tokenizer = AutoTokenizer.from_pretrained(identifier, use_fast=True, local_files_only=True)
identity = tokenizer_identity(tokenizer, requested_identifier=identifier,
    tokenizer_root=identifier if Path(identifier).is_dir() else None)
expected = (assets['tokenizer']['archived_identity_sha256'] if setting == 'samsum'
            else assets['tokenizer']['identity_sha256'])
print('Tokenizer expected / observed:', expected, identity['tokenizer_identity_sha256'])
print('Tokenizer model_max_length:', tokenizer.model_max_length)
assert identity['tokenizer_identity_sha256'] == expected, 'Review tokenizer identity before generation'
print('Preflight PASS; no model construction or generation performed')
PY
```

A tokenizer mismatch is a **review stop**, not a request to refit anything.
For SAMSum, compare against the accepted server run's tokenizer record: the
full digest includes the asset inventory as well as semantic settings. A
cache filename/layout difference alone does not prove tokenization changed,
but do not silently dismiss semantic differences or rewrite the expected hash.
The reported server SAMSum check matched with cached `t5-large`. Finalize a
reproducible preparation recipe before claiming fresh-environment portability.

## 3. Generate a fresh two-sample config

After preflight passes, this creates only a new smoke directory/config. It
does not edit the public examples or existing assets. `mktemp` prevents reuse
of old outputs; the inherited evaluator does not enforce that itself.

```bash
mkdir -p outputs
export SHEER_SMOKE="$(mktemp -d "$PWD/outputs/server-smoke-${SHEER_SETTING}-XXXXXX")"
"$SHEER_PYTHON" - <<'PY'
import json, os
from pathlib import Path

setting = os.environ['SHEER_SETTING']
assert setting in ('samsum', 'multinews')
run = Path(os.environ['SHEER_SMOKE']).resolve()
assert run.parent == (Path.cwd() / 'outputs').resolve() and run.is_dir()
config = json.loads(Path('configs/free_' + setting + '.json').read_text())
assert config['missing_kv_provenance_enabled'] is False
assert config['missing_kv_per_sample_accounting_output'] is None
config.update({
    'model_name_or_path': os.environ['SHEER_MODEL'],
    'tokenizer_name': os.environ['SHEER_TOKENIZER'],
    'kv_runtime_restoration_artifact': os.environ['SHEER_ARTIFACT'],
    'max_eval_samples': 2,
    'output_dir': str(run),
    'eval_predictions_output': str(run / 'eval_predictions.jsonl'),
    'kv_runtime_accounting_output': str(run / 'missing_kv_accounting.json'),
})
with (run / 'config.json').open('x', encoding='utf-8') as handle:
    json.dump(config, handle, indent=2)
    handle.write('\n')
print(run / 'config.json')
PY
```

Only asset locations, sample count and three output paths change. Thresholds,
source layers, precision, lengths, decoding, seeds and restoration flags stay
unchanged. Per-sample accounting remains disabled with provenance disabled;
aggregate counters and prediction records remain enabled.

## 4. Execute the original runner once

This is the first block that loads model weights and runs generation. It is
for the server only. Keep the shell's `pipefail` setting so `tee` cannot hide
a failed Python process.

```bash
"$SHEER_PYTHON" run_summarization.py "$SHEER_SMOKE/config.json" 2>&1 | tee "$SHEER_SMOKE/run.log"
```

## 5. Inspect results and restoration coverage

```bash
"$SHEER_PYTHON" - <<'PY'
import json, os
from pathlib import Path

run = Path(os.environ['SHEER_SMOKE'])
config = json.loads((run / 'config.json').read_text())
metrics = json.loads((run / 'eval_results.json').read_text())
summary = json.loads((run / 'missing_kv_accounting.json').read_text())
predictions = [json.loads(line) for line in
               (run / 'eval_predictions.jsonl').read_text().splitlines() if line.strip()]
assert metrics['eval_samples'] == len(predictions) == config['max_eval_samples']
assert all(row['generated_length'] > 0 for row in predictions)
assert summary['validation']['status'] == 'ok', summary['validation']
counters = summary['counters']
for key in ('restoration_failed_token_layer_units', 'fallback_token_layer_units'):
    assert counters[key] == 0, (key, counters[key])
if os.environ['SHEER_SETTING'] == 'samsum':
    assert summary['restoration_policy_mode'] == 'phase3c_fixed_source6_batched_lazy_insertion'
    assert summary['task_c2_batched_validation']['status'] == 'ok'
    assert summary['task_c2_batched_flush_failures'] == 0
    assert summary['task_c2_batched_fallback_flushes'] == 0
    covered = summary['task_c2_batched_flush_successes'] > 0
else:
    assert summary['restoration_policy_mode'] == 'taskc1_force_restore_all_exact_catchup_then_overwrite'
    covered = counters['restoration_succeeded_token_layer_units'] > 0
print('Accounting policy:', summary['restoration_policy_mode'])
print('Aggregate counters:', json.dumps(counters, sort_keys=True))
print('RUN + RESTORATION COVERAGE PASS' if covered else
      'RUN ONLY PASS: no restored flush observed; restoration coverage still pending')
PY
```

Zero restoration coverage is not a mathematical failure and is not a full
pass for restoration. If needed, prepare a **new** run directory with eight
samples; do not change thresholds, force exits or repeatedly search for a
favorable result. If eight still gives no coverage, report it as untested.
For LongT5, exact catch-up work is expected, not evidence of a failed optimization.
Standard trainer timing/ROUGE fields may be emitted; do not use them as paper
performance or quality measurements.

If a separate extraction-vs-original check is needed, repeat the same small
config with the same server interpreter, assets, cache and seed in the original
checkout, changing only the three output paths to another unused directory.
Compare prediction/reference token digests, generated lengths and semantic
accounting counters. Exclude timing fields and paths. Do not overwrite the
research checkout or resolve a mismatch by changing the restoration method.

Keep configs, environment versions, asset-check results, logs and output
checks locally. Report the tested SHEER snapshot/commit and coverage for both
settings. Review logs/metadata before publishing any new outputs. Providing
these commands does not imply server execution or a successful smoke result.
