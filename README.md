# SHEER

A small inference package for the frozen SHEER restoration method, with two settings:

- **FREE × T5-large × SAMSum**
- **FREE × LongT5-base × Multi-News**

## What you can do with this repository

After installing the dependencies and preparing the original FREE model weights,
you can generate summaries with SHEER's calibrated K/V restoration:

| Setting | Input | Output |
|---|---|---|
| T5-large / SAMSum | A dialogue | A short summary of the conversation |
| LongT5-base / Multi-News | Several news articles about a topic | A combined summary |

The runner saves the generated summaries, reference summaries, basic ROUGE
scores, and counts showing whether restoration was used. It defaults to eight
samples; use two samples for a smaller check. The two calibrated restoration
parameter files are included, so no fitting is required. The original FREE model
weights are a separate required download; downloading this repository alone is
not enough to generate summaries.

You can also inspect the restoration implementation and run the tiny-model CPU
checks without downloading the large model weights. This repository provides
the two inference examples above, not the paper's historical result files or a
script that reproduces every experiment.

## Restoration method and validation

The package includes the calibrated restoration parameters. It restores hidden states with a source/target-specific dimension-wise affine map, applies the target block's existing RMS normalization and K/V projections, then applies K head/channel affine correction and V distance-group/head orthogonal transformation plus bias. V transformations allow reflections.

At FREE's existing flush point, queued exit tokens receive restored self-attention K/V. Those tokens do not run the skipped self-attention or feed-forward blocks. The current non-exit token runs normally and appends its own K/V. Unneeded terminal pending tokens are discarded when generation ends.

**Validation status:** the parameter files were matched to the final experimental SHA-256 values and exported without changing any tensor values. CPU tests cover both decoder families using tiny random models. The real FREE checkpoints and GPU generation have **not** been tested in this packaging environment. The LongT5 direct-insertion connection is a minimal port of the T5 connection; the archived LongT5 quality run used an exact-replay-then-overwrite evaluation path. This package is not a claim of reproducing that evaluation trajectory or all paper results.

## Installation

Get the source with Git, or use **Code → Download ZIP** on GitHub and extract it:

```bash
git clone https://github.com/AIwaterpiggy/SHEER.git
cd SHEER
```

Use **Python 3.11**; Transformers 4.28.1 requires the older Tokenizers 0.13.3 wheel.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
# Install a PyTorch build compatible with your CPU/CUDA device first.
pip install -r requirements.txt
```

For the CPU configuration tested during packaging:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

For CUDA, select a compatible build using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/). The archived experiment records `torch 2.12.0+cu130` and `transformers 4.28.1`; this package was tested with `torch 2.6.0+cpu`. CPU verification does not establish CUDA numerical or performance parity. All supported runs use float32; the package does not quantize the model.

Other pinned dependencies are NumPy 1.26.4, Hugging Face Hub 0.36.2, SentencePiece 0.2.0, Protobuf 3.20.1, NLTK 3.9.1, and rouge-score 0.1.2. Training, `datasets`, `evaluate`, and the private development repository are not dependencies.

## Prepare the model checkpoints

Download the **FREE authors' released, distilled checkpoints** from the [official checkpoint folder](https://drive.google.com/drive/folders/1covxgJtIbFgH_xI-sXIuashX2zsY42w_?usp=share_link), linked by the [FREE repository](https://github.com/raymin0223/fast_robust_early_exit#checkpoints).

Use these folders, each containing the original `config.json` and `pytorch_model.bin`:

```text
checkpoints/
  samsum_t5_large_shallowdeep_kd_dyna/
    config.json
    pytorch_model.bin
  multi_news_longt5_base_shallowdeep_kd_dyna/
    config.json
    pytorch_model.bin
```

The authors' release groups them under SAMSum and Multi-News respectively. Plain pretrained T5/LongT5 models and `weighted_ce` checkpoints do not match these parameters. Model weights are not bundled. The recorded weight sizes are approximately 2.95 GB and 0.99 GB; allow additional memory for the model, restoration banks, encoder activations, and KV caches.

`assets/settings.json` contains the exact config/weight file sizes and SHA-256 values from the accepted experiment inventories. Every run checks them before loading. If a release file has changed or cannot be obtained, request the original named checkpoint; do not substitute a different model or bypass the hash check. Actual large-file download availability was not verified here.

Use `--model /path/to/checkpoint` when storing the checkpoint elsewhere.

## Prepare tokenizers and data

```bash
python prepare_assets.py samsum
python prepare_assets.py multinews
```

These commands download only the tokenizers, validation data, and NLTK sentence-tokenizer resource. They write `tokenizers/<setting>/` and `data/<setting>/validation.jsonl`. They do not download model weights.

| Setting | Public source | Pinned revision |
|---|---|---|
| SAMSum tokenizer | [google-t5/t5-large](https://huggingface.co/google-t5/t5-large) | `150ebc2c4b72291e770f58e6057481c8d2ed331a` |
| Multi-News tokenizer | [google/long-t5-tglobal-base](https://huggingface.co/google/long-t5-tglobal-base) | `5093e24fd41835bd08e5817c58d55f060430c478` |
| SAMSum data | [knkarthick/samsum](https://huggingface.co/datasets/knkarthick/samsum) | `6b929ff10edec703164e3ddb2e94aae058c9ab5f` |
| Multi-News data | [alexfabbri/multi_news](https://huggingface.co/datasets/alexfabbri/multi_news) | `38cb206959a2cca3aa49858914fba76258a3dcaf` |

The archived SAMSum tokenizer identifier was `t5-large`, without an immutable revision. Its public snapshot above is accepted by matching the archived tokenizer backend digest; both settings check the same recorded backend digest before running. The Multi-News revision is directly recorded in the final experiment.

The data revisions are pinned public snapshots resolved during packaging. They are not asserted to identify the original paper dataset snapshot. Data files remain subject to their source terms. Data is downloaded directly from the published CSV/text files; no external dataset script is executed. For offline use, copy the prepared tokenizer folders and JSONL files, and prepare NLTK `punkt_tab` in advance. `--tokenizer` and `--data` accept alternative local locations.

## Frozen restoration parameters

These are **included** as `assets/samsum.npz` and `assets/multinews.npz`. No calibration or fitting step is needed.

| Setting | Original final artifact SHA-256 | Exported float32 parameters |
|---|---|---:|
| SAMSum, fixed source 6 | `8a0a2b7cc1c450673938081d8f02604bbfe4b478926358be0c202add36d357e9` | 468,992 |
| Multi-News, fixed source 3, N128 | `d9cdb61aaf069b9fe76eef4fe546fdaaa3239f0f553af029b2b2e4d4797f7014` | 174,336 |

The NPZ files contain exactly the original hidden/K/V tensor values for threshold 0.9. Server paths, fit-time bookkeeping, and other development metadata are excluded. Their export SHA-256 values are pinned separately in `assets/settings.json`; NPZ container hashes necessarily differ from the original `.pt` hashes. Loading uses `allow_pickle=False`, validates shape/coverage/finiteness, and accepts orthogonal matrices with determinant −1.

## Run

From the package root after preparing files:

```bash
python run.py samsum --max-samples 8
python run.py multinews --max-samples 8
```

These default to CUDA and process eight eligible validation rows. For a smaller check, use `--max-samples 2`. Use `--device cpu` if needed; the full checkpoints can be slow on CPU. To check files without loading model weights into memory:

```bash
python run.py samsum --check-assets
python run.py multinews --check-assets
```

Choose a different subset or output folder with `--offset 8 --max-samples 8 --output outputs/samsum_next8`. Existing nonempty output folders are never overwritten. Add `--require-restoration` to return exit code 2 if generation completes but the subset produces no restoration flush; results are still saved. A small subset may have no exit or only terminal exits, so successful generation alone is not evidence that restoration ran.

Frozen decoding settings:

| Setting | Source depth | Restored blocks, 0-based | Learned targets | Input prefix | Max input/output length |
|---|---:|---|---|---|---|
| SAMSum | 6 | 6–23 | 7–23 | empty | 512 / 128 |
| Multi-News | 3 | 3–11 | 4–11 | `summarize: ` | 2048 / 512 |

Depth 6 means six completed blocks: `h6` is block 5 output/block 6 input. Depth 3 means `h3` is block 2 output/block 3 input. The first missing block uses only its native RMSNorm and K/V projections, with no learned same-layer mapping. Confidence is the float32 softmax top-1 minus top-2 margin, compared with **strict `> 0.9`**. Adaptive thresholds are disabled. Decoding is greedy, batch/beam 1, repetition/length penalties 1, no forced tokens, no minimum length or repeated-ngram restriction. Output length uses `max_length`, including the decoder start token.

## Outputs and basic quality metric

Each run writes:

- `outputs/<setting>/generations.jsonl`: generated text/token IDs, tokenized/truncated reference, per-sample ROUGE, and restoration counts.
- `outputs/<setting>/metrics.json`: arithmetic mean per-sample ROUGE-1/2/L/Lsum F1 on a 0–100 scale and aggregate restoration counts.
- `outputs/<setting>/run_config.json`: resolved settings, input paths, and runtime versions.

ROUGE uses stemming and newline-separated NLTK sentences, following the existing preprocessing convention. No bootstrap is run. The subset is not the historical SAMSum held-out-409 population, and may overlap calibration examples. These basic metrics are for trying the package; they are not reported as paper-quality reproduction. No baseline comparisons, recovery-cost measurements, or historical ablations are included.

## CPU checks and source attribution

```bash
python tests_cpu.py
```

Tests cover real exported parameter loading, tiny-model generation for both decoder families, initial exits with no deep cache, pending-token restoration without skipped-block execution, document reset, normal-path parity with Transformers, and reflection-preserving V correction. Synthetic test parameters are created only in temporary directories and cannot be selected by `run.py`.

See [PROVENANCE.md](PROVENANCE.md) for extraction details and the LongT5 connection boundary, [VALIDATION.md](VALIDATION.md) for completed checks, and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source/license attribution. The FREE upstream snapshot did not contain a repository-level license. Public access to this code does not resolve that licensing status or imply a blanket license for all included code and parameters.
