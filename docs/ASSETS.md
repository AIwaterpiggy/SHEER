# Asset sources and checksums

The repository contains code/configuration, not a second copy of the public
datasets or third-party model weights. Maintainers can reuse existing server
assets; downloading checkpoints to a local PC is not required.

## Datasets

| Setting | Public access / dataset information | Exact loader arguments |
|---|---|---|
| SAMSum | [SAMSum dataset](https://huggingface.co/datasets/knkarthick/samsum) | `knkarthick/samsum`, config `null`, split `validation` |
| Multi-News | [Multi-News dataset](https://huggingface.co/datasets/alexfabbri/multi_news) | `multi_news`, config `null`, split `validation` |

The Multi-News web URL is a source link, **not** an instruction to rename the
JSON loader argument. The inherited model dispatch uses bare `multi_news`.
Do not replace `null` with the string `"default"`. The examples use the first
eight validation samples, not the paper's historical selected populations.
Both dataset builders and one streamed example were checked locally. The
reported server smoke used existing offline, non-streaming validation caches
(818 SAMSum / 5,622 Multi-News rows), generating only the first two samples
per setting. Fresh download/preparation and historical population identity
were not verified by that smoke. Source terms still apply.

## FREE checkpoints

The [FREE authors' repository](https://github.com/raymin0223/fast_robust_early_exit)
links its [finetuned checkpoint folder](https://drive.google.com/drive/folders/1covxgJtIbFgH_xI-sXIuashX2zsY42w_).
Use the exact distilled checkpoint names below, not generic pretrained models
or `weighted_ce` variants. The upstream link was checked on 2026-09-16;
the large checkpoint files were not downloaded in the local review. The
later server report verifies both existing checkpoint configs/weights against
the manifest. It does not test a fresh download from the public folder.

| Setting | Checkpoint directory | Expected `pytorch_model.bin` SHA-256 |
|---|---|---|
| SAMSum | `samsum_t5_large_shallowdeep_kd_dyna` | `9266084a32c8c8e9f55b683c3a60f0122be1010fe8c7dff8d17baa1523d521ba` |
| Multi-News | `multi_news_longt5_base_shallowdeep_kd_dyna` | `bee5fbcbdff018f4f2eb9226db55dfdd9c8a3b4b87a8e69f3a4a341ba0415e03` |

File sizes and `config.json` hashes are in
[ASSET_MANIFEST.json](../assets/ASSET_MANIFEST.json). Do not edit or resave the
checkpoint to make it pass these checks. If upstream access fails, report it;
do not advertise an unverified alternative as the same checkpoint.

## Restoration artifacts

| Setting | Original file | Availability in this assembly |
|---|---|---|
| SAMSum | [native_free_source6_population_matched_phase3c_candidate.pt](../assets/native_free_source6_population_matched_phase3c_candidate.pt) | Original bytes included; local hash/schema/runtime-loader checks passed |
| Multi-News | [longt5_multinews_source3_phase3c_n128.pt](../assets/longt5_multinews_source3_phase3c_n128.pt) | Original bytes included in this server-validation branch |

Required artifact hashes:

- SAMSum: `8a0a2b7cc1c450673938081d8f02604bbfe4b478926358be0c202add36d357e9`
- Multi-News: `d9cdb61aaf069b9fe76eef4fe546fdaaa3239f0f553af029b2b2e4d4797f7014`

Both original files are included in this branch; no separate restoration
artifact download is needed after cloning it. SAMSum is 1,919,017 bytes
(about 1.92 MB), and Multi-News is 724,249 bytes. The supplied SAMSum original
and packaged copy match the frozen server hash exactly. Do not refit,
convert, resave or substitute a preliminary artifact.

These are fitted K/V restoration parameters used as inference inputs, not
generated summaries, evaluation scores, datasets or full model weights.
Their original server-path metadata remains embedded (two model-path fields
in SAMSum). A limited SAMSum metadata scan found no tested credential-token
or private-key patterns; final redistribution review remains separate.

## Tokenizers

- **Multi-News:** [google/long-t5-tglobal-base, frozen revision](https://huggingface.co/google/long-t5-tglobal-base/tree/5093e24fd41835bd08e5817c58d55f060430c478).
  Both the local snapshot and the server's existing local directory matched
  the archived N128 full identity:
  `9d15ad27c161d5afad54065a7b0557de3a8850994efaf070a2d0e57a535e58cf`.
- **SAMSum, accepted existing-cache load:** the server used
  `AutoTokenizer.from_pretrained("t5-large", use_fast=True, local_files_only=True)`.
  Its `models--t5-large` cache reported `refs/main` at
  `150ebc2c4b72291e770f58e6057481c8d2ed331a`; `model_max_length=512` loaded
  without an override. Full identity matched the archive:
  `b6c8d1bf050fb14efe4b02a4c4a658728b89d059132d671bbf0cade5b3af275f`.
  The observed cache ref is not proof that an independently downloaded
  snapshot or a different identifier will load equivalently.
- **SAMSum, unresolved fresh preparation:** a local-directory probe of
  `google-t5/t5-large` at `150ebc2c4b72291e770f58e6057481c8d2ed331a` did not
  match the archived identity. Its behavior config omitted recorded
  `model_max_length=512`, and its asset inventory differed. This revision is
  **not** presented as a verified replacement. Existing-cache success does
  not close this fresh-environment preparation gap.

Use complete assets, including `tokenizer.json`, as the FREE README instructs.
Do not automatically normalize tokenizer settings to pass an identity check.
The server procedure can inspect an existing local directory or an already
cached identifier without downloading anything. See the
[server record](SERVER_VALIDATION_2026-09-16.md) and
[local validation details](../VALIDATION.md).

### Minimum remaining preparation work

1. Record a portable tokenizer preparation procedure from the accepted
   cached inputs. A maintainer must verify both semantics (including
   `model_max_length`) and inventory with the existing identity helper in
   an isolated location before documenting a fresh-load recipe. Do not
   rewrite the expected digest or patch the model to conceal a mismatch.
2. Retain the accepted server environment and cached resources. A clean
   installation is a separate portability check, not a reason to rerun
   paper experiments or modify the validated environment.

3. Finish reader-facing checkpoint access and redistribution review. The
   original model/data source links remain the acquisition route; do not
   bundle large checkpoints or dataset text here.

The restoration-artifact availability gap is closed; fresh-tokenizer
preparation success and final release readiness are not claimed here.

## Default layout and verification

The two JSON examples expect paths relative to the SHEER root:

```text
checkpoints/
  samsum_t5_large_shallowdeep_kd_dyna/{config.json,pytorch_model.bin}
  multi_news_longt5_base_shallowdeep_kd_dyna/{config.json,pytorch_model.bin}
tokenizers/
  samsum/       # complete, verified tokenizer assets
  multinews/
assets/
  native_free_source6_population_matched_phase3c_candidate.pt
  longt5_multinews_source3_phase3c_n128.pt
```

Verify the selected setting **before loading**:

```bash
sha256sum -c assets/samsum.sha256
# OR
sha256sum -c assets/multinews.sha256
```

On Windows, compare `Get-FileHash -Algorithm SHA256` to the manifest. These
checksum files cover the checkpoint config/weights and artifact, not tokenizer
identity or dataset contents. Missing/mismatched files are a stop condition.
The fixed-layer runtime does not universally enforce the configured artifact
SHA, so setting that config field is not a substitute for checking bytes.

To keep server files in their existing locations, use the separate generated
config and external-path hash checks in [SERVER_SMOKE.md](SERVER_SMOKE.md).
No model save, asset copy, tokenizer rewrite or change to the tracked examples
is needed for that check.
