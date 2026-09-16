# Extraction provenance

## Source identity

- Research repository: https://github.com/AIwaterpiggy/MoE_EarlyExit_KVcache
- Branch: `dev/selective-kv-restoration`
- Source commit: `860e12ea29dd86f98709a2381c9f372c337fcce6`
- SHEER starting commit: `e3ede7e5100b66d74cc68c6067a7ea0e3304e990`
- Assembly date: 2026-09-16
- Method: selected whole-file copies, not function extraction or reimplementation.

`SOURCE_MANIFEST.json` enumerates 42 Python files and the original requirements
file. SHA-256 values describe the source's checked-out bytes, including line
endings, not Git blob SHAs. `.gitattributes` disables text normalization for
copied files to retain byte identity across later checkouts.

Config JSON, documentation, inventories, dependency version snapshots,
ignore rules and Git attributes are new packaging material; they do not
implement model behavior. Runtime code does not depend on the private source
checkout's filesystem location.

## Dependency boundary

Package initializers and the existing runner/models import diagnostic,
provenance, training and other experiment support modules. Their presence
does not expand the supported settings. Stage 3 verified complete-runner
imports in an isolated Windows CPU environment. The later maintainer-provided
[server report](docs/SERVER_VALIDATION_2026-09-16.md) records real-checkpoint
generation and restoration for both settings at
`a1ac0024fd77cce8ad829e33b39d6adf98b47eb2` (two samples each).

The disabled calibration compatibility probe imports fitting scripts that
are deliberately absent. Unsupported collection/fitting modes are not
provided. The original tests also need extra fixtures and fitting scripts;
the 42-file assembly does not include them or claim a public tiny-model test
suite. For local stage-3 validation only, an ignored `.validation/probe/`
directory holds byte-identical copies of the extraction and 15 original
test/fixture/support files. The same three selected test files pass 141
tests in both that probe and the original checkout. Those extra fixture
dependencies are not additions to the public 42-file runtime extraction.

The historical LongT5 `run_quality_compat.py` removed an empty dataset auth
argument. The copied current runner handles that case in
`_dataset_auth_kwargs`, so the wrapper is not added.

## Artifacts

Multi-News is copied from the member recorded in `assets/ASSET_MANIFEST.json`.
Its raw file SHA-256 matches the final N128 authority. Stage 3 loaded it
through the unchanged runtime manager and confirmed fixed-layer source 3
compatibility. No resaving, fitting, NPZ export, tensor conversion or
metadata rewriting was performed.
Embedded server-path metadata remains and needs disclosure review before
public release.

SAMSum is not bundled and has no verified public download link. The server
report verifies the genuine file against the archived schema-v2 corrective
SHA-256 and reports 1,919,017 bytes. This documentation update did not obtain
that file; bytes cannot be reconstructed from a digest. No preliminary
artifact or synthetic replacement is supplied.

Checkpoint identities come from the archived SAMSum schema-v2 corrective
inventory and LongT5 N128 final quality inventory. Weights are not bundled.

## Behavioral boundary

SAMSum selects source-6 Batched Task C2; LongT5 selects source-3 Task C1 exact
replay then overwrite. Both retain the original `AdditionalArguments` and
`update_autoconfig` bootstrap. No LongT5 direct-insertion port was made.

Latest-source byte identity does not establish parity with an older paper
execution point. The extracted package's bounded server smoke now has
reported restoration coverage; original-checkout parity was not tested.

## Stage-4 packaging correction

The two example configs now set `missing_kv_per_sample_accounting_output`
to `null`. They deliberately keep `missing_kv_provenance_enabled=false`.
The prior combination requested per-sample records without sample context:
the unchanged trainer returns no context with provenance disabled, and its
per-sample writer then raises `stable_sample_id is required for per-sample
missing-KV accounting`. This was reproduced without model weights or a GPU.
No trainer/runtime change was made; aggregate accounting and predictions
remain enabled. No other config value changed in this stage.

## Publication boundary and checklist

Publish only the reviewed files:

- The 43 unchanged source/dependency files in `SOURCE_MANIFEST.json`.
- `README.md`, `PROVENANCE.md`, `VALIDATION.md`, `THIRD_PARTY_NOTICES.md`,
  `SOURCE_MANIFEST.json`, `.gitignore`, `.gitattributes`.
- The two `configs/free_*.json` examples and two CPU validation requirements
  files (explicitly not a Linux/CUDA install lock).
- `docs/ASSETS.md`, `docs/SERVER_SMOKE.md`,
  `docs/SERVER_VALIDATION_2026-09-16.md`, `assets/ASSET_MANIFEST.json`,
  the two checksum files and approved original restoration artifacts only.

Do not publish local environments/caches, `.validation` fixtures/reports,
checkpoints, tokenizer caches, dataset text, run outputs, private research
history, archives or credentials. The ignore rules exclude those locations
and weight/dump/archive/log extensions, with explicit exceptions only for
the two intended restoration `.pt` filenames. Ignore rules are not a security
review; inspect the final Git file list and diff before staging.

Before advertising a runnable two-setting release:

- [x] Verify checkpoint/artifact hashes and both tokenizer identities using
  existing server assets (maintainer-provided report, 2026-09-16).
- [ ] Confirm that reader-facing checkpoint acquisition is reproducible;
  the server used existing files, not fresh downloads.
- [ ] Supply the genuine SAMSum artifact or a verified public link to it.
- [ ] Finalize fresh-environment tokenizer preparation; the accepted cached
  SAMSum load does not validate arbitrary local-directory snapshots.
- [x] Run the bounded server checks and record actual restoration coverage:
  both settings reported `RUN_AND_RESTORATION_PASS`, two samples each.
- [ ] Resolve third-party redistribution questions and review embedded
  artifact metadata. Do not silently strip metadata and change frozen hashes.
- [x] Review the initial 60-file snapshot for the user-authorized server handoff on
  `work/compact-free-extraction`. No source rewriting, private history,
  checkpoint bundles or generated outputs are included.
- [ ] After server validation and the remaining release review, select a
  release tag/commit for citation. No tag, DOI or release URL is fabricated.

Local absence of third-party weights is **not** a code-publication blocker.
Artifact access, clear scope and accurate availability claims are distinct
from the server smoke check. No full paper rerun is part of this checklist.

The user authorized commit/push of the server-validation snapshot on
2026-09-16. The subsequent server report is summarized in one added public
document, without personal paths or raw generation outputs; the publication
file set is now 61 files. Documentation/inventory updates do not change the
43 copied files, configs or artifact bytes, or confer redistribution rights.
The original Multi-News artifact's recorded path metadata remains unchanged;
no credentials or model weights are added. The branch is not the final paper
release, and the report does not replace the remaining availability review.

### Availability statement draft

Use only after the described code and artifacts are actually publicly
accessible; revise it if either setting remains incomplete:

> A reference implementation and inference configurations for Native FREE
> with T5-large on SAMSum and LongT5-base on Multi-News are available at
> https://github.com/AIwaterpiggy/SHEER. The repository provides the fitted
> K/V restoration artifacts, or links to them, together with instructions
> for obtaining the public datasets and the original FREE checkpoints from
> their respective sources. This release covers these two configurations
> and does not include the complete experimental pipeline of the paper.

This is a description of the planned release, not a certification of journal
policy compliance. Include the actual release tag/commit when finalized and
cite dataset/checkpoint sources in the paper as appropriate.
