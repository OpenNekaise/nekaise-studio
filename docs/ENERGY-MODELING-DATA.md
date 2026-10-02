# Energy and physical modeling data preparation

The operator requested a focused energy-modeling domain dataset on 2026-10-01,
**before deciding to train**. The campaign remains under its explicit pause.
Preparation does not enqueue actions, alter the source cursor, replace the corpus,
change Teacher/Authors, or activate a new recipe. A later training decision must
preserve the authorized CoAPT Mid-training policies and actual model provenance.

The preparation workspace is `workspace/domain-prep/energy-modeling-20261001/`.
Its `README.md` reports measured coverage, counts, checks and outstanding gaps.
It is ignored data, not a tracked dataset. `../nekaise-corpus` is read-only.

## Coverage and source policy

`curricula/energy_modeling_sources.json` declares the public acquisition scope:
Modelica language, MSL, Buildings/IBPSA/AixLib/IDEAS/BuildingSystems/BESMod,
thermofluid libraries, OpenModelica, EnergyPlus/OpenStudio, FMI/Spawn, controls,
calibration, numerical methods, solar/daylight, and public ESP-r/TRNSYS/CHAMPS/CONTAM
references. BES is provisionally Building Energy Simulation. **CHANCES remains an
unresolved name**; collecting independently relevant TRNSYS and CHAMPS does not
claim either is the intended software.

Repositories resolve to an exact commit, with explicit release tags where selected.
HEAD means a development snapshot. Repository files are data: no imports, builds,
simulator runs, submodule execution or model validation. The source archive is kept.
Only declared text/code types are exported; file-type, path, encoding and size
omissions are recorded. A zero-document repository cannot be called complete.

Website acquisition reuses Studio's public-URL checks, robots handling, scope,
per-host pacing and durable fetch evidence in an isolated acquisition workspace.
License declarations are provenance, not a new web-admission gate. The existing
corpus publisher's default-view policy still applies to that read-only source.
Bounded crawl completion is **not** complete coverage of every linked document:
failed/unsupported pages and unvisited frontiers remain in source receipts.
Direct public PDFs retain their originals and `pdftotext -layout` derivatives.
Do not bypass access controls, disable TLS verification or imply private manuals
were acquired. No independent benchmark inputs, private tasks or scores are read.

Claude Opus 5.5 was consulted at high effort. Its recommendations to prioritize
equation semantics, preserve code/LaTeX, pin versions and separate patents were
adopted. We retained all original Modelica annotations rather than stripping them
with text rules. Its proposed private-benchmark overlap check was not performed:
benchmark isolation takes precedence. Source-name exclusions are explicitly not
a guarantee of zero content overlap.

## Reproduce the bundle

Run from the Studio repository with its virtual environment. These commands use
CPU, disk and public network access; they never start training or load weights.

```bash
.venv/bin/python scripts/prepare_energy_modeling.py audit \
  --corpus ../nekaise-corpus --out workspace/domain-prep/energy-modeling-20261001
.venv/bin/python scripts/prepare_energy_modeling.py repositories \
  --sources curricula/energy_modeling_sources.json \
  --out workspace/domain-prep/energy-modeling-20261001
.venv/bin/python scripts/acquire_energy_modeling_web.py \
  --sources curricula/energy_modeling_sources.json \
  --out workspace/domain-prep/energy-modeling-20261001
.venv/bin/python scripts/finalize_energy_modeling.py \
  --sources curricula/energy_modeling_sources.json \
  --out workspace/domain-prep/energy-modeling-20261001
```

Use a fresh output directory for a new dated snapshot. Do not run two writers for
the same acquisition/finalization, or tokenize an export while rebuilding it.
Completed repository receipts reuse their pinned snapshots; preserve receipts and
raw archives to reproduce the acquisition. The corpus audit records each copied
manifest hash and verifies selected document hashes against that snapshot. The
whole published Markdown directory is searched, including body-only matches.
This is lexical discovery; neither recall nor relevance precision is measured.

Optional exact token accounting uses a locally available `tokenizers` installation:

```bash
PATH_TO_ML_PYTHON scripts/count_domain_tokens.py \
  --out workspace/domain-prep/energy-modeling-20261001 \
  --tokenizer PATH_TO_KAI_CHECKPOINT/tokenizer.json
```

This counts each complete core document with the native tokenizer and no special
tokens/truncation. It is not trained exposure or prepared causal-target accounting:
future packing, chat templates, answer masks and passes determine those figures.
The tokenizer, its hash, per-document counts and manifest hash are retained.

## Multilingual additions

The 2026-10-02 addition is in
`workspace/domain-prep/energy-modeling-multilingual-20261002/`. Its report and
`dataset-set.json` reference the original bundle plus the unique addition. Training
remains paused. `curricula/energy_modeling_keywords.json` declares phrase aliases
and 14 search languages; `curricula/energy_modeling_multilingual_sources.json`
declares the acquired public sources. This is a bounded source inventory, not an
exhaustive internet search or a promise that every language has equal coverage.

IDA ICE / IDAICE / IDA-ICE, Indoor Climate and Energy, ESBO and related modeling
terms are included. Here **energy class means building energy performance classes,
certificates and calculation methods**. National terms include energiklass,
energideklaration, primärenergital, E-luku, Energieausweis, DPE and APE. Ambiguous
standalone acronyms such as ICE, EPC, NMF and SAP are not admission phrases.
Generic class/comfort/load terms require building context in titles. Full-text
mentions remain search leads; lexical relevance is not scientific validation.

```bash
.venv/bin/python scripts/prepare_energy_modeling.py audit \
  --corpus ../nekaise-corpus \
  --keywords curricula/energy_modeling_keywords.json \
  --out workspace/domain-prep/energy-modeling-multilingual-20261002
.venv/bin/python scripts/acquire_energy_modeling_web.py \
  --sources curricula/energy_modeling_multilingual_sources.json \
  --out workspace/domain-prep/energy-modeling-multilingual-20261002
.venv/bin/python scripts/finalize_energy_modeling.py \
  --sources curricula/energy_modeling_multilingual_sources.json \
  --out workspace/domain-prep/energy-modeling-multilingual-20261002 \
  --base workspace/domain-prep/energy-modeling-20261001
```

`--base` verifies the ready base's manifest hashes and excludes exact normalized
bodies already present in **any** base split. Base files are never rewritten.
`duplicates.jsonl` identifies `duplicate_scope=base` versus `addition`; the base
readiness hash is recorded. Keep both bundles accessible: cross-bundle duplicate
references are not copies of the original objects. Different translations and
editions remain distinct when their bodies differ. Near-duplicates are not removed.
Token accounting on the addition counts only its unique core export.

Web provenance carries `language_hint`, `jurisdiction`, `version_note` and
`document_role` where declared. A language hint is source metadata, **not detected
language**. National energy classes are not mapped onto a universal scale; dated
manuals are not automatically current regulations. Boverket's handbook explicitly
warned that updates for its October 2026 changes were still in progress at capture.

For the Danish HBEMO handbook, inspection found that the standard extractor kept
cookie text because an outer HTML form enclosed the page. Its explicit
`html_region=role_main` option extracts the declared main region from hash-verified
saved HTML before normal readable extraction. HTTP evidence and previous
extraction hashes remain available. Missing/short regions are recorded as
`extraction_omissions`; other readable pages survive in a partial receipt. This
offline override does not change the live teaching crawler. PDF originals and
layout extractions remain in `reference_pdf` pending equation/figure review.

## Export and extension contract

`readiness.json` contains source receipts, exclusions/errors, counts and SHA256 for
each export. It says `integrity_verified_core_available` only after readable bytes,
object hashes and duplicate-body hashes have been checked. It does not certify the
scientific correctness, simulator compatibility or learning value of every file.

`dataset/core.text.jsonl` is a standalone JSONL export with `text` plus provenance.
Text is original UTF-8 content (a UTF-8 BOM is removed in decoding); model equations,
whitespace and annotations are preserved. Its companion `core.manifest.jsonl`
references exact stored bytes through `object_path`, relative to the preparation
root, and `text_sha256`. Other manifests reference objects using the same contract:

- `research_candidates`: source/title-selected papers, including IBPSA and Modelica
  conferences; document-level relevance and PDF extraction still need review.
- `patent_supplement`: explicit title matches, kept outside the primary core.
- `reference_history`: older corpus repository paths superseded by a pinned snapshot.
- `reference_assets`: old schema dictionaries, bundled EnergyPlus copies, numerical
  tables and package ordering metadata. Useful simulator inputs are preserved.
- `reference_pdf`: public manuals/reports with original PDF paths; mathematical
  layout and image information are not certified by plain-text extraction.
- `duplicates`: identical trimmed bodies, retaining alternate IDs and URLs.

`local-candidates.jsonl` also contains body-only discoveries and rows outside the
publisher view. These are search leads, **not automatically selected training data**.
All original source catalogs and objects remain available. Exact dedup does not
remove renamed/translated/near-duplicate Modelica families. Older HTML versions
can coexist with new source documents and remain labeled by their URLs/commits.

Future loaders must verify export/object hashes, use the declared split, carry
source/version provenance, preserve model package context when chunking, and use
the normal campaign continuation and training accounting paths. This bundle is not
an automatically mounted replacement for `nekaise-corpus`, and it does not invent
SFT answers from raw code. Teacher/Authors can create varied instruction-response
material from it after the operator chooses to proceed.

Validation: `pytest tests/test_energy_modeling_preparation.py tests/test_corpus_policy.py`
checks corpus immutability, original-code recovery, hash quarantine, exact dedup,
source/path exclusions, multilingual ambiguity, immutable cross-bundle dedup,
form-wrapped handbook extraction and reference separation without network,
teacher or GPU work.
