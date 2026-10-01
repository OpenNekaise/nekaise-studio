# Reusable GPC web inventory

The September 30 operator approval adds `curriculum_loop.web_crawl_policy=inventory_v1`
to the continuing workload. It requires enabled web training and
`web_training_policy=teacher_selected_v1`. Historical `bounded_v1` snapshots retain
their original behavior; this feature is activated by a verified continuation.
The primary Teacher is GPT-6.1 Sol (`gpt-6.1-sol`) through Codex, at high effort.
The orchestrator and the four registered Authors retain their roles and budgets.

## Source selection and consumption

The Teacher selects new `ResearchPlan.sources` or explicitly reuses
`reuse_source_ids` from the persistent source catalog. Each unit may nominate four
new sources and four catalog references. An optional source `alternatives` list
contains at most three locations, each with its own URL, title and training scope,
for the same teaching purpose. The worker records the requested and actual source;
it never invents a replacement, silently skips a unit, or declares failed acquisition
to be completed GPC coverage.

For a coherent textbook or documentation section, `training.collection_prefix`
selects the origin and path. `max_pages` is at most 256; `seed_urls` adds up to four
in-scope chapter indexes, and `sitemap_urls` explicitly selects up to four same-origin
XML sitemaps. Up to eight sitemap/index documents are examined, including nested
indexes. Only discovered pages inside the selected content path can enter inventory.
An omitted training selection still means a single page; the host does not enlarge
the Teacher's selection automatically. Licenses are not checked under this policy.

The worker's existing CPU/network producer performs acquisition ahead of its GPU
consumer. Four source collections may execute concurrently across hosts, with four
HTTP requests globally in flight. Host pacing and robots access still apply.
`max_seconds` bounds each collection invocation to 10–300 seconds (default 120),
with a shared 300-second acquisition budget per unit, including alternatives.
Each response is bounded to 2 MB, each collection to 64M extracted characters.
Time-sliced collections preserve their visited/pending frontier and useful pages.
Teacher-selected reuse can continue that frontier; no unbounded background daemon
or extra training lookahead is introduced. Two prepared windows and two unreviewed
cycles remain the training bounds.

Catalog entries are scoped to the existing progression namespace. They reference
immutable collected snapshots and record the original and subsequent selecting
units. The complete catalog is available as an artifact, with a bounded list of
matching entries in Teacher context. Catalog registration means available source
material, not trained exposure. It does not advance curriculum progress.

`training_supply.fresh_chars_after_frontier` reports exact remaining characters after
the supplied source-coverage frontier and deduplicates pages within each unit.
This is not a tokenizer count; a frontier inherited from prepared predecessor windows
is not yet durable training. Shared pages across units can overlap. CPU preparation
deducts the predecessor's exact coverage again, and only verified checkpoint saves
advance the durable namespace. Teacher chooses `web_target_tokens`, corpus allocation,
original Teacher/Author material and dose; there is no new fixed mixture or review gate.

## HTTP and provenance

`web_http` stores actual response bytes and hashes once in ignored workspace storage.
Training prose and 12K reference excerpts derive from those same bytes. Responses
are pinned on first successful retrieval; repeated collection requests do not claim
a fresh network fetch. Robots evidence has a 30-minute cache lifetime and is checked
again for each acquisition session. Previously completed collections are immutable
historical acquisitions, not assertions of current publisher access.

Every live or cached redirect chain is checked against public addresses, the selected
scope and excluded source locations. A same-host HTTP link in an HTTPS collection
becomes a candidate HTTPS URL; its actual response, redirects and robots are checked
before admission. This does not allow a different publisher or an HTTPS downgrade.
Robots redirects are bounded and remain on the selected host. Declared 403/404 errors
are recorded without paid rewriting or access-control bypass. Transient network,
429 and 5xx failures receive bounded retries; a long Retry-After is reported rather
than ignored. Cancellation and pacing also honor acquisition deadlines.
Transient failures retain their frontier entry with an explicit cooldown and a
cumulative limit of three acquisition attempts (each HTTP attempt is itself bounded).
Cooling-down URLs do not block other ready pages. Exhausted URLs retain their failure
receipt; a completed collection denotes bounded acquisition, not that every page
succeeded. Cancellation never consumes a page or sitemap from the pending frontier.

Original research plans, alternative selection, raw response hashes, extracted text
hashes, robots snapshots, URL mappings and full-text source references remain available.
GPQA/MMLU-Pro banks and the excluded general datasets remain outside training sources;
no benchmark results are read or used by this path.

Readable-body extraction records its version. Closing an HTML ancestor also closes
unclosed child regions, so malformed navigation cannot hide the following article.
An extractor upgrade may create a separate bounded journal for a completed, empty
collection whose failures were exclusively insufficient readable text. The original
journal and raw response bytes remain intact. Successful collections and HTTP/access
failures are not reset; the ordinary robots, scope and cache-integrity checks still
apply. This is extraction recovery, not new training exposure or rendered JavaScript.

## Limitations and verification

This implementation supports static HTML/text and explicit XML sitemaps. It does not
render JavaScript, parse PDF, or traverse query-string chapter links. Single nominated
pages can retain their query string. Extraction is the existing readable-body parser;
exact text hashing is not semantic or near-duplicate detection. Indexes can themselves
contain useful prose or navigation; the Teacher owns source choice and dose. Cached
content is pinned, and source-refresh/version scheduling is not implemented.

Tests cover multiple pages, verified HTTPS mapping, robots redirects, sitemap scope,
cache integrity, retries, cancellation with frontier preservation, declared alternatives,
catalog reuse and exact unread coverage. Production activation also requires a bounded
live acquisition check and a new source-fingerprinted continuation with compatible Adam.
Short probes establish functionality, not sustained million-token-per-hour supply,
90% GPU duty, or learning improvement. Codex availability remains a separate dependency.

Official model identity: [OpenAI model documentation](https://learn.chatgpt.com/docs/models#gpt-6.1-sol).
