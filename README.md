# Page EntroKV · v0.2

An open-source research workspace for **GQA-aware KV-cache page selection**. Capture bounded
Qwen2 attention locally, inspect layers and groups, compare configurations, and preserve complete
experiments with SHA-256 checksums.

**[Open the workspace](https://page-entrokv-lab.coolgoose68.chatgpt.site)** ·
[Release notes](CHANGELOG.md) · [Capture and data contract](docs/v0.2.md)

## Who it helps

- **KV-cache researchers:** inspect how entropy weighting changes decisions and find failure cases.
- **Inference engineers:** study quotas and cross-head unions before integrating a serving backend.
- **Students and contributors:** reproduce the operator without a GPU or hosted inference endpoint.

This release makes selection behavior inspectable. Compressed-generation quality, physical GPU
savings, throughput, and vLLM compatibility require separate validation.

## Run the website

```sh
git clone https://github.com/bruce12-glitch/Powered-EntroKV-.git
cd Powered-EntroKV-
python3 -m http.server 8080 --directory dist
```

Open `http://localhost:8080`. Native modules and workers require HTTP rather than a file URL.
SHA-256 exports and project saving require HTTPS or localhost. The website has no build step,
package installation, runtime JavaScript dependency, or model execution in the browser.

## What's new

- **Capture:** a local Qwen2 CLI with CPU/CUDA options, bounded query windows, explicit
  aggregation/normalization, model revisions, environment metadata, and attention checksums.
- **Layer Explorer:** layer × KV-group heatmaps, policy overlays, entropy weights, original token
  inspection, and cancellable background analysis.
- **Compare:** two configurations on the same input, token Jaccard across different page sizes,
  common-reference attention-mass diagnostics, and CSV export.
- **Research projects:** up to 20 complete manifests in local IndexedDB; reopen and recompute.
- **Full exports:** inputs, configurations, selections, metrics, engine version, release code revision,
  and SHA-256 integrity verification before import/recomputation.

The single-layer tool, ideal-payload planner, evidence ledger, standard-library Python reference,
and v0.1 snapshots remain available. The original Notion-inspired design and Folio mascot remain.

## Capture locally

Use Python 3.10+ in an isolated environment. The tested stack is PyTorch 2.8.0,
Transformers 4.57.3, and NumPy 2.2.6. Choose a PyTorch installation matching your hardware.

```sh
python -m venv .venv
# Activate .venv for your operating system.
# CPU-only installation:
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r dist/assets/requirements-capture.txt

# Check mechanics without downloading pretrained weights:
python dist/assets/capture_attention.py --smoke-test --output smoke.json

# Capture a prompt with at least 16 tokens:
python dist/assets/capture_attention.py \
  --model Qwen/Qwen2.5-1.5B-Instruct --revision main \
  --prompt-file dist/assets/example-prompt.txt --output capture.json \
  --device cpu --dtype float32 --max-tokens 1024 --observation-window 8
```

Import the file in Layer Explorer. Prefer an immutable model commit SHA instead of `main`.
Model and tokenizer use the same requested revision; weights download separately and require
sufficient RAM/VRAM. `trust_remote_code=False` is enforced. The site does not provision compute.

Capture prefills the prefix with SDPA, then collects only the final 1–32 query rows with eager
attention. It averages per head, explicitly normalizes in Float64, and records raw row-sum errors.
All layers execute; only requested layers are exported. Truncation keeps the first `--max-tokens`
tokens and is reported. Existing output files are never overwritten.

`dist/assets/qwen2-smoke.json` is an actual tiny **random-weight** Qwen2 CPU capture. It checks
mechanics, not pretrained-model quality. The default six-layer demo uses independently seeded
synthetic observations and is not a model forward pass.

## Scientific contract

Estimate Rényi-2 entropy on each head's normalized residual after excluding the first `sink_size`
positions. Within contiguous GQA groups, use `softmax(-entropy / tau)` to weight the **original**
attention rows. Page scores are maxima over valid token positions.

```text
K = max(1, floor(ceil(T / block_size) * retention_percent / 100))
```

Select exactly K distinct pages per group; ties use ascending page index. The one-page floor,
rounding, partial pages, and realized retention are disclosed. Sink isolation does not protect
sink pages automatically. Independent-head unions can exceed K and are over-budget diagnostics.

Each policy's retained mass is evaluated against the **same arithmetic mean of original sibling
rows**. It is not answer accuracy. A page mask does not release physical backend storage;
shared full-head blocks may require a union across groups. The planner estimates ideal KV payload
under stated compaction assumptions. The supplied manuscript's raw pilot logs are not reproduced.

## Formats and limits

- V2 bundles contain 1–128 aligned layer observations, each with an ID, layer index, and v1 trace.
- Each trace: 1–64 query heads, 16–8,192 tokens, and at most 262,144 attention values.
- Each bundle: at most 1,048,576 values and 32 MiB JSON. The single-layer importer accepts 8 MiB.
- Finite nonnegative rows sum to 1 within 1e-5. Browser imports are never silently normalized.
- Heads are contiguous by KV group; `query_heads` is divisible by `kv_heads`.
- Retention is in (0,100], page size is an integer in [1,512], and tau is in [0.05,10].

Full manifests include SHA-256 checksums for sanitized input, configuration, and stored results.
Import verifies these and recomputes the selections. Checksums detect changed content; they do not
authenticate uploader identity or model provenance. V1 traces and full manifests remain importable.
See [the complete data/hash contract](docs/v0.2.md).

## Verify

```sh
npm test
# After installing the optional capture requirements:
python -B -m unittest discover -s tests -p test_capture.py -v
```

Node 20+ and Python 3.10+ suffice for the dependency-free numerical suite: analytical cases,
quotas, ties, partial pages, union bounds, memory arithmetic, Python/JavaScript operator and binary
hash parity, alignment, common-reference metrics, token overlap, manifests, tampering, and CSV safety.

The capture integration suite uses a tiny random-weight Qwen2 CPU model. It compares bounded
SDPA-prefix observations against full eager last-query attention (`rtol=1e-5`, `atol=1e-7`).
Pretrained CUDA capture and serving performance have not been validated by this release.

## Storage and privacy

Imports are analyzed locally and are not uploaded by the application. Save project stores full
inputs and optional token text in IndexedDB only when requested. Storage can be cleared and is
specific to the current browser/origin; download full manifests for durable records. Up to 50
legacy snapshots remain in localStorage; their FNV-1a identifiers are not cryptographic checksums.
There are no analytics, external fonts, or remote runtime scripts.

The capture CLI may contact Hugging Face for assets. Token IDs are included by default and can
reveal prompt content; decoded text is opt-in. Avoid publishing private captures unintentionally.

## Design and attribution

Original warm editorial design inspired by Notion; not affiliated with Notion. Folio is original
AI-generated artwork, optimized at `dist/assets/folio.webp`.

- Inbasekaran S., *Page-EntroKV: Hardware-Aligned, Entropy-Weighted KV-Cache Eviction under
  Grouped-Query Attention*, supplied manuscript. No venue, DOI, or publication date is asserted.
- [Qwen2.5-1.5B configuration](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json)
- [Transformers attention interface](https://huggingface.co/docs/transformers/attention_interface)
- [Historical vLLM PagedAttention explainer](https://docs.vllm.ai/en/latest/design/paged_attention/):
  background only; integration must inspect a pinned current backend.

## Contributing and next milestones

Open issues with the input schema, configuration, engine version, and expected behavior. Numerical
changes need an independent invariant, analytical case, or reference comparison. Use public fixtures.

Next: matched compressed-generation evaluation, position-preserving cache mutation, then a pinned
serving adapter measuring physical allocation, peak overhead, quality, latency, and concurrency.
Those are separate from v0.2 observation analysis.

MIT license. Original research attribution is retained.
