# Page EntroKV

A browser-local research application for **GQA-aware KV-cache page selection**.
Explore entropy-weighted pooling, inspect exact retained-page indices, estimate ideal KV payload,
and export reproducible experiments. No account, GPU, API key, build step, or runtime JavaScript dependencies.

## Run

```sh
git clone https://github.com/bruce12-glitch/Powered-EntroKV-.git
cd Powered-EntroKV-
python3 -m http.server 8080 --directory dist
```

Open `http://localhost:8080` in a modern browser. Use HTTP rather than opening index.html directly,
because the app uses native JavaScript modules.

## What's included

- **Overview:** original Notion-inspired workspace and Folio, the hand-drawn page mascot.
- **Playground:** seeded synthetic observations or strict JSON attention-trace import; sink-isolated Rényi-2 entropy;
  GQA group pooling; block-max scores; deterministic top-K; mean and independent-head comparisons; page inspector.
- **Memory planner:** model architecture presets, custom dimensions, page rounding, batch size, and element precision.
- **Experiment notebook:** browser-local snapshots; full-manifest export and import/recomputation.
- **Research evidence ledger:** separates conditional theory, manuscript-reported pilot observations, and unmeasured serving claims.
- **Documentation:** input contract, mathematical definitions, memory assumptions, provenance, privacy, local setup.
- **Python reference:** `dist/assets/page_entrokv.py`, standard library only, independently cross-checked against the browser operator.

## Scientific contract

This release is a **numerical reference and research workspace**. It does not load a language model,
modify a model's KV cache, implement a vLLM plugin, or measure task accuracy, GPU allocation, or throughput.
The sample trace is synthetic, reproducible from a seed, and explicitly labeled. It is not a capture from Qwen.
Model presets describe architecture, not validated compression quality or runtime compatibility.

The implementation follows the supplied Page-EntroKV manuscript's sink-isolated entropy weighting,
raw-attention pooling, and page-max selection. It is newly implemented application code, not a claim
that the manuscript's original experimental code or raw logs have been independently reproduced.

For each head, entropy is computed on the residual after excluding the first `sink_size` tokens.
Weights are `softmax(-entropy / tau)` within contiguous GQA groups. The original rows are pooled.
Page scores are maxima over valid token positions. Retained pages per group:

```text
K = max(1, floor(ceil(T / block_size) * retention_percent / 100))
```

Ties use ascending page index. The one-page floor, page rounding, and actual retention are disclosed.
Sink isolation does not automatically protect sink pages. Cross-group unions are accounted separately.
Fixed selection cardinality does not establish physical memory savings or factual accuracy.

## Trace format

```json
{
  "schema_version": "1.0",
  "query_heads": 2,
  "kv_heads": 1,
  "sink_size": 4,
  "attention": [
    [0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625],
    [0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625]
  ],
  "metadata": {"model": "example", "layer": "0"}
}
```

One layer / one observation. Query heads must be ordered contiguously within KV groups.
Rows must be nonnegative finite probabilities summing to 1 within 1e-5. No silent normalization.
Browser limits: 1–64 heads, 16–8,192 tokens, 262,144 total entries, 8 MiB JSON files.

## Python reference

```sh
python3 dist/assets/page_entrokv.py trace.json --retention 20 --block-size 16 --tau 0.5
```

Both references use double-precision arithmetic. Float32 GPU implementations may differ near ties.
The Python CLI outputs selection results; it does not integrate into a serving engine.

## Verify

```sh
npm test
```

Uses Node's built-in test runner (Node 20+), with Python 3.10+ for numerical parity.
Tests cover deterministic fixtures, normalization, entropy closed forms, all-sink guards,
stable ties, partial pages, exact quotas, union bounds, invalid inputs, memory arithmetic,
and Python/JavaScript parity. No dependency installation is required.

## Storage and exports

Attention imports never leave the browser. Full traces remain in memory only.
Up to 50 selection snapshots are stored in localStorage, specific to the current browser and origin.
Save full JSON manifests before closing a session if the original imported data must be retained.
The FNV-1a fingerprint is a change identifier, not a cryptographic checksum.
The application uses no analytics, external fonts, or remote scripts.

## Design and assets

The interface uses an original Notion-inspired editorial direction: warm neutrals, purple controls,
pastel tool surfaces, structured tables, focus states, responsive navigation, and reduced-motion support.
It is not affiliated with Notion. Folio is original AI-generated artwork made for this project.
Asset recipe: friendly paper-stack scholar, black ink contours, round glasses, purple cap/scarf,
warm paper backdrop, restrained lavender/yellow details. Optimized asset: `dist/assets/folio.webp`.

## Research sources

- Supplied manuscript: Inbasekaran S., *Page-EntroKV: Hardware-Aligned, Entropy-Weighted KV-Cache Eviction under Grouped-Query Attention*.
  Venue, DOI, and publication date are intentionally unspecified. Raw manuscript experiment logs are not bundled.
- [Qwen2.5-1.5B configuration](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json)
- [Qwen2.5-7B configuration](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)
- [vLLM PagedAttention design](https://docs.vllm.ai/en/latest/design/paged_attention/)

## Contributing

Open an issue with the exact input schema, configuration, expected result, and engine version.
Do not include sensitive attention traces or private source text. For numerical changes, add
an independent invariant, known analytical case, or reference comparison. Serving integration
must be benchmarked separately from selection analysis.

## Roadmap

1. Validate imported real-model observations and matched downstream baselines.
2. Benchmark the policy in a compressed-cache generation adapter.
3. Integrate backend allocation and position-aware attention.
4. Measure physical memory, peak overhead, quality, latency, and concurrency.

MIT license. Original research attribution is retained in the reference documentation.
