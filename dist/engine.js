/* Page EntroKV reference operators. Float64 JavaScript arithmetic; no model inference. */
export const ENGINE_VERSION = '0.2.0';
export const EPSILON = 1e-12;
export const MAX_TRACE_ENTRIES = 262144;

function integer(value, name, min, max) {
  if (!Number.isSafeInteger(value) || value < min || value > max) {
    throw new Error(`${name} must be an integer from ${min} to ${max}.`);
  }
  return value;
}

export function validateTrace(input) {
  if (!input || typeof input !== 'object' || Array.isArray(input)) throw new Error('Trace must be a JSON object.');
  if (input.schema_version !== '1.0') throw new Error('schema_version must be "1.0".');
  const hq = integer(input.query_heads, 'query_heads', 1, 64);
  const hkv = integer(input.kv_heads, 'kv_heads', 1, hq);
  if (hq % hkv !== 0) throw new Error('query_heads must be divisible by kv_heads.');
  if (!Array.isArray(input.attention) || input.attention.length !== hq) {
    throw new Error('attention must contain exactly query_heads rows.');
  }
  const tokens = input.attention[0]?.length;
  integer(tokens, 'Token count', 16, 8192);
  if (tokens * hq > MAX_TRACE_ENTRIES) throw new Error(`Trace exceeds ${MAX_TRACE_ENTRIES.toLocaleString()} attention entries.`);
  const sinkSize = integer(input.sink_size ?? 4, 'sink_size', 0, tokens - 1);
  const attention = input.attention.map((row, h) => {
    if (!Array.isArray(row) || row.length !== tokens) throw new Error(`Head ${h}: row length must be ${tokens}.`);
    let sum = 0;
    for (const value of row) {
      if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) throw new Error(`Head ${h}: values must be finite, nonnegative numbers.`);
      sum += value;
    }
    if (Math.abs(sum - 1) > 1e-5) throw new Error(`Head ${h}: probabilities sum to ${sum.toPrecision(7)}; expected 1 ± 0.00001. No automatic normalization is applied.`);
    return row.slice();
  });
  const metadata = {};
  for (const key of ['model', 'layer', 'description']) {
    if (input.metadata?.[key] !== undefined) metadata[key] = String(input.metadata[key]).slice(0, 300);
  }
  const result = { schema_version: '1.0', query_heads: hq, kv_heads: hkv, sink_size: sinkSize, attention, metadata };
  if (input.source === 'synthetic' && Number.isSafeInteger(input.seed)) { result.source = 'synthetic'; result.seed = input.seed; }
  return result;
}

export function makeFixture(seed = 42) {
  integer(seed, 'Seed', 0, 2147483647);
  let state = seed >>> 0;
  const random = () => {
    state += 0x6D2B79F5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const tokens = 1024, hq = 12, hkv = 2;
  const attention = Array.from({ length: hq }, (_, h) => {
    const local = h % 6, group = Math.floor(h / 6);
    const peak = 208 + group * 384 + (seed % 7);
    const row = Array.from({ length: tokens }, (_, t) => {
      let v = 0.1 + random() * 0.28;
      if (local === 0) v += 145 * Math.exp(-((t - peak) ** 2) / 5);
      else if (local === 1 && t < 4) v += 85;
      else v += (0.2 + local * 0.2) * Math.exp(-((t - ((h * 133 + seed) % 970 + 20)) ** 2) / 360);
      return v;
    });
    const total = row.reduce((a, b) => a + b, 0);
    return row.map(v => v / total);
  });
  return { schema_version: '1.0', source: 'synthetic', seed, query_heads: hq, kv_heads: hkv, sink_size: 4, attention,
    metadata: { model: 'Synthetic GQA fixture', layer: 'Single observation', description: 'Seeded retrieval-like, sink-like, and diffuse heads. Not a model capture.' } };
}

export function collisionEntropy(row, sinkSize = 4, isolate = true) {
  const start = isolate ? sinkSize : 0;
  let mass = 0, square = 0;
  for (let i = start; i < row.length; i++) mass += row[i];
  const denom = Math.max(mass, EPSILON);
  for (let i = start; i < row.length; i++) square += (row[i] / denom) ** 2;
  return -Math.log(square + EPSILON);
}

export function stableTopK(scores, k) {
  return Array.from(scores, (value, index) => ({ value, index }))
    .sort((a, b) => b.value - a.value || a.index - b.index).slice(0, k).map(v => v.index);
}

export function pageMax(row, blockSize) {
  const scores = [];
  for (let start = 0; start < row.length; start += blockSize) {
    let max = 0;
    for (let i = start; i < Math.min(start + blockSize, row.length); i++) max = Math.max(max, row[i]);
    scores.push(max);
  }
  return scores;
}

export function analyze(trace, config = {}) {
  const blockSize = integer(config.blockSize ?? 16, 'Page size', 1, 512);
  const retention = config.retention ?? 20;
  const tau = config.tau ?? 0.5;
  if (!Number.isFinite(retention) || retention <= 0 || retention > 100) throw new Error('Retention must be greater than 0 and at most 100.');
  if (!Number.isFinite(tau) || tau < 0.05 || tau > 10) throw new Error('Temperature must be from 0.05 to 10.');
  const isolate = config.isolate ?? true;
  if (typeof isolate !== 'boolean') throw new Error('Sink isolation must be a boolean.');
  const tokens = trace.attention[0].length, pages = Math.ceil(tokens / blockSize);
  const quota = Math.max(1, Math.floor(pages * retention / 100));
  const ratio = trace.query_heads / trace.kv_heads;
  const entropy = trace.attention.map(row => collisionEntropy(row, trace.sink_size, isolate));
  const groups = Array.from({ length: trace.kv_heads }, (_, g) => {
    const heads = trace.attention.slice(g * ratio, (g + 1) * ratio);
    const entropies = entropy.slice(g * ratio, (g + 1) * ratio);
    const logits = entropies.map(e => -e / tau);
    const maxLogit = Math.max(...logits);
    const exp = logits.map(e => Math.exp(e - maxLogit));
    const z = exp.reduce((a, b) => a + b, 0), weights = exp.map(e => e / z);
    const pooled = Array.from({ length: tokens }, (_, t) => heads.reduce((s, row, h) => s + weights[h] * row[t], 0));
    const mean = Array.from({ length: tokens }, (_, t) => heads.reduce((s, row) => s + row[t] / ratio, 0));
    const scores = pageMax(pooled, blockSize), meanScores = pageMax(mean, blockSize);
    const selected = stableTopK(scores, quota), meanSelected = stableTopK(meanScores, quota);
    const headSelections = heads.map(row => stableTopK(pageMax(row, blockSize), quota));
    const union = [...new Set(headSelections.flat())].sort((a, b) => a - b);
    const pooledMass = selected.reduce((sum, p) => sum + pooled.slice(p * blockSize, (p + 1) * blockSize).reduce((a, b) => a + b, 0), 0);
    return { group: g, entropies, weights, pooled, scores, selected, meanSelected, meanScores, headSelections, independentUnion: union, uor: union.length / quota, selectedMass: pooledMass };
  });
  const layerUnion = [...new Set(groups.flatMap(g => g.selected))].sort((a, b) => a - b);
  return { engine_version: ENGINE_VERSION, tokens, pages, quota, blockSize, requestedRetention: retention,
    realizedRetention: quota / pages * 100, tau, isolate, groups, layerUnion, crossGroupUnionRatio: layerUnion.length / quota,
    averageIndependentUor: groups.reduce((s, g) => s + g.uor, 0) / groups.length,
    source: trace.source === 'synthetic' ? 'synthetic' : 'imported',
    policy: 'sink-isolated-renyi2-group-pooling-block-max',
    quotaRule: 'max(1, floor(ceil(T / page_size) * retention / 100)); ties use ascending page index',
    note: 'Selection analysis only. No cache allocation, model inference, task accuracy, or serving speed is measured.' };
}

export function planMemory(config) {
  const { layers, kvHeads, headDim, context, batch, blockSize, bytesPerElement, retention } = config;
  integer(layers, 'Layers', 1, 256); integer(kvHeads, 'KV heads', 1, 256);
  integer(headDim, 'Head dimension', 1, 1024); integer(context, 'Context length', 16, 2097152);
  integer(batch, 'Batch size', 1, 1024); integer(blockSize, 'Page size', 1, 512);
  if (![0.5, 1, 2, 4].includes(bytesPerElement)) throw new Error('Unsupported element precision.');
  if (!Number.isFinite(retention) || retention <= 0 || retention > 100) throw new Error('Retention must be in (0, 100].');
  const pages = Math.ceil(context / blockSize), kept = Math.max(1, Math.floor(pages * retention / 100));
  const bytesPerPage = 2 * layers * kvHeads * headDim * blockSize * bytesPerElement * batch;
  if (!Number.isSafeInteger(pages * bytesPerPage)) throw new Error('Payload exceeds exact integer arithmetic. Reduce the architecture or batch dimensions.');
  return { pages, kept, bytesPerPage, fullBytes: pages * bytesPerPage, retainedBytes: kept * bytesPerPage,
    savedBytes: (pages - kept) * bytesPerPage, realizedRetention: kept / pages * 100, reduction: pages / kept,
    rawBytes: 2 * layers * kvHeads * headDim * context * bytesPerElement * batch,
    assumptions: 'Ideal independently compacted KV payload, uniform per-layer/group quotas. Includes page padding; excludes model weights, quantization metadata, allocator metadata, activations, workspace and selection/compaction peaks. Not measured vLLM allocation.' };
}

export function fingerprint(trace) {
  // Non-cryptographic identifier for accidental input changes, never an integrity/security claim.
  const text = JSON.stringify(trace);
  let hash = 2166136261;
  for (let i = 0; i < text.length; i++) { hash ^= text.charCodeAt(i); hash = Math.imul(hash, 16777619); }
  return (hash >>> 0).toString(16).padStart(8, '0');
}
