import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { makeFixture } from '../dist/engine.js';
import { demoBundle,validateBundle,validateConfig,attentionBytes,sha256,jsonDigest,analyzeBundle,summarizeRun,compareRuns,researchManifest,inspectImport,runsCSV,DEFAULT_CONFIG } from '../dist/research.js';

test('legacy traces upgrade without pretending to be model captures',()=>{
  const bundle=validateBundle(makeFixture(42));assert.equal(bundle.source,'imported');assert.equal(bundle.observations.length,1);assert.equal(bundle.observations[0].trace.source,undefined);
});
test('legacy full manifests preserve the recorded configuration when upgraded',async()=>{
  const config={retention:37,blockSize:8,tau:1.2,isolate:false};
  const loaded=await inspectImport({application:'Page EntroKV',schema_version:'1.0',input:makeFixture(42),config});
  assert.deepEqual(loaded.configs,[config]);assert.equal(loaded.expected,null);assert.equal(loaded.bundle.source,'imported');
});
test('multi-layer inputs preserve aligned token positions and unique observation IDs',()=>{
  const b=demoBundle();assert.equal(b.source,'synthetic');assert.equal(b.observations.length,6);
  assert.deepEqual(validateBundle(b),b);
  const repeated=structuredClone(b);repeated.observations[1].id=repeated.observations[0].id;assert.throws(()=>validateBundle(repeated),/unique/);
  const bad=structuredClone(b);bad.token_ids=[1,2];assert.throws(()=>validateBundle(bad),/token_ids/);
  const different=structuredClone(b);different.observations[1].trace.attention=different.observations[1].trace.attention.map(()=>Array(16).fill(1/16));assert.throws(()=>validateBundle(different),/same token/);
});
test('aggregate resource limit and probability validation apply to every observation',()=>{
  const trace={schema_version:'1.0',query_heads:2,kv_heads:1,sink_size:4,attention:[Array(8192).fill(1/8192),Array(8192).fill(1/8192)]};
  const large={schema_version:'2.0',kind:'attention_bundle',observations:Array.from({length:65},(_,i)=>({id:`layer-${i}`,layer:i,trace}))};
  assert.throws(()=>validateBundle(large),/1,048,576/);
  const bad=demoBundle();bad.observations[4].trace.attention[0][3]=-1;assert.throws(()=>validateBundle(bad),/nonnegative/);
});
test('configuration validation rejects strings, invalid isolation and non-finite values',()=>{
  for(const c of [{...DEFAULT_CONFIG,tau:NaN},{...DEFAULT_CONFIG,isolate:'true'},{...DEFAULT_CONFIG,retention:'20'},{...DEFAULT_CONFIG,blockSize:1.1}])assert.throws(()=>validateConfig(c));
});
test('SHA-256 matches a standard known-answer vector',async()=>{
  assert.equal(await sha256(new TextEncoder().encode('abc')),'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
});
test('Python capture and browser use the identical binary attention checksum',async()=>{
  const trace=makeFixture(42),browser=await sha256(attentionBytes(trace));
  const script='import sys,json;sys.path.insert(0,"dist/assets");from capture_attention import attention_sha256;print(attention_sha256(json.load(sys.stdin)))';
  const r=spawnSync('python3',['-B','-c',script],{input:JSON.stringify(trace),encoding:'utf8'});assert.equal(r.status,0,r.stderr);assert.equal(r.stdout.trim(),browser);
  const metadataChanged=structuredClone(trace);metadataChanged.metadata.description='different label';assert.equal(await sha256(attentionBytes(metadataChanged)),browser);
});
test('common-reference mass has the analytical uniform-distribution value',()=>{
  const trace={schema_version:'1.0',query_heads:2,kv_heads:1,sink_size:0,attention:[Array(64).fill(1/64),Array(64).fill(1/64)]};
  const bundle=validateBundle(trace),run=analyzeBundle(bundle,{...DEFAULT_CONFIG,retention:25,blockSize:8});
  const g=run.observations[0].groups[0];for(const metric of Object.values(g.metrics))assert.equal(metric.reference_mass,.25);
  assert.equal(g.weighted_mean_jaccard,1);assert.equal(g.uor,1);
});
test('different page sizes compare original tokens, including partial final pages',()=>{
  const trace={schema_version:'1.0',query_heads:1,kv_heads:1,sink_size:0,attention:[Array(65).fill(1/65)]};
  const b=validateBundle(trace),a=analyzeBundle(b,{...DEFAULT_CONFIG,retention:100,blockSize:8}),c=analyzeBundle(b,{...DEFAULT_CONFIG,retention:100,blockSize:16});
  const rows=compareRuns(a,c);assert.equal(rows[0].jaccard,1);assert.equal(rows[0].intersection,65);assert.notEqual(rows[0].a_pages,rows[0].b_pages);
});
test('an unchanged run has identical results and token Jaccard one across every observation',()=>{
  const b=demoBundle(),a=analyzeBundle(b,DEFAULT_CONFIG),c=analyzeBundle(b,DEFAULT_CONFIG);
  assert.deepEqual(summarizeRun(a),summarizeRun(c));assert.ok(compareRuns(a,c).every(r=>r.jaccard===1));
  for(const o of a.observations)for(const g of o.groups){assert.equal(g.metrics.weighted.pages,o.quota);assert.equal(g.metrics.mean.pages,o.quota);assert.ok(g.metrics.independent.pages>=o.quota);}
});
test('full research exports survive validation and recomputation with matching result checksums',async()=>{
  const b=demoBundle(),run=analyzeBundle(b,DEFAULT_CONFIG),manifest=await researchManifest(b,[run],{code_revision:'test-code'});
  const loaded=await inspectImport(JSON.parse(JSON.stringify(manifest)));
  const recomputed=analyzeBundle(loaded.bundle,loaded.configs[0]);
  assert.equal(await jsonDigest([summarizeRun(recomputed)]),loaded.expected.results_sha256);
  assert.equal(manifest.provenance.code_revision,'test-code');assert.equal(manifest.input.observations.length,6);
});
test('changed attention, configuration, or stored result is detected on manifest import',async()=>{
  const b=demoBundle(),manifest=await researchManifest(b,[analyzeBundle(b,DEFAULT_CONFIG)]);
  const changed=structuredClone(manifest);changed.input.observations[0].trace.attention[0][8]+=.000001;changed.input.observations[0].trace.attention[0][9]-=.000001;await assert.rejects(()=>inspectImport(changed),/SHA-256/);
  const config=structuredClone(manifest);config.configs[0].retention=30;await assert.rejects(()=>inspectImport(config),/Configuration SHA/);
  const result=structuredClone(manifest);result.results[0].observations[0].quota=999;await assert.rejects(()=>inspectImport(result),/result checksum/);
});
test('per-observation capture checksums reject edited source data',async()=>{
  const b=demoBundle();b.observations[0].attention_sha256='0'.repeat(64);await assert.rejects(()=>researchManifest(b,[analyzeBundle(b,DEFAULT_CONFIG)]),/checksum mismatch/);
});
test('CSV exports include every policy and escape user observation IDs',()=>{
  const b=demoBundle();b.observations[0].id='layer,"one"';const run=analyzeBundle(b,DEFAULT_CONFIG),csv=runsCSV([run]);
  assert.equal(csv.trim().split('\r\n').length,1+6*2*3);assert.ok(csv.includes('"layer,""one"""'));assert.ok(csv.includes('independent'));assert.ok(csv.includes('head_mean_attention_mass'));
});
test('CSV user labels cannot become spreadsheet formulas',()=>{
  const b=demoBundle();b.observations[0].id='=1+1';const csv=runsCSV([analyzeBundle(b,DEFAULT_CONFIG)]);assert.ok(csv.includes(",'=1+1,"));assert.ok(!csv.includes(',=1+1,'));
});
