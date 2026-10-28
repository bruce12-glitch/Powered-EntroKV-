import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { makeFixture, validateTrace, collisionEntropy, analyze, stableTopK, pageMax, planMemory, fingerprint } from '../dist/engine.js';

test('fixture seed is reproducible and every row is normalized', () => {
  const a=makeFixture(42), b=makeFixture(42);
  assert.deepEqual(a,b); assert.notEqual(fingerprint(a),fingerprint(makeFixture(43)));
  assert.doesNotThrow(()=>validateTrace(a));
  for(const row of a.attention) assert.ok(Math.abs(row.reduce((s,v)=>s+v,0)-1)<1e-12);
});
test('entropy agrees with the closed form for uniform and concentrated distributions',()=>{
  assert.ok(Math.abs(collisionEntropy(Array(16).fill(1/16),0)-Math.log(16))<1e-9);
  assert.ok(Math.abs(collisionEntropy([1,...Array(15).fill(0)],0))<1e-10);
});
test('all-sink distributions use the finite zero-residual guard',()=>{
  const head=[1,...Array(15).fill(0)];
  assert.ok(Math.abs(collisionEntropy(head,4)+Math.log(1e-12))<1e-12);
  const t={schema_version:'1.0',query_heads:2,kv_heads:1,sink_size:4,attention:[head,Array(16).fill(1/16)]};
  const r=analyze(t,{blockSize:4,retention:25});
  assert.ok(r.groups[0].weights[0]<1e-15);
  assert.ok(r.groups[0].weights.every(Number.isFinite));
});
test('page maxima include a partially filled last page, and ties use ascending index',()=>{
  assert.deepEqual(pageMax([.1,.2,.3,.4,.5],2),[.2,.4,.5]);
  assert.deepEqual(stableTopK([1,1,1,1],2),[0,1]);
});
test('quota, normalization, and union bounds hold across seeds, page sizes and budgets',()=>{
  for(const seed of [0,42,500])for(const blockSize of [4,16,32,64])for(const retention of [5,20,51,100]){
    const t=makeFixture(seed),r=analyze(t,{blockSize,retention,tau:.5});
    for(const g of r.groups){
      assert.equal(new Set(g.selected).size,r.quota);
      assert.equal(g.selected.length,r.quota);
      assert.ok(g.selected.every(v=>v>=0&&v<r.pages));
      assert.ok(g.uor>=1&&g.uor<=6);
      assert.ok(Math.abs(g.weights.reduce((a,b)=>a+b,0)-1)<1e-12);
      assert.ok(Math.abs(g.pooled.reduce((a,b)=>a+b,0)-1)<1e-10);
    }
    assert.ok(r.crossGroupUnionRatio>=1&&r.crossGroupUnionRatio<=t.kv_heads);
  }
});
test('identical sibling heads reproduce equal weights and mean selections',()=>{
  const t=makeFixture(42);t.attention=t.attention.map(()=>t.attention[0].slice());
  const r=analyze(t);
  for(const g of r.groups){assert.deepEqual(g.selected,g.meanSelected);assert.equal(g.uor,1);assert.ok(g.weights.every(w=>Math.abs(w-1/6)<1e-14));}
});
test('a small quota stays explicit rather than returning an empty selection',()=>{
  const t={schema_version:'1.0',query_heads:1,kv_heads:1,sink_size:0,attention:[Array(17).fill(1/17)]};
  const r=analyze(t,{blockSize:16,retention:5});
  assert.equal(r.quota,1);assert.equal(r.realizedRetention,50);assert.equal(r.pages,2);
});
test('trace validation rejects bad GQA geometry, malformed rows and probability sums',()=>{
  const t=makeFixture();
  assert.throws(()=>validateTrace({...t,kv_heads:5}),/divisible/);
  assert.throws(()=>validateTrace({...t,attention:t.attention.slice(1)}),/exactly/);
  const bad=structuredClone(t);bad.attention[0][0]=-1;
  assert.throws(()=>validateTrace(bad),/nonnegative/);
  bad.attention[0][0]=Infinity;assert.throws(()=>validateTrace(bad),/finite/);
  bad.attention[0]=Array(1024).fill(0);assert.throws(()=>validateTrace(bad),/sum/);
  assert.throws(()=>validateTrace({...t,sink_size:1024}),/sink_size/);
});
test('invalid configuration values never produce non-finite output',()=>{
  const t=makeFixture();for(const config of [{tau:0},{tau:NaN},{retention:0},{retention:101},{blockSize:0}])assert.throws(()=>analyze(t,config));
});
test('memory estimate matches known dense FP16 GQA payload and page rounding',()=>{
  const c={layers:32,kvHeads:8,headDim:128,context:32768,batch:1,blockSize:16,bytesPerElement:2,retention:100};
  assert.equal(planMemory(c).fullBytes,4*2**30);
  const p=planMemory({...c,retention:20,context:32769});
  assert.equal(p.pages,2049);assert.equal(p.kept,409);
  assert.equal(p.retainedBytes,2*32*8*128*16*409*2);
  assert.equal(planMemory({...c,batch:2}).fullBytes,2*planMemory(c).fullBytes);
  assert.equal(planMemory({...c,bytesPerElement:.5}).fullBytes,planMemory(c).fullBytes/4);
});
test('Python and browser reference agree on masks and weights for an exported trace',()=>{
  const trace=makeFixture(42),config={blockSize:16,retention:20,tau:.5,isolate:true};
  const browser=analyze(trace,config);
  const script='import sys,json; sys.path.insert(0,"dist/assets"); from page_entrokv import select_pages; print(json.dumps(select_pages(json.load(sys.stdin))))';
  const run=spawnSync('python3',['-B','-c',script],{input:JSON.stringify(trace),encoding:'utf8'});
  assert.equal(run.status,0,run.stderr);const python=JSON.parse(run.stdout);
  assert.equal(python.quota,browser.quota);
  for(let g=0;g<browser.groups.length;g++){
    assert.deepEqual(python.groups[g].selected,browser.groups[g].selected);
    assert.deepEqual(python.groups[g].meanSelected,browser.groups[g].meanSelected);
    assert.deepEqual(python.groups[g].independentUnion,browser.groups[g].independentUnion);
    python.groups[g].weights.forEach((w,h)=>assert.ok(Math.abs(w-browser.groups[g].weights[h])<1e-12));
  }
});
