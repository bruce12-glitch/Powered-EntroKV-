import { ENGINE_VERSION, analyze, makeFixture, validateTrace } from './engine.js';

export const BUNDLE_VERSION = '2.0';
export const MAX_BUNDLE_ENTRIES = 1048576;
export const MAX_IMPORT_BYTES = 32 * 1024 * 1024;
export const DEFAULT_CONFIG = { retention: 20, blockSize: 16, tau: 0.5, isolate: true };
export const DEFAULT_COMPARISON = { retention: 40, blockSize: 16, tau: 0.5, isolate: true };

function metadata(value, depth=0) {
  if(depth>6) throw new Error('Metadata is nested too deeply.');
  if(value===null||typeof value==='boolean')return value;
  if(typeof value==='string')return value.slice(0,2000);
  if(typeof value==='number'){if(!Number.isFinite(value))throw new Error('Metadata numbers must be finite.');return value;}
  if(Array.isArray(value)){if(value.length>2048)throw new Error('Metadata array exceeds 2048 entries.');return value.map(v=>metadata(v,depth+1));}
  if(value&&typeof value==='object'){
    if(Object.keys(value).length>128)throw new Error('Metadata object has too many fields.');
    return Object.fromEntries(Object.entries(value).map(([key,v])=>[key.slice(0,100),metadata(v,depth+1)]));
  }
  throw new Error('Metadata must contain JSON values.');
}

export function validateBundle(raw) {
  if(raw?.application==='Page EntroKV'&&raw.schema_version==='1.0'&&raw.input)raw=raw.input;
  if(raw?.schema_version==='1.0'&&raw.attention)raw={schema_version:BUNDLE_VERSION,kind:'attention_bundle',name:'Imported observation',source:'imported',metadata:raw.metadata??{},observations:[{id:'observation-0',layer:0,trace:raw}]};
  if(!raw||raw.schema_version!==BUNDLE_VERSION||raw.kind!=='attention_bundle')throw new Error('Expected a v2 attention bundle, a v1 trace, or a complete research manifest.');
  if(!Array.isArray(raw.observations)||raw.observations.length<1||raw.observations.length>128)throw new Error('A bundle must contain 1–128 layer observations.');
  let entries=0;const ids=new Set();let sharedTokens=null;
  const observations=raw.observations.map((o,index)=>{
    if(!o||typeof o.id!=='string'||!o.id.trim()||o.id.length>100||ids.has(o.id))throw new Error(`Observation ${index}: IDs must be unique strings, at most 100 characters.`);
    ids.add(o.id);
    if(!Number.isSafeInteger(o.layer)||o.layer<0||o.layer>255)throw new Error(`Observation ${index}: layer must be an integer from 0 to 255.`);
    const h=o.trace?.attention?.length,t=o.trace?.attention?.[0]?.length;
    if(!Number.isSafeInteger(h)||!Number.isSafeInteger(t))throw new Error(`Observation ${index}: missing attention rows.`);
    entries+=h*t;if(entries>MAX_BUNDLE_ENTRIES)throw new Error('Bundle exceeds 1,048,576 attention entries. Capture fewer layers or tokens.');
    const trace=validateTrace(o.trace);delete trace.source;delete trace.seed;
    if(sharedTokens===null)sharedTokens=t;
    if(t!==sharedTokens)throw new Error('All observations must use the same token positions and context length. Use separate bundles for different prompts.');
    const result={id:o.id,layer:o.layer,trace};
    if(o.attention_sha256!==undefined){if(!/^[a-f0-9]{64}$/.test(o.attention_sha256))throw new Error('attention_sha256 must be a lowercase SHA-256 hex digest.');result.attention_sha256=o.attention_sha256;}
    return result;
  });
  const source=['synthetic','model_capture','random_weight_model','imported'].includes(raw.source)?raw.source:'imported';
  const result={schema_version:BUNDLE_VERSION,kind:'attention_bundle',name:String(raw.name??'Untitled bundle').slice(0,160),source,metadata:metadata(raw.metadata??{}),observations};
  if(raw.token_ids!==undefined){if(!Array.isArray(raw.token_ids)||raw.token_ids.length!==sharedTokens||raw.token_ids.some(x=>!Number.isSafeInteger(x)||x<0))throw new Error('token_ids must contain one nonnegative integer per context position.');result.token_ids=raw.token_ids.slice();}
  if(raw.token_text!==undefined){if(!Array.isArray(raw.token_text)||raw.token_text.length!==sharedTokens||raw.token_text.some(x=>typeof x!=='string'||x.length>256))throw new Error('token_text must contain one string of at most 256 characters per context position.');result.token_text=raw.token_text.slice();}
  return result;
}

export function demoBundle() {
  const observations=Array.from({length:6},(_,layer)=>{const trace=makeFixture(42+layer*17);trace.metadata.layer=String(layer);return{id:`demo-layer-${layer}`,layer,trace};});
  return validateBundle({schema_version:BUNDLE_VERSION,kind:'attention_bundle',name:'Six-layer GQA study',source:'synthetic',metadata:{model:'Seeded synthetic observations',seed:42,description:'Six independently seeded fixtures. These do not represent a model forward pass.'},observations});
}

export function validateConfig(raw) {
  if(!raw||typeof raw!=='object')throw new Error('Missing selection configuration.');
  const result={retention:raw.retention,blockSize:raw.blockSize,tau:raw.tau,isolate:raw.isolate};
  if(typeof result.isolate!=='boolean')throw new Error('Sink isolation must be a boolean.');
  if(typeof result.retention!=='number'||!Number.isFinite(result.retention)||result.retention<=0||result.retention>100)throw new Error('Retention must be in (0, 100].');
  if(!Number.isSafeInteger(result.blockSize)||result.blockSize<1||result.blockSize>512)throw new Error('Page size must be an integer in [1, 512].');
  if(typeof result.tau!=='number'||!Number.isFinite(result.tau)||result.tau<.05||result.tau>10)throw new Error('Temperature must be in [0.05, 10].');
  return result;
}

export function retainedPositions(pages,blockSize,tokens) {
  const set=new Set();for(const page of pages)for(let t=page*blockSize;t<Math.min((page+1)*blockSize,tokens);t++)set.add(t);return set;
}
export function tokenJaccard(a,b,tokens) {
  const aa=retainedPositions(a.selected,a.blockSize,tokens),bb=retainedPositions(b.selected,b.blockSize,tokens);
  let intersection=0;for(const t of aa)if(bb.has(t))intersection++;
  return {intersection,union:aa.size+bb.size-intersection,jaccard:intersection/(aa.size+bb.size-intersection||1)};
}
function massInPages(mean,pages,blockSize){let mass=0;for(const page of pages)for(let t=page*blockSize;t<Math.min((page+1)*blockSize,mean.length);t++)mass+=mean[t];return mass;}

export function analyzeBundle(bundle,rawConfig) {
  const config=validateConfig(rawConfig);
  const observations=bundle.observations.map(o=>{
    const r=analyze(o.trace,config),ratio=o.trace.query_heads/o.trace.kv_heads;
    const groups=r.groups.map(g=>{
      const mean=Array.from({length:r.tokens},(_,t)=>o.trace.attention.slice(g.group*ratio,(g.group+1)*ratio).reduce((s,row)=>s+row[t]/ratio,0));
      const policies={weighted:g.selected,mean:g.meanSelected,independent:g.independentUnion};
      const metrics=Object.fromEntries(Object.entries(policies).map(([name,selection])=>[name,{pages:selection.length,reference_mass:massInPages(mean,selection,config.blockSize),realized_retention:selection.length/r.pages*100}]));
      const overlap=tokenJaccard({selected:g.selected,blockSize:config.blockSize},{selected:g.meanSelected,blockSize:config.blockSize},r.tokens);
      return {...g,metrics,weighted_mean_jaccard:overlap.jaccard};
    });
    return {id:o.id,layer:o.layer,tokens:r.tokens,pages:r.pages,quota:r.quota,realizedRetention:r.realizedRetention,groups,crossGroupUnionRatio:r.crossGroupUnionRatio,layerUnion:r.layerUnion};
  });
  return {engine_version:ENGINE_VERSION,config,observations,metric_definition:'reference_mass is the retained mass of the same arithmetic mean of raw sibling attention rows for every policy. It is not task accuracy.',budget_definition:'Weighted and mean pooling share an identical page quota. Independent-head union is an over-budget diagnostic.'};
}

export function summarizeRun(run) {
  return {engine_version:run.engine_version,config:run.config,metric_definition:run.metric_definition,budget_definition:run.budget_definition,observations:run.observations.map(o=>({id:o.id,layer:o.layer,tokens:o.tokens,pages:o.pages,quota:o.quota,realizedRetention:o.realizedRetention,crossGroupUnionRatio:o.crossGroupUnionRatio,layerUnion:o.layerUnion,groups:o.groups.map(g=>({group:g.group,entropies:g.entropies,weights:g.weights,selected:g.selected,meanSelected:g.meanSelected,independentUnion:g.independentUnion,uor:g.uor,metrics:g.metrics,weighted_mean_jaccard:g.weighted_mean_jaccard}))}))};
}

export function compareRuns(a,b) {
  if(a.observations.length!==b.observations.length)throw new Error('Runs use different observations.');
  return a.observations.flatMap((o,i)=>{
    const other=b.observations[i];if(o.id!==other.id||o.tokens!==other.tokens||o.groups.length!==other.groups.length)throw new Error('Runs are not aligned on the same input.');
    return o.groups.map((g,j)=>({observation:o.id,layer:o.layer,group:j,a_mass:g.metrics.weighted.reference_mass,b_mass:other.groups[j].metrics.weighted.reference_mass,a_pages:g.selected.length,b_pages:other.groups[j].selected.length,...tokenJaccard({selected:g.selected,blockSize:a.config.blockSize},{selected:other.groups[j].selected,blockSize:b.config.blockSize},o.tokens)}));
  });
}

// This wire format is identical in Python: magic UTF-8, four uint32 LE fields,
// then head-major IEEE754 float64 LE probabilities. Metadata and token labels are excluded.
export function attentionBytes(trace) {
  const magic=new TextEncoder().encode('PageEntroKV/attention-v2\0'),tokens=trace.attention[0].length;
  const data=new Uint8Array(magic.length+16+8*trace.query_heads*tokens);data.set(magic);
  const view=new DataView(data.buffer);let offset=magic.length;
  for(const v of [trace.query_heads,trace.kv_heads,trace.sink_size,tokens]){view.setUint32(offset,v,true);offset+=4;}
  for(const row of trace.attention)for(const v of row){view.setFloat64(offset,v,true);offset+=8;}
  return data;
}
export async function sha256(bytes) {
  if(!globalThis.crypto?.subtle)throw new Error('SHA-256 needs HTTPS or localhost. Use the published workspace or serve locally at http://localhost:8080.');
  return [...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(b=>b.toString(16).padStart(2,'0')).join('');
}
export function canonicalJSON(value) {
  if(value===null||typeof value==='string'||typeof value==='boolean')return JSON.stringify(value);
  if(typeof value==='number'){if(!Number.isFinite(value))throw new Error('Cannot hash a non-finite number.');return JSON.stringify(value);}
  if(Array.isArray(value))return `[${value.map(canonicalJSON).join(',')}]`;
  if(value&&typeof value==='object')return `{${Object.keys(value).sort().map(key=>`${JSON.stringify(key)}:${canonicalJSON(value[key])}`).join(',')}}`;
  throw new Error('Cannot hash a non-JSON value.');
}
export const jsonDigest=value=>sha256(new TextEncoder().encode(canonicalJSON(value)));
export async function bundleChecksums(bundle) {
  const observations=[];
  for(const o of bundle.observations){const digest=await sha256(attentionBytes(o.trace));if(o.attention_sha256&&o.attention_sha256!==digest)throw new Error(`Attention checksum mismatch in ${o.id}. The observation has changed.`);observations.push({id:o.id,attention_sha256:digest});}
  return {attention_format:'PageEntroKV/attention-v2: uint32-le header + float64-le rows',observations,bundle_sha256:await jsonDigest(bundle)};
}
export async function researchManifest(bundle,runs,release={}) {
  const summaries=runs.map(summarizeRun);
  return {schema_version:BUNDLE_VERSION,kind:'page_entrokv_research_manifest',application:'Page EntroKV',engine_version:ENGINE_VERSION,created_at:new Date().toISOString(),input:bundle,configs:runs.map(r=>r.config),results:summaries,checksums:{...await bundleChecksums(bundle),configs_sha256:await jsonDigest(runs.map(r=>r.config)),results_sha256:await jsonDigest(summaries)},provenance:{arithmetic:'JavaScript Float64',code_revision:release.code_revision??null,source_repository:'https://github.com/bruce12-glitch/Powered-EntroKV-',model_inference_in_browser:false,authenticity:'Checksums detect changes, not uploader identity or truth of model metadata.'}};
}
export async function inspectImport(raw) {
  if(raw?.application==='Page EntroKV'&&raw.schema_version==='1.0'&&raw.input)return {bundle:validateBundle(raw),configs:[validateConfig(raw.config)],expected:null};
  if(raw?.kind!=='page_entrokv_research_manifest')return {bundle:validateBundle(raw),configs:null,expected:null};
  if(raw.schema_version!==BUNDLE_VERSION||!Array.isArray(raw.configs)||raw.configs.length<1||raw.configs.length>12)throw new Error('A research manifest must contain 1–12 configurations.');
  const bundle=validateBundle(raw.input),configs=raw.configs.map(validateConfig);
  const checksums=await bundleChecksums(bundle);
  if(checksums.bundle_sha256!==raw.checksums?.bundle_sha256)throw new Error('Bundle SHA-256 does not match the manifest. Import the source bundle separately if you intentionally edited it.');
  if(await jsonDigest(configs)!==raw.checksums?.configs_sha256)throw new Error('Configuration SHA-256 does not match the manifest.');
  if(await jsonDigest(raw.results)!==raw.checksums?.results_sha256)throw new Error('Stored result checksum does not match the manifest.');
  return {bundle,configs,expected:{engine_version:raw.engine_version,results_sha256:raw.checksums.results_sha256,results:raw.results}};
}

function csvCell(value){let s=String(value);if(typeof value==='string'&&/^[=+\-@\t\r]/.test(s))s="'"+s;return /[",\n\r]/.test(s)?`"${s.replaceAll('"','""')}"`:s;}
export function runsCSV(runs) {
  const fields=['run','observation','layer','group','policy','retention_requested_percent','page_size','tau','sink_isolation','quota_pages','retained_pages','realized_retention_percent','head_mean_attention_mass','independent_union_overhead','weighted_mean_token_jaccard'];
  const rows=[fields];runs.forEach((r,i)=>r.observations.forEach(o=>o.groups.forEach(g=>Object.entries(g.metrics).forEach(([policy,m])=>rows.push([i+1,o.id,o.layer,g.group,policy,r.config.retention,r.config.blockSize,r.config.tau,r.config.isolate,o.quota,m.pages,m.realized_retention,m.reference_mass,g.uor,g.weighted_mean_jaccard])))));
  return rows.map(row=>row.map(csvCell).join(',')).join('\r\n')+'\r\n';
}
