import { analyzeBundle, validateBundle, validateConfig } from './research.js';
self.onmessage=event=>{
  const {id,bundle,configs}=event.data;
  try {
    if(!Array.isArray(configs)||configs.length<1||configs.length>12)throw new Error('Choose 1–12 configurations.');
    const input=validateBundle(bundle),runs=[];
    configs.forEach((c,index)=>{runs.push(analyzeBundle(input,validateConfig(c)));self.postMessage({id,progress:(index+1)/configs.length});});
    self.postMessage({id,runs});
  } catch(error){self.postMessage({id,error:error.message});}
};
