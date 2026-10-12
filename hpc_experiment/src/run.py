"""Local small-LLM feasibility run; no strong-baseline or helpfulness claims."""
from pathlib import Path
import os
R=Path(__file__).resolve().parents[1];os.environ.setdefault('HF_HOME',str(R/'.hf_cache'))
import json,time,hashlib,re,random,csv
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
P=R/'data';O=Path(os.environ.get('CPM_OUTPUT_DIR',str(R/'results/pilot')));O.mkdir(parents=True,exist_ok=True)
def read(p):return [json.loads(x) for x in p.read_text().splitlines()]
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sc=read(P/'scenarios.jsonl');chosen=[next(s for s in sc if s['split']=='dev')];seen=set()
for s in sc:
 if s['split']=='exploratory_test' and s['conversation_id'] not in seen:
  chosen.append(s);seen.add(s['conversation_id'])
  if len(chosen)==3:break
if os.environ.get('CPM_SCOPE','pilot')=='all':chosen=sc
shard=int(os.environ.get('CPM_SHARD_ID','0'));nshards=int(os.environ.get('CPM_SHARDS','1'))
assert 0<=shard<nshards
chosen=chosen[shard::nshards]
if not chosen:raise SystemExit('No scenarios assigned; reduce shard count.')
old=read(P/'rankings.jsonl');mem={m['memory_id']:m for m in read(P/'memories.jsonl')}
pools={s['scenario_id']:next(r for r in old if r['scenario_id']==s['scenario_id'] and r['wording']=='v1' and r['goal']=='understanding' and r['emotion']=='anxious' and r['method']=='event')['top10'][:6] for s in chosen}
for s in chosen:
 for d in pools[s['scenario_id']]:
  m=mem[d];assert m['speaker']==s['person'] and m['conversation_id']==s['conversation_id'] and int(m['session_id'].split('_')[1])<=int(s['cutoff_session'].split('_')[1])
modeldir=Path(os.environ.get('CPM_LLM_MODEL_DIR',str(R/'models/instruct')));prov=json.loads((modeldir/'provenance.json').read_text())
for name,info in prov['files'].items():
 if name.endswith('.safetensors'):assert sha(modeldir/name)==info['sha256'],name
gpu=torch.cuda.is_available();dtype=torch.bfloat16 if gpu and torch.cuda.is_bf16_supported() else (torch.float16 if gpu else torch.float32)
config={'seed':20261012,'model':prov,'quantization':str(dtype)+(' CUDA' if gpu else ' CPU'),'selection':{'scope':os.environ.get('CPM_SCOPE','pilot'),'shard':shard,'shards':nshards,'policy':'pilot: first dev plus first test from two conversations; all: original 20; no outcome selection'},'scenarios':[s['scenario_id'] for s in chosen],'pool':'first six event-only understanding retrieval results, shared across all goals; no anchor injection','returned':3,'temperature':'greedy','max_new_tokens_extraction':256,'max_new_tokens_ranking':96,'new_training':False,'gold':'none; no helpfulness scoring','source_sha256':sha(P/'rankings.jsonl'),'script_sha256':sha(Path(__file__))};save('config.json',config);save('candidate_pools.json',pools)
torch.manual_seed(config['seed']);torch.set_num_threads(4);start=time.time()
tok=AutoTokenizer.from_pretrained(modeldir,local_files_only=True);tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(modeldir,local_files_only=True,torch_dtype=dtype).eval()
if torch.cuda.is_available(): model=model.to('cuda')
# CPU remains float32: dynamic int8 failed the local extraction diagnostic.
log=(O/'model_calls.jsonl').open('w');count=0
# Exact-input cache permits interruption recovery without silently changing prompts/model.
cache={}
cachefile=O/'generation_cache.jsonl'
if cachefile.exists():
 for x in read(cachefile):
  if x.get('model_repository')==prov['repository'] and x.get('model_revision')==prov['revision'] and x.get('quantization')==config['quantization']:cache[x['key']]=x
cf=cachefile.open('a')
def generate(prompt,limit,kind):
 global count
 key=hashlib.sha256((prompt+'|'+str(limit)).encode()).hexdigest()
 if key in cache:x=cache[key];out=x['output']
 else:
  inputs=tok.apply_chat_template([{'role':'system','content':'Follow the requested format. Use only the supplied evidence.'},{'role':'user','content':prompt}],add_generation_prompt=True,return_tensors='pt').to('cuda' if torch.cuda.is_available() else 'cpu')
  t=time.time()
  with torch.inference_mode():v=model.generate(inputs,attention_mask=torch.ones_like(inputs),max_new_tokens=limit,do_sample=False,temperature=None,top_p=None,top_k=None,pad_token_id=tok.eos_token_id)
  out=tok.decode(v[0,inputs.shape[1]:],skip_special_tokens=True)
  x={'key':key,'prompt':prompt,'output':out,'model_repository':prov['repository'],'model_revision':prov['revision'],'quantization':config['quantization'],'seconds':time.time()-t,'input_tokens':inputs.shape[1],'new_tokens':v.shape[1]-inputs.shape[1]};cf.write(json.dumps(x)+'\n');cf.flush();cache[key]=x
 count+=1;log.write(json.dumps({**x,'kind':kind})+'\n');log.flush();print(json.dumps({'call':count,'kind':kind,'seconds':round(time.time()-start,1)}),flush=True);return out
def parse(raw):
 try:return json.loads(raw[raw.index('{'):raw.rindex('}')+1])
 except (ValueError,TypeError):return None
extract_prompt='''Extract evidence from the passage. Return ONLY a JSON object with keys event, emotion, action, outcome. Each value must be one short EXACT quote copied from the passage, or null. Only record actions that actually happened, not future plans or advice. Do not infer success or causality. Do not change who felt or did something.\nPassage: '''
ann={}
for d in sorted({d for ids in pools.values() for d in ids}):
 text=mem[d]['original_text'];raw=generate(extract_prompt+text,256,'extraction');obj=parse(raw);fields={}
 for k in ['event','emotion','action','outcome']:
  q=obj.get(k) if isinstance(obj,dict) else None
  valid=isinstance(q,str) and bool(q) and q in text
  fields[k]={'value':q if valid else 'unknown','evidence_status':'exact_quote_unverified_meaning' if valid else 'unknown','start':text.index(q) if valid else None,'rejected_output':q if q is not None and not valid else None}
 ann[d]={'memory_id':d,'status':'AI-provisional','source_text':text,'fields':fields,'raw_output':raw,'parse_valid':isinstance(obj,dict),'prompt_version':'exact-quote-v1','model_revision':prov['revision']}
(O/'annotations.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in ann.values()))
goals={'fact':'Answer the factual question accurately using evidence.','understanding':'I want my feelings acknowledged and understood, not advice.','action':'I want a practical next step based on my past experience.'}
rows=[];outputs=[];pairs=[]
for s in chosen:
 ids=pools[s['scenario_id']].copy();random.Random(config['seed']+int(s['scenario_id'][2:])).shuffle(ids)
 candidates='\n'.join(json.dumps({'id':i+1,'original':mem[d]['original_text'],'evidence':{k:f['value'] for k,f in ann[d]['fields'].items()}},ensure_ascii=False) for i,d in enumerate(ids))
 for goal in goals:
  for emotion in ['anxious','calm']:
   q=f"I am thinking about {s['topic_en']}. "+(s['fact_question'] if goal=='fact' else '')
   for method in ['A_goal','B_joint','C_mechanism']:
    prompt=f'Current request: {q}\nPurpose: {goals[goal]}\n'
    if method!='A_goal':prompt+=f'Current emotional state: {emotion}.\n'
    prompt+='Select the three memories most useful for this request. All candidate evidence fields are provisional; verify against original text. Do not invent events.\n'
    if method=='C_mechanism':prompt+='Selection policy: For factual recall, prioritize direct evidence and ignore emotional appeal. For feeling understood, consider relevant similar feelings. For practical action, consider relevant past actions and their observed outcomes, including failures; happy memories are not automatically useful. Current emotion may affect usefulness but must not replace event relevance. Missing evidence must stay unknown.\n'
    prompt+='Candidates:\n'+candidates+'\nReturn ONLY JSON: {"ranking": [id, id, id]}. Use three distinct candidate IDs in preferred order.'
    raw=generate(prompt,96,'ranking');obj=parse(raw);rank=obj.get('ranking') if isinstance(obj,dict) else None
    valid=isinstance(rank,list) and len(rank)==3 and all(type(i)==int and 1<=i<=len(ids) for i in rank) and len(set(rank))==3
    top=[ids[i-1] for i in rank] if valid else []
    row={'scenario_id':s['scenario_id'],'split':s['split'],'conversation_id':s['conversation_id'],'goal':goal,'emotion':emotion,'method':method,'valid':valid,'anchor_in_pool':int(s['anchor_memory_id'] in ids),'anchor_hit3':int(s['anchor_memory_id'] in top) if valid else None}
    rows.append(row);outputs.append({**row,'top3':top,'candidate_order':ids,'raw_output':raw,'request':q})
(O/'rankings.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in outputs))
with (O/'per_query.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lookup={(r['scenario_id'],r['goal'],r['emotion'],r['method']):r for r in outputs}
for s in chosen:
 for goal in goals:
  for emotion in ['anxious','calm']:
   for a,b in [('A_goal','B_joint'),('B_joint','C_mechanism')]:
    x=lookup[s['scenario_id'],goal,emotion,a];y=lookup[s['scenario_id'],goal,emotion,b]
    pairs.append({'scenario_id':s['scenario_id'],'split':s['split'],'goal':goal,'emotion':emotion,'contrast':b+'_vs_'+a,'both_valid':x['valid'] and y['valid'],'set_changed':int(set(x['top3'])!=set(y['top3'])) if x['valid'] and y['valid'] else None})
with (O/'contrasts.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
validation={'status':'completed','calls_including_cache':count,'unique_generations':len(cache),'seconds':time.time()-start,'annotations':len(ann),'valid_json_annotations':sum(x['parse_valid'] for x in ann.values()),'rejected_quote_fields':sum(f['rejected_output'] is not None for x in ann.values() for f in x['fields'].values()),'rankings':len(rows),'valid_rankings':sum(x['valid'] for x in rows),'all_candidates_same_person_and_time':True,'human_gold_added':0,'helpfulness':'not evaluated','torch_version':torch.__version__};save('validation.json',validation);print(json.dumps(validation),flush=True)
