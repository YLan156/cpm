"""Tiny constructed capability checks, NOT research gold or exhaustive quality assessment."""
import json,os,sys
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
R=Path(__file__).resolve().parents[1];root=Path(os.environ.get('CPM_LLM_MODEL_DIR',str(R/'models/instruct')));out=Path(os.environ.get('CPM_OUTPUT_DIR',str(R/'results/pilot')));out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);gpu=torch.cuda.is_available();dtype=torch.bfloat16 if gpu and torch.cuda.is_bf16_supported() else torch.float16 if gpu else torch.float32
print({'cuda':gpu,'gpu':torch.cuda.get_device_name(0) if gpu else None,'dtype':str(dtype)},flush=True)
t=AutoTokenizer.from_pretrained(root,local_files_only=True);m=AutoModelForCausalLM.from_pretrained(root,local_files_only=True,torch_dtype=dtype).eval()
if gpu:m=m.to('cuda')
def ask(q):
 x=t.apply_chat_template([{'role':'system','content':'Follow the requested format. Use only the supplied evidence.'},{'role':'user','content':q}],add_generation_prompt=True,return_tensors='pt').to('cuda' if gpu else 'cpu')
 with torch.inference_mode():y=m.generate(x,attention_mask=torch.ones_like(x),max_new_tokens=256,do_sample=False,temperature=None,top_p=None,top_k=None,pad_token_id=t.eos_token_id)
 return t.decode(y[0,x.shape[1]:],skip_special_tokens=True)
cases=[('I felt anxious. I called my friend. I felt calmer afterward.',True),('I will call my friend tomorrow. I hope it helps.',False)]
rows=[]
for text,action_expected in cases:
 raw=ask('Extract event, emotion, action, outcome. Return ONLY JSON with these four keys. Each value must be a short EXACT quote from the passage or null. Action means actually performed, not planned. Outcome means observed, not hoped for. Passage: '+text)
 try:
  obj=json.loads(raw[raw.index('{'):raw.rindex('}')+1]);valid=isinstance(obj,dict) and set(obj)=={'event','emotion','action','outcome'} and all(v is None or isinstance(v,str) and bool(v) and v in text for v in obj.values());correct=valid and (obj['action'] is not None if action_expected else obj['action'] is None and obj['outcome'] is None)
 except (ValueError,TypeError,KeyError):valid=False;correct=False
 rows.append({'text':text,'raw':raw,'format_and_quotes':valid,'minimal_action_check':correct});print(rows[-1],flush=True)
result={'passed':all(x['minimal_action_check'] for x in rows),'cases':rows,'scope':'two constructed software/capability probes only; does not certify general extraction accuracy','model':json.loads((root/'provenance.json').read_text()),'dtype':str(dtype)}
(out/'smoke.json').write_text(json.dumps(result,indent=2));sys.exit(0 if result['passed'] else 2)
