from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[1]
def read(n):return [json.loads(x) for x in (R/'data'/n).read_text().splitlines()]
sc=read('scenarios.jsonl');rr=read('rankings.jsonl');mem={x['memory_id']:x for x in read('memories.jsonl')};manifest=json.loads((R/'data/manifest.json').read_text())
for name,h in manifest.items():assert hashlib.sha256((R/'data'/name).read_bytes()).hexdigest()==h,name
splits={}
for s in sc:
 cid=s['conversation_id'];assert cid not in splits or splits[cid]==s['split'];splits[cid]=s['split']
 row=[x for x in rr if x['scenario_id']==s['scenario_id']];assert len(row)==1
 for d in row[0]['top10']:
  m=mem[d];assert m['speaker']==s['person'] and m['conversation_id']==cid;assert int(m['session_id'].split('_')[1])<=int(s['cutoff_session'].split('_')[1])
print(f'PASS: {len(sc)} scenarios, {len(mem)} memories; hashes, dialogue splits, person/time checked.')
