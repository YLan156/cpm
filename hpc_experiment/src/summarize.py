import argparse,json,csv,random
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
files=list(a.root.rglob('validation.json'));allrows=[];contrasts=[];configs=[]
for vf in files:
 d=vf.parent
 if not (d/'rankings.jsonl').exists():continue
 cfg=json.loads((d/'config.json').read_text());configs.append(cfg)
 rows=[json.loads(x) for x in (d/'rankings.jsonl').read_text().splitlines()]
 for x in rows:
  if x['valid']:assert len(set(x['top3']))==3 and set(x['top3'])<=set(x['candidate_order'])
 allrows+=rows;contrasts+=pd.read_csv(d/'contrasts.csv').to_dict('records')
if not allrows:raise SystemExit('No completed runs. Inspect logs; incomplete runs are not results.')
models={(c['model']['repository'],c['model']['revision'],c['quantization']) for c in configs}
if len(models)!=1:raise SystemExit('Mixed models/dtypes: summarize each model run separately.')
keys=[(r['scenario_id'],r['goal'],r['emotion'],r['method']) for r in allrows]
if len(keys)!=len(set(keys)):raise SystemExit('Duplicate scenario conditions; do not merge pilot and all runs.')
f=pd.DataFrame(allrows);g=f.groupby(['split','goal','method']).agg(conditions=('valid','size'),valid_outputs=('valid','sum'),anchor_in_pool=('anchor_in_pool','mean'),anchor_hit3_valid_only=('anchor_hit3','mean')).reset_index();g.to_csv(a.root/'summary.csv',index=False)
c=pd.DataFrame(contrasts);c.groupby(['split','goal','contrast']).agg(pairs=('both_valid','size'),valid_pairs=('both_valid','sum'),set_change_valid_only=('set_changed','mean')).reset_index().to_csv(a.root/'contrasts_summary.csv',index=False)
# Blind all nonfact test comparisons, including identical sets; no automatic gold.
R=Path(__file__).resolve().parents[1];mem={x['memory_id']:x for x in map(json.loads,(R/'data/memories.jsonl').read_text().splitlines())};lookup={(r['scenario_id'],r['goal'],r['emotion'],r['method']):r for r in allrows};pack=[];key=[];rng=random.Random(20261012)
for r in allrows:
 if r['split']!='exploratory_test' or r['goal']=='fact' or r['method']!='A_goal':continue
 for aa,bb in [('A_goal','B_joint'),('B_joint','C_mechanism')]:
  x=lookup[r['scenario_id'],r['goal'],r['emotion'],aa];y=lookup[r['scenario_id'],r['goal'],r['emotion'],bb]
  if not(x['valid'] and y['valid']):continue
  methods=[aa,bb];rng.shuffle(methods);iid=f'R{len(pack)+1:03d}';row={'id':iid,'request':r['request'],'goal':r['goal'],'emotion':r['emotion'],'rating':None,'status':'unreviewed'}
  for side,m in zip(['left','right'],methods):row[side]=[mem[d] for d in lookup[r['scenario_id'],r['goal'],r['emotion'],m]['top3']]
  pack.append(row);key.append({'id':iid,'left':methods[0],'right':methods[1]})
(a.root/'blind_comparisons.json').write_text(json.dumps(pack,ensure_ascii=False,indent=2));(a.root/'review_key.json').write_text(json.dumps(key,indent=2))
(a.root/'report.md').write_text(f'''# LLM三组对照运行结果

已完成{len(files)}个分片、{len(allrows)}份排序，覆盖{f.scenario_id.nunique()}个情境。模型：{next(iter(models))}。

A=目的；B=目的+情绪；C=相同信息+用途选择策略。完整统计在summary.csv与contrasts_summary.csv。候选池由原事件检索构造，三组相同。目的和情绪由实验给定。

排序变化不是帮助性改善。anchor_hit3只是构造证据代理，只应在fact条件解释；无效模型输出单独计数，未作为成功结果。多种条件重复使用相同情境，不是独立样本。未生成回答、未测量心理疗效、未新增人工标签。

匿名对照保存在blind_comparisons.json，方法映射独立保存；需要独立判断帮助性后才能比较方法好坏。不得将抽取字段或模型自己的排序当金标准。当前GPU执行环境记录在各分片日志及配置中。
''');print(g.to_string(index=False))
