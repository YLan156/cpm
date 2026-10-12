"""Synthetic plumbing tests only; never research results."""
import unittest,tempfile,json,subprocess,sys,csv
from pathlib import Path
R=Path(__file__).resolve().parents[1]
class SummaryTests(unittest.TestCase):
 def make(self,root):
  d=root/'shard_0';d.mkdir();ids=[json.loads(x)['memory_id'] for x in (R/'data/memories.jsonl').read_text().splitlines()][:6];rows=[];cc=[]
  for e in ['anxious','calm']:
   for m in ['A_goal','B_joint','C_mechanism']:rows.append({'scenario_id':'synthetic','split':'exploratory_test','goal':'action','emotion':e,'method':m,'valid':True,'anchor_in_pool':1,'anchor_hit3':1,'top3':ids[:3],'candidate_order':ids,'request':'SYNTHETIC TEST ONLY'})
   for contrast in ['B_joint_vs_A_goal','C_mechanism_vs_B_joint']:cc.append({'split':'exploratory_test','goal':'action','contrast':contrast,'both_valid':True,'set_changed':0})
  (d/'rankings.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));(d/'validation.json').write_text('{}');(d/'config.json').write_text(json.dumps({'model':{'repository':'SYNTHETIC','revision':'TEST'},'quantization':'test'}))
  with (d/'contrasts.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(cc[0]));w.writeheader();w.writerows(cc)
  return d
 def test_reports_and_blind_pairs(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);self.make(p);x=subprocess.run([sys.executable,str(R/'src/summarize.py'),str(p)],capture_output=True,text=True);self.assertEqual(x.returncode,0,x.stderr);self.assertEqual(len(json.loads((p/'blind_comparisons.json').read_text())),4);self.assertTrue((p/'report.md').exists())
 def test_duplicate_shard_rejected(self):
  import shutil
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);d=self.make(p);shutil.copytree(d,p/'duplicate');x=subprocess.run([sys.executable,str(R/'src/summarize.py'),str(p)],capture_output=True,text=True);self.assertNotEqual(x.returncode,0);self.assertIn('Duplicate',x.stderr)
if __name__=='__main__':unittest.main()
