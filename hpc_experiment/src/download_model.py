import argparse,json,hashlib
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
p=argparse.ArgumentParser();p.add_argument('--repo',default='Qwen/Qwen2.5-7B-Instruct');p.add_argument('--revision',default='main');p.add_argument('--output',default='models/instruct');a=p.parse_args()
revision=HfApi().model_info(a.repo,revision=a.revision).sha
root=Path(snapshot_download(a.repo,revision=revision,local_dir=a.output,allow_patterns=['*.json','*.safetensors','*.txt','*.model','LICENSE','README.md']))
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
files={str(f.relative_to(root)):{'sha256':sha(f),'bytes':f.stat().st_size()} for f in root.rglob('*') if f.is_file() and '.cache' not in f.parts and f.name!='provenance.json'}
(root/'provenance.json').write_text(json.dumps({'repository':a.repo,'revision':revision,'files':files},indent=2));print(f'Model pinned to {revision}; saved in {root}')
