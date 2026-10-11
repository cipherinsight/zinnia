import json,pathlib,collections,os
root=pathlib.Path(os.environ['RUN'])/'zk20/epochs'
counts=collections.Counter(); bad=[]; completed=[]
for p in sorted(root.glob('*/raw/*/*.json')):
 if p.parent.name.startswith('_'):continue
 r=json.loads(p.read_text());counts[(p.parent.name,r['status'])]+=1
 if r['status']!='ok':bad.append((p.parent.parent.parent.name,p.parent.name,r['case'],r.get('failure')))
for p in sorted(root.glob('*/manifest.json')):
 m=json.loads(p.read_text())
 completed.append((p.parent.name,m.get('finished_at')))
print('RECORDS',dict((str(k),v) for k,v in counts.items()))
for b in bad: print('FAILURE',json.dumps(b))
print('EPOCHS',len(completed))
for ep,finished in completed:
 if finished:continue
 for system in ['zinnia_halo2','zinnia_noir','sp1','risc0','cairo']:
  steps=list((root/ep/'work'/system).glob('*/steps/*.response.json'))
  if not steps:continue
  okay=[];failed=[]
  for p in steps:
   r=json.loads(p.read_text())
   (okay if r['ok'] else failed).append(p.name.removesuffix('.response.json'))
  print('ACTIVE',ep,system,','.join(sorted(okay)), 'FAILED='+','.join(failed) if failed else '')
