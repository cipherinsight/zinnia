import dataclasses,hashlib,json,os,pathlib,sys
run=pathlib.Path(os.environ['RUN']);wt=run/'zinnia-zk42';out=run/'zk42'
sys.path.insert(0,str(wt))
from benchmarking.harness.adapters.noir_toolchain import verify_proof,public_fields
rows=[];expected=0
for epoch in sorted((out/'epochs').iterdir()):
 m=json.loads((epoch/'manifest.json').read_text())
 if 'zinnia_noir' not in m['config']['systems']:continue
 expected+=5
 work=epoch/'work/_calibration';vk=work/'package/target/vk'
 for i in range(5):
  proof=work/f'proof-{i}'/'proof'
  row={'epoch':epoch.name,'index':i,'proof':str(proof),'vk':str(vk)}
  try:
   row['proof_sha256']=hashlib.sha256(proof.read_bytes()).hexdigest()
   row['public_inputs_sha256']=hashlib.sha256((proof.parent/'public_inputs').read_bytes()).hexdigest()
   row['vk_sha256']=hashlib.sha256(vk.read_bytes()).hexdigest()
   row['output_match']=public_fields(proof.parent)==[7]
   row['verify']=dataclasses.asdict(verify_proof(vk,proof,False,True))
   row['verified']=row['verify']['verified'] and row['output_match']
  except Exception as exc:row.update({'verified':False,'error':str(exc)})
  rows.append(row)
  if len(rows)%25==0:(out/'calibration-verification.json').write_text(json.dumps(rows,indent=2))
(out/'calibration-verification.json').write_text(json.dumps(rows,indent=2))
assert len(rows)==expected and expected>=420,(len(rows),expected)
assert all(r['verified'] for r in rows),[r for r in rows if not r['verified']]
print('ALL_CALIBRATION_PROOFS_VERIFIED',len(rows),flush=True)
