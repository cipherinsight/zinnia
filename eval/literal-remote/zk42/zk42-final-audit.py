import json,pathlib,os,sys,subprocess
run=pathlib.Path(os.environ['RUN']);wt=run/'zinnia-zk42';out=run/'zk42'
sys.path.insert(0,str(wt))
from benchmarking.harness import build,gitstate
subprocess.run([sys.executable,str(out/'zk42-verify-calibrations.py')],check=True)
original=json.loads((out/'build-audit.json').read_text())
current=gitstate.describe(wt)
assert current==original['source'],(current,original['source'])
checks={}
for system in ('sp1','risc0'):
 data=original['artifacts'][system]
 checks[data['table']]=data['table_sha256']
 checks.update(data['guests'])
checks.update(original['artifacts']['cairo'])
for system in ('sp1','risc0'):
 checks.update({str(wt/p):h for p,h in original['baseline_builds'][system]['executables_sha256'].items()})
for path,digest in checks.items():assert build.sha256_file(pathlib.Path(path))==digest,path
proof_count=0;warmup_proof_count=0
for raw in (out/'epochs').glob('*/raw/*/*.json'):
 if raw.parent.name.startswith('_'):continue
 data=json.loads(raw.read_text())
 for item in [data.get('warmup') or {},*data.get('samples',[])]:
  proof=item.get('prove') or {}
  if not proof.get('proof_path'):continue
  path=pathlib.Path(proof['proof_path'])
  assert proof.get('proof_sha256') and build.sha256_file(path)==proof['proof_sha256'],str(path)
  proof_count+=1
  warmup_proof_count+=int(item.get("tag")=="warmup")
record,problems=build.installed_build_problems();assert not problems,problems
warmups=json.loads((out/'warmup-verification.json').read_text());assert len(warmups)==warmup_proof_count
binaries=json.loads((out/'cairo-binary-verification.json').read_text());assert len(binaries)==2*len(list((out/'epochs').glob('*/raw/cairo/*.json'))) and all(r['verified'] for r in binaries)
result={'source':current,'artifact_hashes_rechecked':len(checks),'proof_hashes_rechecked':proof_count,'extension_sha256':record['extension_sha256'],'warmup_count':len(warmups),'cairo_binary_count':len(binaries),'calibration_count':len(json.loads((out/'calibration-verification.json').read_text())),'ok':True}
(out/'final-source-audit.json').write_text(json.dumps(result,indent=2))
print('FINAL_SOURCE_AND_BINARY_AUDIT_OK',len(checks),flush=True)
