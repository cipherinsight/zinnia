import json,pathlib,subprocess,os,hashlib
from benchmarking.harness import build,gitstate,paths
wt=paths.REPO_ROOT
old=wt.parent/'zinnia-literal'
base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=old,text=True).strip()
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=wt,text=True).strip()
changed=subprocess.check_output(['git','diff','--name-only',base,head],cwd=wt,text=True).splitlines()
assert all(p.startswith('benchmarking/datasets/') or p.startswith('benchmarking/baselines/') for p in changed), changed
record=json.loads(paths.BUILD_RECORD.read_text())
original=dict(record)
assert build.sha256_file(paths.installed_extension())==record['extension_sha256']
assert build.sha256_file(old/'zinnia'/paths.installed_extension().name)==record['extension_sha256']
record['reuse']={'method':'independent copy; unchanged compiler source tree','from_checkout':str(old),'from_commit':base,'original_build_record':original,'changed_paths':changed}
record['source']=gitstate.describe(wt)
assert not record['source']['dirty'],record['source']
paths.BUILD_RECORD.write_text(json.dumps(record,indent=2))
assert not build.installed_build_problems()[1],build.installed_build_problems()
(pathlib.Path(os.environ['RUN'])/'zk42/reuse-attestation.json').write_text(json.dumps(record,indent=2))
print('REUSE_ATTESTED',head,record['extension_sha256'])
