import os,pathlib,json,re,hashlib
from benchmarking.harness import paths,build,gitstate
root=paths.REPO_ROOT
state=gitstate.describe(root)
assert not state['dirty'],state
record=json.loads(build.BASELINE_RECORD.read_text())
for name in ('sp1','risc0','cairo'):
 record[name]['source']=state
 for rel,h in record[name]['executables_sha256'].items(): assert build.sha256_file(root/rel)==h
build.BASELINE_RECORD.write_text(json.dumps(record,indent=2))
artifacts={}
for system,filename in [('sp1','guest_elfs.rs'),('risc0','guest_table.rs')]:
 tables=list((root/f'benchmarking/baselines/{system}/target/release/build').glob('*/out/'+filename))
 assert tables,(system,filename)
 table=max(tables,key=lambda p:p.stat().st_mtime)
 text=table.read_text()
 assert 'zinnia-literal' not in text,(system,table)
 files=re.findall(r'"(/[^"\n]+)"',text)
 assert files,(system,table)
 artifacts[system]={'table':str(table),'table_sha256':build.sha256_file(table),'guests':{f:build.sha256_file(pathlib.Path(f)) for f in files}}
artifacts['cairo']={str(f):build.sha256_file(f) for f in (root/'benchmarking/baselines/cairo/target').rglob('*.executable.json')}
assert artifacts['cairo']
assert not build.installed_build_problems()[1]
out=pathlib.Path(os.environ['RUN'])/'zk42'
(out/'build-audit.json').write_text(json.dumps({'source':state,'baseline_builds':record,'artifacts':artifacts},indent=2))
print('BUILD_AUDIT_OK',state['commit'])
(out/'build-ready').write_text(state['commit']+'\n')
