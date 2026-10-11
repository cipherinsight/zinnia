from study import *
import hashlib
inventory=[]
for x in BATCH:
 n=x['program'];d=ROOT/n;p=BASE/x['kernel'];f=p.parent/'fixture.json'
 (d/'source.py.txt').write_bytes(p.read_bytes());(d/'fixture.json').write_bytes(f.read_bytes())
 inventory.append({k:x[k] for k in ['program','dataset','kernel','function']}|{'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'fixture_sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
(ROOT/'manifest.json').write_text(json.dumps(inventory,indent=2)+'\n')
binaries={}
for name in ['circ','zxi']:
 p=pathlib.Path(CIRC)/name;binaries[name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(ROOT/'environment.json').write_text(json.dumps({'binaries':binaries,'numpy_version':np.__version__,'python':sys.executable,'field_modulus':str(P),'real_scale':S,'memory_limit':'8G','timeout_seconds':300,'stdlib':ENV['ZSHARP_STDLIB_PATH'],'tmpdir':ENV['TMPDIR']},indent=2)+'\n')
