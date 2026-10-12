"""Mock-evaluate probe shape metadata and compare with independent NumPy shapes."""
import measure,json,sys,traceback,concurrent.futures,subprocess,os
from zinnia import ZKCircuit
from benchmarking.harness.kernels import load_kernel_module
from benchmarking.harness.costs_halo2 import mock_publics_in_order

def one(p):
 name=measure.ident(p);r=json.loads((measure.HERE/'probes'/(name+'.json')).read_text())
 rows=[x for x in json.loads((measure.HERE/'arrays.json').read_text()) if x['program']==name]
 base=ZKCircuit.from_method(getattr(load_kernel_module(p.kernel_path),p.function))
 c=ZKCircuit(p.function,r['source'],chips=list(base.chips.values()),externals=list(base.externals.values())).compile()
 fixture=measure.load_fixture(p.fixture_path);out=dict(id=name,checks=[])
 for f in ['A','B']:
  mock=c.prove(*fixture.arguments(f),backend='mock')
  assert mock.proof_bytes_hex=='mock_satisfied',mock.proof_bytes_hex
  publics=mock_publics_in_order(mock.public_values)
  layout={tuple(x['output'].get('return',[])):x for x in c.get_output_layout()}
  for row in rows:
   if row['source']!='expression_probe':continue
   d=layout[tuple(row['path'])]['dynamic']
   def slot(s):return s['constant'] if 'constant'in s else publics[s['public']]
   length=slot(d['length']);shape=[slot(s) for s in d['dims']][:slot(d['rank'])]
   out['checks'].append(dict(array=row['array'],fixture=f,length=length,shape=shape,match=length==row['lengths'][f] and shape==row['shapes'][f]))
 assert all(x['match'] for x in out['checks']),out
 return out
if __name__=='__main__':
 folder=measure.HERE/'validation';folder.mkdir(exist_ok=True)
 if len(sys.argv)>1:
  p=next(p for p in measure.entries() if measure.ident(p)==sys.argv[1])
  try:r=one(p)
  except BaseException as e:r=dict(id=measure.ident(p),error=str(e),traceback=traceback.format_exc())
  (folder/(measure.ident(p)+'.json')).write_text(json.dumps(r,indent=2))
 else:
  ids={x['program'] for x in json.loads((measure.HERE/'arrays.json').read_text())}
  def run(p):
   name=measure.ident(p)
   with (folder/(name+'.log')).open('w') as log:
    try:r=subprocess.run([sys.executable,'-B',__file__,name],stdout=log,stderr=log,timeout=300,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    except subprocess.TimeoutExpired:(folder/(name+'.json')).write_text(json.dumps(dict(id=name,error='resource timeout 300s')))
   print(name,flush=True)
  with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get('B_JOBS','16'))) as pool:list(pool.map(run,[p for p in measure.entries() if measure.ident(p) in ids]))
