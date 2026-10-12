"""Check all bounded programs on A/B against the harness reference and original output metadata."""
import measure,json,sys,traceback,concurrent.futures,subprocess,os
from benchmarking.harness.reference import run_kernel_as_numpy
from benchmarking.harness.costs_halo2 import mock_publics_in_order
from benchmarking.harness.adapters.zinnia_common import describe_compiled,zinnia_publics
from benchmarking.harness.outputs import check_outputs,reference_leaves

def one(p):
 c=measure.compile_kernel(p.kernel_path,p.function);f=measure.load_fixture(p.fixture_path);out=dict(id=measure.ident(p),fixtures={})
 for label in ['A','B']:
  mock=c.prove(*f.arguments(label),backend='mock');assert mock.proof_bytes_hex=='mock_satisfied',mock.proof_bytes_hex
  public=mock_publics_in_order(mock.public_values);ref=run_kernel_as_numpy(p.kernel_path,p.function,f.arguments(label));leaves,_=reference_leaves(ref)
  check=check_outputs(leaves,zinnia_publics(public,describe_compiled(c)),p.tolerance)
  shapes=[]
  for part in c.get_output_layout():
   if 'dynamic' not in part:continue
   d=part['dynamic'];value=ref
   for i in part['output']['return']:value=value[i]
   def slot(x):return x['constant']if 'constant'in x else public[x['public']]
   length=slot(d['length']);shape=[slot(s)for s in d['dims']][:slot(d['rank'])]
   assert length==value.size and shape==list(value.shape)
   shapes.append(dict(output=part['output'],length=length,shape=shape))
  out['fixtures'][label]=dict(satisfied=True,check=check.status,detail=check.detail,dynamic_outputs=shapes)
  assert check.status=='match',out
 return out
if __name__=='__main__':
 folder=measure.HERE/'original_validation';folder.mkdir(exist_ok=True)
 if len(sys.argv)>1:
  p=next(p for p in measure.entries()if measure.ident(p)==sys.argv[1])
  try:r=one(p)
  except BaseException as e:r=dict(id=measure.ident(p),error=str(e),traceback=traceback.format_exc())
  (folder/(measure.ident(p)+'.json')).write_text(json.dumps(r,indent=2))
 else:
  ids={x['program']for x in json.loads((measure.HERE/'arrays.json').read_text())}
  def run(p):
   name=measure.ident(p)
   with (folder/(name+'.log')).open('w') as log:
    try:subprocess.run([sys.executable,'-B',__file__,name],stdout=log,stderr=log,timeout=300,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    except subprocess.TimeoutExpired:(folder/(name+'.json')).write_text(json.dumps(dict(id=name,error='resource timeout 300s')))
   print(name,flush=True)
  with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get('B_JOBS','16')))as pool:list(pool.map(run,[p for p in measure.entries()if measure.ident(p)in ids]))
