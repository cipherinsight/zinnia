"""Checked shape-control specialization; compile/mock/cell stats/bb gates only."""
import measure,ast,json,copy,sys,os,traceback,subprocess
from zinnia import ZKCircuit
from zinnia.config.zinnia_config import ZinniaConfig
from zinnia._zinnia_core import halo2_base_cell_stats
from benchmarking.harness.kernels import load_kernel_module
from benchmarking.noir_gates import gate_count,NOIR_FPC
from benchmarking.harness.reference import run_kernel_as_numpy

# Shape-control inputs only. No payload values are specialized.
CONTROLS={'0301':['ncol'],'0302':['nrow'],'0303':['ncol'],'0304':['ncol'],'0383':['N'],'0389':['h','w'],'0391':['low','high'],'0392':['low','high'],'0393':['low','high'],'0414':['bin_size'],'0415':['bin_size'],'0416':['bin_size'],'0417':['bin_size'],'0418':['bin_size'],'0419':['bin_size'],'0467':['size'],'0468':['size']}
def variant(base,p,which):
 f=measure.load_fixture(p.fixture_path);values=f.values(which)
 tree=ast.parse(base.source);fn=tree.body[0];prefix=[]
 for name in CONTROLS[p.program[-4:]]:
  value=values[name]
  if isinstance(value,list):
   for i,v in enumerate(value):prefix+=ast.parse(f'assert {name}[{i}] == {v!r}').body
   prefix+=ast.parse(f'{name} = np.array({value!r})').body
  else:prefix+=ast.parse(f'assert {name} == {value!r}\n{name} = {value!r}').body
 fn.body=prefix+fn.body
 return ast.unparse(ast.fix_missing_locations(tree))
def stats(c):return json.loads(halo2_base_cell_stats(c._compiled_ir))
def flat_shapes(v):
 import numpy as np
 if isinstance(v,np.ndarray):return [dict(shape=list(v.shape),length=int(v.size))]
 if isinstance(v,(tuple,list)):return sum((flat_shapes(x) for x in v),[])
 return []
def work(p):
 name=measure.ident(p);folder=measure.HERE/'costs'/name;folder.mkdir(parents=True,exist_ok=True)
 fixture=measure.load_fixture(p.fixture_path)
 base=ZKCircuit.from_method(getattr(load_kernel_module(p.kernel_path),p.function))
 out=dict(id=name,controls=CONTROLS[p.program[-4:]],variants={})
 for v in ['default','A','B']:
  source=base.source if v=='default' else variant(base,p,v)
  (folder/(v+'.py')).write_text(source)
  rec={};out['variants'][v]=rec
  try:
   c=ZKCircuit(p.function,source,chips=list(base.chips.values()),externals=list(base.externals.values())).compile()
   rec['layout']=c.get_output_layout();rec['halo2']=stats(c)
   for label in (['A','B'] if v=='default' else [v]):
    mock=c.prove(*fixture.arguments(label),backend='mock');assert mock.proof_bytes_hex=='mock_satisfied',mock.proof_bytes_hex;rec['mock_'+label]=dict(public_values=mock.public_values,satisfied=True)
   # Compare logical outputs with NumPy through the harness decoder.
   from benchmarking.harness.adapters.zinnia_common import describe_compiled,zinnia_publics
   from benchmarking.harness.outputs import field_to_le_hex,check_outputs,reference_leaves
   from benchmarking.harness.costs_halo2 import mock_publics_in_order
   for label in (['A','B'] if v=='default' else [v]):
    ref=run_kernel_as_numpy(p.kernel_path,p.function,fixture.arguments(label));rec['numpy_'+label]=flat_shapes(ref)
    publics=mock_publics_in_order(rec['mock_'+label]['public_values'])
    # zinnia_publics expects numeric field elements.
    leaves,_=reference_leaves(ref)
    checked=check_outputs(leaves,zinnia_publics(publics,describe_compiled(c)),p.tolerance)
    rec['check_'+label]=dict(status=checked.status,detail=checked.detail)
   nc=ZKCircuit(p.function,source,chips=list(base.chips.values()),externals=list(base.externals.values()),config=ZinniaConfig(backend='noir')).compile()
   nr=nc.emit_structural(*fixture.arguments('A' if v=='default' else v),backend='noir');(folder/(v+'.nr')).write_text(nr)
   rec['noir']=gate_count(nr,'noir_fpc',NOIR_FPC)
  except BaseException as e:rec['error']=str(e);rec['traceback']=traceback.format_exc()
  (folder/'result.json').write_text(json.dumps(out,indent=2))
 # Match lookup widths pairwise (no key generation).
 for v in ['A','B']:
  a=out['variants']['default'];b=out['variants'][v]
  if 'halo2' not in a or 'halo2' not in b:continue
  lb=max(a['halo2']['lookup_bits'],b['halo2']['lookup_bits'])
  pair={}
  for label,r in [('default',a),(v,b)]:
   c=ZKCircuit(p.function,(folder/(label+'.py')).read_text(),chips=list(base.chips.values()),externals=list(base.externals.values())).compile()
   pair[label]=json.loads(halo2_base_cell_stats(c._compiled_ir,json.dumps(dict(k=r['halo2']['k'],lookup_bits=lb,precision_bits=32))))
  out.setdefault('common',{})[v]=pair
 (folder/'result.json').write_text(json.dumps(out,indent=2));return out
if __name__=='__main__':
 os.environ['TMPDIR']=str(measure.HERE/'tmp');(measure.HERE/'tmp').mkdir(exist_ok=True)
 import tempfile;tempfile.tempdir=os.environ['TMPDIR']
 p=next(p for p in measure.entries() if p.program=='case'+sys.argv[1]);work(p)
