import measure,json,copy,traceback
from zinnia import ZKCircuit
from benchmarking.harness.kernels import load_kernel_module
from bounds_cost import CONTROLS
from benchmarking.harness.reference import kernel_namespace
out=[]
for p in measure.entries():
 if p.program[-4:] not in CONTROLS:continue
 base=ZKCircuit.from_method(getattr(load_kernel_module(p.kernel_path),p.function));folder=measure.HERE/'costs'/measure.ident(p);r=json.loads((folder/'result.json').read_text())
 fixture=measure.load_fixture(p.fixture_path)
 for f in ['A','B']:
  c=ZKCircuit(p.function,(folder/(f+'.py')).read_text(),chips=list(base.chips.values()),externals=list(base.externals.values())).compile()
  args=fixture.arguments(f);names=[x.name for x in fixture.parameters];first=CONTROLS[p.program[-4:]][0];i=names.index(first)
  if hasattr(args[i],'shape'):args[i].flat[0]+=1
  else:args[i]+=1
  rejected=False;detail=None
  try:
   proof=c.prove(*args,backend='mock');rejected=proof.proof_bytes_hex.startswith('mock_unsatisfied');detail=proof.proof_bytes_hex
  except BaseException as e:rejected=True;detail=str(e)
  # Specialized observable buffers have exactly the reference lengths.
  layout=r['variants'][f]['layout'];expected=r['variants'][f]['numpy_'+f]
  capacities=[len(x.get('dynamic',{}).get('cells',x.get('cells',[])))//(2 if x.get('complex') else 1) for x in layout]
  if len(expected)>1:
   from collections import defaultdict
   grouped=defaultdict(int)
   for x,cap in zip(layout,capacities):grouped[x['output']['return'][0]]+=cap
   capacities=[grouped[i] for i in sorted(grouped)]
  exact=capacities==[x['length'] for x in expected]
  valid=c.prove(*fixture.arguments(f),backend='mock')
  assert valid.proof_bytes_hex=='mock_satisfied',valid.proof_bytes_hex
  out.append(dict(id=measure.ident(p),fixture=f,modified_control=first,rejected=rejected,detail=detail,capacities=capacities,actual_lengths=[x['length'] for x in expected],exact_output_capacity=exact))
assert all(x['rejected'] and x['exact_output_capacity'] for x in out),out
(measure.HERE/'cost_validation.json').write_text(json.dumps(out,indent=2));print(len(out),'domain rejection and exact output capacity checks passed')
