"""Document why preconditions alone do not give exact-capacity variants."""
import measure,ast,json
from zinnia import ZKCircuit
from benchmarking.harness.kernels import load_kernel_module
spec={'0298':'np.unique(a).size == 3','0303':'ncol == 5','0383':'N == 7','0391':'low == 4 and high == 7','0412':'np.sum(x >= 0) == 2'}
out=[]
for case,cond in spec.items():
 p=next(p for p in measure.entries() if p.program=='case'+case);base=ZKCircuit.from_method(getattr(load_kernel_module(p.kernel_path),p.function));names=[a.arg for a in ast.parse(base.source).body[0].args.args]
 for kind in ['default','assert','requires']:
  tree=ast.parse(base.source);fn=tree.body[0]
  if kind=='assert':fn.body=ast.parse('assert '+cond).body+fn.body
  elif kind=='requires':fn.decorator_list.append(ast.parse('@requires(lambda '+', '.join(names)+': '+cond+')\ndef f(): pass').body[0].decorator_list[0])
  source=ast.unparse(ast.fix_missing_locations(tree));row=dict(case=case,kind=kind,condition=cond,source=source)
  try:
   c=ZKCircuit(p.function,source,chips=list(base.chips.values())).compile();row['layout']=c.get_output_layout()
   row['capacity']=[len(x.get('dynamic',{}).get('cells',x.get('cells',[])))for x in row['layout']]
  except BaseException as e:row['error']=str(e)
  out.append(row)
(measure.HERE/'requirements_probe.json').write_text(json.dumps(out,indent=2))
print([(x['case'],x['kind'],x.get('capacity',x.get('error')))for x in out])
