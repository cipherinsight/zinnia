"""Gate actual participant engines on loop, array and natural alternative solutions."""
import ast,json,tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from ..site.execution import Runner
from ..tasks import chained_indices
from ..revision2.tasks import WhileLoops
from . import content as c
from .reference import PYTHON,LOOPS,RUST,RUST_LOOPS

def main():
 jobs=[]
 for task in c.ORDER:
  inputs=c.cases(task)+c.cases(task,True)
  py={'array':PYTHON[task],'loops':LOOPS[task], 'default_zeros':LOOPS[task].replace(',dtype=int',''), 'chained':chained_indices(LOOPS[task])}
  py['while']=ast.unparse(ast.fix_missing_locations(WhileLoops().visit(ast.parse(LOOPS[task]))))
  if task==c.ORDER[0]:py['where']='def kernel(x,gain,bias):\n    a=x*gain+bias\n    a=np.where(a<0,0,np.where(a>255,255,a))\n    d=a-np.min(a,axis=0,keepdims=True)\n    return np.sum(d*d,axis=0)\n'
  if task==c.ORDER[1]:py['broadcast']='def kernel(x,y):\n    z=x.shape[0]*x-x.sum(axis=0)\n    w=y.shape[0]*y-y.sum(axis=0)\n    return (z[:,:,None]*w[:,None,:]).sum(axis=0)\n'
  if task==c.ORDER[2]:py['comprehension']='def kernel(x,y,weight):\n    return np.array([[sum(weight[d]*(x[i,d]-y[j,d])**2 for d in range(3)) for j in range(y.shape[0])] for i in range(x.shape[0])])\n'
  if task==c.ORDER[3]:py['windows']='def kernel(x,filter):\n    from numpy.lib.stride_tricks import sliding_window_view\n    windows=sliding_window_view(x,filter.shape)\n    return (windows*filter).sum(axis=(2,3))\n'
  if task==c.ORDER[4]:py['reshape']='def kernel(x):\n    h,w,c=x.shape\n    p=x.reshape((h//2,2,w//2,2,c)).transpose(0,2,1,3,4).reshape((h//2,w//2,4,c))\n    d=p-p.min(axis=2,keepdims=True)\n    return (d*d).sum(axis=2)\n'
  for variant,code in py.items():jobs.append(('zinnia',task,variant,code,inputs))
  for variant,code in [('array',RUST[task]),('loops',RUST_LOOPS[task])]:jobs.append(('rust',task,variant,code,inputs))
 with tempfile.TemporaryDirectory() as runtime:
  runner=Runner(runtime,timeout=180)
  def run(job):
   lang,task,variant,code,inputs=job;r=runner.execute(lang,task,code,inputs)
   row={'language':lang,'task':task,'variant':variant,'cases':len(inputs),'status':r['status'],'engine':r.get('engine')}
   if r['status']!='passed':row['error']=r
   print(lang,task,variant,r['status'],flush=True)
   return row
  try:
   with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(run,jobs))
  finally:runner.pool.shutdown()
 report={'rows':rows,'passed':sum(r['status']=='passed' for r in rows),'total':len(rows)}
 Path(__file__).with_name('validation.json').write_text(json.dumps(report,indent=2)+'\n')
 assert report['passed']==report['total'],f"{report['passed']}/{report['total']} passed"
if __name__=='__main__':main()
