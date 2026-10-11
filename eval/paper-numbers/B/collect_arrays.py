"""Produce source-expression array inventory with A/B lengths and compiler capacities."""
import measure,probe,ast,json,collections,sys,copy
from benchmarking.harness.reference import run_kernel_as_numpy
import numpy as np

def getpath(v,path):
 for i in path:v=v[i]
 return v

def collect(p):
 name=measure.ident(p);r=json.loads((measure.HERE/'probes'/(name+'.json')).read_text());c=json.loads((measure.HERE/'census'/(name+'.json')).read_text())
 rows=[];issues=[]
 labels={}
 for x in r.get('layout',[]):
  path=x['output'].get('return',[])
  if len(path)==2 and path[1]==0 and 'cells'in x and len(x['cells'])==1 and 'constant'in x['cells'][0]:labels[path[0]]=r['selected'][x['cells'][0]['constant']]
 observed={}
 for f in ['A','B']:
  bysite=collections.defaultdict(list)
  for x in r.get('numpy',{}).get(f,[]):bysite[x['site']].append(x['value'])
  observed[f]=bysite
 occurrences=collections.defaultdict(int)
 itemocc={}
 for item,site in sorted(labels.items()):
  itemocc[item]=occurrences[site];occurrences[site]+=1
 for x in r.get('dynamic',[]):
  path=x['output'].get('return',[])
  if len(path)<2 or path[1]!=1:continue # instrumentation trace's own dynamic shape is not a source array
  d=x['dynamic']
  # Runtime-shaped means a shape component is witness-dependent, including fixed-total reshapes.
  if not any('public'in slot for slot in [d['length'],d['rank'],*d['dims']]):continue
  item=path[0];site=labels[item];nth=itemocc[item];lengths={};shapes={}
  for f in ['A','B']:
   vals=observed[f][site]
   try:
    val=getpath(vals[nth],path[2:]);lengths[f]=val['length'];shapes[f]=val['shape']
   except (IndexError,TypeError,KeyError) as e:issues.append(f'{site} occurrence {nth} fixture {f}: {e}');lengths[f]=None
  rows.append(dict(program=name,array=f'{site}#{nth}'+str(path[2:]),expression=r['sites'][site],capacity=len(d['cells'])//(2 if x.get('complex') else 1),lengths=lengths,shapes=shapes,source='expression_probe',path=path,layout=x))
 # Crop-result probes were deliberately truncated before the branch.
 if p.program in ['case0507','case0509','case0510']:
  x=next(x for x in c['layout'] if 'dynamic'in x)
  lens={};shapes={}
  for f in ['A','B']:
   v=run_kernel_as_numpy(p.kernel_path,p.function,measure.load_fixture(p.fixture_path).arguments(f));lens[f]=v.size;shapes[f]=list(v.shape)
  rows.append(dict(program=name,array='return:crop',expression='result (branch-merged crop)',capacity=len(x['dynamic']['cells']),lengths=lens,shapes=shapes,source='original_output',layout=x))
 return rows,issues

def main():
 rows=[];issues={};audit=[]
 for p in measure.entries():
  path=measure.HERE/'probes'/(measure.ident(p)+'.json')
  if not path.exists():issues[measure.ident(p)]=['missing probe'];continue
  r=json.loads(path.read_text())
  if 'error'in r:issues[measure.ident(p)]=[r['error']];continue
  try:rr,ii=collect(p)
  except BaseException as e:issues[measure.ident(p)]=[repr(e)];continue
  rows+=rr
  if ii:issues[measure.ident(p)]=ii
  audit.append(dict(id=measure.ident(p),bounded=bool(rr),arrays=len(rr),repair=r.get('repair')))
 (measure.HERE/'arrays.json').write_text(json.dumps(rows,indent=2));(measure.HERE/'array_issues.json').write_text(json.dumps(issues,indent=2));(measure.HERE/'audit.json').write_text(json.dumps(audit,indent=2))
 print('programs',len(audit),'bounded',sum(x['bounded'] for x in audit),'arrays',len(rows),'issues',issues)
if __name__=='__main__':main()
