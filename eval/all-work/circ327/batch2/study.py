import os,json,pathlib,subprocess,re,time,ast,sys,copy
import numpy as np
BASE=pathlib.Path(__file__).resolve().parent
ROOT=pathlib.Path('/home/zhantong/Projects/zinnia-project/wt/literal')
CIRC=pathlib.Path('/home/zhantong/.cache/zkcc-reconfirm/circ-stock/target/release/examples')
MEM='/home/zhantong/Projects/zinnia-project/zinnia-src/eval/tools/memrun.sh'
P=52435875175126190479447740508185965837690552500527637822603658699938581184513
Q=2**32
ENV={**os.environ,'TMPDIR':str(BASE/'tmp'),'MEMRUN_MAX':'8G','ZSHARP_STDLIB_PATH':'/home/zhantong/.cache/zkcc-reconfirm/circ-stock/src/third_party/ZoKrates/zokrates_stdlib/stdlib','PYTHONDONTWRITEBYTECODE':'1'}
ENTRIES={x['program']:{k:v for k,v in x.items() if k!='zinnia_outcome'} for x in json.load(open(BASE.parent/'batch2.json'))}
SPECS={}
HELP=f'''from "EMBED" import u32_to_field
from "utils/casts/field_to_u32" import main as idx

def neg(field x) -> bool:
    return x > {P//2}

def lt(field x, field y) -> bool:
    return if neg(x) != neg(y) then neg(x) else x < y fi

def ab(field x) -> field:
    return if neg(x) then -x else x fi

'''
def spec(n,body,out='float',rewrite=None,reason='',helpers=True):
 SPECS[n]=dict(code=(HELP if helpers else '')+body,out=out,rewrites=rewrite,reason=reason)
def blocked(n,category,reason,rewrite=None):
 SPECS[n]=dict(category=category,reason=reason,rewrites=rewrite)
def fixture(n):
 p=ROOT/ENTRIES[n]['kernel']; f=p.with_name('fixture.json')
 if not f.exists():f=p.with_name('fixture.'+ENTRIES[n]['function']+'.json')
 return f,json.loads(f.read_text())
def orig(n,label):
 p=ROOT/ENTRIES[n]['kernel']; tree=ast.parse(p.read_text())
 for node in ast.walk(tree):
  if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
   node.decorator_list=[];node.returns=None
   for a in node.args.args:a.annotation=None
 tree.body=[node for node in tree.body if not(isinstance(node,ast.ImportFrom) and node.module=='zinnia')]
 ns={};exec(compile(ast.fix_missing_locations(tree),str(p),'exec'),ns)
 _,f=fixture(n)
 def co(v,d):
  if d['kind']=='tuple':return tuple(co(x,t) for x,t in zip(v,d['items']))
  if d['dtype']=='complex':
   a=np.asarray(v);v=a[...,0]+1j*a[...,1]
  if d['kind']=='array':return np.array(v,dtype={'float':float,'int':np.int64,'bool':bool,'complex':complex}[d['dtype']])
  return v
 args={d['name']:co(f['inputs'][label][d['name']],d) for d in f['parameters']}
 args.update(f.get('constants',{}));return ns[ENTRIES[n]['function']](**args)
def pin(n,label):
 _,f=fixture(n); lines=[]
 def walk(name,v,d):
  if d['kind']=='tuple':
   for i,(x,t) in enumerate(zip(v,d['items'])):walk(name+'.'+str(i),x,t)
  elif isinstance(v,list):
   for i,x in enumerate(v):walk(name+'.'+str(i),x,{**d,'kind':'scalar'})
  else:
   val=round(v*Q) if d['dtype'] in ['float','complex'] else int(v)
   lines.append(f'({name} #f{val})')
 for d in f['parameters']:walk(d['name'],f['inputs'][label][d['name']],d)
 return f'(set_default_modulus {P}\n(let (\n'+ '\n'.join(lines)+'\n) true))\n'
def run_cmd(cmd,log):
 t=time.monotonic();r=subprocess.run(cmd,env=ENV,cwd=BASE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 log.write_text(r.stdout);return r,time.monotonic()-t

def check(n,mode,s):
 d=BASE/n; result={}
 for label in ['A','B']:
  (d/(label+'.pin')).write_text(pin(n,label))
  r,elapsed=run_cmd([MEM,'timeout','300',str(CIRC/'zxi'),str(d/(mode+'.zok')),str(d/(label+'.pin'))],d/(mode+'.'+label+'.log'))
  if r.returncode:result[label]={'status':'error','error':r.stdout[:1000]};continue
  vals=re.findall(r'(-?\d+)f\b',r.stdout)
  if s['out']=='bool':vals=re.findall(r'\b(true|false)\b',r.stdout);actual=np.array([v=='true' for v in vals])
  else:
   actual=np.array([float(int(v)-P if int(v)>P//2 else int(v)) for v in vals]);actual/= Q if s['out']=='float' else 1
  expected=orig(n,label)
  if n=='pythran_pivot':
   if expected is None:
    expected=np.array([0.0]);actual=actual[:1]
   else:
    vals=np.concatenate([np.asarray(x).reshape(-1) for x in expected])
    expected=np.concatenate([[1.0],np.stack([vals.real,vals.imag],axis=-1).reshape(-1)])
  actual=decode(n,actual,expected,label)
  exp=np.concatenate([np.asarray(x).reshape(-1) for x in expected]) if n=='case0467' else np.asarray(expected).reshape(-1)
  passed=actual.shape==exp.shape and np.allclose(actual,exp,rtol=1e-6,atol=1e-6,equal_nan=True)
  result[label]={'status':'pass' if passed else 'wrong','elements':len(exp),'max_abs_error':float(np.max(np.abs(actual.astype(float)-exp.astype(float)))) if actual.shape==exp.shape and len(exp) else None}
  (d/(label+'.expected.json')).write_text(json.dumps([np.asarray(x).tolist() for x in expected] if n=='case0467' else np.asarray(expected).tolist()))
  (d/(mode+'.'+label+'.decoded.json')).write_text(json.dumps(actual.tolist()))
 return result

def decode(n,a,e,label):
 if n=='case0302':
  assert tuple(a[:2].astype(int))==np.asarray(e).shape
  return a[2:22]
 if n=='case0393':return a[1:].reshape(20,10)[:,:int(a[0])].reshape(-1) # metadata field encoded at Q scale
 if n=='pythran_zero':
  _,f=fixture(n)
  assert tuple(a[:2].astype(int))==(f['inputs'][label]['m'],f['inputs'][label]['n'])
  return a[2:].reshape(1024,1024)[:len(e),:len(e[0])].reshape(-1) if len(e) and len(e[0]) else np.array([])
 if n=='case0467':
  rows=a.reshape(16,18)
  for row,win in zip(rows,e):assert tuple(row[:2].astype(int))==win.shape
  return np.concatenate([row[2:].reshape(4,4)[:win.shape[0],:win.shape[1]].reshape(-1) for row,win in zip(rows,e)])
 return a

def execute(n):
 s=SPECS[n];d=BASE/n;d.mkdir(exist_ok=True)
 (d/'source.py').write_text((ROOT/ENTRIES[n]['kernel']).read_text())
 result={'program':n,'category':s.get('category'),'as_written':{'status':'not expressible','error':s['reason'],'constraints':None},'rewrite':None,'correctness':{'A':'not run: no accepted translation','B':'not run: no accepted translation'},'notes':s['reason']}
 if 'code' not in s:
  result['rewrite']={'status':'not expressible' if s['category'].startswith('not_expressible') else 'needs gadget' if s['category'].startswith('needs_gadget') else 'unsupported','error':s['reason'],'constraints':None,'rewrites':s.get('rewrites') or []}
 else:
  mode='rewrite' if s['rewrites'] else 'as_written';(d/(mode+'.zok')).write_text(s['code'])
  cmd=[MEM,'timeout','300',str(CIRC/'circ'),'--language','zsharp',str(d/(mode+'.zok')),'r1cs','--action','count']
  r,elapsed=run_cmd(cmd,d/(mode+'.compile.log'))
  status='accepted' if r.returncode==0 else 'timeout' if r.returncode==124 else 'out of memory' if r.returncode in [137,-9] else 'rejected'
  counts=re.findall(r'(\d+)\s+(?:constraints|constraints,)',r.stdout)
  if not counts:counts=re.findall(r'constraints[^\d\n]*(\d+)',r.stdout,re.I)
  attempt={'status':status,'error':None if not r.returncode else '\n'.join(r.stdout.splitlines()[:12]),'constraints':int(counts[-1]) if counts else None,'seconds':elapsed,'command':cmd}
  if mode=='rewrite':attempt['rewrites']=s['rewrites'];result['rewrite']=attempt
  else:result['as_written']=attempt
  result['category']='accepted_'+('with_rewrite' if mode=='rewrite' else 'as_written') if status=='accepted' else 'timeout_or_oom' if status in ['timeout','out of memory'] else 'circ_error'
  if status=='accepted':
   result['correctness']=check(n,mode,s)
   if any(v['status']!='pass' for v in result['correctness'].values()):result['category']='wrong'
  elif s.get('gadget'):
   
   if mode=='as_written':result['rewrite']={'status':'needs gadget','error':s['gadget'],'constraints':None,'rewrites':[]}
   else:result['rewrite']['remaining_blocker']='needs gadget: '+s['gadget']
   result['category']='needs_gadget: '+s['gadget']
 (d/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(n,result['category'],flush=True)
