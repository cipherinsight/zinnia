import ast,json,pathlib,subprocess,re,sys,os,traceback
import numpy as np
from run import BASE,ENV,BIN,PREFIX
P=52435875175126190479447740508185965837690552500527637822603658699938581184513
M=json.load(open(BASE/'manifest.json'));B={p['program']:p for p in json.load(open(BASE.parent/'batch1.json'))}
def numpy_run(p,fx,side):
 tree=ast.parse((BASE/p/'kernel.py').read_text())
 tree.body=[n for n in tree.body if not isinstance(n,ast.ImportFrom) or n.module!='zinnia']
 for n in ast.walk(tree):
  if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
   n.decorator_list=[];n.returns=None
   for a in n.args.args:a.annotation=None
 ns={};exec(compile(ast.fix_missing_locations(tree),str(BASE/p/'kernel.py'),'exec'),ns)
 args={}
 for spec in fx['parameters']:
  k=spec['name'];v=fx['inputs'][side][k]
  if spec['dtype']=='complex':
   a=np.array(v);v=a[...,0]+1j*a[...,1]
  elif spec['kind']=='array':v=np.array(v,dtype={'float':float,'int':np.int64,'bool':bool}.get(spec['dtype'],float))
  args[k]=v
 result=ns[B[p]['function']](**args)
 if result is None:
  names={'pythran_insertion_sort':['list2'],'pythran_ccopy':['CY'],'npbench_polybench_gemver_gemver':['A','w','x'],'npbench_polybench_k2mm_k2mm':['D']}.get(p,[])
  result=tuple(args[n] for n in names)
 return result

def flatten(x):
 if isinstance(x,(tuple,list)):return [v for y in x for v in flatten(y)]
 a=np.asarray(x)
 if np.iscomplexobj(a):return np.stack([a.real,a.imag],axis=-1).reshape(-1).tolist()
 return a.reshape(-1).tolist()
def inputs(p,fx,side,attempt):
 code=(BASE/p/(attempt+'.zok')).read_text();rows=[]
 def emit(name,v,scale,ty):
  if isinstance(v,list):
   for i,x in enumerate(v):emit(name+'.'+str(i),x,scale,ty)
  else:
   n=round(float(v)*scale) if scale!=1 else int(v)
   lit=('#x'+format(n%(1<<int(ty[1:])),f'0{int(ty[1:])//4}x')) if ty.startswith('u') else str(n) if ty=='integer' else '#f'+str(n%P)
   rows.append(f'    ({name} {lit})')
 for s in fx['parameters']:
  name=s['name'];match=re.search(r'private (\w+)(?:\[\d+\])* '+re.escape(name)+r'\b',code)
  if not match:raise ValueError('missing parameter '+name)
  emit(name,fx['inputs'][side][name],2**32 if s['dtype'] in ['float','complex'] else 1,match[1])
 out=BASE/p/(side+'.pin');out.write_text(f'(set_default_modulus {P}\n(let (\n'+'\n'.join(rows)+'\n) true))\n');return out

def parse(text):
 # Interpreter prints structural punctuation plus scalar f/u/hex literals; fields are centered.
 vals=[]
 for token in re.findall(r'(?<![\w.])(?:0x[0-9a-fA-F]+|-?\d+f|-?\d+u(?:8|16|32|64)|true|false)(?!\w)',text):
  if token in ['true','false']:v=int(token=='true')
  elif token.startswith('0x'):v=int(token,16)
  elif token.endswith('f'):
   v=int(token[:-1]);v=v-P if v>P//2 else v
  else:v=int(token.split('u')[0])
  vals.append(v)
 return vals

def check(p):
 d=BASE/p;r=json.load(open(d/'result.json'));m=M[p]
 a=r.get('accepted_attempt')
 if not a:return
 fx=json.load(open(d/'fixture.json'))
 for side in ['A','B']:
  try:
   pin=inputs(p,fx,side,a)
   proc=subprocess.run(PREFIX+[BIN+'zxi',str(d/(a+'.zok')),str(pin)],env=ENV,capture_output=True,text=True,cwd=BASE)
   (d/(a+'.'+side+'.zxi.log')).write_text(proc.stdout+proc.stderr)
   if proc.returncode:raise ValueError(f'zxi exit {proc.returncode}: '+proc.stderr[:600])
   vals=parse(proc.stdout);want=flatten(numpy_run(p,fx,side));sc=m['scale']
   if m['decode']=='compact':
    # Z# structs preserve declared order: values then length.
    n=int(vals[0]);vals=vals[1:];width=3 if p=='case0354' else 10 if p=='case0392' else 1;vals=vals[:n*width]
   elif m['decode']=='reshape':
    rows,cols=vals[-2:];expected=np.shape(numpy_run(p,fx,side))
    if (rows,cols)!=expected:raise ValueError(f'shape mismatch {(rows,cols)} != {expected}')
    vals=vals[:-2]
   elif m['decode']=='square':
    n=vals[0];vals=np.array(vals[1:]).reshape(1024,1024)[:n,:n].reshape(-1).tolist()
   got=np.asarray(vals,dtype=float)/sc;want=np.asarray(want,dtype=float)
   ok=got.shape==want.shape and np.allclose(got,want,rtol=1e-6,atol=1e-6,equal_nan=True)
   detail={'status':'pass' if ok else 'wrong','expected_elements':len(want),'actual_elements':len(got),'max_absolute_error':float(np.max(np.abs(got-want))) if got.shape==want.shape and len(want) else None,'rtol':1e-6,'atol':1e-6}
   (d/(side+'.comparison.json')).write_text(json.dumps({'expected':want.tolist(),'actual':got.tolist(),**detail},indent=2)+'\n')
   r['correctness'][side]=detail
  except Exception as e:r['correctness'][side]={'status':'error','error':str(e)}
 good=all(isinstance(x,dict) and x.get('status')=='pass' for x in r['correctness'].values())
 r['category']=('accepted_as_written' if a=='as_written' else 'accepted_with_rewrite') if good else 'wrong'
 (d/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(p,r['category'],r['correctness'],flush=True)
if __name__=='__main__':
 for p in (sys.argv[1:] or M):
  if (BASE/p/'result.json').exists():check(p)
