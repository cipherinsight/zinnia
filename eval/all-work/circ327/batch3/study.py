import pathlib,json,os,subprocess,re,time,ast,copy,collections,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
BASE=pathlib.Path('/home/zhantong/Projects/zinnia-project/wt/literal')
CIRC='/home/zhantong/.cache/zkcc-reconfirm/circ-stock/target/release/examples/'
MEM='/home/zhantong/Projects/zinnia-project/zinnia-src/eval/tools/memrun.sh'
P=52435875175126190479447740508185965837690552500527637822603658699938581184513
S=2**32
ENV=dict(os.environ,TMPDIR=str(ROOT/'tmp'),MEMRUN_MAX='8G',ZSHARP_STDLIB_PATH='/home/zhantong/.cache/zkcc-reconfirm/circ-stock/src/third_party/ZoKrates/zokrates_stdlib/stdlib',PYTHONDONTWRITEBYTECODE='1')
BATCH=json.load(open(ROOT.parent/'batch3.json'))
DATA={x['program']:x for x in BATCH}
HELP=f'''// Signed integers use their canonical field residues; real inputs are Q32.
def neg(field x) -> bool:
    return x > {P//2}
def absv(field x) -> field:
    return if neg(x) then -x else x fi
def lt(field a, field b) -> bool:
    return if neg(a) != neg(b) then neg(a) else a < b fi
// Constant integer rescaling uses native remainder and field division only.
def divc(field x, field c) -> field:
    field m = absv(x)
    field q = (m - m % c) / c
    return if neg(x) then -q else q fi
def mulq(field a, field b) -> field:
    return divc(a * b, 4294967296)
'''
def fixture(name):return json.load(open(BASE/pathlib.Path(DATA[name]['kernel']).parent/'fixture.json'))
def sig(name,types=None):
 return ', '.join('private '+(types or {}).get(p['name'],'field')+''.join('['+str(n)+']' for n in p.get('shape',[]))+' '+p['name'] for p in fixture(name)['parameters'])
def program(name,rettype,body,attempt='as_written',extra='',types=None):
 d=ROOT/name; d.mkdir(exist_ok=True)
 (d/(attempt+'.zok')).write_text('// '+DATA[name]['kernel']+'\n'+HELP+extra+'\ndef main('+sig(name,types)+') -> '+rettype+':\n'+body.strip('\n')+'\n')
def result(name,category,reason,status='not_expressible',rewrites=None):
 d=ROOT/name;d.mkdir(exist_ok=True)
 r={'program':name,'category':category,'as_written':{'status':status,'error':reason,'constraints':None},'rewrite':{'status':'not_expressible','error':reason,'constraints':None,'rewrites':rewrites or []},'correctness':{'A':{'status':'not_run','reason':'No accepted attempt'},'B':{'status':'not_run','reason':'No accepted attempt'}},'notes':reason,'kernel':DATA[name]['kernel']}
 (d/'result.json').write_text(json.dumps(r,indent=2)+'\n'); (d/'assessment.md').write_text(reason+'\n')
 return r
def compiled(name,attempt='as_written',rewrites=None,initial=None):
 d=ROOT/name
 cmd=[MEM,'timeout','300',CIRC+'circ','--language','zsharp',str(d/(attempt+'.zok')),'r1cs','--action','count']
 t=time.monotonic()
 run=subprocess.run(cmd,env=ENV,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 log=run.stdout;(d/(attempt+'.compile.log')).write_text(log)
 (d/(attempt+'.command.json')).write_text(json.dumps({'argv':cmd,'env':{k:ENV[k] for k in ['TMPDIR','MEMRUN_MAX','ZSHARP_STDLIB_PATH']},'returncode':run.returncode,'seconds':time.monotonic()-t},indent=2))
 m=re.search(r'Final r1cs: ([\d,]+) constraints',log)
 status='accepted' if run.returncode==0 and m else 'timeout' if run.returncode==124 else 'out_of_memory' if run.returncode in [137,-9] else 'rejected'
 err=None if status=='accepted' else '\n'.join(log.splitlines()[:18])
 a={'status':status,'error':err,'constraints':int(m[1].replace(',','')) if m else None}
 r=initial or {'program':name,'as_written':a,'rewrite':None,'correctness':{'A':{'status':'not_run'},'B':{'status':'not_run'}},'notes':'Scale 2^32; signed field residues. Native arithmetic and array operations only.','kernel':DATA[name]['kernel']}
 r[attempt]=dict(a,**({'rewrites':rewrites or []} if attempt=='rewrite' else {}))
 r['category']='accepted_'+('as_written' if attempt=='as_written' else 'with_rewrite') if status=='accepted' else 'timeout_or_oom' if status in ['timeout','out_of_memory'] else 'circ_error'
 (d/'result.json').write_text(json.dumps(r,indent=2)+'\n')
 print(name,attempt,status,a['constraints'],flush=True)
 return r

def oracle(name,label):
 x=DATA[name]; tree=ast.parse((BASE/x['kernel']).read_text())
 for n in tree.body:
  if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
   n.decorator_list=[]
   for a in n.args.args:a.annotation=None
   n.returns=None
 tree.body=[n for n in tree.body if not(isinstance(n,ast.ImportFrom) and n.module=='zinnia')]
 ns={};exec(compile(ast.fix_missing_locations(tree),x['kernel'],'exec'),ns)
 f=fixture(name); inputs={}
 for p in f['parameters']:
  v=copy.deepcopy(f['inputs'][label][p['name']])
  if p['dtype']=='complex':
   if isinstance(v,dict):v=np.array(v['real'])+1j*np.array(v['imag'])
   else:
    v=np.array(v);v=v[...,0]+1j*v[...,1]
  if p['kind']=='array':v=np.array(v,dtype={'float':float,'int':np.int64,'complex':complex}[p['dtype']])
  inputs[p['name']]=v
 out=ns[x['function']](**inputs)
 if out is None:
  mutated={'npbench_polybench_syr2k_syr2k':'C','npbench_polybench_lu_lu':'A','npbench_polybench_cholesky_cholesky':'A','pythran_quicksort':'l'}
  out=inputs[mutated[name]]
 return out

def pin(name,label):
 f=fixture(name);entries=[]
 def rec(key,v,dtype):
  if isinstance(v,list):
   for i,item in enumerate(v):rec(key+'.'+str(i),item,dtype)
  else:entries.append('('+key+' #f'+str((round(float(v)*S) if dtype=='float' else int(v))%P)+')')
 for p in f['parameters']:
  v=f['inputs'][label][p['name']]
  if p['dtype']=='complex':
   def comp(key,v,depth):
    if depth:
     for i,el in enumerate(v):comp(key+'.'+str(i),el,depth-1)
    else:
     rec(key+'.re',v[0],'float');rec(key+'.im',v[1],'float')
   comp(p['name'],v,len(p.get('shape',[])))
  elif name=='pythran_factorize_naive':entries.append('('+p['name']+' #x'+format(int(v),'08x')+')')
  else:rec(p['name'],v,p['dtype'])
 path=ROOT/name/(label+'.pin');path.write_text('(set_default_modulus '+str(P)+' (let (\n'+'\n'.join(entries)+'\n) true))\n');return path

def validate(name,attempt='as_written',decode=None,scale=S):
 d=ROOT/name;r=json.load(open(d/'result.json'))
 if r[attempt]['status']!='accepted':return
 for label in ['A','B']:
  inp=pin(name,label);run=subprocess.run([CIRC+'zxi',str(d/(attempt+'.zok')),str(inp)],env=ENV,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  (d/(attempt+'.'+label+'.zxi.log')).write_text(run.stdout)
  vals=[]
  if run.returncode==0 and name=='pythran_factorize_naive':
   vals=[int(v,16) for v in re.findall(r'0x([0-9a-fA-F]+)',run.stdout)]
  if run.returncode==0:
   for s in re.findall(r'(-?\d+)f\b|\b(true|false)\b',run.stdout):
    if s[1]:vals.append(s[1]=='true')
    else:
     v=int(s[0]);vals.append(v-P if v>P//2 else v)
  try:
   expected=oracle(name,label)
   actual=decode(vals,label) if decode else np.array(vals,dtype=float)/scale
   def flatten(v):
    if isinstance(v,(list,tuple)):return [z for x in v for z in flatten(x)]
    a=np.asarray(v)
    if np.iscomplexobj(a):return np.stack([a.real,a.imag],axis=-1).reshape(-1).tolist()
    return a.reshape(-1).tolist()
   a=np.array(flatten(actual));e=np.array(flatten(expected))
   ok=run.returncode==0 and a.shape==e.shape and np.allclose(a,e,rtol=1e-6,atol=1e-6,equal_nan=True)
   info={'status':'pass' if ok else 'wrong','numpy_shape':list(np.shape(expected)) if not isinstance(expected,list) else 'list','compared_values':len(e),'max_absolute_error':float(np.max(np.abs(a-e))) if a.shape==e.shape and len(e) else None,'actual':a.tolist(),'expected':e.tolist()}
  except Exception as ex:info={'status':'error','error':str(ex),'interpreter_returncode':run.returncode}
  r['correctness'][label]=info
  print('CHECK',name,label,info['status'],info.get('max_absolute_error'),flush=True)
 if any(v['status']!='pass' for v in r['correctness'].values()):r[attempt]['status']='wrong';r['category']='wrong'
 (d/'result.json').write_text(json.dumps(r,indent=2)+'\n')
