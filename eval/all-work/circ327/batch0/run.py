import os,json,pathlib,subprocess,time,re,ast,sys,concurrent.futures
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
BIN=pathlib.Path('/home/zhantong/.cache/zkcc-reconfirm/circ-stock/target/release/examples')
MEM='/home/zhantong/Projects/zinnia-project/zinnia-src/eval/tools/memrun.sh'
P=52435875175126190479447740508185965837690552500527637822603658699938581184513
ENV=dict(os.environ,TMPDIR=str(ROOT/'tmp'),MEMRUN_MAX='8G',ZSHARP_STDLIB_PATH='/home/zhantong/.cache/zkcc-reconfirm/circ-stock/src/third_party/ZoKrates/zokrates_stdlib/stdlib',PYTHONDONTWRITEBYTECODE='1')
spec=json.load(open(ROOT/'spec.json'))
batch={p['program']:p for p in json.load(open(ROOT.parent/'batch0.json'))}
def command(cmd,log):
 t=time.monotonic()
 with open(log,'w') as out:
  r=subprocess.run([MEM,'timeout','300']+list(map(str,cmd)),env=ENV,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
 s=pathlib.Path(log).read_text();status='accepted' if r.returncode==0 else 'timeout' if r.returncode==124 else 'out of memory' if r.returncode in [137,-9] else 'rejected'
 return dict(status=status,error=None if status=='accepted' else s[:2200],constraints=None,returncode=r.returncode,seconds=round(time.monotonic()-t,3),command=list(map(str,[MEM,'timeout','300']+cmd))),s
def flatten_input(name,v,dtype):
 if isinstance(v,list):
  return sum((flatten_input(f'{name}.{i}',x,dtype) for i,x in enumerate(v)),[])
 if dtype=='bool':s=str(v).lower()
 else:s='#f'+str(round(v*2**32) if dtype=='float' else int(v))
 return [f'({name} {s})']
def pin(f,label,path):
 a=[]
 for p in f['parameters']:a+=flatten_input(p['name'],f['inputs'][label][p['name']],p['dtype'])
 path.write_text(f'(set_default_modulus {P}\n(let (\n'+ '\n'.join(a)+'\n) false))\n')
def reference(d,f,label):
 tree=ast.parse((d/'kernel.py').read_text())
 tree.body=[n for n in tree.body if not isinstance(n,ast.ImportFrom) or n.module!='zinnia']
 for node in ast.walk(tree):
  if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
   node.decorator_list=[];node.returns=None
   for arg in node.args.args:arg.annotation=None
 ns={};exec(compile(ast.fix_missing_locations(tree),str(d/'kernel.py'),'exec'),ns)
 if not hasattr(np,'in1d'):np.in1d=np.isin
 args={p['name']:np.array(f['inputs'][label][p['name']],dtype={'float':float,'int':np.int64,'bool':bool}.get(p['dtype'])) if p['kind']=='array' else f['inputs'][label][p['name']] for p in f['parameters']}
 result=ns[f['function']](**args)
 return result

def run(n):
 d=ROOT/n;s=spec[n];mode='rewrite' if s.get('rewrite') else 'as_written'
 original=None
 if mode=='rewrite' and not (d/'as_written.zok').read_text().lstrip().startswith('//'):
  original,original_log=command([BIN/'circ','--language','zsharp',d/'as_written.zok','r1cs','--action','count'],d/'as_written.compile.log')
  original_counts=re.findall(r'(\d+) constraints',original_log)
  if original_counts:original['constraints']=int(original_counts[-1])
 r,log=command([BIN/'circ','--language','zsharp',d/f'{mode}.zok','r1cs','--action','count'],d/f'{mode}.compile.log')
 counts=re.findall(r'(\d+) constraints',log)
 if counts:r['constraints']=int(counts[-1])
 result=dict(program=n,category='accepted_'+('with_rewrite' if mode=='rewrite' else 'as_written') if r['status']=='accepted' else 'timeout_or_oom' if r['status'] in ['timeout','out of memory'] else 'circ_error',as_written=r if mode=='as_written' else original or dict(status='not expressible',error=s['reason'],constraints=None),rewrite=None if mode=='as_written' else dict(r,rewrites=s['rewrite']),correctness={'A':'not run','B':'not run'},notes=s['reason'])
 if r['status']=='accepted':
  f=json.load(open(d/'fixture.json'))
  for label in ['A','B']:
   pin(f,label,d/f'{label}.pin')
   ir,output=command([BIN/'zxi',d/f'{mode}.zok',d/f'{label}.pin'],d/f'{mode}.{label}.zxi.log')
   try:
    ref=reference(d,f,label)
    # parse in a later pass once the concrete pretty-printer representation is known.
    def serial(x):
     if isinstance(x,np.ndarray):return x.tolist()
     if isinstance(x,(list,tuple)):return [serial(y) for y in x]
     if isinstance(x,np.generic):return x.item()
     return x
    (d/f'{label}.expected.json').write_text(json.dumps(serial(ref)))
    result['correctness'][label]={'status':'pending comparison','interpreter_status':ir['status']}
   except Exception as e:result['correctness'][label]={'status':'reference error','error':repr(e)}
 (d/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(n,r['status'],r['constraints'],flush=True)
if __name__=='__main__':
 names=sys.argv[1:] or list(spec)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,names))
