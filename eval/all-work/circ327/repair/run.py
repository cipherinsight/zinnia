import pathlib,json,os,subprocess,re,time,ast,copy,sys,concurrent.futures,traceback
import numpy as np
BASE=pathlib.Path(__file__).resolve().parent
ROOT=pathlib.Path('/home/zhantong/Projects/zinnia-project/wt/literal')
BIN='/home/zhantong/.cache/zkcc-reconfirm/circ-stock/target/release/examples/'
MEM='/home/zhantong/Projects/zinnia-project/zinnia-src/eval/tools/memrun.sh'
ENV=dict(os.environ,TMPDIR=str(BASE/'tmp'),MEMRUN_MAX='8G',ZSHARP_STDLIB_PATH='/home/zhantong/.cache/zkcc-reconfirm/circ-stock/src/third_party/ZoKrates/zokrates_stdlib/stdlib',PYTHONDONTWRITEBYTECODE='1')
Q=2**32
P=52435875175126190479447740508185965837690552500527637822603658699938581184513
MUT={'pythran_fbcorr':['output'],'npbench_polybench_k2mm_k2mm':['D'],'npbench_polybench_gemm_gemm':['C'],'npbench_polybench_gemver_gemver':['A','w','x'],'npbench_polybench_trmm_trmm':['B'],'npbench_polybench_mvt_mvt':['x1','x2'],'npbench_polybench_symm_symm':['C'],'npbench_polybench_syr2k_syr2k':['C'],'npbench_polybench_lu_lu':['A'],'npbench_polybench_trisolv_trisolv':['x'],'npbench_polybench_doitgen_doitgen':['A'],'npbench_scattering_self_energies_scattering_self_energies':['Sigma'],'npbench_weather_stencils_hdiff_hdiff':['out_field'],'npbench_weather_stencils_vadv_vadv':['utens_stage'],'pythran_quicksort':['l'],'pythran_lu':['M'],'npbench_cavity_flow_cavity_flow':['u','v','p']}
def fixture(p):
 path=ROOT/p['kernel'];f=path.with_name('fixture.'+p['function']+'.json')
 if not f.exists():f=path.with_name('fixture.json')
 return f,json.loads(f.read_text())
def oracle(p,f,tag):
 tree=ast.parse((ROOT/p['kernel']).read_text())
 for n in ast.walk(tree):
  if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
   n.decorator_list=[];n.returns=None
   for a in n.args.args:a.annotation=None
 tree.body=[n for n in tree.body if not(isinstance(n,ast.ImportFrom) and n.module=='zinnia')]
 ns={};exec(compile(ast.fix_missing_locations(tree),p['kernel'],'exec'),ns)
 def val(v,d):
  if d.get('kind')=='tuple':return tuple(val(x,t) for x,t in zip(v,d['items']))
  dt=d['dtype']
  if dt=='complex':
   if isinstance(v,dict):v=np.asarray(v['real'])+1j*np.asarray(v['imag'])
   else:a=np.asarray(v);v=a[...,0]+1j*a[...,1]
  if d.get('shape'):return np.array(v,dtype={'float':float,'int':np.int64,'integer':np.int64,'complex':complex,'bool':bool,'boolean':bool}[dt])
  return v
 args={d['name']:val(f['inputs'][tag].get(d['name'],f.get('constants',{}).get(d['name'])),d) for d in f['parameters']}
 args.update(f.get('constants',{}));res=ns[p['function']](**args)
 if p['program']=='pythran_spectraldns' and p['function']=='add_pressure_diffusion':res=(res,args['dU'])
 if res is None and p['program'] in MUT:
  res=tuple(args[k] for k in MUT[p['program']]);res=res[0] if len(res)==1 else res
 return res

def flat(v):
 if v is None:return []
 if isinstance(v,tuple):return sum([flat(x) for x in v],[])
 a=np.asarray(v)
 if np.iscomplexobj(a):return [x for z in a.reshape(-1) for x in (float(z.real),float(z.imag))]
 return a.reshape(-1).tolist()

def pin(p,f,tag,code):
 pins=[]
 sig=re.search(r'def main\((.*?)\)\s*->',code,re.S)[1]
 def rec(n,v,dt,typ,depth):
  if depth:
   for i,x in enumerate(v):rec(n+'.'+str(i),x,dt,typ,depth-1)
  elif dt=='complex':
   if isinstance(v,dict):v=[v['real'],v['imag']]
   for suffix,x in zip(['re','im'] if typ=='Complex' else ['0','1'],v):rec(n+'.'+suffix,x,'float',('field' if typ=='field' or (typ=='Complex' and 'field re' in code) else 'u64'),0)
  else:
   if dt in ['bool','boolean']:enc=str(bool(v)).lower()
   else:
    num=round(float(v)*Q) if dt=='float' else int(v)
    enc=('#f'+str(num%P)) if typ=='field' else '#x'+format(num%(2**(32 if typ=='u32' else 64)), '08x' if typ=='u32' else '016x')
   pins.append('('+n+' '+enc+')')
 for d in f['parameters']:
  n=d['name'];m=re.search(r'private\s+(\w+)(?:\[[^]]+\])*\s+'+re.escape(n)+r'\b',sig)
  if not m:continue
  v=f['inputs'][tag].get(n,f.get('constants',{}).get(n));rec(n,v,d['dtype'],m[1],len(d.get('shape',[])))
 return '(set_default_modulus '+str(P)+' (let (\n'+'\n'.join(pins)+'\n) false))\n'

def check(d,p,mode,code):
 fp,f=fixture(p);out={}
 for tag in ['A','B']:
  try:
   expected=oracle(p,f,tag);ef=flat(expected)
   if p['program']=='pythran_laplace':ef=[float(v) for v in ef]
   (d/(tag+'.expected.json')).write_text(json.dumps(ef))
   (d/(tag+'.pin')).write_text(pin(p,f,tag,code))
   cmd=[MEM,'timeout','300',BIN+'zxi',str(d/(mode+'.zok')),str(d/(tag+'.pin'))]
   proc=subprocess.run(cmd,env=ENV,cwd=BASE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
   (d/(mode+'.'+tag+'.zxi.log')).write_text(proc.stdout)
   if proc.returncode:out[tag]={'status':'timeout' if proc.returncode==124 else 'out_of_memory' if proc.returncode in [-9,137] else 'interpreter_error','returncode':proc.returncode,'error':proc.stdout[:2000],'command':cmd};continue
   txt=re.sub(r'Complex \{ im: ([^,]+), re: ([^,]+), \}', r'Complex { re: \2, im: \1, }', proc.stdout)
   toks=re.findall(r'0x[0-9a-fA-F]+|\d+f\b|\btrue\b|\bfalse\b',txt)
   vals=[int(t[:-1])-P if t.endswith('f') and not t.startswith('0x') and int(t[:-1])>P//2 else int(t[:-1]) if t.endswith('f') and not t.startswith('0x') else int(t,16) if t.startswith('0x') else t=='true' for t in toks]
   vals=[v-2**64 if t.startswith('0x') and isinstance(v,int) and 2**63<=v<2**64 else v for v,t in zip(vals,toks)]
   n=p['program']
   if n in ['case0417']:vals=vals[1:1+vals[0]]
   if n in ['case0416','case0418']:w=vals[0];vals=[vals[1+i*6+j] for i in range(8) for j in range(w)]
   if n=='case0388':count,a,b=vals[:3];vals=vals[3:3+count*a*b]
   if n=='npbench_polybench_covariance_covariance':m=vals[0];vals=[vals[1+i*14+j] for i in range(m) for j in range(m)]
   if n=='pythran_lttb_pt':m=vals[0];vals=vals[1:1+2*m]
   if n in ['pythran_cdotc','pythran_cdotu']:vals=vals[1:] if expected is not None else []
   if n=='pythran_caxpy':vals=vals[1:] if expected is not None else []
   if n in ['pythran_fibo','pythran_fibo_seq']:
    vals=[sum((v%2**64)*2**(64*i) for i,v in enumerate(vals))]
   if n=='pythran_laplace':
    side=vals[0];vals=[vals[1+i*1024+j] for i in range(side) for j in range(side)]
   if n=='pythran_fannkuch':
    assert vals==[False], 'None presence tag must be false'
    vals=[]
   actual=[v/Q if isinstance(e,(float,np.floating)) else v for v,e in zip(vals,ef)]
   good=len(vals)==len(ef) and all(abs(v-e)<=1e-6+1e-6*abs(e) if isinstance(e,(float,np.floating)) else v==e for v,e in zip(actual,ef))
   err=max([abs(float(v)-float(e)) for v,e in zip(actual,ef)],default=0)
   out[tag]={'status':'pass' if good else 'wrong','expected_elements':len(ef),'actual_elements':len(vals),'max_abs_error':err,'command':cmd}
   (d/(tag+'.decoded.json')).write_text(json.dumps(actual))
  except Exception:out[tag]={'status':'check_error','error':traceback.format_exc()}
 return out

def process(d):
 p=json.load(open(d/'entry.json'));old=json.load(open(d/'old-result.json'))
 meta=json.load(open(d/'translation.json'));
 if meta.get('preserve_attempt'):old=json.load(open(d/'before-native-field.result.json'))
 mode=meta['mode'];code=(d/(mode+'.zok')).read_text()
 r={'program':p['program'],'function':p['function'],'category':None,'as_written':copy.deepcopy(old.get('as_written')) if mode=='rewrite' else None,'rewrite':None,'correctness':{t:{'status':'not_run','reason':'No accepted compilation'} for t in ['A','B']},'notes':meta.get('notes','Q32 values use signed u64; native operations, explicit signs and 32-bit product parts. No arbitrary-integer backend operations.'),'source':p,'numeric_encoding':{'real_scale':Q,'signed':'two-complement u64','rtol':1e-6,'atol':1e-6}}
 cmd=[MEM,'timeout','300',BIN+'circ','--language','zsharp',str(d/(mode+'.zok')),'r1cs','--action','count']
 t=time.monotonic();proc=subprocess.run(cmd,env=ENV,cwd=BASE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);elapsed=time.monotonic()-t
 (d/(mode+'.compile.log')).write_text(proc.stdout)
 (d/(mode+'.command.json')).write_text(json.dumps({'argv':cmd,'env':{k:ENV[k] for k in ['TMPDIR','MEMRUN_MAX','ZSHARP_STDLIB_PATH']},'returncode':proc.returncode,'seconds':elapsed},indent=2))
 match=re.search(r'Final r1cs: (\d+) constraints',proc.stdout)
 status='accepted' if proc.returncode==0 and match else 'timeout' if proc.returncode==124 else 'out_of_memory' if proc.returncode in [137,-9] else 'rejected'
 attempt={'status':status,'constraints':int(match[1]) if match else None,'error':None if status=='accepted' else '\n'.join(proc.stdout.splitlines()[:20]),'seconds':elapsed,'command':cmd}
 if mode=='rewrite':attempt['rewrites']=meta.get('rewrites',[])
 r[mode]=attempt;r['category']='accepted_as_written' if mode=='as_written' else 'accepted_with_rewrite'
 if status=='accepted':
  r['correctness']=check(d,p,mode,code)
  if not all(x['status']=='pass' for x in r['correctness'].values()):r['category']='wrong'
 else:r['category']='timeout_or_oom' if status in ['timeout','out_of_memory'] else 'circ_error'
 (d/'result.json').write_text(json.dumps(r,indent=2)+'\n')
 print(d.name,r['category'],{k:v['status'] for k,v in r['correctness'].items()},flush=True)
 return r

if __name__=='__main__':
 ds=[BASE/n for n in sys.argv[1:]] if len(sys.argv)>1 else [p.parent for p in BASE.glob('*/translation.json')]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  fs={pool.submit(process,d):d for d in ds}
  for f in concurrent.futures.as_completed(fs):
   try:f.result()
   except Exception:print('RUN_ERROR',fs[f],traceback.format_exc(),flush=True)
