import ast,json,pathlib,subprocess,os,re,time,copy
import numpy as np
BASE=pathlib.Path(__file__).resolve().parent
ROOT=pathlib.Path('/home/zhantong/Projects/zinnia-project/wt/literal')
CIRC=pathlib.Path('/home/zhantong/.cache/zkcc-reconfirm/circ-stock')
ENV=dict(os.environ,TMPDIR=str(BASE/'tmp'),MEMRUN_MAX='8G',ZSHARP_STDLIB_PATH=str(CIRC/'src/third_party/ZoKrates/zokrates_stdlib/stdlib'))
BATCH=json.load(open(BASE.parent/'batch5.json'))
S=2**32
spec={}
LT='def lt(u64 a, u64 b) -> bool:\n    return (a ^ 0x8000000000000000) < (b ^ 0x8000000000000000)\n\n'
def add(n,params,out,body,**kw): spec[n]=dict(params=params,out=out,body=body,**kw)
def ztype(t,sh): return t+''.join('['+str(x)+']' for x in sh)
def zero(t,sh):
 v={'u64':'0u64','field':'0f','integer':'0int','bool':'false'}[t]
 for n in reversed(sh):v='['+v+'; '+str(n)+']'
 return v
def source(p,s):
 f=fixture_for(p)
 args=[]
 for a in f['parameters']:
  t=s.get('types',{}).get(a['name'],s.get('type','u64'))
  if a['dtype'] in ('bool','boolean'):t='bool'
  args.append('private '+ztype(t,a.get('shape',[]))+' '+a['name'])
 return s.get('prefix','')+'def main('+', '.join(args)+') -> '+s['out']+':\n'+s['body'].strip('\n')+'\n'
add('case0309',None,'u64','''    u64 best = a[0][0]
    u64 idx = 0u64
    for u32 i in 0..8 do
        for u32 j in 0..6 do
            bool take = lt(best, a[i][j])
            idx = if take then u64idx(i, j) else idx fi
            best = if take then a[i][j] else best fi
        endfor
    endfor
    return idx''',prefix=LT+'from "utils/casts/u32_to_u64" import main as widen\ndef u64idx(u32 i, u32 j) -> u64:\n    return widen(i * 6u32 + j)\n')
add('case0317',None,'u64[10][20][5]','    u64[10][20][5] result = a\n    return result')
add('case0336',None,'u64[8]','''    u64[8][8] flipped = [[0u64; 8]; 8]
    for u32 i in 0..8 do
        for u32 j in 0..8 do
            flipped[i][j] = a[i][7u32-j]
        endfor
    endfor
    u64[8] result = [0u64; 8]
    for u32 i in 0..8 do
        result[i] = flipped[i][i]
    endfor
    return result''')
add('case0342',None,'u64[30]','''    u64[30] result = [0u64; 30]
    for u32 i in 0..5 do
        for u32 j in 0..6 do
            result[i*6u32+j] = X[i][j]
        endfor
    endfor
    return result''',notes='Append count is statically exactly 30; list is represented by a 30-element array.')
add('case0349',None,'u64','    return 6u64')
add('case0364',None,'u64[101][10]','''    u64 p = if lt(pos, 0u64) then pos+100u64 else pos fi
    u64[101][10] result = [[0u64; 10]; 101]
    for u64 i in 0..101 do
        for u32 j in 0..10 do
            u64 ix = if i > p then i-1u64 else i fi
            ix = if ix < 100u64 then ix else 99u64 fi
            result[i][j] = if i == p then element[j] else a[ix][j] fi
        endfor
    endfor
    return result''',prefix=LT,notes='Scalar insertion always has shape 101x10; normalize valid negative positions. Guard unused gather index at inserted final row.')
add('case0400',None,'u64[0]','    return [0u64; 0]')
add('case0436',None,'bool[10][5]','''    bool[10][5] result = [[false; 5]; 10]
    for u32 i in 0..10 do
        u64 m = a[i][0]
        for u32 j in 1..5 do
            m = if lt(m, a[i][j]) then a[i][j] else m fi
        endfor
        for u32 j in 0..5 do
            result[i][j] = m == a[i][j]
        endfor
    endfor
    return result''',prefix=LT)
add('case0451',None,'u64[20][10][10][2]','    return '+zero('u64',[20,10,10,2]))
add('case0472',None,'u64[1][2][9][3][7][4]','    return [a[5]]')
add('case0502',None,'u64[13][13]','''    u64[13][13] result = [[0u64; 13]; 13]
    for u32 i in 0..13 do
        for u32 j in 0..13 do
            u64 ix = if lt(b[i][j], 0u64) then b[i][j]+2u64 else b[i][j] fi
            result[i][j] = a[i][j][ix]
        endfor
    endfor
    return result''',prefix=LT)
add('npbench_compute_compute',None,'u64[16][16]','''    u64[16][16] result = [[0u64; 16]; 16]
    for u32 i in 0..16 do
        for u32 j in 0..16 do
            u64 x = if lt(array_1[i][j], 2u64) then 2u64 else if lt(10u64, array_1[i][j]) then 10u64 else array_1[i][j] fi fi
            result[i][j] = x*a+array_2[i][j]*b+c
        endfor
    endfor
    return result''',prefix=LT)
add('npbench_polybench_floyd_warshall_floyd_warshall',None,'u64[16][16]','''    for u32 k in 0..16 do
        u64[16][16] old = path
        for u32 i in 0..16 do
            for u32 j in 0..16 do
                u64 v = old[i][k]+old[k][j]
                path[i][j] = if lt(v, old[i][j]) then v else old[i][j] fi
            endfor
        endfor
    endfor
    return path''',prefix=LT,mutated=['path'])
add('pythran_create_grid',None,'u64[256][3]','''    u64[16][16][3] z = [[[0u64; 3]; 16]; 16]
    for u32 i in 0..16 do
        for u32 j in 0..16 do
            z[i][j][0] = x[i]
            z[i][j][1] = x[j]
        endfor
    endfor
    u64[256][3] result = [[0u64; 3]; 256]
    for u32 i in 0..16 do
        for u32 j in 0..16 do
            result[i*16u32+j] = z[i][j]
        endfor
    endfor
    return result''')
add('pythran_multitype',None,'(u64[1], bool[1])','    return if y[0] then x else [x[0]+x[0]] fi, y')
add('pythran_periodic_bc',None,'u64[3][3][3]','''    for u32 i in 0..3 do
        for u32 k in 0..3 do
            f[i][0][k] = f[i][1][k]
        endfor
    endfor
    return f''')
add('pythran_reverse_cumsum',None,'u64[10]','''    u64[10] rev = [0u64; 10]
    for u32 i in 0..10 do
        rev[i] = x[9u32-i]
    endfor
    u64 total = 0u64
    u64[10] result = [0u64; 10]
    for u32 i in 0..10 do
        total = total + rev[i]
        result[9u32-i] = total
    endfor
    return result''')
# Bounded filtering: output includes logical length and zero-padded storage.
add('case0442',None,'(u64, u64[20])','''    u64[20] C = [0u64; 20]
    u64 n = 0u64
    for u32 i in 0..20 do
        bool found = false
        for u32 j in 0..3 do
            found = found || A[i] == B[j]
        endfor
        C[n] = if !found then A[i] else C[n] fi
        n = if !found then n+1u64 else n fi
    endfor
    return n, C''',attempt='rewrite',rewrites=['Replace boolean-indexed output with capacity 20, masked stable compaction, and public logical length; at most all 20 entries of A survive.'],dynamic=('dynamic shape','C = A[~np.in1d(A,B)]'))
# Remove dead sorting only in the second attempt, preserving every returned value.
add('case0373',None,'u64[50]','    return ['+', '.join(str(round((i+1)*S/50))+'u64' for i in range(50))+']',attempt='rewrite',rewrites=['Remove unused sorted values: ecdf_result returns only arange(1,51)/50; len(sort(grades)) is always 50. Constant output is encoded at scale 2^32.'],gadget='sort')
def run_cmd(cmd,log):
 start=time.time()
 p=subprocess.run(cmd,env=ENV,cwd=BASE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 log.write_text(p.stdout)
 return p.returncode,p.stdout,time.time()-start

def compile_zok(f):
 cmd=['/home/zhantong/Projects/zinnia-project/zinnia-src/eval/tools/memrun.sh','timeout','300',str(CIRC/'target/release/examples/circ'),'--language','zsharp',str(f),'r1cs','--action','count']
 rc,txt,t=run_cmd(cmd,f.with_suffix('.compile.log'))
 status='accepted' if rc==0 else 'timeout' if rc==124 else 'out of memory' if rc in (137,-9) else 'rejected'
 counts=re.findall(r'(\d+)\s+constraints',txt)
 if not counts: counts=re.findall(r'constraints[:\s]+(\d+)',txt)
 return dict(status=status,error=None if rc==0 else '\n'.join(txt.splitlines()[:14]),constraints=int(counts[-1]) if counts else None,seconds=t,command=cmd)

def load_reference(p):
 tree=ast.parse((ROOT/p['kernel']).read_text())
 tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module=='zinnia')]
 for n in ast.walk(tree):
  if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
   n.decorator_list=[];n.returns=None
   for a in n.args.args:a.annotation=None
 ns={};exec(compile(ast.fix_missing_locations(tree),str(ROOT/p['kernel']),'exec'),ns)
 return ns[p['function']]
def flat(v):
 if isinstance(v,tuple):return sum((flat(x) for x in v),[])
 return np.asarray(v).reshape(-1).tolist()
def correctness(p,s,f):
 fixture=fixture_for(p)
 fn=load_reference(p); result={}
 for tag in ['A','B']:
  values=fixture['inputs'][tag];args={};pins=[]
  for param in fixture['parameters']:
   name=param['name'];kind=param['dtype'];val=values.get(name,fixture.get('constants',{}).get(name))
   arr=np.asarray(val,dtype={'float':float,'int':np.int64,'integer':np.int64,'bool':bool,'boolean':bool}.get(kind,float))
   args[name]=arr.copy() if param.get('shape') else arr.item()
   for ix in np.ndindex(arr.shape):
    v=arr[ix].item();n=name+''.join('.'+str(k) for k in ix)
    typ=s.get('types',{}).get(name,s.get('type','u64'))
    if kind in ('bool','boolean'):enc=str(bool(v)).lower()
    elif typ=='integer':enc=str(round(v*S) if kind=='float' else int(v))
    else:enc='#x'+format((round(v*S) if kind=='float' else int(v))%(2**64),'016x')
    pins.append('('+n+' '+enc+')')
  pin=f.parent/(tag+'.pin');pin.write_text('(let (\n'+'\n'.join(pins)+'\n) false)\n')
  expected=fn(**args)
  if s.get('mutated'): expected=tuple(args[k] for k in s['mutated']);expected=expected[0] if len(expected)==1 else expected
  rc,txt,t=run_cmd(['timeout','300',str(CIRC/'target/release/examples/zxi'),str(f),str(pin)],f.parent/(tag+'.zxi.log'))
  if rc:result[tag]=dict(status='interpreter_error',error=txt[:1500]);continue
  tokens=re.findall(r'0x[0-9a-fA-F]+|\btrue\b|\bfalse\b',txt)
  actual=[(int(x,16) if int(x,16)<2**63 else int(x,16)-2**64) if x.startswith('0x') else x=='true' for x in tokens]
  if p['program'] in ('case0442','case0414'):actual=actual[1:1+actual[0]]
  if p['program']=='case0509':
   h,w=actual[:2];padded=actual[2:];actual=[padded[i*10+j] for i in range(h) for j in range(w)]
   assert (h,w)==(expected.shape if expected.size else (0,0))
  if p['program']=='case0390':
   count,side=actual[:2];actual=actual[2:2+count*side*side]
   assert (count,side,side)==expected.shape
  if p['program']=='case0400':actual=[]
  ef=flat(expected)
  # Each numerical output's scale comes from the reference dtype (not its value).
  def dec(v):
   if isinstance(v,tuple):return sum((dec(q) for q in v),[])
   a=np.asarray(v);return [S if a.dtype.kind in 'fc' else 1]*a.size
  scales=dec(expected)
  actual=[v/sc for v,sc in zip(actual,scales)] if len(actual)==len(scales) else actual
  ok=len(actual)==len(ef) and np.allclose(actual,ef,rtol=1e-6,atol=1e-6)
  exact=all(a==e for a,e,sc in zip(actual,ef,scales) if sc==1)
  ok=ok and exact
  result[tag]=dict(status='pass' if ok else 'wrong',elements=len(ef),actual_elements=len(actual),max_abs_error=float(np.max(np.abs(np.array(actual)-np.array(ef)))) if len(ef) and len(actual)==len(ef) else None)
  (f.parent/(tag+'.comparison.json')).write_text(json.dumps(dict(actual=actual,expected=ef,check=result[tag]),indent=1))
 return result

def process(p):
 n=p['program'];d=BASE/n;d.mkdir(exist_ok=True);s=spec[n];attempt=s.get('attempt','as_written')
 src=source(p,s);f=d/(attempt+'.zok');f.write_text(src)
 r=dict(program=n,category=None,as_written=None,rewrite=None,correctness={'A':'not run','B':'not run'},notes=s.get('notes',''))
 if attempt=='rewrite':
  err=s.get('dynamic',('','needs gadget: '+s.get('gadget','')))[1]
  r['as_written']=dict(status='not expressible' if s.get('dynamic') else 'needs gadget',error=err,constraints=None)
  if p['program']=='case0400':
   original=f.parent/'as_written.zok'
   original.write_text('def main() -> u64[0]:\n    return [0u64; 0]\n')
   r['as_written']=compile_zok(original)
  r['rewrite']=compile_zok(f);r['rewrite']['rewrites']=s['rewrites'];a=r['rewrite']
 else:r['as_written']=compile_zok(f);a=r['as_written']
 if a['status']=='accepted':
  r['correctness']=correctness(p,s,f)
  r['category']='accepted_'+('with_rewrite' if attempt=='rewrite' else 'as_written') if all(v.get('status')=='pass' for v in r['correctness'].values()) else 'wrong'
 elif a['status'] in ('timeout','out of memory'):r['category']='timeout_or_oom'
 else:r['category']='circ_error'
 if a['status']=='rejected' and s.get('fallback'):
  r['category']='needs_gadget: '+s['fallback']
  r['rewrite']=dict(status='needs gadget',error='Native integer Q32 arithmetic cannot be lowered to R1CS; a custom wide fixed-point rescaling/division gadget would be required.',constraints=None,rewrites=[])
 (d/'result.json').write_text(json.dumps(r,indent=2));print(n,r['category'],a['constraints'],flush=True)
 return r
# Additional bounded crop rewrite.
add('case0509',None,'(u64, u64, u64[100])','''    bool[10][10] mask = [[false; 10]; 10]
    u64 rmin = 10u64
    u64 rmax = 0u64
    u64 cmin = 10u64
    u64 cmax = 0u64
    for u64 i in 0..10 do
        for u64 j in 0..10 do
            mask[i][j] = im[i][j] == 0u64
        endfor
    endfor
    for u64 i in 0..10 do
        bool row = false
        bool col = false
        for u64 j in 0..10 do
            row = row || mask[i][j]
            col = col || mask[j][i]
        endfor
        rmin = if row && i < rmin then i else rmin fi
        rmax = if row then i else rmax fi
        cmin = if col && i < cmin then i else cmin fi
        cmax = if col then i else cmax fi
    endfor
    u64 h = if rmin == 10u64 then 0u64 else rmax-rmin+1u64 fi
    u64 w = if cmin == 10u64 then 0u64 else cmax-cmin+1u64 fi
    u64[100] out = [0u64; 100]
    for u64 i in 0..10 do
        for u64 j in 0..10 do
            bool active = i < h && j < w
            u64 r = if active then rmin+i else 0u64 fi
            u64 c = if active then cmin+j else 0u64 fi
            u64 dest = if active then i*w+j else 99u64 fi
            out[dest] = if active then im[r][c] else out[dest] fi
        endfor
    endfor
    return h, w, out''',attempt='rewrite',dynamic=('dynamic shape','rows/cols = np.flatnonzero(...), and value-dependent crop at source lines 12-18'),rewrites=['Replace variable-length nonzero-index lists by guarded min/max reductions over the 10 rows/columns.','Pad the crop to capacity 100 and return its logical height and width; a crop of a 10x10 input contains at most 100 values.'])
add('case0390',None,'(u64, u64, u64[20000])','''    u64 p = if patch_size > 100u64 then 1u64 else patch_size fi
    u64 nr = 100u64 / p
    u64 nc = 200u64 / p
    u64[20000] result = [0u64; 20000]
    for u64 i in 0..100 do
        for u64 j in 0..200 do
            bool active = patch_size <= 100u64 && i < nr*p && j < nc*p
            u64 dest = ((j/p)*nr+i/p)*p*p+(i%p)*p+j%p
            dest = if active then dest else 19999u64 fi
            result[dest] = if active then a[i][j] else result[dest] fi
        endfor
    endfor
    return if patch_size > 100u64 then 0u64 else nr*nc fi, patch_size, result''',attempt='rewrite',dynamic=('dynamic shape','x = a[:a.shape[0] // patch_size * patch_size, ...]; dynamic reshape on lines 11-12'),rewrites=['Use a flat padded output capacity 20000, with public patch count and patch side; retained patches contain no more elements than the 100x200 input.','Guard source elements outside whole patches and scatter to the exact transpose/reshape order. Treat patch_size > 100 as empty output without squaring the unbounded side.'])
# Native arbitrary integers avoid silently overflowing Q32 products; test whether the backend supports them.
add('case0420',None,'integer','    return if x < x_min*4294967296int then x_min*4294967296int else if x > x_max*4294967296int then x_max*4294967296int else 3int*(x*x/4294967296int)-2int*((x*x/4294967296int)*x/4294967296int) fi fi',type='integer',fallback='division')
add('pythran_matmul',None,'integer[16][16]','''    integer[16][16] result = [[0int; 16]; 16]
    for u32 i in 0..16 do
        for u32 j in 0..16 do
            integer r = 0int
            for u32 k in 0..16 do
                r = r + m0[i][k]*m1[k][j]/4294967296int
            endfor
            result[i][j] = r
        endfor
    endfor
    return result''',type='integer',fallback='division')
add('npbench_polybench_mvt_mvt',None,'(integer[16], integer[16])','''    for u32 i in 0..16 do
        integer p = 0int
        integer q = 0int
        for u32 j in 0..16 do
            p = p+A[i][j]*y_1[j]/4294967296int
            q = q+y_2[j]*A[j][i]/4294967296int
        endfor
        x1[i] = x1[i]+p
        x2[i] = x2[i]+q
    endfor
    return x1, x2''',type='integer',fallback='division',mutated=['x1','x2'])
add('pythran_fbcorr',None,'integer[4][6][4][4]','''    for u32 ii in 0..4 do
        for u32 rr in 0..4 do
            for u32 cc in 0..4 do
                for u32 hh in 0..5 do
                    for u32 ww in 0..5 do
                        for u32 jj in 0..3 do
                            for u32 ff in 0..6 do
                                integer imgval = imgs[ii][rr+hh][cc+ww][jj]
                                integer filterval = filters[ff][hh][ww][jj]
                                output[ii][ff][rr][cc] = output[ii][ff][rr][cc]+imgval*filterval/4294967296int
                            endfor
                        endfor
                    endfor
                endfor
            endfor
        endfor
    endfor
    return output''',type='integer',fallback='division',mutated=['output'])
add('npbench_polybench_covariance2_covariance2',None,'integer[14][14]','''    integer[14] mean = [0int; 14]
    for u32 j in 0..14 do
        for u32 i in 0..16 do
            mean[j] = mean[j]+data[i][j]
        endfor
        mean[j] = mean[j]/16int
    endfor
    integer[16][14] centered = data
    for u32 i in 0..16 do
        for u32 j in 0..14 do
            centered[i][j] = data[i][j]-mean[j]
        endfor
    endfor
    integer[14][14] out = [[0int; 14]; 14]
    for u32 i in 0..14 do
        for u32 j in 0..14 do
            for u32 k in 0..16 do
                out[i][j] = out[i][j]+centered[k][i]*centered[k][j]/4294967296int
            endfor
            out[i][j] = out[i][j]/15int
        endfor
    endfor
    return out''',type='integer',fallback='division')
# Z# has one return value; use ordinary structs to encode NumPy tuples.
for name,fields,expr in [
 ('case0442',[('u64','a'),('u64[20]','b')],'Result {a: n, b: C}'),
 ('case0509',[('u64','a'),('u64','b'),('u64[100]','c')],'Result {a: h, b: w, c: out}'),
 ('case0390',[('u64','a'),('u64','b'),('u64[20000]','c')],'Result {a: if patch_size > 100u64 then 0u64 else nr*nc fi, b: patch_size, c: result}'),
 ('pythran_multitype',[('u64[1]','a'),('bool[1]','b')],'Result {a: if y[0] then x else [x[0]+x[0]] fi, b: y}'),
 ('npbench_polybench_mvt_mvt',[('integer[16]','a'),('integer[16]','b')],'Result {a: x1, b: x2}')]:
 s=spec[name];s['out']='Result';s['prefix']=s.get('prefix','')+'struct Result {\n'+'\n'.join('    '+t+' '+n for t,n in fields)+'\n}\n'
 s['body']=s['body'].rsplit('    return ',1)[0]+'    return '+expr
add('case0400',None,'u64[1]','    return [0u64]',attempt='rewrite',rewrites=['Encode the empty output by one zero padding slot; logical length is statically zero.'],notes='Original zero-length array is rejected by CirC (Empty array); padding has a known, zero logical length.')

def fixture_for(p):
 d=(ROOT/p['kernel']).parent
 f=d/'fixture.json'
 if not f.exists():f=d/('fixture.'+p['function']+'.json')
 return json.load(open(f))
add('pythran_lapl2d',None,'Result','''    u64[4][4] left = In
    u64[4][4] right = Out
    bool swapped = false
    for u64 it in 0..1024 do
        u64[4][4] next = right
        for u32 i in 1..3 do
            for u32 j in 1..3 do
                u64 v = left[i-1u32][j]+left[i][j-1u32]-4u64*left[i][j]+left[i+1u32][j]+left[i][j+1u32]
                next[i][j] = (v >> 4u32) | (if (v & 0x8000000000000000) != 0u64 then 0xf000000000000000 else 0u64 fi)
            endfor
        endfor
        bool active = it < niter
        right = if active then left else right fi
        left = if active then next else left fi
        swapped = if active then !swapped else swapped fi
    endfor
    return Result {a: if swapped then right else left fi, b: if swapped then left else right fi}''',prefix='struct Result {\n    u64[4][4] a\n    u64[4][4] b\n}\n',attempt='rewrite',dynamic=('dynamic loop','for it in range(0, niter), line 13'),rewrites=['Unroll to 1024 with it < niter guards, directly justified by the declared 0 <= niter <= 1024 precondition.','Expose both original mutated input buffers, preserving local In/Out swaps; h2 is exactly 1/16 and Q32 rescaling uses a sign-extended right shift.'],mutated=['In','Out'])
add('npbench_weather_stencils_hdiff_hdiff',None,'integer[11][11][10]','''    integer[13][13][10] lap = [[[0int; 10]; 13]; 13]
    for u32 i in 0..13 do
        for u32 j in 0..13 do
            for u32 k in 0..10 do
                lap[i][j][k] = 4int*in_field[i+1u32][j+1u32][k]-(in_field[i+2u32][j+1u32][k]+in_field[i][j+1u32][k]+in_field[i+1u32][j+2u32][k]+in_field[i+1u32][j][k])
            endfor
        endfor
    endfor
    integer[12][11][10] flx = [[[0int; 10]; 11]; 12]
    integer[11][12][10] fly = [[[0int; 10]; 12]; 11]
    for u32 i in 0..12 do
        for u32 j in 0..11 do
            for u32 k in 0..10 do
                integer res = lap[i+1u32][j+1u32][k]-lap[i][j+1u32][k]
                flx[i][j][k] = if res*(in_field[i+2u32][j+2u32][k]-in_field[i+1u32][j+2u32][k]) > 0int then 0int else res fi
            endfor
        endfor
    endfor
    for u32 i in 0..11 do
        for u32 j in 0..12 do
            for u32 k in 0..10 do
                integer res = lap[i+1u32][j+1u32][k]-lap[i+1u32][j][k]
                fly[i][j][k] = if res*(in_field[i+2u32][j+2u32][k]-in_field[i+2u32][j+1u32][k]) > 0int then 0int else res fi
            endfor
        endfor
    endfor
    for u32 i in 0..11 do
        for u32 j in 0..11 do
            for u32 k in 0..10 do
                out_field[i][j][k] = in_field[i+2u32][j+2u32][k]-coeff[i][j][k]*(flx[i+1u32][j][k]-flx[i][j][k]+fly[i][j+1u32][k]-fly[i][j][k])/4294967296int
            endfor
        endfor
    endfor
    return out_field''',type='integer',fallback='division',mutated=['out_field'])
# Rectangular padding avoids an unnecessary dynamic scatter when materializing a crop.
spec['case0509']['body']=spec['case0509']['body'].replace('u64 dest = if active then i*w+j else 99u64 fi\n            out[dest] = if active then im[r][c] else out[dest] fi','out[i*10u64+j] = if active then im[r][c] else 0u64 fi')
spec['case0509']['rewrites'][-1]='Pad the crop to a 10x10 rectangle with public logical height and width; each dimension of a crop of a 10x10 input is at most 10.'
add('pythran_sobelfilter',None,'u64[16]','''    u64[16] edge = [0u64,1u64,2u64,3u64,4u64,5u64,6u64,7u64,8u64,9u64,10u64,11u64,12u64,13u64,14u64,15u64]
    // For cols == 0, every iteration writes the same two positions.
    bool repeat_zero = cols == 0u64 && lt(0u64, rows)
    edge[0] = if repeat_zero then 255u64 else edge[0] fi
    edge[15] = if repeat_zero then 255u64 else edge[15] fi
    for u64 i in 0..16 do
        bool active = cols != 0u64 && lt(i, rows)
        u64 a = if active then norm(i*cols) else 0u64 fi
        u64 b = if active then norm((i+1u64)*cols-1u64) else 0u64 fi
        edge[a] = if active then 255u64 else edge[a] fi
        edge[b] = if active then 255u64 else edge[b] fi
    endfor
    for u64 i in 1..17 do
        bool active = lt(i+1u64, cols)
        u64 a = if active then i else 0u64 fi
        u64 b = if active then norm(i+(rows-1u64)*cols) else 0u64 fi
        edge[a] = if active then 255u64 else edge[a] fi
        edge[b] = if active then 255u64 else edge[b] fi
    endfor
    for u64 iy in 1..16 do
        for u64 ix in 1..16 do
            bool active = lt(iy+1u64, rows) && lt(ix+1u64, cols)
            u64 p00 = if active then norm(ix-1u64+(iy-1u64)*cols) else 0u64 fi
            u64 p01 = if active then norm(ix+(iy-1u64)*cols) else 0u64 fi
            u64 p02 = if active then norm(ix+1u64+(iy-1u64)*cols) else 0u64 fi
            u64 p10 = if active then norm(ix-1u64+iy*cols) else 0u64 fi
            u64 p12 = if active then norm(ix+1u64+iy*cols) else 0u64 fi
            u64 p20 = if active then norm(ix-1u64+(iy+1u64)*cols) else 0u64 fi
            u64 p21 = if active then norm(ix+(iy+1u64)*cols) else 0u64 fi
            u64 p22 = if active then norm(ix+1u64+(iy+1u64)*cols) else 0u64 fi
            u64 sx = 0u64-original_image[p00]-2u64*original_image[p01]-original_image[p02]+original_image[p20]+2u64*original_image[p21]+original_image[p22]
            sx = clamp(sx)
            u64 sy = original_image[p00]-original_image[p02]+2u64*original_image[p10]-2u64*original_image[p12]+original_image[p20]-original_image[p22]
            sy = clamp(sy)
            u64 sum = sx+sy
            u64 pos = if active then norm(ix+iy*cols) else 0u64 fi
            edge[pos] = if active then 255u64-(255u64 & sum) else edge[pos] fi
        endfor
    endfor
    return edge''',prefix=LT+'''def norm(u64 x) -> u64:
    return if lt(x, 0u64) then x+16u64 else x fi

def clamp(u64 x) -> u64:
    return if lt(x, 0u64) then 0u64 else if lt(255u64, x) then 255u64 else x fi fi

''',attempt='rewrite',dynamic=('dynamic loop','for i in range(rows); for i in range(1, cols-1); nested ranges(rows-1, cols-1), lines 11-19'),rewrites=['For valid executions with cols != 0, the first loop has at most 16 iterations because its edge-image accesses must lie in [-16,15]; replace by a 16-step guarded loop.','For cols == 0, collapse repetitions of the same two boundary writes into one guarded pair; this preserves arbitrary positive rows without inventing a bound.','The second loop has at most 16 iterations because edge_image[i] must be in bounds; inner image loops are bounded by the first two loops. Use guarded iterations and normalize Python negative indices.'])

# Mean by an integer divisor retains Q32 scale: native uint division suffices,
# without a widened fixed-point division algorithm or witness gadget.
add('case0414',None,'Result','''    u64 bs = if bin_size > 8u64 then 1u64 else bin_size fi
    u64 n = if bin_size > 8u64 then 0u64 else 8u64/bs fi
    u64[8] result = [0u64; 8]
    for u64 i in 0..8 do
        u64 total = 0u64
        for u64 j in 0..8 do
            bool selected = i < n && j >= i*bs && j < (i+1u64)*bs
            total = total + if selected then data[j] else 0u64 fi
        endfor
        bool neg = (total & 0x8000000000000000) != 0u64
        u64 magnitude = if neg then 0u64-total else total fi
        u64 quotient = magnitude/bs
        result[i] = if neg then 0u64-quotient else quotient fi
    endfor
    return Result {a: n, b: result}''',prefix='struct Result {\n    u64 a\n    u64[8] b\n}\n',attempt='rewrite',dynamic=('dynamic shape','data[:(data.size // bin_size) * bin_size].reshape(-1, bin_size), line 11'),rewrites=['Pad the result to eight bins with a public logical count; at most eight nonempty bins fit in eight input elements.','Replace dynamic reshape and row means by guarded static reductions over the eight input elements. Division of a Q32 sum by a positive integer uses CirC native unsigned integer division with sign conversion; it requires no wide Q32-by-Q32 division gadget.'])

if __name__=='__main__':
 import sys
 for p in BATCH:
  if p['program'] in spec and (len(sys.argv)==1 or p['program'] in sys.argv[1:]):process(p)
