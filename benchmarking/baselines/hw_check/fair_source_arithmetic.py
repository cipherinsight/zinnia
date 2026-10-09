"""Exact-Q32 source-loop reference versus Noir execute; never setup/prove.
Run under memrun.sh. The reference uses individually rounded products and
literal source loops, independently of any baseline CSE or gate construction.
"""
import argparse
import json
import math
import subprocess
import tomllib
from pathlib import Path
import numpy as np
from .entries import resolve
from .noir_driver import NARGO, compile_package, stage_package
from .prover_toml import BN254_P, SCALE, quantize, render_prover_toml

CASES = ['pythran_deuxd_convolution__conv', 'pythran_make_decision__md',
         'npbench_weather_stencils_hdiff_hdiff__hdiff', 'pythran_tax__tax']

def mul(a,b): return int(a)*int(b)//SCALE

def reference(case, fixture, inp):
    a = {p['name']: np.vectorize(quantize,otypes=[object])(
         np.array(fixture['inputs'][inp][p['name']],dtype=float))
         for p in fixture['parameters']}
    if case == CASES[0]:
        x,w=a['x'],a['weights']; out=np.zeros(x.shape,dtype=object)
        for i,j,ii,jj in np.ndindex(*x.shape,*w.shape):
            r=min(max(0,i+ii-w.shape[0]//2),w.shape[0])
            c=min(max(0,j+jj-w.shape[0]//2),w.shape[0])
            out[i,j]+=mul(x[r,c],w[ii,jj])
    elif case == CASES[1]:
        e,s=a['E'],a['symbols'];out=[]
        for v in e:
            distances=[]
            for symbol in s:
                dr,di=v-symbol
                magnitude=math.isqrt((mul(dr,dr)+mul(di,di))*SCALE)
                distances.append(mul(magnitude,magnitude))
            out.extend(s[min(range(len(s)),key=distances.__getitem__)])
        return [int(v) for v in out]
    elif case == CASES[2]:
        x,coeff=a['in_field'],a['coeff'];I,J,K=coeff.shape
        lap=4*x[1:I+3,1:J+3,:]-(x[2:I+4,1:J+3,:]+x[:I+2,1:J+3,:]+x[1:I+3,2:J+4,:]+x[1:I+3,:J+2,:])
        res=lap[1:,1:J+1,:]-lap[:-1,1:J+1,:]
        diff=x[2:I+3,2:J+2,:]-x[1:I+2,2:J+2,:]
        flx=np.where(np.vectorize(mul,otypes=[object])(res,diff)>0,0,res)
        res=lap[1:I+1,1:,:]-lap[1:I+1,:-1,:]
        diff=x[2:I+2,2:J+3,:]-x[2:I+2,1:J+2,:]
        fly=np.where(np.vectorize(mul,otypes=[object])(res,diff)>0,0,res)
        divergence=flx[1:]-flx[:-1]+fly[:,1:]-fly[:,:-1]
        out=x[2:I+2,2:J+2,:]-np.vectorize(mul,otypes=[object])(coeff,divergence)
    else:
        assert case==CASES[3]
        d=a['d'];seg1=d[d>quantize(256303)];seg2=d[(d>quantize(54057))&(d<=quantize(256303))]
        seg3=d[(d>quantize(13769))&(d<=quantize(54057))]-quantize(13769)
        seg4=d[(d>quantize(8820))&(d<=quantize(13769))]-quantize(8820)
        total=sum(mul(x,quantize(.45))-quantize(16164.53) for x in seg1)
        total+=sum(mul(x,quantize(.42))-quantize(8475.44) for x in seg2)
        total+=sum(mul(x,mul(x,quantize(2.2376e-6))+quantize(.2397))+quantize(939.57) for x in seg3)
        total+=sum(mul(x,mul(x,quantize(1.00727e-5))+quantize(.14)) for x in seg4)
        return [int(total)*SCALE//int(sum(d))]
    return [int(v) for v in out.flat]

def flatten(value):
    if isinstance(value,list): return [x for v in value for x in flatten(v)]
    code = int(value,0) if isinstance(value,str) else int(value)
    return [code-BN254_P if code>BN254_P//2 else code]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('cases',nargs='*')
    args=ap.parse_args(); results={}
    for name in args.cases or CASES:
        entry=resolve(name); fixture=json.loads(entry.fixture.read_text())
        pkg=stage_package(entry.directory,args.work/name,entry.function)
        compiled=compile_package(pkg); row={}
        for inp in ('A','B'):
            expected=reference(name,fixture,inp)
            (pkg/'Prover.toml').write_text(render_prover_toml(compiled['abi'],fixture,inp))
            proc=subprocess.run([NARGO,'execute','--silence-warnings','--overwrite-return'],cwd=pkg,capture_output=True,text=True)
            (pkg/f'execute-{inp}.log').write_text(proc.stdout+proc.stderr)
            if proc.returncode: raise RuntimeError(f'{name} {inp}: {proc.stderr[-1500:]}')
            actual=flatten(tomllib.loads((pkg/'Prover.toml').read_text())['return'])
            assert actual==expected,(name,inp,[(i,x,y) for i,(x,y) in enumerate(zip(actual,expected)) if x!=y][:5])
            row[inp]={'exact_q32_match':True,'public_cells':len(actual)}
        results[name]=row
        print(name,'A/B exact Q32 match (execute only)',flush=True)
        args.out.write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__': main()
