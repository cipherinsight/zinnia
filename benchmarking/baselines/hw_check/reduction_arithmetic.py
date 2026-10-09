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

CASES = ['pythran_fbcorr__fbcorr', 'pythran_matmul__matrix_multiply',
         'pythran_allpairs__sqr_dists', 'pythran_allpairs__sqr_dists_loops',
         'pythran_pairwise__pairwise', 'pythran_l2norm__l2_norm',
         'pythran_sumarray3d__summation',
         'pythran_wdist__slow_wdist',
         'pythran_ln__ln', 'npbench_polybench_covariance2_covariance2__kernel',
         'npbench_scattering_self_energies_scattering_self_energies__scattering_self_energies']

def mul(a,b): return int(a)*int(b)//SCALE

def root(x): return math.isqrt(int(x)*SCALE)

def reference(case, fixture, inp):
    a = {p['name']: np.array(fixture['inputs'][inp][p['name']], dtype=object) if p['dtype']=='int'
         else np.vectorize(quantize,otypes=[object])(np.array(fixture['inputs'][inp][p['name']],dtype=float))
         for p in fixture['parameters']}
    if case == CASES[0]:
        imgs,filters,out = a['imgs'],a['filters'],a['output'].copy()
        ni,rows,cols,nc = imgs.shape
        nf,h,w,_ = filters.shape
        for ii,rr,cc,hh,ww,jj,ff in np.ndindex(ni,rows-h+1,cols-w+1,h,w,nc,nf):
            out[ii,ff,rr,cc] += mul(imgs[ii,rr+hh,cc+ww,jj],filters[ff,hh,ww,jj])
    elif case == CASES[1]:
        x,y = a['m0'],a['m1']
        out = np.zeros((len(x),y.shape[1]),dtype=object)
        for i,j,k in np.ndindex(len(x),y.shape[1],len(y)):
            out[i,j] += mul(x[i,k],y[k,j])
    elif case in CASES[2:4]:
        x,y = a['X'],a['Y']
        out = np.zeros((len(x),len(y)),dtype=object)
        for i,j in np.ndindex(out.shape):
            out[i,j] = sum(mul(x[i,k]-y[j,k],x[i,k]-y[j,k]) for k in range(x.shape[1]))
        if case == CASES[2]: out = out.T
    elif case == CASES[4]:
        x = a['X']
        return [root(sum(mul(x[i,k]-x[j,k],x[i,k]-x[j,k]) for k in range(x.shape[1])))
                for i in range(len(x)) for j in range(len(x)) if i!=j]
    elif case == CASES[5]:
        return [root(sum(mul(abs(x),abs(x)) for x in row)) for row in a['x']]
    elif case == CASES[6]:
        pos,weights,points = a['pos'],a['weights'],a['points']
        total = [0]*len(points)
        xyz = np.zeros((len(points),3),dtype=object)
        for i,j in np.ndindex(len(points),len(weights)):
            delta = points[i]-pos[j]
            dr = SCALE*SCALE//root(sum(mul(d,d) for d in delta))
            total[i] += mul(weights[j],dr)
            for k in range(3): xyz[i,k] += mul(weights[j],delta[k])
        return total+[int(x) for x in xyz.flat]
    elif case == 'pythran_wdist__slow_wdist':
        x,y,w = a['A'],a['B'],a['W']
        out = np.zeros((x.shape[1],y.shape[1]),dtype=object)
        for i,j in np.ndindex(out.shape):
            # Euclidean quotient: numerator = quotient*divisor + remainder,
            # with nonnegative remainder below abs(divisor), for either sign.
            diffs = [(int(x[k,i]-y[k,j])*SCALE//abs(int(w[k,i]))) * (1 if w[k,i]>0 else -1)
                     for k in range(len(x))]
            out[i,j] = root(sum(mul(d,d) for d in diffs))
    elif case == 'pythran_ln__ln':
        def power(x,n):
            y=x
            for bit in bin(n)[3:]:
                y=mul(y,y)
                if bit=='1': y=mul(y,x)
            return y
        return [sum((-1 if n%2==0 else 1)*(power(int(x)-SCALE,n)//n) for n in range(1,10)) for x in a['X']]
    elif case == 'npbench_polybench_covariance2_covariance2__kernel':
        data=a['data']; no,nv=data.shape
        means=[sum(data[:,i])//no for i in range(nv)]
        out=np.zeros((nv,nv),dtype=object)
        for i,j in np.ndindex(nv,nv):
            raw=sum((data[o,i]-means[i])*(data[o,j]-means[j]) for o in range(no))
            out[i,j]=(raw//SCALE)//(no-1)
    else:
        assert case == CASES[-1]
        def product(u,v):
            return np.array([mul(u[0],v[0])-mul(u[1],v[1]),mul(u[0],v[1])+mul(u[1],v[0])],dtype=object)
        def matmul(x,y):
            out = np.zeros((x.shape[0],y.shape[1],2),dtype=object)
            for i,j,k in np.ndindex(x.shape[0],y.shape[1],y.shape[0]): out[i,j] += product(x[i,k],y[k,j])
            return out
        g,dh,d,out,neigh = a['G'],a['dH'],a['D'],a['Sigma'].copy(),a['neigh_idx']
        for k,e,q,w,i,j,aa,b in np.ndindex(g.shape[0],g.shape[1],d.shape[0],d.shape[1],d.shape[-3],d.shape[-2],neigh.shape[0],neigh.shape[1]):
            if e<w: continue
            left = matmul(g[k,e-w,int(neigh[aa,b])],dh[aa,b,i])
            right = np.array([[product(v,d[q,w,aa,b,i,j]) for v in row] for row in dh[aa,b,j]],dtype=object)
            out[k,e,aa] += matmul(left,right)
    return [int(x) for x in out.flat]

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
