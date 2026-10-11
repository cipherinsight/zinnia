# This builder imports only the helper definitions, not manual.py's top-level mutations.
import pathlib,json,textwrap
B=pathlib.Path(__file__).resolve().parent
d=B/'npbench_cavity_flow_cavity_flow'
head='from "../native" import neg, absv, lt, divc, mulq, divq\n\n'
code='''struct Result {
    u64[16][16] u
    u64[16][16] v
    u64[16][16] p
}
def main(private u64 nx, private u64 ny, private u64 nt, private u64 nit, private u64[16][16] u, private u64[16][16] v, private u64 dt, private u64 dx, private u64 dy, private u64[16][16] p, private u64 rho, private u64 nu) -> Result:
    u64[16][16] b = [[0u64; 16]; 16]
    for u64 n in 0..1024 do
        bool active = n < nt
        u64[16][16] un = u
        u64[16][16] vn = v
        for u32 i in 1..15 do
            for u32 j in 1..15 do
                u64 dux = divq(u[i][j+1u32]-u[i][j-1u32],2u64*dx)
                u64 dvy = divq(v[i+1u32][j]-v[i-1u32][j],2u64*dy)
                u64 duy = divq(u[i+1u32][j]-u[i-1u32][j],2u64*dy)
                u64 dvx = divq(v[i][j+1u32]-v[i][j-1u32],2u64*dx)
                b[i][j] = mulq(rho,mulq(divq(4294967296u64,dt),dux+dvy)-mulq(dux,dux)-2u64*mulq(duy,dvx)-mulq(dvy,dvy))
            endfor
        endfor
        u64 dx2 = mulq(dx,dx)
        u64 dy2 = mulq(dy,dy)
        for u64 q in 0..1024 do
            bool step = active && q < nit
            u64[16][16] pn = p
            for u32 i in 1..15 do
                for u32 j in 1..15 do
                    u64 val = divq(mulq(pn[i][j+1u32]+pn[i][j-1u32],dy2)+mulq(pn[i+1u32][j]+pn[i-1u32][j],dx2),2u64*(dx2+dy2))-mulq(divq(mulq(dx2,dy2),2u64*(dx2+dy2)),b[i][j])
                    p[i][j] = if step then val else p[i][j] fi
                endfor
            endfor
            for u32 i in 0..16 do
                p[i][15] = if step then p[i][14] else p[i][15] fi
            endfor
            for u32 j in 0..16 do
                p[0][j] = if step then p[1][j] else p[0][j] fi
            endfor
            for u32 i in 0..16 do
                p[i][0] = if step then p[i][1] else p[i][0] fi
            endfor
            for u32 j in 0..16 do
                p[15][j] = if step then 0u64 else p[15][j] fi
            endfor
        endfor
        for u32 i in 1..15 do
            for u32 j in 1..15 do
                u64 uval = un[i][j]-mulq(divq(mulq(un[i][j],dt),dx),un[i][j]-un[i][j-1u32])-mulq(divq(mulq(vn[i][j],dt),dy),un[i][j]-un[i-1u32][j])-mulq(divq(dt,2u64*mulq(rho,dx)),p[i][j+1u32]-p[i][j-1u32])+mulq(nu,mulq(divq(dt,dx2),un[i][j+1u32]-2u64*un[i][j]+un[i][j-1u32])+mulq(divq(dt,dy2),un[i+1u32][j]-2u64*un[i][j]+un[i-1u32][j]))
                u64 vval = vn[i][j]-mulq(divq(mulq(un[i][j],dt),dx),vn[i][j]-vn[i][j-1u32])-mulq(divq(mulq(vn[i][j],dt),dy),vn[i][j]-vn[i-1u32][j])-mulq(divq(dt,2u64*mulq(rho,dy)),p[i+1u32][j]-p[i-1u32][j])+mulq(nu,mulq(divq(dt,dx2),vn[i][j+1u32]-2u64*vn[i][j]+vn[i][j-1u32])+mulq(divq(dt,dy2),vn[i+1u32][j]-2u64*vn[i][j]+vn[i-1u32][j]))
                u[i][j] = if active then uval else u[i][j] fi
                v[i][j] = if active then vval else v[i][j] fi
            endfor
        endfor
        for u32 j in 0..16 do
            u[0][j] = if active then 0u64 else u[0][j] fi
            u[j][0] = if active then 0u64 else u[j][0] fi
            u[j][15] = if active then 0u64 else u[j][15] fi
            v[0][j] = if active then 0u64 else v[0][j] fi
            v[15][j] = if active then 0u64 else v[15][j] fi
            v[j][0] = if active then 0u64 else v[j][0] fi
            v[j][15] = if active then 0u64 else v[j][15] fi
        endfor
        for u32 j in 0..16 do
            u[15][j] = if active then 4294967296u64 else u[15][j] fi
        endfor
    endfor
    return Result {u: u,v: v,p: p}
'''
(d/'rewrite.zok').write_text(head+code)
(d/'translation.json').write_text(json.dumps({'mode':'rewrite','rewrites':['Guard both time and pressure loops at 1024 using the explicit nt/nit contracts.','Use a 16x16 b on active iterations: assignment of the 14x14 u/v stencil to b[1:-1,1:-1] requires nx=ny=16 whenever nt>0. On nt=0 b is unused, so omit its unbounded allocation.','Return all three mutated arrays u, v and p.']},indent=2))
# Remove incorrect huge integer bounds: express the identical factorial trip count as nested static loops.
d=B/'pythran_fannkuch';f=d/'rewrite.zok';s=f.read_text();rows=s.splitlines();out=[];stack=[]
for line in rows:
 indent=len(line)-len(line.lstrip());stripped=line.strip()
 if stripped.startswith('for u64 ') and len(stripped)>200:
  prefix=' '*indent
  name=stripped.split()[2]
  for k in range(2,1025):out.append(prefix+'    '*(k-2)+f'for u32 {name}_digit{k} in 0..{k} do')
  stack.append((indent,1022));continue
 extra=sum(x[1] for x in stack)
 if stripped=='endfor' and stack and indent==stack[-1][0]:
  base,n=stack.pop()
  for k in reversed(range(n+1)):out.append(' '*(base+4*k)+'endfor')
 else:out.append(' '*(4*extra)+line)
f.write_text('\n'.join(out)+'\n')
m=json.load(open(d/'translation.json'));m['rewrites'].append('Express each 1024! bound by nested native u32 loops with radices 2 through 1024; no enormous unsupported integer literal and no smaller bound.');(d/'translation.json').write_text(json.dumps(m,indent=2))
