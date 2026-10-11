from study import *
# Fix interpreter struct presentation order, not the circuit.
n='pythran_make_decision';r=json.load(open(ROOT/n/'result.json'));r['rewrite']['status']='accepted';r['category']='accepted_with_rewrite';r['notes']+=' zxi prints struct fields alphabetically (im,re); decoder reorders to NumPy (re,im).';(ROOT/n/'result.json').write_text(json.dumps(r,indent=2)+'\n')
validate(n,'rewrite',lambda v,l:np.array(v).reshape(-1,2)[:,::-1]/S)
# Saturate only a dead factor-list write index after the maximum ten factors.
n='pythran_factorize_naive';p=ROOT/n/'rewrite.zok';s=p.read_text().replace('out[1+count] = if factor then p else if last then n else out[1+count] fi fi','u32 slot = if count < 10 then 1+count else 10 fi\n        out[slot] = if factor then p else if last then n else out[slot] fi fi');p.write_text(s)
r=json.load(open(ROOT/n/'result.json'));r['rewrite']['rewrites'].append('Clamp the unused write index after ten factors, so later inactive iterations never access past the padding.');compiled(n,'rewrite',r['rewrite']['rewrites'],r);validate(n,'rewrite',lambda v,l:v[1:1+int(v[0])],scale=1)
# Window clipping avoids computing an unbounded half-size. Values beyond +-4
# give identical clipped Python slices on a length-four axis.
n='case0468';p=ROOT/n/'rewrite.zok';s=p.read_text();start=s.index('    field rw =');end=s.index('    field[288] out',start)
s=s[:start]+'''    field rw = -4
    field cw = -4
    for field t in 1..9 do
        field threshold = 2*(t-4)
        rw = if !lt(size[0], threshold) then t-4 else rw fi
        cw = if !lt(size[1], threshold) then t-4 else cw fi
    endfor
'''+s[end:];p.write_text(s)
r=json.load(open(ROOT/n/'result.json'));r['rewrite']['rewrites'].append('Compute floor(size/2) saturated to [-4,4] by eight threshold comparisons. On a four-element axis, radii >=4 yield the full slice and radii <=-4 yield empty slices, including Python negative-end normalization.');compiled(n,'rewrite',r['rewrite']['rewrites'],r)
def decode(v,l):
 out=[]
 for k in range(16):
  b=v[k*18:(k+1)*18];out.append(np.array(b[2:]).reshape(4,4)[:int(b[0]),:int(b[1])])
 return out
validate(n,'rewrite',decode,scale=1)
# The temporary b allocation can be guarded when it is unused.
n='npbench_cavity_flow_cavity_flow'
why='kernel.py:33-34: private b shape and private nt/nit loops are not expressible as written. For nt>0, build_up_b assigns a 14x14 RHS into b[1:-1,1:-1], forcing ny=nx=16 on valid calls; for nt=0 b is unobserved. A guarded 16x16 allocation and two loops bounded by declared 1024 preserve all outputs. Lines 10,18,39-40 still require private fixed-point division.'
r=result(n,'needs_gadget: division',why,'not_expressible',['Guard the unused b allocation when nt=0; use 16x16 when nt>0, as implied by assignment shape compatibility.','Bound nt and nit loops at the declared 1024 with activity guards.']);r['rewrite']['status']='needs_gadget';(ROOT/n/'result.json').write_text(json.dumps(r,indent=2)+'\n')
