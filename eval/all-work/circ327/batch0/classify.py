import json,pathlib
R=pathlib.Path('.')
B=json.load(open('../batch0.json'))
# Constructs for which the brief explicitly prohibits inventing an implementation.
blocked={
'case0300':('sort','kernel.py:11: np.percentile requires ordering 20 private real values and interpolating.','The input length is 20, but supplying a handwritten sort/selection gadget is prohibited.'),
'case0324':('cos','kernel.py:11: cos(deg2rad(degree)) requires a real cosine implementation.','No native fixed-point cosine exists in the installed Z# frontend or standard library.'),
'case0330':('pow','kernel.py:11: a ** power uses an unrestricted signed runtime exponent.','Negative exponents require reciprocal division; a bounded nonnegative-only expansion would narrow the signature.'),
'case0374':('sort','kernel.py:11: ecdf_result calls np.sort(grades).','Fifty entries give a size bound, but sorting is a prohibited handwritten gadget.'),
'case0430':('sqrt','kernel.py:11: np.std computes square root of variance.','The fixed input size bounds reductions, but the real square root needs a gadget.'),
'npbench_deep_learning_softmax_softmax':('exp','kernel.py:11,13: np.exp and real normalization division.','Static 2x2x16x16 shape; no fixed-point exp implementation in Z#.'),
'npbench_weather_stencils_vadv_vadv':('division','kernel.py:23,36,46: 1.0 / bcol and recurrence-dependent real denominators.','Shapes and loop bounds are fixed; field inversion is not Q32 division, and a signed fixed-point division gadget is prohibited.'),
'pythran_cronbach':('division','kernel.py:9,12: variance normalization and division by variance of total scores.','A fixed 16x16 shape bounds all reductions; the final variable-denominator fixed-point quotient needs a gadget.'),
'pythran_lu':('division','kernel.py:40 in factorMatrix1: v = M[j, line] / pivot.','All loops are bounded by 5 and dynamic pivot row indexing is expressible; signed fixed-point pivot division needs a gadget.'),
'pythran_periodic_dist':('sqrt','kernel.py:27: np.sqrt(dx ** 2 + dy ** 2 + dz ** 2).','Replace masked assignments with 16x16 guarded elementwise loops if needed; the final real square root still needs a gadget.'),
'pythran_wdist':('division','kernel.py:15-16: real division by W and square root.','The 2x3 inputs bound all loops; fixed-point division and sqrt require gadgets.'),
}
bounded={
'npbench_contour_integral_contour_integral':('complex pow and linear solve','kernel.py:10-15: arrays and loop bound depend on NR, NM and slab_per_bc; lines 15,18,20 use complex powers/inverse/solve.',['Use 5x5 Tz and a 5x15 output capacity with public dimensions: valid addition to Ham[n] forces NR=5, and successful accumulation forces NM=5 (inverse branch) or NM=15 (solve branch).','Guard at most three slab iterations, as indexing Ham[n] requires slab_per_bc <= 2 on valid executions.'],'The bounded rewrite still requires complex exponentiation and a matrix inverse/linear solve, including real division.'),
'npbench_polybench_covariance_covariance':('division','kernel.py:12-14: M determines covariance array and loop bounds.',['Pad covariance to 14x14 and guard iterations: successful data[:, i] reads require 0 <= M <= 14.','Return public M and retain the mutation data -= mean as an observable output.'],'kernel.py:14 divides by private float_n - 1.0; fixed-point division remains necessary.'),
'pythran_caxpy':('fixed-point multiplication','kernel.py:16,25: loops have runtime N; line 17/26 multiplies private complex values.',['Guard 1024 iterations using the explicit 0 <= N <= 1024 precondition; preserve increment-dependent indices and early-return tag.','Represent complex values by real/imaginary Q32 components; compare both components to zero for abs(CA)==0.'],'Signed products of Q32 components require scaling without overflow; native integer R1CS lowering fails and a custom arithmetic gadget is prohibited.'),
'pythran_clip2':('sqrt','kernel.py:19-20: complex magnitude mask and dynamically selected z; helper lines 11,14 use magnitude and division.',['Use a capacity of two for selected z, with guards, since its declared input length is two.'],'Complex absolute value requires sqrt and helper normalization needs fixed-point division.'),
'pythran_matrix_class_distance':('sqrt','kernel.py:11-12: runtime iteration bound and filtered row count.',['Guard at most 16 iterations: every positive iteration unconditionally reads dat_points[i], so valid inputs require iterations <= 16 (nonpositive iterations return zero).','Use sixteen row slots and masks for filtering the sixteen input rows.'],'The norm over ten columns needs a real sqrt gadget.'),
'pythran_tax':('division','kernel.py:10-13: boolean selection creates variable-length arrays.',['Use ten slots and guards for each selected segment; every segment is a subset of the ten input entries.'],'kernel.py:16 divides the tax sum by private np.sum(d), requiring fixed-point division; products also require Q32 scaling.'),
}
unbounded={
'case0408':('dynamic shape','kernel.py:15: np.zeros(np.amax(index)+1).','The signature permits arbitrarily large nonnegative group labels; 100 labels do not bound the largest label or dense output length. Unique and min-mask lengths are bounded by 100, but the returned dense array is not.'),
'case0497':('dynamic shape','kernel.py:11: np.pad to private shape[0], shape[1].','The declared integer target dimensions have no upper bound; padding is valid for arbitrarily larger dimensions, so input shape 42x10 gives no output capacity.'),
'npbench_polybench_jacobi_2d_jacobi_2d':('dynamic loop','kernel.py:10: range(1, TSTEPS).','TSTEPS is unrestricted by the signature; fixed array extents do not bound the number of stencil time steps.'),
'pythran_fdtd':('dynamic loop','kernel.py:15: range(steps).','steps is unrestricted by the signature; fixed 16x16 arrays do not bound the number of repeated time steps.'),
'pythran_hyantes_core':('dynamic shape','kernel.py:12: [[0] * range_y for _ in range(range_x)].','range_x and range_y have no upper bounds; only nonzero step and range_ are required. Arbitrarily large positive grids are valid. Trigonometry and division are additional blockers.'),
}
def write(n,cat,aw,rw,notes):
 d=R/n
 res=dict(program=n,category=cat,as_written=aw,rewrite=rw,correctness={'A':{'status':'not applicable','reason':'No accepted full translation.'},'B':{'status':'not applicable','reason':'No accepted full translation.'}},notes=notes)
 (d/'result.json').write_text(json.dumps(res,indent=2)+'\n')
 (d/'as_written.zok').write_text('// NO EXECUTABLE TRANSLATION: '+aw['error']+'\n// '+notes+'\n// No dummy program was compiled or counted as this kernel.\n')
 if rw:(d/'rewrite.zok').write_text('// NO EXECUTABLE TRANSLATION: '+str(rw['error'])+'\n'+''.join('// '+v+'\n' for v in rw['rewrites']))
for n,(g,where,why) in blocked.items():
 aw=dict(status='needs gadget',error=where,constraints=None)
 rw=dict(status='needs gadget',error='needs gadget: '+g,constraints=None,rewrites=[],bound_argument=why)
 write(n,'needs_gadget: '+g,aw,rw,why+' Not compiled: the brief forbids implementing this missing gadget.')
for n,(g,where,rewrites,why) in bounded.items():
 aw=dict(status='not expressible',error=where,constraints=None)
 rw=dict(status='needs gadget',error='needs gadget: '+g+'; '+why,constraints=None,rewrites=rewrites)
 write(n,'needs_gadget: '+g,aw,rw,why+' Bounded rewrite analyzed but not emitted as a dummy executable; missing gadget explicitly prohibited.')
for n,(kind,where,why) in unbounded.items():
 aw=dict(status='not expressible',error=where,constraints=None)
 rw=dict(status='not expressible',error='No domain-implied finite bound. '+why,constraints=None,rewrites=[])
 write(n,'not_expressible: '+kind,aw,rw,why+' Not compiled because no faithful finite Z# shape/loop exists.')
# Preserve the actual compiler rejection for full native integer translations.
for n in ['case0422','case0485','npbench_polybench_atax_atax','npbench_polybench_gemm_gemm','npbench_polybench_trmm_trmm','pythran_rosen','case0452','pythran_lapl3d']:
 d=R/n;r=json.load(open(d/'result.json'))
 if r['as_written']['status']=='accepted':continue
 g='division'
 r['category']='needs_gadget: '+g
 r['rewrite']=dict(status='needs gadget',error='Signed fixed-point division (including rescaling a Q32 product) requires a gadget once native integer R1CS lowering fails.',constraints=None,rewrites=[])
 r['notes']+=' Native integer arithmetic was actually compiled, not inferred unsupported. No unsigned/narrowed input domain or modular field-division substitution was used.'
 (d/'rewrite.zok').write_text('// No executable rewrite: needs signed fixed-point division/rescaling gadget.\n// See as_written.zok and as_written.compile.log for the full native attempt.\n')
 (d/'result.json').write_text(json.dumps(r,indent=2)+'\n')
