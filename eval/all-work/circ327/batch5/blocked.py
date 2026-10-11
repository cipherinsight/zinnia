import study as s
import json
# Each blocker refers only to source/domain, never to another compiler's outcome.
D={
'case0299':('dynamic shape','temp.max()+1','The 200 input integers have no declared value-range bound; max(a)-min(a)+1 is not bounded by their count.'),
'case0407':('dynamic shape','np.bincount(accmap, weights = a)','The 100 map entries have no declared upper bound; bincount length is max(adjusted accmap)+1, not 100.'),
'case0483':('dynamic shape','np.polyfit(x, y, degree)','degree is unconstrained; even with 100 samples, NumPy can return degree+1 coefficients for higher-degree underdetermined fits. Padding has no domain-implied capacity.'),
'case0496':('dynamic shape','np.pad(a,','The two requested shape values have no declared upper bounds; input shape 42x10 supplies only lower bounds for valid padding.'),
'npbench_polybench_adi_adi':('dynamic loop','for t in range(1, TSTEPS + 1)','TSTEPS != 0 gives no upper bound. The N <= 1024 precondition does not bound the time loop. Fixed-point division is an additional blocker.'),
'npbench_polybench_jacobi_1d_jacobi_1d':('dynamic loop','for t in range(1, TSTEPS)','TSTEPS is an unconstrained integer and is independent of the 16-element buffer size.'),
'pythran_calculate_u':('dynamic loop','while t <= t_stop','dt != 0 gives neither a positive-step guarantee nor a bound on (t_stop-t)/dt; do not infer a cap from fixture values. Fixed-point division is an additional blocker.'),
'pythran_diffusion':('dynamic loop','for n in range(iterNum)','iterNum has no upper-bound precondition, independently of the fixed 16x16 grid.'),
'pythran_histogram':('dynamic shape','out_data = [0] * (1 + bin_width)','bin_width has no declared upper bound; 16 samples do not limit the number of bins. Real division is also needed.'),
'pythran_wave_simulation':('dynamic shape','masspoints = np.empty((2, count, 2)','PARTICLE_COUNT has no declared upper bound. The fixed two-coordinate return does not bound intermediate particle arrays. sqrt/division are additional blockers.'),
}
G={
'case0323':('sin','np.sin(np.deg2rad(degree))','No native/stdlib Q32 sine; arbitrary signed degrees cannot be substituted by fixture constants.'),
'case0329':('power','a = a ** power','The exponent is a private, signed, unconstrained integer; negative exponents require real reciprocals. Native field exponentiation does not implement fixed-point real powers.'),
'case0358':('sort','np.argsort(a.sum(axis = (1, 2)))','Sorting is required to preserve the data-dependent reordering of b; the brief prohibits writing a sorting gadget.'),
'case0384':('matrix inverse','np.linalg.matrix_power(A, n)','Signed private n includes negative powers, requiring matrix inversion and real division; no native fixed-point matrix-power/inverse routine exists. Repeated powers also have no declared exponent bound.'),
'case0414':('division','bin_data_mean =','Capacity eight is implied by the data length, but each bin mean still needs signed fixed-point division by the private bin_size; no custom division gadget is permitted.'),
'case0429':('sqrt','a.std()','Standard deviation needs a square root of the variance; no built-in fixed-point square-root route exists.'),
'case0458':('sqrt','np.linalg.norm','Euclidean norms require a square root; replacing them by squared distances changes the output.'),
'npbench_deep_learning_resnet_resnet':('sqrt','std = np.std','Batch normalization needs standard deviation, another sqrt(std+eps), and real division; cannot omit normalization or supply its values as witnesses.'),
'npbench_polybench_trisolv_trisolv':('division','x[i] = (b[i]','Triangular solve divides each computed real numerator by private L[i,i]; native field division is modular inverse, not Q32 real division.'),
'pythran_clip':('sqrt','if abs(z[i]) > _max','Complex absolute value needs sqrt(real^2+imag^2), and limit1 divides by that magnitude; complex numbers could otherwise be represented by real/imaginary pairs.'),
'pythran_goodexpomeasure':('exp','goodExpoR = numpy.exp','Needs fixed-point exp and division by private sigma; cannot precompute from fixtures.'),
'pythran_lttb_pt':('division','c = next_bin.mean(axis=0)','Valid n_out lies in [3,16], giving capacity 16 and at most 14 bins. Native integer division can compute bin means, but triangle areas multiply Q32 differences; full-width products and their Q32 rescaling need a custom wide fixed-point arithmetic gadget after native arbitrary-integer lowering fails. Guarded padding alone cannot supply that arithmetic.'),
'pythran_projection_simplex':('sort','u = np.sort(v)[::-1]','Sort is required; capacity ten bounds conditional selections but does not eliminate sorting and real division.'),
'pythran_sumarray3d':('sqrt','dr = 1.0 / np.sqrt','Both returned arrays must be retained; the first requires inverse square roots. Returning only sum_array3d would weaken the program.'),
'pythran_another_quicksort':('sort','while i <= idx','The first attempt has value-dependent partition loops, variable-length recursive subarrays and recursion (lines 23-56). The input capacity is ten, but replacing recursion by a bespoke bounded sorting implementation would be a forbidden sort gadget. No such replacement was made.'),
}
for p in s.BATCH:
 n=p['program']
 if n in s.spec or (n not in D and n not in G):continue
 d=s.BASE/n;d.mkdir(exist_ok=True)
 src=(s.ROOT/p['kernel']).read_text();rows=src.splitlines()
 kind,needle,note=(D.get(n) or G[n]);line=next(i+1 for i,v in enumerate(rows) if needle in v)
 reason=f"kernel.py:{line}: {rows[line-1].strip()}. {note}"
 isdyn=n in D
 aw=dict(status='not expressible' if isdyn else 'needs gadget',error=reason,constraints=None)
 if n in ('case0414','pythran_lttb_pt','pythran_another_quicksort'):
  aw['status']='not expressible';aw['error']=reason
 rw=dict(status='not expressible' if isdyn else 'needs gadget',error=note,constraints=None,rewrites=[])
 r=dict(program=n,category=('not_expressible: '+kind if isdyn else 'needs_gadget: '+kind),as_written=aw,rewrite=rw,correctness={'A':'not run: no executable accepted translation','B':'not run: no executable accepted translation'},notes=note)
 (d/'result.json').write_text(json.dumps(r,indent=2))
 (d/'ATTEMPTS.md').write_text('# '+n+'\n\n'+reason+'\n\nAs-written and minimal-rewrite analysis are in result.json. No placeholder program was compiled or counted as the kernel. No bound was inferred from fixtures.\n')
 print(n,r['category'])
