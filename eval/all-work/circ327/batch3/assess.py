from study import *
# No finite capacity follows from these signature domains. Fixtures do not constrain inputs.
blocked={
'case0297':('dynamic shape','kernel.py:12: zeros((50, temp.max()+1)); unrestricted Integer values make max(a)-min(a)+1 unbounded.'),
'case0327':('dynamic shape','kernel.py:11: np.pad output length is private unrestricted length; length >= 10 has no upper bound.'),
'case0405':('dynamic shape','kernel.py:11: bincount produces max(accmap)+1 entries; nonnegative accmap has no upper bound in Integer[100].'),
'case0427':('dynamic shape','kernel.py:14: arange(m) and the output have private unrestricted width m; input count 20 does not bound m.'),
'npbench_mandelbrot2_mandelbrot2':('dynamic shape','kernel.py:11-18: mgrid and complex arrays have xn*yn elements; xn and yn have no bound. The itermax <= 1024 precondition only bounds iterations.'),
'pythran_allpairs_distances':('dynamic shape','kernel.py:14-15: arange(600*d) and arange(200*d) have unbounded private sizes; d has no declared bound. No algebraic closed-form replacement is a minimal statement translation.'),
'pythran_mulmod':('dynamic loop','kernel.py:11: while x > 0 has bit_length(x) iterations; only m is bounded by the precondition, not x. Assuming a 32/64-bit x would narrow the Integer domain.'),
'pythran_nqueens':('dynamic shape','kernel.py:45: list(range(queen_count)) and solution tuples have unbounded width; queen_count has no bound. Permutations and solution list sizes are also unbounded.'),
'pythran_primes_sieve':('dynamic shape','kernel.py:9: [True] * limit has unrestricted private length; no signature bound on limit. The returned prime list is also dynamic.'),
'pythran_stone':('dynamic loop','kernel.py:18,37: N2=12*loopstart and range(1,N2+1) have no declared upper bound. Replacing the full kernel with its final KIP expression would discard source statements; transcendental calls at lines 70-71,94 additionally need gadgets.'),
'npbench_cavity_flow_cavity_flow':('dynamic shape','kernel.py:33: zeros((ny,nx)) is unbounded for valid nt=0 inputs, for which no subsequent shape compatibility constrains nx or ny. nt/nit <=1024 bounds loops but not this allocation. Nonzero iteration paths additionally require fixed-point division at lines 10,18,39-40.')}
for n,(kind,why) in blocked.items():result(n,'not_expressible: '+kind,why)
gadgets={
'case0347':('division','kernel.py:11: a[row-1,:] /= divide_number requires division by a private Q32 real, not field inversion.'),
'case0356':('sort','kernel.py:11: np.argsort(a,axis=0) needs a sorting/argsort implementation; there is no native Z# sort in the available stdlib.'),
'case0382':('sort','kernel.py:11: np.argsort(a) requires sorting 100 private reals. No handwritten sorting network or permutation argument is permitted.'),
'case0440':('sqrt','kernel.py:13: np.sqrt(np.diag(mat)) requires fixed-point square root.'),
'case0456':('sqrt','kernel.py:11: Euclidean norm requires square root of each sum of squares.'),
'npbench_polybench_cholesky_cholesky':('sqrt','kernel.py:10,16: Cholesky requires private square roots; line 14 also needs fixed-point division.'),
'npbench_polybench_durbin_durbin':('division','kernel.py:16: alpha divides by private real beta; modular field division is not fixed-point division.'),
'npbench_polybench_gramschmidt_gramschmidt':('sqrt','kernel.py:14: sqrt(nrm); line 15 also divides by the private real R[k,k].'),
'npbench_polybench_lu_lu':('division','kernel.py:13: A[i,j] /= A[j,j] divides private fixed-point quantities.'),
'pythran_l2norm':('sqrt','kernel.py:10: np.sqrt of a sum of squares needs a fixed-point square-root gadget.'),
'pythran_value_index':('sort','kernel.py:11: np.argsort(score[:,1]) is not a native Z# operation; implementing a new sorting gadget is prohibited.'),
'pythran_blacksholes':('sqrt','kernel.py:20 is a dynamic loop, bounded by nb_opt <=1024 (and 14 array entries on valid calls); lines 21-30 require private sqrt/log/exp/division. Pad two outputs to 14 entries and guard 14 iterations, but the numerical gadgets remain required.'),
'pythran_compute_subpix_2d_gaussian2':('log','kernel.py:10-12: crop/where lists are dynamic but at most 4x4; masks and padding can bound them. Lines 21-26 call log on private crop entries; lines 28-29 additionally need fixed-point division.'),
'pythran_slowparts':('tanh','kernel.py:12: tanh is required on private real array elements. Nonempty valid assignments at line 16 imply d=8,re=5 from the 9x5 RHS; d=0 skips the loop and has empty output with variable shape metadata. A 9x5x16x8 capacity plus shape metadata suffices, but tanh remains necessary.')}
for n,(g,why) in gadgets.items():
 r=result(n,'needs_gadget: '+g,why,'needs_gadget')
 r['rewrite']['status']='needs_gadget';r['rewrite']['rewrites']=['No prohibited numerical/sorting gadget was hand-written.']
 if n in ['pythran_blacksholes','pythran_compute_subpix_2d_gaussian2','pythran_slowparts']:
  r['as_written']['status']='not_expressible'
 (ROOT/n/'result.json').write_text(json.dumps(r,indent=2)+'\n')
