from study import *
for name in ['case0334','pythran_harris','pythran_ln','npbench_polybench_syr2k_syr2k']:
 r=json.load(open(ROOT/name/'result.json'))
 compiled(name,'rewrite',['Use native arbitrary integer Q32 arithmetic instead of native field remainder; no custom gadget or new bound.'],r)
 validate(name,'rewrite')
for name in ['case0418','case0468','npbench_spmv_spmv','pythran_factorize_naive','pythran_fibo_seq','case0388','pythran_quicksort']:
 r=json.load(open(ROOT/name/'result.json'));compiled(name,'rewrite',r['rewrite']['rewrites'],r)
