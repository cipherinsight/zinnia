from study import *
validate('pythran_factorize_naive','rewrite',lambda v,l:v[1:1+int(v[0])],scale=1)
for name in ['pythran_make_decision','pythran_deriv','pythran_cdotu','npbench_deep_learning_lenet_lenet']:
 r=json.load(open(ROOT/name/'result.json'));compiled(name,'rewrite',r['rewrite']['rewrites'],r)
 validate(name,'rewrite')
