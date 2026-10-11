from study import *
static=['case0313','case0321','case0334','case0340','case0369','case0398','case0434','case0449','case0491','case0500','pythran_harris','pythran_ln','npbench_polybench_syr2k_syr2k']
for n in static:
 r=compiled(n)
 validate(n,scale=1 if n in ['case0313','case0321','case0340','case0369','case0434','case0491'] else S)
for n in ['case0303','case0362','case0412','case0481','case0507']:
 old=json.load(open(ROOT/n/'result.json'));compiled(n,'rewrite',old['rewrite']['rewrites'],old)
 def decode(v,l):
  if n=='case0303':return np.array(v[2:2+int(v[0])*int(v[1])])/S
  if n=='case0362':return np.array(v[1:]).reshape(8,6)[:,:int(v[0])]/S
  if n=='case0412':return np.array(v[1:1+int(v[0])])/S
  if n=='case0481':return v[1:1+int(v[0])]
  if n=='case0507':return np.array(v[2:]).reshape(5,6)[:int(v[0]),:int(v[1])]
 validate(n,'rewrite',decode=decode)
