from study import *
p=ROOT/'case0303/rewrite.zok'
s=p.read_text().replace('field col = divc(23, ncol) * ncol','field nr = 0\n    for field k in 1..24 do\n        nr = nr + if k * ncol <= 23 then 1 else 0 fi\n    endfor\n    field col = nr * ncol').replace('out[0] = divc(23, ncol)','out[0] = nr')
p.write_text(s)
r=json.load(open(ROOT/'case0303/result.json'));r['rewrite']['rewrites'].append('Determine row count by scanning 1..23 and comparing k*ncol to 23, avoiding native field remainder backend crash without narrowing ncol.')
compiled('case0303','rewrite',r['rewrite']['rewrites'],r)
validate('case0303','rewrite',lambda v,l:np.array(v[2:2+int(v[0])*int(v[1])])/S)
