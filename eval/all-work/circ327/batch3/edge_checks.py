from study import *
checks=[]
for n in [0,1,2,1021,1024]:
 d=ROOT/'pythran_factorize_naive';path=d/f'edge_{n}.pin';path.write_text(f'(let ((n #x{n:08x})) true)\n')
 run=subprocess.run([CIRC+'zxi',str(d/'rewrite.zok'),str(path)],env=ENV,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (d/f'edge_{n}.log').write_text(run.stdout)
 vals=[int(v,16) for v in re.findall(r'0x([0-9a-fA-F]+)',run.stdout)]
 factors=vals[1:1+vals[0]] if vals else None
 expected=[];v=n
 if n>=2:
  p=2
  while p*p<=v:
   while v%p==0:expected.append(p);v//=p
   p+=1
  if v>1:expected.append(v)
 checks.append({'program':'pythran_factorize_naive','n':n,'pass':factors==expected,'actual':factors,'expected':expected})
print(json.dumps(checks,indent=2));(ROOT/'edge_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
assert all(c['pass'] for c in checks)
