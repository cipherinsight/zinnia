import pathlib,json,shutil,math
from translate import transform
B=pathlib.Path(__file__).resolve().parent;d=B/'pythran_fannkuch'
for n in ['result.json','rewrite.zok','rewrite.compile.log','rewrite.command.json']:
 if (d/n).exists():shutil.copyfile(d/n,d/('ungrouped.'+n))
s=transform((B.parent/'batch4/pythran_fannkuch/rewrite.zok').read_text())
groups=[];cur=1
for k in range(2,1025):
 if cur*k >= 2**63:groups.append(cur);cur=1
 cur*=k
groups.append(cur)
assert math.prod(groups)==math.factorial(1024)
out=[];stack=[]
for line in s.splitlines():
 indent=len(line)-len(line.lstrip());strip=line.strip()
 if strip.startswith('for u64 ') and len(strip)>200:
  name=strip.split()[2];prefix=' '*indent
  for i,cap in enumerate(groups):out.append(prefix+'    '*i+f'for u64 {name}_block{i} in 0..{cap} do')
  stack.append((indent,len(groups)-1));continue
 extra=sum(x[1] for x in stack)
 if strip=='endfor' and stack and indent==stack[-1][0]:
  base,n=stack.pop()
  for k in reversed(range(n+1)):out.append(' '*(base+4*k)+'endfor')
 else:out.append(' '*(4*extra)+line)
(d/'rewrite.zok').write_text('\n'.join(out)+'\n')
m=json.load(open(d/'translation.json'));m['rewrites']=[v for v in m['rewrites'] if not v.startswith('Express each')]+[f'Group the factors of each 1024! loop bound into {len(groups)} native u64 bounds, each below 2^63; their product is exactly 1024!. This avoids unnecessary one-factor-per-loop parser depth without reducing the domain.'];(d/'translation.json').write_text(json.dumps(m,indent=2))
print('groups',len(groups))
