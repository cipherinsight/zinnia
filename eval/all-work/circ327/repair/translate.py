import re,pathlib,json
BASE=pathlib.Path(__file__).resolve().parent
Q=4294967296
class Expr:
 def __init__(self,s,types):
  self.ts=re.findall(r'0x[0-9a-fA-F]+|\d+(?:u64|u32|int|f)?|[A-Za-z_]\w*|==|!=|<=|>=|&&|\|\||<<|>>|\*\*|\.\.|[^\s]',s);self.i=0;self.types=types
 def peek(self):return self.ts[self.i] if self.i<len(self.ts) else ''
 def pop(self):t=self.peek();self.i+=1;return t
 def parse(self,minp=0):
  t=self.pop()
  if t=='if':
   c=self.parse();assert self.pop()=='then';a=self.parse();assert self.pop()=='else';b=self.parse();assert self.pop()=='fi';n=('if',c,a,b)
  elif t in ['-','!','+']:n=('un',t,self.parse(12))
  elif t=='(':
   n=self.parse();assert self.pop()==')'
  elif t=='[':
   ns=[]
   if self.peek()!=']':
    ns.append(self.parse())
    while self.peek() in [',',';']:
     sep=self.pop();ns.append(sep);ns.append(self.parse())
   assert self.pop()==']';n=('arr',ns)
  else:n=('atom',t)
  while True:
   t=self.peek()
   if t=='(':
    self.pop();args=[]
    if self.peek()!=')':
     args.append(self.parse())
     while self.peek()==',':self.pop();args.append(self.parse())
    assert self.pop()==')';n=('call',n,args);continue
   if t=='[':
    self.pop();a=self.parse();assert self.pop()==']';n=('index',n,a);continue
   if t=='.':self.pop();n=('member',n,self.pop());continue
   if t=='{' and n[0]=='atom':
    self.pop();items=[]
    while self.peek()!='}':
     key=self.pop();assert self.pop()==':';items.append((key,self.parse()))
     if self.peek()==',':self.pop()
     else:break
    assert self.pop()=='}';n=('struct',n[1],items);continue
   prec={'||':1,'&&':2,'|':3,'^':4,'&':5,'==':6,'!=':6,'<':7,'>':7,'<=':7,'>=':7,'<<':8,'>>':8,'+':9,'-':9,'*':10,'/':10,'%':10,'**':11}.get(t,0)
   if not prec or prec<minp:break
   self.pop();n=('bin',t,n,self.parse(prec+1))
  return n
 def typ(self,n):
  k=n[0]
  if k=='atom':
   s=n[1]
   if s in ['true','false']:return 'bool'
   if re.fullmatch(r'\d+',s):return 'lit'
   if s.endswith('u32'):return 'u32'
   if s.endswith('u64') or s.startswith('0x'):return 'u64'
   return self.types.get(s,'u64')
  if k in ['index','member']:return self.typ(n[1])
  if k=='call':return 'u32' if n[1][1] in ['ui','idx'] else 'bool' if n[1][1] in ['neg','lt'] else 'u64'
  if k=='un':return self.typ(n[2])
  if k=='bin':
   a,b=self.typ(n[2]),self.typ(n[3]);return b if a=='lit' else a
  if k=='if':return self.typ(n[2])
  return 'u64'
 def num(self,n,v):return n[0]=='atom' and re.sub(r'u64$|u32$','',n[1])==str(v)
 def emit(self,n):
  k=n[0]
  if k=='atom':return n[1]
  if k=='index':return self.emit(n[1])+'['+self.emit(n[2])+']'
  if k=='member':return self.emit(n[1])+'.'+n[2]
  if k=='un':return '('+('0u64-' if n[1]=='-' and self.typ(n[2])!='u32' else n[1])+self.emit(n[2])+')'
  if k=='if':return 'if '+self.emit(n[1])+' then '+self.emit(n[2])+' else '+self.emit(n[3])+' fi'
  if k=='call':return self.emit(n[1])+'('+', '.join(self.emit(a) for a in n[2])+')'
  if k=='struct':return n[1]+' {'+', '.join(key+': '+self.emit(val) for key,val in n[2])+'}'
  if k=='arr':return '['+''.join(a if isinstance(a,str) else self.emit(a) for a in n[1])+']'
  if k=='bin':
   op,a,b=n[1:];x,y=self.emit(a),self.emit(b)
   if op=='/' and self.typ(n)!='u32':
    if self.num(b,Q):return self.down(a)
    if a[0]=='bin' and a[1]=='*' and (self.num(a[2],Q) or self.num(a[3],Q)):
     return 'divq('+self.emit(a[3] if self.num(a[2],Q) else a[2])+', '+y+')'
    return 'divc('+x+', '+y+')'
   if op in ['<','>','<=','>='] and self.typ(a)!='u32' and self.typ(b)!='u32':
    if self.num(b,0):return {'<':'neg('+x+')','>':'(!neg('+x+') && '+x+' != 0u64)','<=':'(neg('+x+') || '+x+' == 0u64)','>=':'!neg('+x+')'}[op]
    return ('!' if op in ['<=','>='] else '')+'lt('+(', '.join([y,x]) if op in ['>','<='] else ', '.join([x,y]))+')'
   return '('+x+' '+op+' '+y+')'
  raise Exception(n)
 def down(self,n):
  if n[0]=='bin' and n[1]=='*':return 'mulq('+self.emit(n[2])+', '+self.emit(n[3])+')'
  if n[0]=='bin' and n[1] in ['+','-']:return '('+self.down(n[2])+n[1]+self.down(n[3])+')'
  return 'divc('+self.emit(n)+', 4294967296u64)'

def transform(src):
 # Replace only representation helpers; preserve kernel functions and source loops.
 src=re.sub(r'^from .*\n','',src,flags=re.M)
 src=re.sub(r'^def (neg|absv|ab|lt|divc|mulq|mul|ui|it)\([^\n]*\n(?:[ \t]+[^\n]*\n|\n)*','',src,flags=re.M)
 src=re.sub(r'\b(?:integer|field)\b','u64',src)
 src=re.sub(r'(\d+)(?:int|f)\b',r'\1u64',src)
 types={name:typ for typ,name in re.findall(r'\b(u32|u64|bool|Complex)(?:\[\d+\])*\s+(\w+)',src)}
 out=['from "../native" import neg, absv, ab, lt, divc, mulq, mul, divq, ui, it, integer_to_field','']
 for line in src.splitlines():
  if not line.strip() or line.lstrip().startswith('//'):continue
  if line.startswith('const u64 HALF'):continue
  if line.lstrip().startswith(('def ','struct ','for ','endfor','assert(')) or line.strip() in ['}','{']:out.append(line);continue
  m=re.match(r'^(\s*(?:return |.*? = ))(.*)$',line)
  if not m:out.append(line);continue
  pref,e=m.groups()
  ex=Expr(e,types)
  try:
   node=ex.parse();assert ex.i==len(ex.ts),(ex.ts[ex.i:]);e=ex.emit(node)
  except Exception as er:raise ValueError((line,er))
  out.append(pref+e)
 return '\n'.join(out)+'\n'

if __name__=='__main__':
 root=BASE.parent;lists=json.load(open(root/'repair-lists.json'));multi=set(lists['multi_function']);ns=set(sum(lists.values(),[]))
 for p in json.load(open(root/'programs.json')):
  n=p['program'];f=p['function']
  if n not in ns:continue
  d=BASE/(n+'__'+f if n in multi else n);d.mkdir(exist_ok=True)
  (d/'entry.json').write_text(json.dumps({k:v for k,v in p.items() if k!='zinnia_outcome'},indent=2))
  old=list(root.glob('batch*/'+n))[0];r=json.load(open(old/'result.json'))
  (d/'old-result.json').write_text(json.dumps(r,indent=2))
  candidates=[x for x in [old/'as_written.zok',old/'rewrite.zok'] if x.exists() and 'def main' in x.read_text()]
  if not candidates:continue
  candidate=candidates[0]
  # Batch3's first candidate uses field arithmetic; preserve its actual bounded-rewrite mode.
  mode='rewrite' if candidate.name=='rewrite.zok' else 'as_written'
  if mode=='rewrite' and r.get('as_written',{}).get('status') not in ['not_expressible','not expressible']:mode='as_written'
  try:
   code=transform(candidate.read_text())
   (d/(mode+'.zok')).write_text(code)
   (d/'translation.json').write_text(json.dumps({'mode':mode,'source':str(candidate),'rewrites':(r.get('rewrite') or {}).get('rewrites',[]) if mode=='rewrite' else []},indent=2))
   print(n,f,mode)
  except Exception as e:print('ERROR',n,str(e))
