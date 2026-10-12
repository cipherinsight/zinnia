"""Observe ndarray-producing source expressions, then expose them through ordinary outputs.
No compiler modifications. Observations retain references; only shape/size is measured.
"""
import ast,copy,json,traceback,sys,os
from pathlib import Path
import measure
from benchmarking.harness.reference import kernel_namespace
from zinnia import ZKCircuit
from zinnia.api.zk_chip import ZKChip
import numpy as np

KINDS=(ast.Call,ast.Subscript,ast.BinOp,ast.UnaryOp,ast.Attribute,ast.Compare)
def key(n):return f'{n.lineno}:{n.col_offset}:{n.end_lineno}:{n.end_col_offset}:{type(n).__name__}'
def candidates(tree):return {key(n):ast.unparse(n) for n in ast.walk(tree) if isinstance(n,KINDS) and (not hasattr(n,'ctx') or isinstance(n.ctx,ast.Load))}
class Observe(ast.NodeTransformer):
 def visit(self,n):
  if isinstance(n,KINDS) and (not hasattr(n,'ctx') or isinstance(n.ctx,ast.Load)):
   k=key(n); n=self.generic_visit(n)
   return ast.copy_location(ast.Call(ast.Name('_observe',ast.Load()),[ast.Constant(k),n],[]),n)
  return super().visit(n)
 def visit_FunctionDef(self,n):
  n.decorator_list=[];n.body=[self.visit(x) for x in n.body];return n
 def visit_ImportFrom(self,n):return None if n.module=='zinnia' else n

class DesugarSubscriptAugAssign(ast.NodeTransformer):
 def visit_AugAssign(self,n):
  if not isinstance(n.target,ast.Subscript):return self.generic_visit(n)
  # In this accepted corpus, only Boolean-mask augmented subscripts can
  # introduce a runtime shape. Other augmented subscripts have fixed extents.
  if not any(isinstance(x,ast.Compare) for x in ast.walk(n.target.slice)):return self.generic_visit(n)
  lhs=copy.deepcopy(n.target);lhs.ctx=ast.Load()
  rhs=ast.copy_location(ast.BinOp(lhs,n.op,n.value),n)
  return ast.copy_location(ast.Assign([n.target],rhs),n)

def observations(p):
 tree=ast.fix_missing_locations(DesugarSubscriptAugAssign().visit(ast.parse(p.kernel_path.read_text())));names=candidates(tree);found={}; runs={}
 for fixture in ['A','B']:
  seen=[]
  def observe(k,v):
   if isinstance(v,np.ndarray) or (isinstance(v,(tuple,list)) and any(isinstance(x,np.ndarray) for x in v)):
    def sizes(x):
     if isinstance(x,np.ndarray):return dict(shape=list(x.shape),length=int(x.size))
     if isinstance(x,(tuple,list)):return [sizes(y) for y in x]
     return None
    found[k]=names[k];seen.append(dict(site=k,value=sizes(v)))
   return v
  ns=kernel_namespace();ns['_observe']=observe
  exec(compile(ast.fix_missing_locations(Observe().visit(copy.deepcopy(tree))),str(p.kernel_path),'exec'),ns)
  try:ns[p.function](*measure.load_fixture(p.fixture_path).arguments(fixture));runs[fixture]=seen
  except BaseException as e:runs[fixture]={'error':str(e)}
 return tree,found,runs

class Lift:
 def __init__(self,selected):self.selected=selected;self.i=0;self.prefix=[]
 def expr(self,n):
  if not isinstance(n,ast.AST):return n
  # Do not lift from comprehensions/lambdas/short-circuit expressions.
  if isinstance(n,(ast.ListComp,ast.GeneratorExp,ast.Lambda,ast.BoolOp,ast.IfExp)):return n
  for field,value in ast.iter_fields(n):
   if isinstance(value,list):setattr(n,field,[self.expr(v) if isinstance(v,ast.AST) else v for v in value])
   elif isinstance(value,ast.AST):setattr(n,field,self.expr(value))
  if isinstance(n,KINDS) and hasattr(n,'lineno') and key(n) in self.selected and (not hasattr(n,'ctx') or isinstance(n.ctx,ast.Load)):
   name=f'bprobe_v{self.i}';self.i+=1
   self.prefix.append(ast.Assign([ast.Name(name,ast.Store())],n))
   self.prefix.append(ast.Expr(ast.Call(ast.Attribute(ast.Name('bprobe_trace',ast.Load()),'append',ast.Load()),[ast.Tuple([ast.Constant(list(self.selected).index(key(n))),ast.Name(name,ast.Load())],ast.Load())],[])))
   return ast.Name(name,ast.Load())
  return n
 def block(self,body):
  out=[]
  for s in body:
   self.prefix=[]
   if isinstance(s,ast.Return):
    if s.value is not None:self.expr(s.value)
    out+=self.prefix+[ast.Return(ast.Name('bprobe_trace',ast.Load()))];continue
   if isinstance(s,(ast.FunctionDef,ast.AsyncFunctionDef)):
    out.append(s);continue
   for f,v in ast.iter_fields(s):
    if f in ('body','orelse','finalbody'):continue
    if isinstance(v,ast.AST):setattr(s,f,self.expr(v))
    elif isinstance(v,list):setattr(s,f,[self.expr(x) if isinstance(x,ast.AST) else x for x in v])
   prefix=self.prefix
   for f in ('body','orelse','finalbody'):
    if hasattr(s,f):setattr(s,f,self.block(getattr(s,f)))
   out+=prefix+[s]
  return out

def instrument(p,tree,found):
 tree=copy.deepcopy(tree); fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==p.function)
 selected=sorted({key(n) for n in ast.walk(fn) if isinstance(n,KINDS) and key(n) in found})
 lift=Lift(selected);fn.body=[ast.Assign([ast.Name('bprobe_trace',ast.Store())],ast.List([],ast.Load()))]+lift.block(fn.body)
 if not isinstance(fn.body[-1],ast.Return):fn.body.append(ast.Return(ast.Name('bprobe_trace',ast.Load())))
 fn.decorator_list=[]
 source=ast.unparse(ast.fix_missing_locations(fn))
 chips=[ZKChip.from_source(n.name,ast.unparse(n)) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name!=p.function]
 return source,chips,selected

def probe(p):
 tree,found,runs=observations(p)
 out=dict(id=measure.ident(p),sites=found,numpy=runs)
 source,chips,selected=instrument(p,tree,found)
 out['selected']=selected
 out['source']=source
 try:
  c=ZKCircuit(p.function,source,chips=chips).compile();layout=c.get_output_layout()
  out['layout']=layout
  # trace item = (constant string label (no layout), array): output path [item,1]
  out['dynamic']=[x for x in layout if 'dynamic'in x]
 except BaseException as e:out['error']=str(e);out['traceback']=traceback.format_exc()
 return out
if __name__=='__main__':
 p=next(p for p in measure.entries() if measure.ident(p)==sys.argv[1])
 try:r=probe(p)
 except BaseException as e:r=dict(id=measure.ident(p),error=str(e),traceback=traceback.format_exc())
 d=measure.HERE/'probes';d.mkdir(exist_ok=True);(d/(measure.ident(p)+'.json')).write_text(json.dumps(r,indent=2))
