"""Normalize canonical integer while counters before symbolic execution.

A counter initialized by a literal and incremented once at the end of each
iteration can use range when its bound is an invariant integer shape dimension.
Keep the increment in the body to preserve the counter's post-loop value.
"""
import ast
import copy


def normalize_counted_loops(tree):
    class Normalize(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            # Shape-unpacked locals are integers. Only accept names whose every
            # binding in this function comes from a shape-unpacking assignment.
            shape_names=set()
            writes={}
            for stmt in ast.walk(node):
                if isinstance(stmt,ast.Assign) and isinstance(stmt.value,ast.Attribute) and stmt.value.attr=='shape':
                    for target in stmt.targets:
                        if isinstance(target,(ast.Tuple,ast.List)):
                            for name in target.elts:
                                if isinstance(name,ast.Name):shape_names.add(name.id)
                if isinstance(stmt,ast.Name) and isinstance(stmt.ctx,ast.Store):writes[stmt.id]=writes.get(stmt.id,0)+1
            safe=set()
            for name in shape_names:
                shape_writes=sum(1 for stmt in ast.walk(node) if isinstance(stmt,ast.Assign) and isinstance(stmt.value,ast.Attribute) and stmt.value.attr=='shape' for target in stmt.targets if isinstance(target,(ast.Tuple,ast.List)) for item in target.elts if isinstance(item,ast.Name) and item.id==name)
                if shape_writes==writes[name]:safe.add(name)
            # Also recognize the usual individual dimension bindings, e.g.
            # C = x.shape[2], B = len(x), and invariant integer expressions.
            # Every write must have an integer source; an increment or unknown
            # reassignment disqualifies the name from use as a static bound.
            candidates={}
            for stmt in ast.walk(node):
                if isinstance(stmt,ast.Assign) and len(stmt.targets)==1 and isinstance(stmt.targets[0],ast.Name):
                    candidates.setdefault(stmt.targets[0].id,[]).append(stmt.value)
            len_shadowed=any(isinstance(n,ast.Name) and n.id=='len' and isinstance(n.ctx,ast.Store) or isinstance(n,ast.arg) and n.arg=='len' or isinstance(n,ast.FunctionDef) and n.name=='len' for n in ast.walk(node))
            def dimension(expr):
                if isinstance(expr,ast.Constant):return type(expr.value) is int
                if isinstance(expr,ast.Name):return expr.id in safe
                if isinstance(expr,ast.Subscript) and isinstance(expr.value,ast.Attribute) and expr.value.attr=='shape':return True
                if isinstance(expr,ast.Call) and isinstance(expr.func,ast.Name) and expr.func.id=='len' and not len_shadowed:return True
                return isinstance(expr,ast.BinOp) and isinstance(expr.op,(ast.Add,ast.Sub,ast.Mult,ast.FloorDiv)) and dimension(expr.left) and dimension(expr.right)
            for _ in range(len(candidates)):
                added={name for name,values in candidates.items() if len(values)==writes.get(name) and all(dimension(v) for v in values)}
                if added<=safe:break
                safe.update(added)
            # range may be shadowed by a parameter/local or nested definition.
            shadowed=any(isinstance(n,ast.Name) and n.id=='range' and isinstance(n.ctx,ast.Store) or isinstance(n,ast.arg) and n.arg=='range' or isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name=='range' for n in ast.walk(node))
            if shadowed:return node
            def integer(expr):
                return (isinstance(expr,ast.Constant) and type(expr.value) is int or isinstance(expr,ast.Name) and expr.id in safe or isinstance(expr,ast.BinOp) and isinstance(expr.op,(ast.Add,ast.Sub,ast.Mult,ast.FloorDiv)) and integer(expr.left) and integer(expr.right))
            def block(stmts):
                result=[]
                for stmt in stmts:
                    # Inspect the unmodified body, then recurse.
                    replacement=None
                    if isinstance(stmt,ast.While) and result:
                        init=result[-1];test=stmt.test
                        if (isinstance(init,ast.Assign) and len(init.targets)==1 and isinstance(init.targets[0],ast.Name)
                            and isinstance(init.value,ast.Constant) and type(init.value.value) is int
                            and isinstance(test,ast.Compare) and isinstance(test.left,ast.Name)
                            and test.left.id==init.targets[0].id and len(test.ops)==1 and isinstance(test.ops[0],ast.Lt)
                            and integer(test.comparators[0]) and stmt.body):
                            name=test.left.id;end=stmt.body[-1]
                            increments=[n for n in ast.walk(stmt) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store) and n.id==name]
                            control=any(isinstance(n,(ast.Break,ast.Continue,ast.Return,ast.Yield,ast.YieldFrom,ast.NamedExpr)) for n in ast.walk(stmt))
                            bound_names={n.id for n in ast.walk(test.comparators[0]) if isinstance(n,ast.Name)}
                            bound_changes=any(isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store) and n.id in bound_names for n in ast.walk(stmt))
                            if (isinstance(end,ast.AugAssign) and isinstance(end.target,ast.Name) and end.target.id==name
                                and isinstance(end.op,ast.Add) and isinstance(end.value,ast.Constant) and type(end.value.value) is int and end.value.value==1
                                and len(increments)==1 and not control and not bound_changes):
                                replacement=ast.copy_location(ast.For(target=ast.Name(id=name,ctx=ast.Store()),iter=ast.Call(func=ast.Name(id='range',ctx=ast.Load()),args=[copy.deepcopy(init.value),copy.deepcopy(test.comparators[0])],keywords=[]),body=stmt.body,orelse=stmt.orelse,type_comment=None),stmt)
                    stmt=replacement or stmt
                    if isinstance(stmt,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                        result.append(stmt)
                        continue
                    for field in ('body','orelse','finalbody'):
                        children=getattr(stmt,field,None)
                        if isinstance(children,list):setattr(stmt,field,block(children))
                    result.append(stmt)
                return result
            node.body=block(node.body)
            return node
    return ast.fix_missing_locations(Normalize().visit(tree))
