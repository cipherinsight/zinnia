"""Lower finite Python callbacks and lexical helpers to ordinary circuit chips.

This pass operates on syntax, never on task names or expected answers. Generated
names are hygienic. It intentionally rejects escaping/reassigned closures,
nonlocal writes, recursion, and unsupported callback options rather than silently
changing Python's binding or evaluation semantics.
"""
from __future__ import annotations

import ast
import builtins
import copy
from dataclasses import dataclass
import textwrap

from zinnia.compile.import_scope import check_import_statement


class FunctionLoweringError(ValueError):
    """A callable cannot be faithfully represented by the circuit source API."""


def _module_expr(module):
    """The circuit expression for an allowed module: NumPy is always spelled
    ``np``; ``math`` keeps its name; submodules are attribute chains."""
    root, *parts = module.split('.')
    value = _name('np' if root == 'numpy' else root)
    for part in parts:
        value = ast.Attribute(value=value, attr=part, ctx=ast.Load())
    return value


@dataclass
class _Helper:
    name: str
    captures: tuple[str, ...]
    args: ast.arguments


@dataclass
class _Iterator:
    expression: ast.GeneratorExp
    consumed: bool = False


@dataclass
class _Vectorized:
    callback: object
    otypes: ast.expr | None = None
    cache: bool = False


def _name(s, ctx=None):
    return ast.Name(id=s, ctx=ctx or ast.Load())


def _call(s, args, keywords=()):
    return ast.Call(func=_name(s), args=args, keywords=list(keywords))


def _immutable_literal(value):
    return isinstance(value,(int,float,bool,str,bytes,type(None))) or isinstance(value,tuple) and all(_immutable_literal(item) for item in value)


def _bound(fn):
    result = {a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs}
    if fn.args.vararg: result.add(fn.args.vararg.arg)
    if fn.args.kwarg: result.add(fn.args.kwarg.arg)
    class Bind(ast.NodeVisitor):
        def visit_Name(self, n):
            if isinstance(n.ctx, (ast.Store, ast.Del)): result.add(n.id)
        def visit_FunctionDef(self, n): result.add(n.name)
        def visit_Lambda(self, n): pass
        def visit_ListComp(self, n): pass
        visit_GeneratorExp = visit_ListComp
        visit_SetComp = visit_ListComp
        visit_DictComp = visit_ListComp
    for s in fn.body: Bind().visit(s)
    return result


def _free(fn):
    bound = _bound(fn)
    free = set()
    class Read(ast.NodeVisitor):
        def visit_Name(self, n):
            if isinstance(n.ctx, ast.Load) and n.id not in bound: free.add(n.id)
        def visit_FunctionDef(self, n): pass
        def visit_Lambda(self, n): pass
        def visit_Call(self,n):
            # Unrewritten bare call names resolve through the compiler's chip
            # / external / builtin registry. They are not closure value args.
            if not isinstance(n.func,ast.Name):self.visit(n.func)
            for argument in n.args:self.visit(argument)
            for keyword in n.keywords:self.visit(keyword.value)
        def comp(self, n):
            old = set(bound)
            for g in n.generators:
                self.visit(g.iter)
                bound.update(x.id for x in ast.walk(g.target) if isinstance(x, ast.Name))
                for condition in g.ifs: self.visit(condition)
            if isinstance(n, ast.DictComp): self.visit(n.key); self.visit(n.value)
            else: self.visit(n.elt)
            bound.clear(); bound.update(old)
        visit_ListComp = comp
        visit_GeneratorExp = comp
        visit_SetComp = comp
        visit_DictComp = comp
    for s in fn.body: Read().visit(s)
    return free


class _Lower:
    def __init__(self, tree):
        self.used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        self.used.update(n.arg for n in ast.walk(tree) if isinstance(n, ast.arg))
        self.used.update(n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)))
        self.helpers = []
        self.generated = set()
        self.constants = {}
        self.active = set()
        self.scopes = []
        self.global_env = {}

    def fresh(self, stem):
        stem = '_zinnia_fn_' + stem
        candidate = stem
        i = 0
        while candidate in self.used:
            i += 1; candidate = stem + '_' + str(i)
        self.used.add(candidate)
        return candidate

    def lift(self, fn, env):
        if fn.name in self.active:
            raise FunctionLoweringError('Recursive Python helpers require an explicit supported chip')
        if fn.decorator_list:
            raise FunctionLoweringError('Decorated local functions cannot be lifted implicitly')
        if fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs or fn.args.posonlyargs:
            raise FunctionLoweringError('Implicit helper lowering supports ordinary positional/keyword parameters')
        for d in fn.args.defaults:
            try:
                value = ast.literal_eval(d)
            except (ValueError, TypeError):
                raise FunctionLoweringError('Helper defaults must be immutable literals; definition-time expressions are unsupported')
            if not _immutable_literal(value):
                raise FunctionLoweringError('Mutable default arguments cannot be lifted implicitly')
        self.active.add(fn.name)
        original = copy.deepcopy(fn)
        fn = copy.deepcopy(fn)
        # Retain the existing circuit-helper result convention used by the
        # decorator's older pure-helper lifter, including captured helpers.
        from zinnia.compile.nested_def_lift import _rewrite_zinnia_result_to_return
        fn.body = _rewrite_zinnia_result_to_return(fn.body)
        local_env = dict(env)
        for arg in fn.args.args: local_env.pop(arg.arg, None)
        self.scopes.append(_bound(fn))
        fn.body = self.block(fn.body, local_env)
        self.scopes.pop()
        self.active.remove(original.name)
        enclosing = set().union(*self.scopes) if self.scopes else set()
        captures = sorted(_free(fn) - (set(dir(builtins)) - enclosing) - {'np'} - self.generated)
        # Mutation of an enclosing variable needs closure cells, not value args.
        for n in ast.walk(fn):
            if isinstance(n, (ast.Nonlocal, ast.Global)):
                raise FunctionLoweringError('nonlocal/global writes are unsupported by implicit helper lowering')
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Del):
                raise FunctionLoweringError('Deleting helper bindings is unsupported')
        fn.name = self.fresh(original.name)
        self.generated.add(fn.name)
        fn.args.args = [ast.arg(arg=c) for c in captures] + fn.args.args
        self.helpers.append((fn.name, ast.unparse(ast.fix_missing_locations(fn))))
        return _Helper(fn.name, tuple(captures), original.args)

    def callback(self, expr, env):
        if isinstance(expr, ast.Lambda):
            fn = ast.FunctionDef(name=self.fresh('lambda'), args=copy.deepcopy(expr.args),
                body=[ast.Return(value=copy.deepcopy(expr.body))], decorator_list=[])
            return self.lift(fn, env)
        if isinstance(expr, ast.Name) and expr.id in env:
            value = env[expr.id]
            if isinstance(value, ast.FunctionDef):
                scopes = self.scopes
                self.scopes = []
                try:
                    value = self.lift(value, self.global_env)
                finally:
                    self.scopes = scopes
                env[expr.id] = value
            return value
        if isinstance(expr, ast.Name) and expr.id in self.active:
            raise FunctionLoweringError('Recursive Python helpers require an explicit supported chip')
        if isinstance(expr, (ast.Name, ast.Attribute)):
            return copy.deepcopy(expr)
        raise FunctionLoweringError('Callback must be a statically bound function or lambda')

    def invoke(self, callback, args, keywords=()):
        if isinstance(callback, _Helper):
            return _call(callback.name, [_name(c) for c in callback.captures] + args, keywords)
        if isinstance(callback, (ast.Name, ast.Attribute)):
            return ast.Call(func=copy.deepcopy(callback), args=args, keywords=list(keywords))
        raise FunctionLoweringError('This callable cannot be invoked in a scalar context')

    def block(self, statements, env):
        result = []
        for stmt in statements:
            if isinstance(stmt, ast.FunctionDef):
                if stmt.name in env and isinstance(env[stmt.name], (_Helper, _Vectorized)):
                    raise FunctionLoweringError('Rebinding an existing helper requires runtime function cells')
                env[stmt.name] = self.lift(stmt, env)
                continue
            if isinstance(stmt, (ast.Nonlocal, ast.Global)):
                raise FunctionLoweringError('nonlocal/global writes are unsupported by implicit helper lowering')
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                target = stmt.targets[0].id
                if target in env and isinstance(env[target], (_Helper, _Vectorized)):
                    raise FunctionLoweringError('Rebinding an existing helper requires runtime function cells')
                value = stmt.value
                if isinstance(value, ast.Lambda) or isinstance(value, ast.Name) and value.id in env:
                    env[target] = self.callback(value, env)
                    continue
                if self.is_numpy(value, 'vectorize'):
                    env[target] = self.vectorize(value,env)
                    continue
                env.pop(target, None)
            rewritten = _Expr(self, env).visit(copy.deepcopy(stmt))
            if isinstance(rewritten, ast.Assign) and len(rewritten.targets)==1 and isinstance(rewritten.targets[0], ast.Name) and isinstance(rewritten.value, ast.GeneratorExp):
                # Bind the outer iterable now; callback captures remain late.
                iterable = self.fresh('iterable')
                result.append(ast.Assign(targets=[_name(iterable, ast.Store())], value=rewritten.value.generators[0].iter))
                rewritten.value.generators[0].iter = _name(iterable)
                env[rewritten.targets[0].id] = _Iterator(rewritten.value)
            else:
                result.append(rewritten)
        return result

    @staticmethod
    def is_numpy(n, member):
        return isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'np' and n.func.attr == member

    def wrapper(self, stem, arguments, body, env):
        fn = ast.FunctionDef(name=self.fresh(stem),
            args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=a) for a in arguments], vararg=None,
                kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
            body=body, decorator_list=[])
        return self.lift(fn, env)

    def vectorize(self,value,env):
        if not value.args: raise FunctionLoweringError('np.vectorize requires a callback')
        opts = {k.arg: k.value for k in value.keywords}
        if len(value.args)>2 or set(opts)-{'otypes','cache'}:
            raise FunctionLoweringError('np.vectorize signature/excluded options are unsupported')
        cache = opts.get('cache', ast.Constant(False))
        if not isinstance(cache, ast.Constant) or not isinstance(cache.value, bool):
            raise FunctionLoweringError('np.vectorize cache must be a literal bool')
        otypes = value.args[1] if len(value.args)>1 else opts.get('otypes')
        if isinstance(otypes,ast.Constant) and otypes.value is None: otypes=None
        if otypes is not None and not (isinstance(otypes,ast.Constant) and isinstance(otypes.value,str) and len(otypes.value)==1 or isinstance(otypes,(ast.List,ast.Tuple)) and len(otypes.elts)==1):
            raise FunctionLoweringError('np.vectorize lowering supports exactly one scalar output dtype')
        return _Vectorized(self.callback(value.args[0],env),otypes,cache.value)

    def vectorized_call(self, cb, args, keywords, env):
        if len(args)!=1 or keywords:
            raise FunctionLoweringError('np.vectorize lowering currently accepts one array argument')
        a, flat, item, out = [self.fresh(s) for s in ('array','flat','item','result')]
        statements = ast.parse(f'{flat}=np.asarray({a}).reshape(-1)').body
        if cb.otypes is None:
            statements.extend(ast.parse(f'assert len({flat}) > 0').body)
        first_name = self.fresh('first')
        dtype = []
        if cb.otypes is None:
            first = self.invoke(cb.callback, [ast.Subscript(value=_name(flat), slice=ast.Constant(0), ctx=ast.Load())])
            statements.append(ast.Assign(targets=[_name(first_name,ast.Store())],value=first))
            dtype=[ast.keyword(arg='dtype',value=ast.Attribute(value=ast.Call(func=ast.Attribute(value=_name('np'),attr='asarray',ctx=ast.Load()),args=[_name(first_name)],keywords=[]),attr='dtype',ctx=ast.Load()))]
        else:
            dtype=[ast.keyword(arg='dtype', value=ast.Subscript(value=copy.deepcopy(cb.otypes),slice=ast.Constant(0),ctx=ast.Load()))]
        iterable = _name(flat)
        if cb.otypes is None and cb.cache:
            iterable=ast.Subscript(value=_name(flat),slice=ast.Slice(lower=ast.Constant(1)),ctx=ast.Load())
        values = ast.ListComp(elt=self.invoke(cb.callback, [_name(item)]), generators=[ast.comprehension(target=_name(item,ast.Store()),iter=iterable,ifs=[],is_async=0)])
        if cb.otypes is None and cb.cache:
            values=ast.BinOp(left=ast.List(elts=[_name(first_name)],ctx=ast.Load()),op=ast.Add(),right=values)
        value = ast.Call(func=ast.Attribute(value=_name('np'),attr='array',ctx=ast.Load()),args=[values],keywords=dtype)
        statements.append(ast.Assign(targets=[_name(out,ast.Store())],value=value))
        statements.extend(ast.parse(f'return {out}.reshape(np.asarray({a}).shape)').body)
        wrapper=self.wrapper('vectorize', [a], statements, env)
        return self.invoke(wrapper,args)

    def piecewise(self, node, env):
        if len(node.args)!=3 or node.keywords:
            raise FunctionLoweringError('np.piecewise lowering requires x, a literal condition list, and a literal function list')
        x, conditions, funcs = node.args
        if not isinstance(conditions,(ast.List,ast.Tuple)) or not isinstance(funcs,(ast.List,ast.Tuple)):
            raise FunctionLoweringError('np.piecewise conditions/functions must be syntactic lists or tuples')
        if len(funcs.elts) not in (len(conditions.elts),len(conditions.elts)+1):
            raise FunctionLoweringError('np.piecewise function count must equal condition count or exceed it by one')
        arg, out, covered = [self.fresh(s) for s in ('array','result','covered')]
        masks=[self.fresh('mask') for _ in conditions.elts]
        params=[arg]+masks
        args=[_Expr(self,env).visit(x)]+[_Expr(self,env).visit(c) for c in conditions.elts]
        body=ast.parse(f'{out}=np.zeros_like({arg})\n{covered}=np.zeros_like({arg},dtype=bool)').body
        for mask in masks: body.extend(ast.parse(f'{covered}={covered} | {mask}').body)
        if len(funcs.elts)>len(masks):
            default=self.fresh('mask'); body.extend(ast.parse(f'{default}=~{covered}').body);masks.append(default)
        for mask,func in zip(masks,funcs.elts):
            if isinstance(func,(ast.Lambda,ast.Attribute)) or isinstance(func,ast.Name) and (func.id in env or callable(getattr(builtins,func.id,None))):
                cb=self.callback(func,env)
                value=self.invoke(cb,[ast.Subscript(value=_name(arg),slice=_name(mask),ctx=ast.Load())])
                update=ast.Assign(targets=[ast.Subscript(value=_name(out),slice=_name(mask),ctx=ast.Store())],value=value)
                body.append(ast.If(test=ast.Call(func=ast.Attribute(value=_name('np'),attr='any',ctx=ast.Load()),args=[_name(mask)],keywords=[]),body=[update],orelse=[]))
            else:
                scalar=self.fresh('choice');params.append(scalar);args.append(_Expr(self,env).visit(func))
                body.append(ast.Assign(targets=[ast.Subscript(value=_name(out),slice=_name(mask),ctx=ast.Store())],value=_name(scalar)))
        body.append(ast.Return(value=_name(out)))
        return self.invoke(self.wrapper('piecewise',params,body,env),args)

    def apply_axis(self,node,env):
        if len(node.args)<3:
            raise FunctionLoweringError('np.apply_along_axis requires callback, axis, and array positionally')
        cb=self.callback(node.args[0],env)
        axis=node.args[1]
        try: ax=ast.literal_eval(axis)
        except (ValueError,TypeError): raise FunctionLoweringError('np.apply_along_axis axis must be a literal integer')
        if not isinstance(ax,int): raise FunctionLoweringError('np.apply_along_axis axis must be a literal integer')
        arr, normalized, moved, rows, i, values, result = [self.fresh(s) for s in ('array','axis','moved','rows','i','values','result')]
        extra=[self.fresh('arg') for _ in node.args[3:]]
        kwparams=[self.fresh('kwarg') for _ in node.keywords]
        if any(k.arg is None for k in node.keywords): raise FunctionLoweringError('Expanded callback keyword arguments are unsupported')
        body=ast.parse(f'''{normalized}={ax}
if {normalized}<0:
    {normalized}={normalized}+{arr}.ndim
assert 0 <= {normalized} < {arr}.ndim
{moved}=np.moveaxis({arr},{normalized},-1)
{rows}={moved}.reshape((-1,{moved}.shape[-1]))
assert len({rows}) > 0
''').body
        value=self.invoke(cb,[ast.Subscript(value=_name(rows),slice=_name(i),ctx=ast.Load())]+[_name(a) for a in extra],
            [ast.keyword(arg=k.arg,value=_name(a)) for k,a in zip(node.keywords,kwparams)])
        body.append(ast.Assign(targets=[_name(values,ast.Store())],value=ast.ListComp(elt=value,generators=[ast.comprehension(target=_name(i,ast.Store()),iter=_call('range',[_call('len',[_name(rows)])]),ifs=[],is_async=0)])))
        body.extend(ast.parse(f'''{result}=np.array({values})
{result}={result}.reshape({moved}.shape[:-1]+{result}.shape[1:])
return {result}.transpose(tuple(range({normalized}))+tuple(range({moved}.ndim-1,{result}.ndim))+tuple(range({normalized},{moved}.ndim-1)))''').body)
        wrapper=self.wrapper('apply_axis',[arr]+extra+kwparams,body,env)
        return self.invoke(wrapper,[_Expr(self,env).visit(a) for a in node.args[2:]]+[_Expr(self,env).visit(k.value) for k in node.keywords])


class _Expr(ast.NodeTransformer):
    def __init__(self, lower, env): self.lower=lower;self.env=env

    def visit_Nonlocal(self,n): raise FunctionLoweringError('nonlocal writes require closure cells and cannot be lifted')
    def visit_Global(self,n): raise FunctionLoweringError('global writes cannot be lifted')
    def visit_NamedExpr(self,n): raise FunctionLoweringError('Assignment expressions in callbacks are unsupported')
    def visit_FunctionDef(self,n): raise FunctionLoweringError('Conditional/loop-defined helpers are unsupported')
    def visit_Lambda(self,n): raise FunctionLoweringError('A lambda cannot escape its statically known call site')
    def visit_Name(self,n):
        if isinstance(n.ctx,ast.Load) and n.id in self.env and isinstance(self.env[n.id], _Iterator):
            iterator = self.env[n.id]
            if iterator.consumed:
                raise FunctionLoweringError('Reusing a one-shot iterator requires runtime iterator state')
            iterator.consumed = True
            return copy.deepcopy(iterator.expression)
        if isinstance(n.ctx,ast.Load) and n.id in self.env:
            raise FunctionLoweringError('Function values may only be invoked, aliased, or passed to supported callback APIs')
        if isinstance(n.ctx,ast.Load) and n.id in self.lower.constants and not any(n.id in scope for scope in self.lower.scopes):
            return copy.deepcopy(self.lower.constants[n.id])
        return n

    def comprehension(self,n):
        env=dict(self.env)
        visitor=_Expr(self.lower,env)
        local=set()
        self.lower.scopes.append(local)
        for gen in n.generators:
            gen.iter=visitor.visit(gen.iter)
            for target in ast.walk(gen.target):
                if isinstance(target,ast.Name):
                    env.pop(target.id,None)
                    local.add(target.id)
            gen.ifs=[visitor.visit(test) for test in gen.ifs]
        if isinstance(n,ast.DictComp):n.key=visitor.visit(n.key);n.value=visitor.visit(n.value)
        else:n.elt=visitor.visit(n.elt)
        self.lower.scopes.pop()
        return n
    visit_ListComp=comprehension
    visit_GeneratorExp=comprehension
    visit_SetComp=comprehension
    visit_DictComp=comprehension

    def visit_For(self,n):
        n.iter=self.visit(n.iter)
        env=dict(self.env)
        for a in ast.walk(n.target):
            if isinstance(a,ast.Name):
                if isinstance(env.get(a.id),(_Helper,_Vectorized)):
                    raise FunctionLoweringError('Loop targets cannot rebind a static helper')
                env.pop(a.id,None)
        n.body=self.lower.block(n.body,env)
        n.orelse=self.lower.block(n.orelse,dict(self.env))
        return n

    def visit_If(self,n):
        # Definitions/rebinding of callable names in dynamic branches would
        # require a runtime function value, not a static chip identity.
        for s in n.body+n.orelse:
            if isinstance(s,ast.FunctionDef) or isinstance(s,ast.Assign) and (isinstance(s.value,ast.Lambda) or any(isinstance(t,ast.Name) and t.id in self.env for t in s.targets)):
                raise FunctionLoweringError('Conditional helper bindings are unsupported')
        return self.generic_visit(n)

    def visit_Call(self,n):
        l=self.lower
        if l.is_numpy(n.func,'vectorize'):
            callback=l.vectorize(n.func,self.env)
            return l.vectorized_call(callback,[self.visit(a) for a in n.args],[ast.keyword(arg=k.arg,value=self.visit(k.value)) for k in n.keywords],self.env)
        if l.is_numpy(n,'piecewise'): return l.piecewise(n,self.env)
        if l.is_numpy(n,'apply_along_axis'): return l.apply_axis(n,self.env)
        if l.is_numpy(n,'fromiter'):
            if not n.args: raise FunctionLoweringError('np.fromiter requires an iterable')
            opts={k.arg:k.value for k in n.keywords}
            if len(n.args)>3 or set(opts)-{'dtype','count'}:raise FunctionLoweringError('Unsupported np.fromiter arguments')
            dtype=n.args[1] if len(n.args)>1 else opts.get('dtype')
            if dtype is None:raise FunctionLoweringError('np.fromiter requires dtype')
            count=n.args[2] if len(n.args)>2 else opts.get('count',ast.Constant(-1))
            it,ct,dt,out,i=[l.fresh(s) for s in ('iter','count','dtype','result','i')]
            # The circuit iterable representation is finite. Enumerating and
            # filtering avoids indexing a generator and retains count checks.
            body=ast.parse(f'''{out}=[]
if {ct} != 0:
    for {i} in {it}:
        {out}.append({i})
        if {ct} >= 0 and len({out}) == {ct}:
            break
assert {ct} < 0 or len({out}) == {ct}
return np.array({out},dtype={dt})''').body
            wrapper=l.wrapper('fromiter',[it,dt,ct],body,self.env)
            return l.invoke(wrapper,[self.visit(n.args[0]),self.visit(dtype),self.visit(count)])
        if isinstance(n.func,ast.Name) and n.func.id in ('map','filter') and n.func.id not in self.env:
            kind=n.func.id
            if n.keywords or len(n.args)<2 or kind=='filter' and len(n.args)!=2:
                raise FunctionLoweringError('Unsupported map/filter call signature')
            callback = None if isinstance(n.args[0],ast.Constant) and n.args[0].value is None else l.callback(n.args[0],self.env)
            if callback is None and kind=='map': raise FunctionLoweringError('map requires a callable')
            names=[l.fresh('item') for _ in n.args[1:]]
            iterable=self.visit(n.args[1]) if len(names)==1 else _call('zip',[self.visit(a) for a in n.args[1:]])
            target=_name(names[0],ast.Store()) if len(names)==1 else ast.Tuple(elts=[_name(a,ast.Store()) for a in names],ctx=ast.Store())
            value=l.invoke(callback,[_name(a) for a in names]) if callback is not None else _name(names[0])
            elt=value if kind=='map' else _name(names[0])
            conditions=[value] if kind=='filter' else []
            if isinstance(iterable,ast.GeneratorExp) and len(names)==1:
                if isinstance(iterable.elt,ast.Name):
                    class Replace(ast.NodeTransformer):
                        def visit_Name(self,node):
                            return copy.deepcopy(iterable.elt) if node.id==names[0] and isinstance(node.ctx,ast.Load) else node
                    elt=Replace().visit(elt)
                    conditions=[Replace().visit(c) for c in conditions]
                    generators=copy.deepcopy(iterable.generators)
                    generators[-1].ifs.extend(conditions)
                else:
                    generators=copy.deepcopy(iterable.generators)+[ast.comprehension(target=target,iter=ast.Tuple(elts=[iterable.elt],ctx=ast.Load()),ifs=conditions,is_async=0)]
                return ast.GeneratorExp(elt=elt,generators=generators)
            return ast.GeneratorExp(elt=elt,generators=[ast.comprehension(target=target,iter=iterable,ifs=conditions,is_async=0)])
        if isinstance(n.func,ast.Lambda) or isinstance(n.func,ast.Name) and n.func.id in self.env:
            cb=l.callback(n.func,self.env)
            args=[self.visit(a) for a in n.args];kws=[ast.keyword(arg=k.arg,value=self.visit(k.value)) for k in n.keywords]
            if isinstance(cb,_Vectorized):return l.vectorized_call(cb,args,kws,self.env)
            return l.invoke(cb,args,kws)
        if isinstance(n.func,ast.Name) and n.func.id in l.active:
            raise FunctionLoweringError('Recursive Python helpers require an explicit supported chip')
        return self.generic_visit(n)


class _NumpyImports(ast.NodeTransformer):
    """Resolve statically bound NumPy names without overriding lexical locals."""

    def __init__(self, aliases=None):
        self.aliases = dict(aliases or {})

    @staticmethod
    def bindings(node):
        """Names bound by an allowed import, mapped to the expression the
        circuit uses for them (``np...`` or ``math...``). Imports of other
        modules are out of scope (``import_scope``)."""
        check_import_statement(node)
        result = {}
        if isinstance(node, ast.Import):
            for alias in node.names:
                # `import numpy.linalg` binds `numpy`; with `as`, the alias
                # names the submodule itself.
                module = alias.name if alias.asname else alias.name.split('.')[0]
                result[alias.asname or module] = _module_expr(module)
        else:
            for alias in node.names:
                if alias.name == '*':
                    raise FunctionLoweringError('Wildcard imports require explicit member names')
                result[alias.asname or alias.name] = ast.Attribute(
                    value=_module_expr(node.module), attr=alias.name, ctx=ast.Load())
        return result

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.aliases:
            return ast.copy_location(copy.deepcopy(self.aliases[node.id]), node)
        return node

    def visit_FunctionDef(self, node):
        # Defaults/decorators are evaluated in the enclosing scope.
        node.decorator_list = [self.visit(x) for x in node.decorator_list]
        node.args.defaults = [self.visit(x) for x in node.args.defaults]
        node.args.kw_defaults = [self.visit(x) if x else None for x in node.args.kw_defaults]
        inherited = dict(self.aliases)
        for name in _bound(node):
            inherited.pop(name, None)
        imported = {}
        imports = []
        for statement in node.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                imports.append(statement)
                imported.update(self.bindings(statement))
        # Imports are local bindings even on paths before their execution.
        for name in imported:
            inherited.pop(name, None)
        assigned = _bound(node)
        if set(imported) & assigned:
            raise FunctionLoweringError('A NumPy import cannot share a local binding with a parameter or assignment')
        aliases = dict(inherited)
        visitor = _NumpyImports(aliases)
        body = []
        for statement in node.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                visitor.aliases.update(self.bindings(statement))
                continue
            # Reject references before a local import rather than resolving to
            # an inherited module binding that Python would regard as unbound.
            pending = set(imported) - set(visitor.aliases)
            if any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in pending for n in ast.walk(statement)):
                raise FunctionLoweringError('Local NumPy imports must precede their uses and nested helper definitions')
            body.append(visitor.visit(statement))
        if 'np' in assigned and any(isinstance(n, ast.Name) and n.id == 'np' and isinstance(n.ctx, ast.Load) for item in body for n in ast.walk(item)) and (imported or inherited):
            raise FunctionLoweringError('The canonical NumPy name np is shadowed; rename that local binding')
        node.body = body
        return node

    def visit_Lambda(self, node):
        if self.aliases and any(arg.arg == 'np' for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs):
            raise FunctionLoweringError('A lambda parameter shadows the canonical NumPy name np')
        aliases = dict(self.aliases)
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            aliases.pop(arg.arg, None)
        node.body = _NumpyImports(aliases).visit(node.body)
        return node

    def comprehension(self, node):
        visitor = _NumpyImports(self.aliases)
        for generator in node.generators:
            generator.iter = visitor.visit(generator.iter)
            for target in ast.walk(generator.target):
                if isinstance(target, ast.Name):
                    if target.id == 'np' and visitor.aliases:
                        raise FunctionLoweringError('A comprehension target shadows the canonical NumPy name np')
                    visitor.aliases.pop(target.id, None)
            generator.ifs = [visitor.visit(condition) for condition in generator.ifs]
        if isinstance(node, ast.DictComp):
            node.key = visitor.visit(node.key)
            node.value = visitor.visit(node.value)
        else:
            node.elt = visitor.visit(node.elt)
        return node

    visit_ListComp = comprehension
    visit_GeneratorExp = comprehension
    visit_SetComp = comprehension
    visit_DictComp = comprehension

    def visit_Import(self, node):
        check_import_statement(node)
        raise FunctionLoweringError('An import must occur directly in a module or function body')

    visit_ImportFrom = visit_Import


def _normalize_numpy_imports(tree):
    aliases = {}
    body = []
    for statement in tree.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            aliases.update(_NumpyImports.bindings(statement))
        else:
            body.append(statement)
    # Module bindings are resolved when functions execute, after module setup.
    for statement in body:
        if isinstance(statement, ast.FunctionDef):
            aliases.pop(statement.name, None)
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name):
                        aliases.pop(name.id, None)
    visitor = _NumpyImports(aliases)
    tree.body = [visitor.visit(statement) for statement in body]
    return tree


def lower_function_source(source: str, name: str) -> tuple[str, list[tuple[str,str]]]:
    """Return a single entry function plus generated ``(name, source)`` chips.

    Call before the circuit transformer. Parse each returned helper with
    ``ZinniaCompiler.chip_ast_parse`` and merge into the chips dictionary.
    This pass must run once for the entry source; generated chips are already
    normalized. Existing explicit chips are not rewritten.
    """
    tree=_normalize_numpy_imports(ast.parse(textwrap.dedent(source)))
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    if name not in functions:raise FunctionLoweringError(f'No function named {name!r} in source')
    lower=_Lower(tree)
    for stmt in tree.body:
        if isinstance(stmt,(ast.FunctionDef,ast.Import,ast.ImportFrom)):continue
        if isinstance(stmt,ast.Expr) and isinstance(stmt.value,ast.Constant) and isinstance(stmt.value.value,str):continue
        if isinstance(stmt,ast.Assign) and len(stmt.targets)==1 and isinstance(stmt.targets[0],ast.Name):
            try: value=ast.literal_eval(stmt.value)
            except (ValueError,TypeError):raise FunctionLoweringError('Source module globals must be immutable literal constants')
            if not _immutable_literal(value):
                raise FunctionLoweringError('Mutable module globals cannot be implicitly captured')
            lower.constants[stmt.targets[0].id]=stmt.value
            continue
        raise FunctionLoweringError('Unsupported executable module-level statement')
    entry=copy.deepcopy(functions[name])
    env={key:value for key,value in functions.items() if key!=name}
    lower.global_env=dict(env)
    for arg in entry.args.args:env.pop(arg.arg,None)
    lower.scopes.append(_bound(entry))
    entry.body=lower.block(entry.body,env)
    lower.scopes.pop()
    return ast.unparse(ast.fix_missing_locations(entry)),lower.helpers
