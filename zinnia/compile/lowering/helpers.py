"""Lift a circuit's local functions and lambdas into generated chips (compile stage).

Python lets a circuit define functions in its body, close over its
variables, pass functions to other functions and bind them to names. The
compiler only knows chips: functions it inlines at each call. This pass
turns every function value whose identity is known statically into a chip:

- A local ``def``, a lambda, or a module-level function of the same source
  becomes a generated chip ``_zinnia_fn_<name>`` whose first parameters are
  the variables it reads from enclosing scopes (its *captures*); a call
  ``f(a)`` becomes ``_zinnia_fn_f(c1, c2, a)``. Captures are read at the
  call, so a later rebinding of a captured variable is seen, as in Python.
- ``g = f`` and ``g = lambda ...`` bind ``g`` to the same chip; calls of
  ``g`` call it.
- Calls of NumPy/Python APIs that take a function argument are lowered by
  `callbacks.py`.
- ``it = (expr for ...)`` makes ``it`` a one-shot iterator: its single use
  is replaced by the generator; a second use is rejected.

Anything whose meaning needs a runtime function value or a closure cell is
rejected with `FunctionLoweringError`: rebinding a helper name, defining or
rebinding one in a branch, recursion, ``nonlocal``/``global``, ``del`` of a
helper binding, decorators, ``*args``/``**kwargs``, keyword-only and
positional-only parameters, non-literal defaults, a function value used other
than by calling it or passing it to a supported API, and module-level
statements other than literal constants.

Generated names start with ``_zinnia_fn_`` and never collide with a name of
the source.
"""
from __future__ import annotations

import ast
import builtins
import copy
from dataclasses import dataclass

from zinnia.compile.lowering import callbacks
from zinnia.compile.lowering.callbacks import VectorizedCallback
from zinnia.compile.lowering.imports import resolve_import_aliases
from zinnia.compile.lowering.scopes import bound_names, free_names
from zinnia.compile.lowering.syntax import (
    FunctionLoweringError, call, dedent_source, is_immutable_literal, is_numpy_call, load,
    names_bound_by_target, result_assignments_to_returns, store,
)

_BUILTIN_NAMES = frozenset(dir(builtins))


@dataclass
class LiftedHelper:
    """A function value lifted to the chip ``name``; calls pass ``captures`` first."""

    name: str
    captures: tuple[str, ...]


@dataclass
class OneShotIterator:
    """A name bound to a generator expression, which may be consumed once."""

    expression: ast.GeneratorExp
    consumed: bool = False


#: What a name can be bound to in a lowering environment, besides an
#: unlifted module-level ``ast.FunctionDef``.
_FUNCTION_VALUES = (LiftedHelper, VectorizedCallback)


class HelperLowering:
    """State of one lowering run over a source file.

    An *environment* (``env``) maps names to function values visible at a
    point of the program: ``LiftedHelper``, ``VectorizedCallback``,
    ``OneShotIterator``, or a module-level ``ast.FunctionDef`` not lifted yet.
    Environments are copied when entering a scope.
    """

    def __init__(self, module: ast.Module):
        self.used_names = {n.id for n in ast.walk(module) if isinstance(n, ast.Name)}
        self.used_names.update(n.arg for n in ast.walk(module) if isinstance(n, ast.arg))
        self.used_names.update(n.name for n in ast.walk(module) if isinstance(n, (ast.FunctionDef, ast.ClassDef)))
        #: Generated chips, ``(name, source)``, in generation order.
        self.helpers: list[tuple[str, str]] = []
        self.generated_names: set[str] = set()
        #: Module-level literal constants of the source: name -> expression.
        self.constants: dict[str, ast.expr] = {}
        #: Names of the functions being lifted (to detect recursion).
        self.lifting: set[str] = set()
        #: Names bound in each enclosing function being lifted, innermost last.
        self.scopes: list[set[str]] = []
        #: Module-level functions of the source, the environment they are lifted in.
        self.module_functions: dict[str, ast.FunctionDef] = {}

    def fresh(self, stem: str) -> str:
        """A new name ``_zinnia_fn_<stem>`` (with a numeric suffix if taken)."""
        stem = '_zinnia_fn_' + stem
        candidate = stem
        suffix = 0
        while candidate in self.used_names:
            suffix += 1
            candidate = f'{stem}_{suffix}'
        self.used_names.add(candidate)
        return candidate

    # --- lifting -------------------------------------------------------------

    def lift(self, function: ast.FunctionDef, env: dict) -> LiftedHelper:
        """Lift ``function`` (lowering its body first) into a generated chip."""
        self._check_liftable(function)
        self.lifting.add(function.name)
        lifted = copy.deepcopy(function)
        lifted.body = result_assignments_to_returns(lifted.body)
        local_env = dict(env)
        for argument in lifted.args.args:
            local_env.pop(argument.arg, None)
        self.scopes.append(bound_names(lifted))
        lifted.body = self.lower_block(lifted.body, local_env)
        self.scopes.pop()
        self.lifting.remove(function.name)

        captures = self._captures(lifted)
        for node in ast.walk(lifted):
            if isinstance(node, (ast.Nonlocal, ast.Global)):
                raise FunctionLoweringError('nonlocal/global writes are unsupported by implicit helper lowering')
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Del):
                raise FunctionLoweringError('Deleting helper bindings is unsupported')
        lifted.name = self.fresh(function.name)
        self.generated_names.add(lifted.name)
        lifted.args.args = [ast.arg(arg=name) for name in captures] + lifted.args.args
        self.helpers.append((lifted.name, ast.unparse(ast.fix_missing_locations(lifted))))
        return LiftedHelper(lifted.name, tuple(captures))

    def _check_liftable(self, function: ast.FunctionDef) -> None:
        if function.name in self.lifting:
            raise FunctionLoweringError('Recursive Python helpers require an explicit supported chip')
        if function.decorator_list:
            raise FunctionLoweringError('Decorated local functions cannot be lifted implicitly')
        arguments = function.args
        if arguments.vararg or arguments.kwarg or arguments.kwonlyargs or arguments.posonlyargs:
            raise FunctionLoweringError('Implicit helper lowering supports ordinary positional/keyword parameters')
        for default in arguments.defaults:
            try:
                value = ast.literal_eval(default)
            except (ValueError, TypeError):
                raise FunctionLoweringError('Helper defaults must be immutable literals; definition-time expressions are unsupported')
            if not is_immutable_literal(value):
                raise FunctionLoweringError('Mutable default arguments cannot be lifted implicitly')

    def _captures(self, function: ast.FunctionDef) -> list[str]:
        """The variables ``function`` reads from enclosing scopes, sorted.

        Builtins are not captured unless an enclosing function rebinds the
        name; ``np`` and generated helpers are global to every chip.
        """
        enclosing = set().union(*self.scopes) if self.scopes else set()
        globally_known = (_BUILTIN_NAMES - enclosing) | {'np'} | self.generated_names
        return sorted(free_names(function) - globally_known)

    def generated_helper(self, stem: str, parameters: list[str], body: list[ast.stmt], env: dict) -> LiftedHelper:
        """Lift a function the lowering wrote itself (see `callbacks.py`)."""
        function = ast.FunctionDef(
            name=self.fresh(stem),
            args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=p) for p in parameters], vararg=None,
                               kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
            body=body,
            decorator_list=[],
        )
        return self.lift(function, env)

    # --- function values -------------------------------------------------------

    def callback(self, expression: ast.expr, env: dict):
        """The function value ``expression`` denotes, lifting it if needed.

        Returns a ``LiftedHelper`` or ``VectorizedCallback``, or a copy of
        ``expression`` when it names a function the compiler resolves itself
        (a chip, builtin or ``np.<function>``).
        """
        if isinstance(expression, ast.Lambda):
            function = ast.FunctionDef(name=self.fresh('lambda'), args=copy.deepcopy(expression.args),
                                       body=[ast.Return(value=copy.deepcopy(expression.body))], decorator_list=[])
            return self.lift(function, env)
        if isinstance(expression, ast.Name) and expression.id in env:
            value = env[expression.id]
            if isinstance(value, ast.FunctionDef):
                # A module-level function: lifted on first use, in the module scope.
                enclosing_scopes = self.scopes
                self.scopes = []
                try:
                    value = self.lift(value, self.module_functions)
                finally:
                    self.scopes = enclosing_scopes
                env[expression.id] = value
            return value
        if isinstance(expression, ast.Name) and expression.id in self.lifting:
            raise FunctionLoweringError('Recursive Python helpers require an explicit supported chip')
        if isinstance(expression, (ast.Name, ast.Attribute)):
            return copy.deepcopy(expression)
        raise FunctionLoweringError('Callback must be a statically bound function or lambda')

    def invoke(self, function, args: list[ast.expr], keywords=()) -> ast.Call:
        """A call of a function value from `callback`."""
        if isinstance(function, LiftedHelper):
            return call(function.name, [load(c) for c in function.captures] + args, keywords)
        if isinstance(function, (ast.Name, ast.Attribute)):
            return ast.Call(func=copy.deepcopy(function), args=args, keywords=list(keywords))
        raise FunctionLoweringError('This callable cannot be invoked in a scalar context')

    # --- statements and expressions --------------------------------------------

    def lower_block(self, statements: list[ast.stmt], env: dict) -> list[ast.stmt]:
        """Lower a statement list; function bindings update ``env`` and emit nothing."""
        result = []
        for statement in statements:
            if isinstance(statement, ast.FunctionDef):
                if isinstance(env.get(statement.name), _FUNCTION_VALUES):
                    raise FunctionLoweringError('Rebinding an existing helper requires runtime function cells')
                env[statement.name] = self.lift(statement, env)
                continue
            if isinstance(statement, (ast.Nonlocal, ast.Global)):
                raise FunctionLoweringError('nonlocal/global writes are unsupported by implicit helper lowering')
            if _assigns_one_name(statement):
                target = statement.targets[0].id
                if isinstance(env.get(target), _FUNCTION_VALUES):
                    raise FunctionLoweringError('Rebinding an existing helper requires runtime function cells')
                value = statement.value
                if isinstance(value, ast.Lambda) or isinstance(value, ast.Name) and value.id in env:
                    env[target] = self.callback(value, env)
                    continue
                if is_numpy_call(value, 'vectorize'):
                    env[target] = callbacks.vectorize(self, value, env)
                    continue
                env.pop(target, None)
            lowered = ExpressionLowering(self, env).visit(copy.deepcopy(statement))
            if _assigns_one_name(lowered) and isinstance(lowered.value, ast.GeneratorExp):
                # `it = (... for x in xs)`: evaluate `xs` now, as Python does;
                # the generator itself runs where `it` is consumed.
                iterable = self.fresh('iterable')
                generator = lowered.value
                result.append(ast.Assign(targets=[store(iterable)], value=generator.generators[0].iter))
                generator.generators[0].iter = load(iterable)
                env[lowered.targets[0].id] = OneShotIterator(generator)
            else:
                result.append(lowered)
        return result

    def lower_expression(self, node: ast.expr, env: dict) -> ast.expr:
        return ExpressionLowering(self, env).visit(node)


def _assigns_one_name(statement: ast.stmt) -> bool:
    return (isinstance(statement, ast.Assign) and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name))


class ExpressionLowering(ast.NodeTransformer):
    """Lowers one statement or expression in the environment ``env``."""

    def __init__(self, lowering: HelperLowering, env: dict):
        self.lowering = lowering
        self.env = env

    def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.AST:
        raise FunctionLoweringError('nonlocal writes require closure cells and cannot be lifted')

    def visit_Global(self, node: ast.Global) -> ast.AST:
        raise FunctionLoweringError('global writes cannot be lifted')

    def visit_NamedExpr(self, node: ast.NamedExpr) -> ast.AST:
        raise FunctionLoweringError('Assignment expressions in callbacks are unsupported')

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        raise FunctionLoweringError('Conditional/loop-defined helpers are unsupported')

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        raise FunctionLoweringError('A lambda cannot escape its statically known call site')

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if not isinstance(node.ctx, ast.Load):
            return node
        value = self.env.get(node.id)
        if isinstance(value, OneShotIterator):
            if value.consumed:
                raise FunctionLoweringError('Reusing a one-shot iterator requires runtime iterator state')
            value.consumed = True
            return copy.deepcopy(value.expression)
        if node.id in self.env:
            raise FunctionLoweringError('Function values may only be invoked, aliased, or passed to supported callback APIs')
        constants = self.lowering.constants
        if node.id in constants and not any(node.id in scope for scope in self.lowering.scopes):
            return copy.deepcopy(constants[node.id])
        return node

    def _comprehension(self, node: ast.ListComp | ast.GeneratorExp | ast.SetComp | ast.DictComp) -> ast.AST:
        env = dict(self.env)
        inner = ExpressionLowering(self.lowering, env)
        local_names: set[str] = set()
        self.lowering.scopes.append(local_names)
        for generator in node.generators:
            generator.iter = inner.visit(generator.iter)
            for name in names_bound_by_target(generator.target):
                env.pop(name, None)
                local_names.add(name)
            generator.ifs = [inner.visit(condition) for condition in generator.ifs]
        if isinstance(node, ast.DictComp):
            node.key = inner.visit(node.key)
            node.value = inner.visit(node.value)
        else:
            node.elt = inner.visit(node.elt)
        self.lowering.scopes.pop()
        return node

    visit_ListComp = _comprehension
    visit_GeneratorExp = _comprehension
    visit_SetComp = _comprehension
    visit_DictComp = _comprehension

    def visit_For(self, node: ast.For) -> ast.AST:
        node.iter = self.visit(node.iter)
        env = dict(self.env)
        for name in names_bound_by_target(node.target):
            if isinstance(env.get(name), _FUNCTION_VALUES):
                raise FunctionLoweringError('Loop targets cannot rebind a static helper')
            env.pop(name, None)
        node.body = self.lowering.lower_block(node.body, env)
        node.orelse = self.lowering.lower_block(node.orelse, dict(self.env))
        return node

    def visit_If(self, node: ast.If) -> ast.AST:
        # Defining or rebinding a function name in a branch would need a
        # runtime function value, not a static chip identity.
        for statement in node.body + node.orelse:
            if isinstance(statement, ast.FunctionDef) or isinstance(statement, ast.Assign) and (
                    isinstance(statement.value, ast.Lambda)
                    or any(isinstance(t, ast.Name) and t.id in self.env for t in statement.targets)):
                raise FunctionLoweringError('Conditional helper bindings are unsupported')
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        lowering = self.lowering
        lowered = callbacks.lower_callback_api(lowering, node, self.env)
        if lowered is not None:
            return lowered
        if isinstance(node.func, ast.Lambda) or isinstance(node.func, ast.Name) and node.func.id in self.env:
            function = lowering.callback(node.func, self.env)
            args = [self.visit(argument) for argument in node.args]
            keywords = [ast.keyword(arg=k.arg, value=self.visit(k.value)) for k in node.keywords]
            if isinstance(function, VectorizedCallback):
                return callbacks.call_vectorized(lowering, function, args, keywords, self.env)
            return lowering.invoke(function, args, keywords)
        if isinstance(node.func, ast.Name) and node.func.id in lowering.lifting:
            raise FunctionLoweringError('Recursive Python helpers require an explicit supported chip')
        return self.generic_visit(node)


def _module_constants(module: ast.Module) -> dict[str, ast.expr]:
    """The module-level literal constants of a source file.

    A source given as a whole file may define constants next to its
    functions; any other executable module-level statement is rejected.
    """
    constants = {}
    for statement in module.body:
        if isinstance(statement, (ast.FunctionDef, ast.Import, ast.ImportFrom)):
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) \
                and isinstance(statement.value.value, str):
            continue  # a docstring
        if _assigns_one_name(statement):
            try:
                value = ast.literal_eval(statement.value)
            except (ValueError, TypeError):
                raise FunctionLoweringError('Source module globals must be immutable literal constants')
            if not is_immutable_literal(value):
                raise FunctionLoweringError('Mutable module globals cannot be implicitly captured')
            constants[statement.targets[0].id] = statement.value
            continue
        raise FunctionLoweringError('Unsupported executable module-level statement')
    return constants


def lower_circuit_source(source: str, name: str) -> tuple[str, list[tuple[str, str]]]:
    """Lower the circuit ``name`` defined in ``source``.

    Returns the lowered circuit source and the generated chips as
    ``(name, source)`` pairs, to be parsed with
    ``ZinniaCompiler.chip_ast_parse``. Generated chips are already lowered;
    explicit chips are not passed through this pass.
    """
    module = resolve_import_aliases(ast.parse(dedent_source(source)))
    functions = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}
    if name not in functions:
        raise FunctionLoweringError(f'No function named {name!r} in source')
    lowering = HelperLowering(module)
    lowering.constants = _module_constants(module)
    entry = copy.deepcopy(functions[name])
    env = {key: function for key, function in functions.items() if key != name}
    lowering.module_functions = dict(env)
    for argument in entry.args.args:
        env.pop(argument.arg, None)
    lowering.scopes.append(bound_names(entry))
    entry.body = lowering.lower_block(entry.body, env)
    lowering.scopes.pop()
    return ast.unparse(ast.fix_missing_locations(entry)), lowering.helpers
