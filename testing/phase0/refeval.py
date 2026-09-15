"""The reference evaluator of `revision/phase0/acceptance.md` §2.1 and §2.2.

The AST JSON carries no source positions, so the adequacy harness cannot map
phase 0's descriptors to source lines. Instead it *executes the exported AST*
with NumPy semantics and keys every value exactly the way phase 0 keys it
(`implementation.md` §5a). Comparing the two records is then a dictionary
lookup.

What this module implements itself: control flow, the assignment forms, chip
inlining, comprehensions, the key discipline. What it does not implement:
shapes. Every operator is handed to NumPy through the table in `ops.py`, so a
disagreement between phase 0 and this module is a disagreement between phase 0
and NumPy, which is what `definition.md` §7 asks the test to find.

# The key discipline (`implementation.md` §5a)

`KeyCursor` below is a transcription of `src/phase0/desc.rs`'s `KeyCursor`,
and every method that takes a key names the §5a rule it implements. The rules,
in one list:

1. circuit input `k` takes `at(k)`, the empty statement path;
2. every expression node takes the next key in preorder (`Evaluator.expr`);
3. every name an assignment, a loop target or an unpacking binds takes one
   further key (`Evaluator._bind_name`);
4. iterating or unpacking an *array* takes one key per element, allocated
   before the loop is entered (`Evaluator._iterable_elements`);
5. the `if` merge takes one key per name whose binding differs between the
   arms, in sorted order (`Evaluator._merge_keys`);
6. the delivery set takes `at(u32::MAX)`;
7. each arm of an `if` and of a conditional expression pushes its own
   component — the word `then` or `else` — onto the statement path
   (`KeyCursor.enter_arm`), so the two arms cannot write the same key;
8. a list method that changes its receiver takes one further key, after its
   arguments, for the receiver's new version (`Evaluator._method`);
9. a subscript assignment takes two keys: the write key of the indexed
   region, and one further key for the new version of the value written into
   (`Evaluator._setitem`).

# Where this evaluator deliberately differs from the walker

Phase 0 walks both arms of an `if` and every iteration of a bounded loop; a
real run takes one arm and the iterations the data calls for. The harness does
not treat that as a mismatch: `adequacy.py` requires a descriptor key the run
never reached to have an execution condition $\\psi$ that is false under
$\\theta(u)$, and classifies the rest.

Three smaller differences, all deliberate:

* a comprehension evaluates its element expression only when the `if` clauses
  keep the element, as Python does; phase 0 evaluates it whenever the clauses
  are not *statically* false, so the two disagree on the keys inside a
  filtered-out element (`_expand_generators`);
* the two Zinnia-specific readings of `np.where(m)` and of `int` / `float` /
  `bool` on an array, which `ops.py` documents;
* `evaluate` runs on deep copies of the inputs, because a circuit may write
  its inputs in place and the caller's values must come back untouched.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, NamedTuple

import numpy as np

from testing.phase0 import ops
from testing.phase0.ops import UnsupportedConstruct

#: The expression index of the delivery set (`walk/mod.rs::DELIVERY_EXPR_INDEX`).
DELIVERY_EXPR_INDEX = 0xFFFFFFFF

#: The `x.m(...)` targets that name a namespace rather than a value in scope
#: (`walk/calls.rs::named_attribute`). Every other target is a bound name, and
#: a method call on it can rebind it (§5a rule 8).
NUMPY_NAMESPACES = frozenset({"np", "numpy", "zinnia", "DynamicNDArray"})
MATH_NAMESPACES = frozenset({"math", "cmath"})
COPY_NAMESPACE = "copy"
NAMESPACES = NUMPY_NAMESPACES | MATH_NAMESPACES | {COPY_NAMESPACE}


# ---------------------------------------------------------------------------
# Value keys (`acceptance.md` §2.2, `implementation.md` §5a)
# ---------------------------------------------------------------------------


class KeyCursor:
    """Produces the value keys of a traversal.

    A transcription of `src/phase0/desc.rs::KeyCursor`: `enter_block` /
    `set_stmt` / `exit_block` around each block, `enter_loop` /
    `set_iteration` / `exit_loop` around each unrolled loop, `enter_call` /
    `exit_call` around each inlined chip body, and `take` once per keyed value.
    """

    def __init__(self) -> None:
        #: Statement indices and arm words, as `PathStep::as_string` spells
        #: them: a decimal number, or `"then"` / `"else"`.
        self.stmt_path: list[str] = []
        self.iters: list[int] = []
        self.calls: list[tuple[str, int]] = []
        self.next_expr = 0

    def enter_block(self) -> int:
        self.stmt_path.append("0")
        return self.next_expr

    def set_stmt(self, index: int) -> None:
        if self.stmt_path:
            self.stmt_path[-1] = str(index)
        self.next_expr = 0

    def exit_block(self, mark: int) -> None:
        self.stmt_path.pop()
        self.next_expr = mark

    def enter_arm(self, arm: str) -> int:
        """One arm of an `if` or of a conditional expression.

        The arm joins the statement path and the expression counter restarts,
        so statement `j` of the `then` arm and statement `j` of the `else` arm
        do not collide.
        """
        self.stmt_path.append(arm)
        mark, self.next_expr = self.next_expr, 0
        return mark

    def exit_arm(self, mark: int) -> None:
        self.stmt_path.pop()
        self.next_expr = mark

    def enter_loop(self) -> int:
        self.iters.append(0)
        return self.next_expr

    def set_iteration(self, index: int) -> None:
        if self.iters:
            self.iters[-1] = index
        self.next_expr = 0

    def exit_loop(self, mark: int) -> None:
        self.iters.pop()
        self.next_expr = mark

    def enter_call(self, callee: str, site: int) -> int:
        self.calls.append((callee, site))
        mark, self.next_expr = self.next_expr, 0
        return mark

    def exit_call(self, mark: int) -> None:
        self.calls.pop()
        self.next_expr = mark

    def take(self) -> str:
        """The key of the next expression node in preorder."""
        key = self.at(self.next_expr)
        self.next_expr += 1
        return key

    def at(self, expr_index: int) -> str:
        """The canonical string of `ValueKey::encode` (`src/phase0/desc.rs`):
        `"<stmt path>|<iterations>|<calls>|<expression index>"`, the statement
        path dot-joined over statement indices and arm words.
        """
        path = ".".join(self.stmt_path)
        iters = ".".join(str(n) for n in self.iters)
        calls = ".".join(f"{name}@{site}" for name, site in self.calls)
        return f"{path}|{iters}|{calls}|{expr_index}"


class ParsedKey(NamedTuple):
    """A value key taken apart, in the four fields `KeyCursor.at` joins.

    The encoding belongs to [`KeyCursor`], so reading one back belongs here
    too: every consumer that wants the iteration counters or the call site of
    a key goes through :meth:`parse` rather than splitting the string again.
    """

    stmt_path: tuple[str, ...]
    iters: tuple[int, ...]
    calls: tuple[str, ...]
    expr_index: int

    @staticmethod
    def parse(key: str) -> "ParsedKey":
        path, iters, calls, expr = key.split("|")
        return ParsedKey(
            stmt_path=tuple(path.split(".")) if path else (),
            iters=tuple(int(n) for n in iters.split(".")) if iters else (),
            calls=tuple(calls.split(".")) if calls else (),
            expr_index=int(expr),
        )

    def encode(self) -> str:
        return "{}|{}|{}|{}".format(
            ".".join(self.stmt_path),
            ".".join(str(n) for n in self.iters),
            ".".join(self.calls),
            self.expr_index,
        )

    def component(self, index: int) -> "ParsedKey":
        """The key of element `index` of the composite this key names
        (`ValueKey::component`, `src/phase0/desc.rs`): the position joins the
        iteration field. `E-Input-Composite` (§5.1) is the one rule that uses
        it, so that two scalar elements of one input do not share a key.
        """
        return self._replace(iters=self.iters + (index,))


# ---------------------------------------------------------------------------
# What the run records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Observed:
    """The actual value at one key, compressed to what adequacy compares.

    `kind` mirrors the `desc` kinds of the export: `array`, `scalar`,
    `composite`, `class`, `none`.
    """

    kind: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    number: Any = None
    elems: tuple["Observed", ...] | None = None

    @property
    def leading_extent(self) -> int | None:
        """The extent a `Count`, a `UniqueCount` or a `ValueSized` variable
        stands for: axis 0 of an array, the length of a list.

        A *tuple* of arrays is `np.nonzero`'s result, whose members share the
        one count (§5.5), so the count is their common leading extent.
        """
        if self.kind == "array":
            return self.shape[0] if self.shape else None
        if self.kind == "composite" and self.elems is not None:
            if self.dtype == "tuple" and self.elems and self.elems[0].kind == "array":
                return self.elems[0].leading_extent
            return len(self.elems)
        return None


def observe(value: Any) -> Observed:
    """Compress a runtime value into an `Observed`."""
    if value is None:
        return Observed(kind="none")
    if isinstance(value, np.ndarray):
        return Observed(
            kind="array", shape=tuple(int(n) for n in value.shape),
            dtype=str(value.dtype),
        )
    if isinstance(value, (list, tuple)):
        return Observed(
            kind="composite",
            dtype="tuple" if isinstance(value, tuple) else "list",
            elems=tuple(observe(item) for item in value),
        )
    if isinstance(value, ops.DtypeClass):
        return Observed(kind="class", dtype=value.name)
    if isinstance(value, str):
        return Observed(kind="scalar", dtype="String", number=value)
    if isinstance(value, (bool, np.bool_)):
        return Observed(kind="scalar", dtype="Bool", number=bool(value))
    if isinstance(value, (int, np.integer)):
        return Observed(kind="scalar", dtype="Int", number=int(value))
    if isinstance(value, (float, np.floating)):
        return Observed(kind="scalar", dtype="Float", number=float(value))
    if isinstance(value, (complex, np.complexfloating)):
        return Observed(kind="scalar", dtype="Complex", number=complex(value))
    return Observed(kind="scalar", dtype=type(value).__name__, number=None)


@dataclass
class Trace:
    """One reference run: what it computed, and how it ended."""

    values: dict[str, Observed] = field(default_factory=dict)
    result: Observed | None = None
    #: `None` when the run succeeded; otherwise the Python/NumPy error.
    error: str | None = None
    #: Set when that error is a `TypeError`, which NumPy raises for an
    #: operand of the wrong *type* — a string where a number belongs, a
    #: bitwise operator on floats. Every shape condition raises `ValueError`
    #: or `IndexError` instead, so the flag separates an ill-typed program
    #: from a failing one (`adequacy.py`).
    ill_typed: bool = False
    #: The key the cursor stood at when the run failed.
    error_key: str | None = None
    #: Set when the evaluator itself could not run the program
    #: (`acceptance.md` §2.5: a harness gap, not a phase 0 failure).
    gap: str | None = None
    #: The declared input names, in order, so that `theta(u)` can find the
    #: value an `input_length` or `shape_symbol` variable names (§2.3).
    input_names: list[str] = field(default_factory=list)
    #: The actual value of every `SymbolicNDArray` shape symbol (§5.1).
    shape_symbols: dict[str, int] = field(default_factory=dict)
    @property
    def ok(self) -> bool:
        return self.error is None and self.gap is None


# ---------------------------------------------------------------------------
# Control flow signals
# ---------------------------------------------------------------------------


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


class _Return(Exception):
    def __init__(self, value: Any) -> None:
        super().__init__("return")
        self.value = value


class EvaluatorBug(Exception):
    """The reference evaluator is missing something, not the program.

    A `visit_` handler that reaches a name it never bound, an `ops.py`
    signature that does not match NumPy's, a recursion the evaluator did not
    bound: none of these says anything about the circuit, and reporting them
    as a failing run would make `adequacy.check` take its G3 branch and pass
    as soon as any check is false (`acceptance.md` §2.5). They are harness
    gaps, and `_run_guarded` reports them as such.
    """


class _LoopCapExceeded(Exception):
    """A `while` that did not exit within `loop_limit`.

    `S-While-Bounded` (§6.2) makes that a failing run: phase 0 records
    `(φ ∧ b_1 ∧ … ∧ b_C) ⇒ false`, so G3 applies.
    """


# ---------------------------------------------------------------------------
# The evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    """Executes the exported AST with NumPy semantics, keying every value."""

    def __init__(
        self,
        ast: dict[str, Any],
        config: dict[str, Any],
        chips: dict[str, Any],
        externals: dict[str, Any],
        external_impls: dict[str, Callable[..., Any]] | None = None,
    ) -> None:
        self.ast = ast
        self.loop_limit = int(config.get("loop_limit", 1000))
        self.recursion_limit = int(config.get("recursion_limit", 16))
        self.chips = chips
        self.externals = externals
        self.external_impls = external_impls or {}
        self.keys = KeyCursor()
        self.frames: list[dict[str, Any]] = []
        self.trace = Trace()
        self._current_key = "|||0"

    # ── entry point ───────────────────────────────────────────────────

    def run(self, inputs: list[Any]) -> Trace:
        """Execute the circuit on `inputs`, one value per declared input."""
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            self._run_guarded(inputs)
        return self.trace

    def _run_guarded(self, inputs: list[Any]) -> None:
        try:
            self._bind_inputs(inputs)
            result = self._run_body()
        except UnsupportedConstruct as gap:
            self.trace.gap = str(gap)
        # Not a property of the program: the evaluator's own limits. A
        # `KeyError` is a lookup this module got wrong, an `AttributeError`
        # an `ops.py` entry that does not match NumPy, a `RecursionError` a
        # nesting it did not bound.
        except (EvaluatorBug, KeyError, AttributeError, RecursionError) as bug:
            self.trace.gap = f"the reference evaluator raised {type(bug).__name__}: {bug}"
        except _LoopCapExceeded as exc:
            self.trace.error = str(exc)
            self.trace.error_key = self._current_key
        except (ArithmeticError, ValueError, IndexError, TypeError,
                AssertionError, np.exceptions.AxisError) as exc:
            self.trace.error = f"{type(exc).__name__}: {exc}"
            self.trace.error_key = self._current_key
            self.trace.ill_typed = isinstance(exc, TypeError)
        else:
            self.trace.result = observe(result)

    def _run_body(self) -> Any:
        """The circuit body; the returned value takes the delivery key
        (§5a rule 6).
        """
        try:
            self.block(self.ast.get("block", []))
            returned = None
        except _Return as exc:
            returned = exc.value
        if returned is not None:
            self._record(self.keys.at(DELIVERY_EXPR_INDEX), returned)
        return returned

    def _bind_inputs(self, inputs: list[Any]) -> None:
        """§5a rule 1: input `k` takes the key with an empty statement path
        and expression index `k`.
        """
        declared = self.ast.get("inputs", [])
        if len(inputs) != len(declared):
            raise UnsupportedConstruct(
                f"{len(inputs)} values for {len(declared)} declared inputs"
            )
        frame = dict(ops.GLOBAL_DTYPES)
        self.frames.append(frame)
        self.trace.input_names = [d["name"] for d in declared]
        for position, (declaration, value) in enumerate(zip(declared, inputs)):
            dt = declaration["annotation"]["dt"]
            value = _coerce_input(dt, value)
            key = self.keys.at(position)
            self._record(key, value)
            self._record_components(key, dt, value)
            frame[declaration["name"]] = value
            self._note_shape_symbols(dt, value)

    def _record_components(self, key: str, dt: Any, value: Any) -> None:
        """Record each element of a `List` / `Tuple` input under the key
        `E-Input-Composite` gives it (`ParsedKey.component`), so that a scalar
        element's variable can be read off the run.
        """
        elements = _composite_elements(dt)
        if elements is None or not isinstance(value, (list, tuple)):
            return
        for index, (element_dt, element) in enumerate(zip(elements, value)):
            element_key = ParsedKey.parse(key).component(index).encode()
            self.trace.values[element_key] = observe(element)
            self._record_components(element_key, element_dt, element)

    def _note_shape_symbols(self, dt: Any, value: Any) -> None:
        """Record the actual value of every `SymbolicNDArray` shape symbol, so
        that `theta(u)` can read it back (`acceptance.md` §2.3).
        """
        if not isinstance(dt, dict) or "SymbolicNDArray" not in dt:
            return
        shape = getattr(value, "shape", ())
        for axis, dim in enumerate(dt["SymbolicNDArray"]["shape"]):
            symbol = dim["symbol"] if isinstance(dim, dict) else dim
            if isinstance(symbol, str) and axis < len(shape):
                self.trace.shape_symbols.setdefault(symbol, int(shape[axis]))

    # ── recording ─────────────────────────────────────────────────────

    def _record(self, key: str, value: Any) -> Any:
        self.trace.values[key] = observe(value)
        self._current_key = key
        return value

    @property
    def frame(self) -> dict[str, Any]:
        return self.frames[-1]

    # ── blocks and statements ─────────────────────────────────────────

    def block(self, statements: list[dict[str, Any]]) -> None:
        """Walk a block, pushing one slot on the statement path (§5a)."""
        mark = self.keys.enter_block()
        try:
            for index, statement in enumerate(statements):
                self.keys.set_stmt(index)
                self.statement(statement)
        finally:
            self.keys.exit_block(mark)

    def statement(self, node: dict[str, Any]) -> None:
        handler = getattr(self, f"visit_{node['__class__']}", None)
        if handler is None:
            raise UnsupportedConstruct(f"{node['__class__']} in statement position")
        handler(node)

    def visit_ASTPassStatement(self, node: dict[str, Any]) -> None:
        del node

    def visit_ASTExprStatement(self, node: dict[str, Any]) -> None:
        self.expr(node["expr"])

    def visit_ASTAssignStatement(self, node: dict[str, Any]) -> None:
        """`S-Assign` (§5.10): the value once, then every target left to right."""
        value = self.expr(node["value"])
        for target in node["targets"]:
            self.assign_target(target, value)

    def visit_ASTReturnStatement(self, node: dict[str, Any]) -> None:
        expression = node.get("expr")
        raise _Return(self.expr(expression) if expression is not None else None)

    def visit_ASTAssertStatement(self, node: dict[str, Any]) -> None:
        """`S-Assert`: the condition takes the statement's first key (§5a)."""
        key = self.keys.take()
        if not _truth(self.expr_at(node["expr"], key)):
            raise AssertionError("assertion failed")

    def visit_ASTBreakStatement(self, node: dict[str, Any]) -> None:
        del node
        raise _Break()

    def visit_ASTContinueStatement(self, node: dict[str, Any]) -> None:
        del node
        raise _Continue()

    def visit_ASTCondStatement(self, node: dict[str, Any]) -> None:
        """`S-If` (§6.1). The run takes one arm; the merge keys of §5a rule 5
        are allocated afterwards for every name both arms leave bound, in
        sorted order, so that the fresh variables of the merge can be resolved.
        """
        key = self.keys.take()
        guard = self.expr_at(node["cond"], key)
        before = set(self.frame)
        taken = _truth(guard)
        block = node["t_block"] if taken else node["f_block"]
        mark = self.keys.enter_arm("then" if taken else "else")
        reached_the_merge = False
        try:
            self.block(block)
            reached_the_merge = True
        finally:
            self.keys.exit_arm(mark)
            self._merge_keys(before, node, reached_the_merge)

    def _merge_keys(
        self, before: set[str], node: dict[str, Any], reached: bool
    ) -> None:
        """One key per name whose binding differs between the arms, in sorted
        order (§5a rule 5).

        Phase 0 decides "differs" by the binding's key: a name neither arm
        rebound is the same value on both sides, so the merge is the identity
        and it takes no key. The prediction here is syntactic, because the run
        takes one arm and never observes what the other would have rebound: a
        name bound before the `if` is merged when either arm can rebind it,
        and a name born inside the branch is merged when both arms bind it.
        The two answers differ only where the walker does not walk a statement
        the arm contains — an `if` whose condition folds, a loop with no
        iterations, a statement after a `return` — and adequacy reports that
        as a key mismatch.

        The keys are taken whatever the arm did, because the walker takes
        them; the *values* are recorded only when the arm ran to its end.
        An arm that left through `break`, `continue` or `return` never
        reaches the merge, and phase 0 gives the merged binding a $\\psi$
        that says so.
        """
        rebound = rebound_names(node["t_block"]) | rebound_names(node["f_block"])
        born = assigned_names(node["t_block"]) & assigned_names(node["f_block"])
        for name in sorted((before & rebound) | (born - before)):
            key = self.keys.take()
            if reached:
                self._record(key, self.frame.get(name))

    def visit_ASTForInStatement(self, node: dict[str, Any]) -> None:
        """`S-For-*` (§6.2): the iterable, then one iteration per element."""
        iterable = self.expr(node["iter_expr"])
        elements = self._iterable_elements(iterable, "a `for` loop")
        completed = True
        mark = self.keys.enter_loop()
        try:
            for index, element in enumerate(elements):
                self.keys.set_iteration(index)
                self.bind_target(node["target"], element)
                try:
                    self.block(node["block"])
                except _Continue:
                    continue
                except _Break:
                    completed = False
                    break
        finally:
            self.keys.exit_loop(mark)
        if completed and node.get("orelse"):
            self.block(node["orelse"])

    def visit_ASTWhileStatement(self, node: dict[str, Any]) -> None:
        """`S-While-Bounded` (§6.2): the guard is expression 0 of each
        iteration; `loop_limit` is the cap phase 0 records a check for.
        """
        completed = True
        mark = self.keys.enter_loop()
        try:
            for iteration in range(self.loop_limit + 1):
                if iteration == self.loop_limit:
                    raise _LoopCapExceeded(
                        f"`while` did not exit within loop_limit={self.loop_limit}"
                    )
                self.keys.set_iteration(iteration)
                key = self.keys.take()
                if not _truth(self.expr_at(node["test_expr"], key)):
                    break
                try:
                    self.block(node["block"])
                except _Continue:
                    continue
                except _Break:
                    completed = False
                    break
        finally:
            self.keys.exit_loop(mark)
        if completed and node.get("orelse"):
            self.block(node["orelse"])

    # ── assignment targets (`S-Unpack`, `S-SetItem`, §5.11) ───────────

    def assign_target(self, target: dict[str, Any], value: Any) -> None:
        kind = target["__class__"]
        if kind == "ASTNameAssignTarget":
            self._bind_name(target["name"], value)
        elif kind in ("ASTTupleAssignTarget", "ASTListAssignTarget"):
            self._unpack(target["targets"], value)
        elif kind == "ASTSubscriptAssignTarget":
            self._setitem(target, value)
        else:
            raise UnsupportedConstruct(f"{kind} as an assignment target")

    def bind_target(self, target: dict[str, Any], value: Any) -> None:
        """A loop or generator target, which binds like an assignment."""
        self.assign_target(target, value)

    def _bind_name(self, name: str, value: Any) -> None:
        """§5a rule 3: one key per bound name, which is its entry in Δ."""
        self._record(self.keys.take(), value)
        self.frame[name] = value

    def _unpack(self, targets: list[dict[str, Any]], value: Any) -> None:
        """`S-Unpack` (§5.11), including a starred target."""
        elements = self._iterable_elements(value, "an unpacking")
        star = next(
            (i for i, t in enumerate(targets) if t.get("star")), None
        )
        if star is None:
            if len(targets) != len(elements):
                raise ValueError(
                    f"unpacking {len(elements)} values into {len(targets)} targets"
                )
            for target, element in zip(targets, elements):
                self.assign_target(target, element)
            return
        after = len(targets) - star - 1
        if len(elements) < star + after:
            raise ValueError("too few values to unpack with a starred target")
        middle = len(elements) - star - after
        for index in range(star):
            self.assign_target(targets[index], elements[index])
        self.assign_target(targets[star], list(elements[star:star + middle]))
        for index in range(after):
            self.assign_target(
                targets[star + 1 + index], elements[star + middle + index]
            )

    def _setitem(self, target: dict[str, Any], value: Any) -> None:
        """`S-SetItem` (§5.11). §5a rule 9 fixes the order: the statement's
        value expression (already evaluated), the write key, the target
        expression, the slice entries, then one further key for the new
        version of the value written into.
        """
        key = self.keys.take()
        base = self.expr(target["target"])
        index = self._subscript_index(self.slice_indices(target["slicing"]))
        if not isinstance(base, (np.ndarray, list)):
            raise UnsupportedConstruct(
                f"subscript assignment to a {type(base).__name__}"
            )
        # The write's key is the one phase 0 gives the *indexed region*: it is
        # what `subscript_array` describes there, and what `E-Take`'s `inrange`
        # witness is anchored to (§5.4). Recording the written value instead
        # would make that witness read the right-hand side.
        self._record(key, base[index])
        base[index] = value
        # The further key is the value after the write, whether or not the
        # target is a name; only a name is rebound to it.
        self._record(self.keys.take(), base)

    # ── expressions ───────────────────────────────────────────────────

    def expr(self, node: dict[str, Any]) -> Any:
        """§5a rule 2: every expression node takes the next key in preorder,
        *before* its children are visited.
        """
        return self.expr_at(node, self.keys.take())

    def expr_at(self, node: dict[str, Any], key: str) -> Any:
        """Evaluate an expression whose key the caller already took."""
        handler = getattr(self, f"visit_{node['__class__']}", None)
        if handler is None:
            raise UnsupportedConstruct(f"{node['__class__']} in expression position")
        self._current_key = key
        return self._record(key, handler(node, key))

    def visit_ASTConstantInteger(self, node: dict[str, Any], key: str) -> int:
        del key
        return int(node["value"])

    def visit_ASTConstantFloat(self, node: dict[str, Any], key: str) -> float:
        del key
        return float(node["value"])

    def visit_ASTConstantBoolean(self, node: dict[str, Any], key: str) -> bool:
        del key
        return bool(node["value"])

    def visit_ASTConstantComplex(self, node: dict[str, Any], key: str) -> complex:
        del key
        return complex(node["real"], node["imag"])

    def visit_ASTConstantString(self, node: dict[str, Any], key: str) -> str:
        del key
        return str(node["value"])

    def visit_ASTConstantNone(self, node: dict[str, Any], key: str) -> None:
        del node, key
        return None

    def visit_ASTLoad(self, node: dict[str, Any], key: str) -> Any:
        del key
        name = node["name"]
        if name not in self.frame:
            raise EvaluatorBug(f"unbound name `{name}`")
        return self.frame[name]

    def visit_ASTBinaryOperator(self, node: dict[str, Any], key: str) -> Any:
        del key
        lhs = self.expr(node["lhs"])
        rhs = self.expr(node["rhs"])
        return ops.binary(node["operator"], lhs, rhs)

    def visit_ASTUnaryOperator(self, node: dict[str, Any], key: str) -> Any:
        del key
        return ops.unary(node["operator"], self.expr(node["operand"]))

    def visit_ASTSquareBrackets(self, node: dict[str, Any], key: str) -> list:
        del key
        return list(self._composite_elements(node["values"]))

    def visit_ASTParenthesis(self, node: dict[str, Any], key: str) -> tuple:
        del key
        return tuple(self._composite_elements(node["values"]))

    def visit_ASTJoinedStr(self, node: dict[str, Any], key: str) -> str:
        del key
        return "".join(str(self.expr(value)) for value in node["values"])

    def visit_ASTFormattedValue(self, node: dict[str, Any], key: str) -> str:
        del key
        return str(self.expr(node["value"]))

    def visit_ASTSubscriptExp(self, node: dict[str, Any], key: str) -> Any:
        del key
        value = self.expr(node["val"])
        index = self._subscript_index(self.slice_indices(node["slicing"]))
        if isinstance(value, (list, tuple)) and not isinstance(index, (int, slice)):
            value = np.asarray(value)
        return value[index]

    def visit_ASTCondExp(self, node: dict[str, Any], key: str) -> Any:
        """`E-CondExp` (§5.11): each arm takes its keys inside its own arm
        component of the statement path, exactly as an `if` statement's do
        (§5a).
        """
        del key
        taken = _truth(self.expr(node["cond"]))
        mark = self.keys.enter_arm("then" if taken else "else")
        try:
            return self.expr(node["t_expr"] if taken else node["f_expr"])
        finally:
            self.keys.exit_arm(mark)

    def visit_ASTGeneratorExp(self, node: dict[str, Any], key: str) -> Any:
        """`E-Comp` (§5.11), expanded generator by generator."""
        del key
        collected: list[Any] = []
        self._expand_generators(node["generators"], 0, node["elt"], collected)
        return tuple(collected) if node.get("kind") == "tuple" else collected

    def _expand_generators(
        self,
        generators: list[dict[str, Any]],
        index: int,
        element_expr: dict[str, Any],
        out: list[Any],
    ) -> None:
        if index == len(generators):
            out.append(self.expr(element_expr))
            return
        generator = generators[index]
        iterable = self.expr(generator["iter"])
        elements = self._iterable_elements(iterable, "a comprehension")
        mark = self.keys.enter_loop()
        try:
            for iteration, element in enumerate(elements):
                self.keys.set_iteration(iteration)
                self.bind_target(generator["target"], element)
                if self._generator_filter(generator.get("ifs", [])):
                    self._expand_generators(generators, index + 1, element_expr, out)
        finally:
            self.keys.exit_loop(mark)

    def _generator_filter(self, conditions: list[dict[str, Any]]) -> bool:
        """Every `if` clause is evaluated, as `exprs.rs::generator_filter`
        does, so the keys line up whatever the clauses answer.
        """
        kept = True
        for condition in conditions:
            key = self.keys.take()
            if not _truth(self.expr_at(condition, key)):
                kept = False
        return kept

    def _composite_elements(self, values: list[dict[str, Any]]) -> Iterable[Any]:
        """`E-Composite` (§5.11), with starred elements flattened into place."""
        for value in values:
            if value["__class__"] == "ASTStarredExpr":
                inner = self.expr(value["value"])
                yield from self._spread(inner)
            else:
                yield self.expr(value)

    def _spread(self, inner: Any) -> list[Any]:
        if isinstance(inner, (list, tuple)):
            return list(inner)
        if isinstance(inner, np.ndarray):
            return self._iterable_elements(inner, "a starred expression")
        return [inner]

    # ── subscripts (§5.4, §5.5) ───────────────────────────────────────

    def slice_indices(self, slicing: dict[str, Any]) -> list[Any]:
        """The per-axis entries of a subscript, left to right.

        An omitted slice endpoint arrives as an `ASTConstantNone` node and
        does take a key, as it does in the baseline (§5a).
        """
        entries: list[Any] = []
        for entry in slicing["data"]:
            if isinstance(entry, dict):
                marker = entry.get("__class__")
                if marker == "ASTSliceEllipsis":
                    entries.append(Ellipsis)
                    continue
                if marker == "ASTSliceNewAxis":
                    entries.append(None)
                    continue
                entries.append(self.expr(entry))
                continue
            triple = [self._slice_endpoint(entry, position) for position in range(3)]
            entries.append(slice(*triple))
        return entries

    def _slice_endpoint(self, triple: list[Any], position: int) -> Any:
        if position >= len(triple) or triple[position] is None:
            return None
        value = self.expr(triple[position])
        return None if value is None else _as_index(value)

    @staticmethod
    def _subscript_index(entries: list[Any]) -> Any:
        """The Python index a list of subscript entries denotes."""
        prepared = [
            entry if isinstance(entry, (slice, type(Ellipsis), type(None)))
            else _as_operand(entry)
            for entry in entries
        ]
        return prepared[0] if len(prepared) == 1 else tuple(prepared)

    # ── iteration (§6.2, `S-Unpack`) ──────────────────────────────────

    def _iterable_elements(self, value: Any, context: str) -> list[Any]:
        """The elements a value yields when iterated or unpacked.

        §5a rule 4: an *array* takes one key per element, allocated before the
        loop is entered; a composite takes none, because its elements already
        have their own keys.
        """
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, np.ndarray):
            return [
                self._record(self.keys.take(), element) for element in value
            ]
        raise UnsupportedConstruct(f"iteration over a {type(value).__name__}"
                                   f" in {context}")

    # ── calls (§5.10, §5.11) ──────────────────────────────────────────

    def visit_ASTNamedAttribute(self, node: dict[str, Any], key: str) -> Any:
        """`f(...)`, `np.f(...)` and `x.m(...)` with `x` a name in scope."""
        target = node.get("target")
        if target == "np" and node["member"] == "linalg":
            raise UnsupportedConstruct("`np.linalg` used as a value")
        args, kwargs = self.call_args(node)
        member = node["member"]
        if target is None:
            return self._bare_call(member, args, kwargs, key)
        if target in NUMPY_NAMESPACES:
            return self._numpy_call(member, args, kwargs)
        if target in MATH_NAMESPACES:
            return self._namespace_call(ops.MATH, member, args, kwargs, "math")
        if target == COPY_NAMESPACE:
            return args[0]
        if target not in self.frame:
            raise EvaluatorBug(f"unbound name `{target}` in `{target}.{member}`")
        return self._method(target, self.frame[target], member, args, kwargs)

    def visit_ASTExprAttribute(self, node: dict[str, Any], key: str) -> Any:
        """`<expr>.m(...)`: the target expression first, then the arguments."""
        del key
        inner = node["target"]
        if (
            inner.get("__class__") == "ASTNamedAttribute"
            and inner.get("target") == "np"
            and inner.get("member") == "linalg"
        ):
            args, kwargs = self.call_args(node)
            return self._namespace_call(
                ops.LINALG, node["member"], args, kwargs, "np.linalg"
            )
        receiver = self.expr(inner)
        args, kwargs = self.call_args(node)
        return ops.method_call(receiver, node["member"], args, _dtypes(kwargs))

    def call_args(self, node: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
        """Positional arguments left to right with starred ones flattened,
        then the keyword arguments **sorted by name** (§5a).
        """
        positional: list[Any] = []
        for argument in node.get("args", []):
            if argument["__class__"] == "ASTStarredExpr":
                positional.extend(self._spread(self.expr(argument["value"])))
            else:
                positional.append(self.expr(argument))
        keywords = node.get("kwargs", {}) or {}
        return positional, {
            name: self.expr(keywords[name]) for name in sorted(keywords)
        }

    def _bare_call(
        self, member: str, args: list[Any], kwargs: dict[str, Any], key: str
    ) -> Any:
        """A bare name: a chip first, then a builtin, then an external
        (`calls.rs::bare_call`).
        """
        if member in self.chips:
            return self._inline_chip(member, args, kwargs, key)
        if member in ops.BUILTINS:
            return ops.builtin_call(member, self._iterated(member, args), kwargs)
        if member in self.externals:
            return self._call_external(member, args, kwargs)
        raise UnsupportedConstruct(f"call to `{member}`")

    def _iterated(self, member: str, args: list[Any]) -> list[Any]:
        """A builtin that iterates its first argument reads an array element
        by element, and each element takes a key (§5a rule 4).

        Replacing the array by that list of elements is the identity for
        `enumerate`, `list` and `tuple`, and is what takes the keys.
        """
        if member not in ops.ITERATING_BUILTINS or not args:
            return args
        if not isinstance(args[0], np.ndarray):
            return args
        return [self._iterable_elements(args[0], f"`{member}`"), *args[1:]]

    def _numpy_call(
        self, member: str, args: list[Any], kwargs: dict[str, Any]
    ) -> Any:
        """`np.<member>(...)`, including the `np.<dtype>` names (§5.1)."""
        if member in ops.DTYPE_OBJECTS:
            dtype = ops.DTYPE_OBJECTS[member]
            return ops.DtypeClass(member, dtype) if not args \
                else ops.cast_to(args[0], dtype)
        return self._namespace_call(ops.NUMPY, member, args, kwargs, "np")

    @staticmethod
    def _namespace_call(
        table: dict[str, Callable[..., Any]],
        member: str,
        args: list[Any],
        kwargs: dict[str, Any],
        namespace: str,
    ) -> Any:
        handler = table.get(member)
        if handler is None:
            raise UnsupportedConstruct(f"`{namespace}.{member}`")
        return handler(*ops.collapse_varargs(member, args), **_dtypes(kwargs))

    def _method(
        self,
        name: str,
        receiver: Any,
        member: str,
        args: list[Any],
        kwargs: dict[str, Any],
    ) -> Any:
        """`x.m(...)`. A list method that changes its receiver rebinds the
        name under one further key, taken after the arguments (§5a rule 8).
        """
        if member in ops.LIST_METHODS and isinstance(receiver, list):
            if member == "extend":
                # `extend` iterates its argument, so an array argument takes
                # one key per element (`walk/values.rs::mutate_list`).
                args = self._iterated("list", args)
            result = _list_method(self.frame, name, receiver, member, args)
            if member in ops.REBINDING_LIST_METHODS:
                self._record(self.keys.take(), self.frame[name])
            return result
        return ops.method_call(receiver, member, args, _dtypes(kwargs))

    def _inline_chip(
        self, name: str, args: list[Any], kwargs: dict[str, Any], key: str
    ) -> Any:
        """`S-Call` (§5.10): a fresh environment, arguments bound positionally
        then by keyword then by default, and the body walked as a block.
        """
        if len(self.frames) > self.recursion_limit:
            raise RecursionError(f"chip recursion deeper than {self.recursion_limit}")
        chip = self.chips[name]["chip_ast"]
        # The call node's own expression index, which tells two calls of
        # the same chip in one statement apart (`acceptance.md` §2.2).
        site = ParsedKey.parse(key).expr_index
        self.frames.append(dict(ops.GLOBAL_DTYPES))
        mark = self.keys.enter_call(name, site)
        try:
            self._bind_chip_inputs(chip, args, kwargs)
            self.block(chip.get("block", []))
            return None
        except _Return as exc:
            return exc.value
        finally:
            self.keys.exit_call(mark)
            self.frames.pop()

    def _bind_chip_inputs(
        self, chip: dict[str, Any], args: list[Any], kwargs: dict[str, Any]
    ) -> None:
        for index, declaration in enumerate(chip.get("inputs", [])):
            name = declaration["name"]
            if index < len(args):
                self.frame[name] = args[index]
            elif name in kwargs:
                self.frame[name] = kwargs[name]
            elif declaration.get("default") is not None:
                self.frame[name] = self.expr(declaration["default"])

    def _call_external(
        self, member: str, args: list[Any], kwargs: dict[str, Any]
    ) -> Any:
        """`E-External` (§5.11). The harness must supply the real callable;
        without it the program is a gap, not a failure (`acceptance.md` §2.5).
        """
        implementation = self.external_impls.get(member)
        if implementation is None:
            raise UnsupportedConstruct(f"external `{member}` has no implementation")
        return implementation(*args, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


#: Annotations that denote an array, whatever Python object the caller passes.
ARRAY_ANNOTATIONS = frozenset({"NDArray", "SymbolicNDArray", "DynamicNDArray"})


def _composite_elements(dt: Any) -> list[Any] | None:
    """The element annotations of a `List` / `Tuple`, through a
    `PoseidonHashed` wrapper; `None` for anything else.
    """
    if isinstance(dt, dict) and "PoseidonHashed" in dt:
        return _composite_elements(dt["PoseidonHashed"]["dtype"])
    if isinstance(dt, dict) and set(dt) & {"List", "Tuple"}:
        return dt[next(iter(dt))]["elements"]
    return None


def _coerce_input(dt: Any, value: Any) -> Any:
    """Read an input the way the baseline reads it.

    A test may pass a nested list where the annotation says `NDArray`; the
    baseline builds the array it denotes, so the reference run must too, or the
    descriptor would be compared against a composite.
    """
    if isinstance(dt, dict) and "PoseidonHashed" in dt:
        return _coerce_input(dt["PoseidonHashed"]["dtype"], value)
    if isinstance(dt, dict) and set(dt) & ARRAY_ANNOTATIONS:
        return np.asarray(value) if isinstance(value, (list, tuple)) else value
    elements = _composite_elements(dt)
    if elements is not None:
        if isinstance(value, (list, tuple)) and len(value) == len(elements):
            coerced = [_coerce_input(e, v) for e, v in zip(elements, value)]
            return tuple(coerced) if "Tuple" in dt else coerced
    return value


def _truth(value: Any) -> bool:
    """`to_scalar_bool`: the Boolean a value contributes as a condition.

    An array condition is the conjunction of its elements; Python would raise
    for a multi-element array, and phase 0 makes it one fresh Boolean variable,
    so the reference takes `np.all`.
    """
    if isinstance(value, np.ndarray):
        return bool(np.all(value))
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    return bool(value)


def _as_index(value: Any) -> Any:
    """A slice endpoint, which must be a Python integer."""
    return int(value) if isinstance(value, (bool, int, float, np.generic)) else value


def _as_operand(value: Any) -> Any:
    """A subscript entry that is not a slice: an integer, a mask or an index
    array.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return value


def _dtypes(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Turn a `dtype=` argument written as a Zinnia dtype object into the
    NumPy dtype NumPy expects.
    """
    return {
        name: (value.numpy_dtype if isinstance(value, ops.DtypeClass) else value)
        for name, value in kwargs.items()
    }


def _list_method(
    frame: dict[str, Any], name: str, receiver: list, member: str, args: list[Any]
) -> Any:
    """`S-ListMethod` (§5.11) on a bound name, with Python's own semantics."""
    if member == "copy":
        return list(receiver)
    if member in ("count", "index"):
        return getattr(receiver, member)(args[0])
    result = getattr(receiver, member)(*args)
    frame[name] = receiver
    return result


def rebound_names(block: list[dict[str, Any]]) -> set[str]:
    """Every name a block can bind to a new value.

    The walker gives every version of a name a key of its own and the `if`
    merge compares those keys (§5a rule 5), so this is the syntactic form of
    "this arm rebound the name": the assignment forms of
    :func:`assigned_names`, plus the receiver of a list method that changes it
    (§5a rule 8). Such a receiver is always a name, because
    `walk/calls.rs::method_call` accepts no other target.
    """
    return assigned_names(block) | _list_method_receivers(block)


def _list_method_receivers(node: Any) -> set[str]:
    """The names a list method rebinds anywhere inside `node`, which is any
    AST node, list of nodes, or leaf of the exported JSON.
    """
    if isinstance(node, list):
        names: set[str] = set()
        for item in node:
            names |= _list_method_receivers(item)
        return names
    if not isinstance(node, dict):
        return set()
    target = node.get("target")
    names = {target} if (
        node.get("__class__") == "ASTNamedAttribute"
        and isinstance(target, str)
        and target not in NAMESPACES
        and node.get("member") in ops.REBINDING_LIST_METHODS
    ) else set()
    for child in node.values():
        names |= _list_method_receivers(child)
    return names


def assigned_names(block: list[dict[str, Any]]) -> set[str]:
    """Every name a block can assign, nested blocks included.

    A transcription of `walk/while_cap.rs::assigned_names`; :func:`rebound_names`
    is what predicts the merged name set of an `if` (§5a rule 5).
    """
    names: set[str] = set()
    for statement in block:
        kind = statement["__class__"]
        if kind in ("ASTAssignStatement", "ASTAugAssignStatement"):
            for target in statement["targets"]:
                names |= _target_names(target)
        elif kind == "ASTCondStatement":
            names |= assigned_names(statement["t_block"])
            names |= assigned_names(statement["f_block"])
        elif kind == "ASTForInStatement":
            names |= _target_names(statement["target"])
            names |= assigned_names(statement["block"])
            names |= assigned_names(statement.get("orelse", []))
        elif kind == "ASTWhileStatement":
            names |= assigned_names(statement["block"])
            names |= assigned_names(statement.get("orelse", []))
    return names


def _target_names(target: dict[str, Any]) -> set[str]:
    kind = target["__class__"]
    if kind == "ASTNameAssignTarget":
        return {target["name"]}
    if kind in ("ASTTupleAssignTarget", "ASTListAssignTarget"):
        return set().union(*(_target_names(t) for t in target["targets"])) \
            if target["targets"] else set()
    if kind == "ASTSubscriptAssignTarget":
        inner = target["target"]
        return {inner["name"]} if inner.get("__class__") == "ASTLoad" else set()
    return set()


def evaluate(
    payloads: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
    inputs: list[Any],
    external_impls: dict[str, Callable[..., Any]] | None = None,
) -> Trace:
    """Run one reference execution.

    `payloads` is `(ast, config, chips, externals)`, the four dicts
    `ZinniaCompiler._core_payloads` builds and the Rust core reads.
    """
    ast, config, chips, externals = payloads
    # A circuit may write its inputs in place (`S-SetItem`, §5.11), so the
    # reference run gets its own copies: the caller's values must come back
    # untouched, or the adequacy hook would change what the suites observe.
    values = [copy.deepcopy(value) for value in inputs]
    return Evaluator(ast, config, chips, externals, external_impls).run(values)
