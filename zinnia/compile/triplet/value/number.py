import random
from typing import Union
import time
from z3 import z3
import gc
import multiprocessing


from zinnia.compile.triplet.value.atomic import AtomicValue
from zinnia.compile.type_sys import NumberDTDescriptor
from zinnia.compile.triplet.store import ValueTriplet, ValueStore


class SMTUtils:
    """Safe SMT expression resolver using subprocess isolation (spawn).
    Z3 expressions are serialized as SMT-LIB strings to avoid pickling errors.
    """
    ACCUMULATED_TIME = 0
    ENABLE_RESOLVE = True

    @staticmethod
    def resolve_expr(expr, constraints=None, total_ms=30, per_step_ms=10):
        """Resolve `expr` under `constraints` to a constant int/float if possible.
        Each heuristic runs in a spawned subprocess (≤ per_step_ms), with total ≤ total_ms.
        """
        constraints = constraints or []
        start_time = time.time()

        # Serialize expressions as strings
        expr_str = expr.sexpr()
        constraint_strs = [c.sexpr() for c in constraints]

        ctx = multiprocessing.get_context("spawn")
        heuristics = [
            SMTUtils._try_simplify_proc,
            # SMTUtils._try_substitute_proc,
            SMTUtils._try_solve_eqs_proc,
            # SMTUtils._try_ctx_simplify_proc,
            SMTUtils._try_solver_check_proc,
        ]

        result = None
        for h in heuristics:
            remaining = total_ms - (time.time() - start_time) * 1000
            if remaining <= 0:
                break

            q = ctx.Queue()
            p = ctx.Process(target=h, args=(expr_str, constraint_strs, q))
            p.start()
            p.join(min(per_step_ms, remaining) / 1000)

            if p.is_alive():
                p.terminate()
                p.join()

            if not q.empty():
                result = q.get()

            # Ensure resources are freed
            q.close()
            q.join_thread()
            del q, p
            gc.collect()

            if result is not None:
                break

        return result

    # ---------------- Utility ----------------
    @staticmethod
    def _is_value(e):
        return (
            z3.is_int_value(e)
            or z3.is_rational_value(e)
            or z3.is_algebraic_value(e)
            or z3.is_true(e)
            or z3.is_false(e)
        )

    @staticmethod
    def _to_python_value(e):
        if z3.is_int_value(e):
            return int(e.as_long())
        if z3.is_rational_value(e):
            return float(e.as_fraction())
        if z3.is_algebraic_value(e):
            return float(e.approx(10))
        return None

    # ---------------- Internal reconstruction helpers ----------------
    @staticmethod
    def _rebuild(expr_str, constraint_strs):
        """Rebuild expr and constraint objects in a fresh Z3 context."""
        expr = z3.parse_smt2_string(f"(assert {expr_str})")[0]
        constraints = [z3.parse_smt2_string(f"(assert {c})")[0] for c in constraint_strs]
        return expr, constraints

    # ---------------- Heuristic workers (run in subprocess) ----------------
    @staticmethod
    def _try_simplify_proc(expr_str, constraint_strs, q):
        try:
            expr, constraints = SMTUtils._rebuild(expr_str, constraint_strs)
            s_expr = z3.simplify(expr)
            if SMTUtils._is_value(s_expr):
                q.put(SMTUtils._to_python_value(s_expr))
                return
        except Exception:
            pass
        q.put(None)

    @staticmethod
    def _try_substitute_proc(expr_str, constraint_strs, q):
        try:
            expr, constraints = SMTUtils._rebuild(expr_str, constraint_strs)
            subs = []
            for c in constraints:
                if c.decl().kind() == z3.Z3_OP_EQ:
                    lhs, rhs = c.children()
                    if z3.is_const(lhs) and SMTUtils._is_value(rhs):
                        subs.append((lhs, rhs))
                    elif z3.is_const(rhs) and SMTUtils._is_value(lhs):
                        subs.append((rhs, lhs))
            if subs:
                s_expr = z3.simplify(z3.substitute(expr, subs))
                if SMTUtils._is_value(s_expr):
                    q.put(SMTUtils._to_python_value(s_expr))
                    return
        except Exception:
            pass
        q.put(None)

    @staticmethod
    def _try_solve_eqs_proc(expr_str, constraint_strs, q):
        try:
            expr, constraints = SMTUtils._rebuild(expr_str, constraint_strs)
            g = z3.Goal()
            g.add(*constraints)
            g.add(expr == z3.Const("tmp_expr", expr.sort()))
            t = z3.Tactic("solve-eqs")
            subgoals = t(g)
            if subgoals and len(subgoals[0]) == 0:
                s_expr = z3.simplify(expr)
                if SMTUtils._is_value(s_expr):
                    q.put(SMTUtils._to_python_value(s_expr))
                    return
        except Exception:
            pass
        q.put(None)

    @staticmethod
    def _try_ctx_simplify_proc(expr_str, constraint_strs, q):
        try:
            expr, constraints = SMTUtils._rebuild(expr_str, constraint_strs)
            g = z3.Goal()
            g.add(*(constraints + [expr == z3.Const("tmp_expr", expr.sort())]))
            t = z3.Tactic("ctx-solver-simplify")
            result = t(g)
            for sub in result:
                for f in sub:
                    simp = z3.simplify(f)
                    if SMTUtils._is_value(simp):
                        q.put(SMTUtils._to_python_value(simp))
                        return
        except Exception:
            pass
        q.put(None)

    @staticmethod
    def _try_solver_check_proc(expr_str, constraint_strs, q):
        try:
            expr, constraints = SMTUtils._rebuild(expr_str, constraint_strs)
            s = z3.Solver()
            s.add(constraints)
            if s.check() == z3.sat:
                m = s.model()
                val = m.eval(expr, model_completion=True)
                s.push()
                s.add(expr != val)
                if s.check() == z3.unsat:
                    q.put(SMTUtils._to_python_value(val))
                    return
                s.pop()
        except Exception:
            pass
        q.put(None)


class NumberValue(AtomicValue):
    def __init__(self, triplet: ValueTriplet):
        super().__init__(triplet)

    def val(self) -> int | float | None:
        return self._triplet.get_s()

    def c_val(self) -> int | float | None:
        return self._triplet.get_s()

    def ptr(self) -> int | None:
        return self._triplet.get_v()

    def type(self) -> NumberDTDescriptor:
        assert isinstance(self._triplet.get_t(), NumberDTDescriptor)
        return self._triplet.get_t()

    def assign(self, value: 'NumberValue', force: bool = False) -> 'NumberValue':
        if self.type_locked():
            assert force or value._triplet.get_s() == self._triplet.get_s()
        self._triplet.assign(value._triplet)
        return self

    def __copy__(self):
        raise NotImplementedError()

    def __deepcopy__(self, memo):
        raise NotImplementedError()

    @classmethod
    def from_value_store(cls, store: ValueStore, type_locked: bool = False) -> Union['NumberValue', None]:
        raise NotImplementedError()

    @staticmethod
    def smt_resolve_expr(expr, constraints=None, timeout_ms=30):
        if not SMTUtils.ENABLE_RESOLVE:
            return None
        if len(constraints) > 2:
            # early reject requests that are too complex
            return None
        result = SMTUtils.resolve_expr(expr, constraints, timeout_ms)
        SMTUtils.ACCUMULATED_TIME += timeout_ms / 1000
        return result

    # @staticmethod
    # def smt_resolve_expr(expr, constraints=None, timeout_ms=100):
    #     if len(constraints) >= 10:
    #         # early reject to avoid long solving times
    #         return None
    #
    #     print('Resolving SMT expression...' + str(random.randint(0, 20000)))
    #
    #     def to_python_value(e):
    #         """Convert Z3 value to Python int or float."""
    #         if z3.is_int_value(e):
    #             return int(e.as_long())
    #         if z3.is_rational_value(e):
    #             return float(e.as_fraction())
    #         if z3.is_algebraic_value(e):
    #             # Algebraic numbers may not be rational; approximate numerically
    #             return float(e.approx(10))
    #         # For booleans or others, we return None
    #         return None
    #
    #     s = z3.Solver()
    #     s.set(timeout=timeout_ms)
    #     s.add(constraints)
    #     if s.check() == z3.sat:
    #         m = s.model()
    #         val = m.eval(expr, model_completion=True)
    #         s.push()
    #         s.add(expr != val)
    #         if s.check() == z3.unsat:
    #             print('Done')
    #             return to_python_value(val)
    #         s.pop()
    #
    #     print('Done')
    #     return None
