import z3

from zinnia.compile.solver.symbolic_expr import SymbolicExpr


class SymbolicInt(SymbolicExpr):
    concrete_value: int
    has_concrete_value: bool
    has_solved: bool

    def __init__(self, expr: z3.ExprRef, constraints: list[z3.ExprRef] = None):
        super().__init__(expr, constraints or [])
        self.has_solved = False
        self.concrete_value = 0
        self.has_concrete_value = False

    @staticmethod
    def from_val(val: int) -> 'SymbolicInt':
        return SymbolicInt(z3.IntVal(val))

    @staticmethod
    def from_var(name: str) -> 'SymbolicInt':
        return SymbolicInt(z3.Int(name))

    @staticmethod
    def from_expr(expr: z3.ExprRef, constraints: list[z3.ExprRef] = None) -> 'SymbolicInt':
        return SymbolicInt(expr, constraints or [])

    def _solve(self):
        if self.has_solved:
            return
        solver = z3.Solver()
        for constraint in self.constraints:
            solver.add(constraint)
        assert solver.check() == z3.sat
        model = solver.model()
        the_value = model.eval(self.expr, model_completion=True)
        concrete_value = the_value.as_long()
        # Add a constraint to exclude this value in future checks
        solver.add(self.expr != the_value)
        self.has_solved = True
        if solver.check() == z3.sat:
            # There are more possible values
            self.has_concrete_value = False
            return
        self.concrete_value = concrete_value

    def determine_value(self) -> int:
        self._solve()
        assert self.has_concrete_value
        return self.concrete_value

    def is_determined(self) -> bool:
        self._solve()
        return self.has_concrete_value

    def test_equals(self, other: 'SymbolicInt') -> bool:
        solver = z3.Solver()
        for constraint in self.constraints:
            solver.add(constraint)
        for constraint in other.constraints:
            solver.add(constraint)
        solver.add(self.expr != other.expr)
        return solver.check() == z3.unsat

    def test_not_equals(self, other: 'SymbolicInt') -> bool:
        solver = z3.Solver()
        for constraint in self.constraints:
            solver.add(constraint)
        for constraint in other.constraints:
            solver.add(constraint)
        solver.add(self.expr == other.expr)
        return solver.check() == z3.unsat

    def test_true(self) -> bool:
        return self.test_not_equals(SymbolicInt.from_val(0))

    def test_false(self) -> bool:
        return self.test_equals(SymbolicInt.from_val(0))

    @staticmethod
    def Div(lhs: 'SymbolicInt', rhs: 'SymbolicInt') -> 'SymbolicInt':
        return SymbolicInt.from_expr(lhs.expr // rhs.expr, lhs.constraints + rhs.constraints)

    @staticmethod
    def FloorDiv(lhs: 'SymbolicInt', rhs: 'SymbolicInt') -> 'SymbolicInt':
        return SymbolicInt.from_expr(lhs.expr // rhs.expr, lhs.constraints + rhs.constraints)

    @staticmethod
    def Mod(lhs: 'SymbolicInt', rhs: 'SymbolicInt') -> 'SymbolicInt':
        return SymbolicInt.from_expr(lhs.expr % rhs.expr, lhs.constraints + rhs.constraints)

    @staticmethod
    def Add(lhs: 'SymbolicInt', rhs: 'SymbolicInt') -> 'SymbolicInt':
        return SymbolicInt.from_expr(lhs.expr + rhs.expr, lhs.constraints + rhs.constraints)

    @staticmethod
    def Sub(lhs: 'SymbolicInt', rhs: 'SymbolicInt') -> 'SymbolicInt':
        return SymbolicInt.from_expr(lhs.expr - rhs.expr, lhs.constraints + rhs.constraints)

    @staticmethod
    def Mul(lhs: 'SymbolicInt', rhs: 'SymbolicInt') -> 'SymbolicInt':
        return SymbolicInt.from_expr(lhs.expr * rhs.expr, lhs.constraints + rhs.constraints)

    @staticmethod
    def Pow(lhs: 'SymbolicInt', rhs: 'SymbolicInt') -> 'SymbolicInt':
        return SymbolicInt.from_expr(lhs.expr ** rhs.expr, lhs.constraints + rhs.constraints)

    @staticmethod
    def Eq(lhs: 'SymbolicInt', rhs: 'SymbolicInt') -> 'SymbolicInt':
        return SymbolicInt.from_expr(lhs.expr == rhs.expr, lhs.constraints + rhs.constraints)

    @staticmethod
    def Ne(lhs: 'SymbolicInt', rhs: 'SymbolicInt') -> 'SymbolicInt':
        return SymbolicInt.from_expr(lhs.expr != rhs.expr, lhs.constraints + rhs.constraints)

    @staticmethod
    def Lt(lhs: 'SymbolicInt', rhs: 'SymbolicInt') -> 'SymbolicInt':
        return SymbolicInt.from_expr(lhs.expr < rhs.expr, lhs.constraints + rhs.constraints)

    @staticmethod
    def LtE(lhs: 'SymbolicInt', rhs: 'SymbolicInt') -> 'SymbolicInt':
        return SymbolicInt.from_expr(lhs.expr <= rhs.expr, lhs.constraints + rhs.constraints)

    @staticmethod
    def Gt(lhs: 'SymbolicInt', rhs: 'SymbolicInt') -> 'SymbolicInt':
        return SymbolicInt.from_expr(lhs.expr > rhs.expr, lhs.constraints + rhs.constraints)

    @staticmethod
    def GtE(lhs: 'SymbolicInt', rhs: 'SymbolicInt') -> 'SymbolicInt':
        return SymbolicInt.from_expr(lhs.expr >= rhs.expr, lhs.constraints + rhs.constraints)

    @staticmethod
    def And(lhs: 'SymbolicInt', rhs: 'SymbolicInt') -> 'SymbolicInt':
        return SymbolicInt.from_expr(z3.If(z3.And(lhs.expr != 0, lhs.expr != 0), z3.IntVal(1), z3.IntVal(0)), lhs.constraints + rhs.constraints)

    @staticmethod
    def Or(lhs: 'SymbolicInt', rhs: 'SymbolicInt') -> 'SymbolicInt':
        return SymbolicInt.from_expr(z3.If(z3.Or(lhs.expr != 0, lhs.expr != 0), z3.IntVal(1), z3.IntVal(0)), lhs.constraints + rhs.constraints)

    @staticmethod
    def Not(x: 'SymbolicInt') -> 'SymbolicInt':
        return SymbolicInt.from_expr(z3.If(x.expr == 0, z3.IntVal(1), z3.IntVal(0)), x.constraints)

    @staticmethod
    def Select(cond: 'SymbolicInt', t: 'SymbolicInt', f: 'SymbolicInt') -> 'SymbolicInt':
        return SymbolicInt.from_expr(z3.If(cond.expr, t.expr, f.expr), cond.constraints + t.constraints + f.constraints)

    @staticmethod
    def Neg(x: 'SymbolicInt') -> 'SymbolicInt':
        return SymbolicInt.from_expr(-x.expr, x.constraints)

    @staticmethod
    def Abs(x: 'SymbolicInt') -> 'SymbolicInt':
        return SymbolicInt.from_expr(z3.If(x.expr < 0, -x.expr, x.expr), x.constraints)

    @staticmethod
    def Sign(x: 'SymbolicInt') -> 'SymbolicInt':
        return SymbolicInt.from_expr(z3.If(x.expr < 0, -1, z3.If(x.expr > 0, 1, 0)), x.constraints)

    @staticmethod
    def IntCast(x: 'SymbolicReal') -> 'SymbolicInt':
        return SymbolicInt.from_expr(z3.ToInt(x.expr), x.constraints)

    @staticmethod
    def BoolCast(x: 'SymbolicInt') -> 'SymbolicReal':
        return SymbolicInt.from_expr(z3.If(x.expr == 0, z3.IntVal(0), z3.IntVal(1)), x.constraints)
