import z3

from zinnia.compile.solver.symbolic_expr import SymbolicExpr


class SymbolicReal(SymbolicExpr):
    concrete_value: float
    has_concrete_value: bool
    has_solved: bool

    def __init__(self, expr: z3.ExprRef, constraints: list[z3.ExprRef] = None):
        super().__init__(expr, constraints or [])
        self.has_solved = False
        self.concrete_value = 0.0
        self.has_concrete_value = False

    @staticmethod
    def from_val(val: float) -> 'SymbolicReal':
        return SymbolicReal(z3.RealVal(val))

    @staticmethod
    def from_var(name: str) -> 'SymbolicReal':
        return SymbolicReal(z3.Real(name))

    @staticmethod
    def from_expr(expr: z3.ExprRef, constraints: list[z3.ExprRef] = None) -> 'SymbolicReal':
        return SymbolicReal(expr, constraints or [])

    def _solve(self):
        if self.has_solved:
            return
        solver = z3.Solver()
        for constraint in self.constraints:
            solver.add(constraint)
        assert solver.check() == z3.sat
        model = solver.model()
        the_value = model.eval(self.expr, model_completion=True)
        concrete_value = the_value.as_decimal(10)
        # Add a constraint to exclude this value in future checks
        solver.add(self.expr != the_value)
        self.has_solved = True
        if solver.check() == z3.sat:
            # There are more possible values
            self.has_concrete_value = False
            return
        self.concrete_value = concrete_value

    def determine_value(self) -> float:
        self._solve()
        assert self.has_concrete_value
        return self.concrete_value

    def is_determined(self) -> bool:
        self._solve()
        return self.has_concrete_value

    def test_equals(self, other: 'SymbolicReal') -> bool:
        solver = z3.Solver()
        for constraint in self.constraints:
            solver.add(constraint)
        for constraint in other.constraints:
            solver.add(constraint)
        solver.add(self.expr != other.expr)
        return solver.check() == z3.unsat

    def test_not_equals(self, other: 'SymbolicReal') -> bool:
        solver = z3.Solver()
        for constraint in self.constraints:
            solver.add(constraint)
        for constraint in other.constraints:
            solver.add(constraint)
        solver.add(self.expr == other.expr)
        return solver.check() == z3.unsat

    @staticmethod
    def Div(lhs: 'SymbolicReal', rhs: 'SymbolicReal') -> 'SymbolicReal':
        return SymbolicReal.from_expr(lhs.expr / rhs.expr, lhs.constraints + rhs.constraints)

    @staticmethod
    def FloorDiv(lhs: 'SymbolicReal', rhs: 'SymbolicReal') -> 'SymbolicReal':
        return SymbolicReal.from_expr(lhs.expr // rhs.expr, lhs.constraints + rhs.constraints)

    @staticmethod
    def Mod(lhs: 'SymbolicReal', rhs: 'SymbolicReal') -> 'SymbolicReal':
        return SymbolicReal.from_expr(lhs.expr % rhs.expr, lhs.constraints + rhs.constraints)

    @staticmethod
    def Add(lhs: 'SymbolicReal', rhs: 'SymbolicReal') -> 'SymbolicReal':
        return SymbolicReal.from_expr(lhs.expr + rhs.expr, lhs.constraints + rhs.constraints)

    @staticmethod
    def Sub(lhs: 'SymbolicReal', rhs: 'SymbolicReal') -> 'SymbolicReal':
        return SymbolicReal.from_expr(lhs.expr - rhs.expr, lhs.constraints + rhs.constraints)

    @staticmethod
    def Mul(lhs: 'SymbolicReal', rhs: 'SymbolicReal') -> 'SymbolicReal':
        return SymbolicReal.from_expr(lhs.expr * rhs.expr, lhs.constraints + rhs.constraints)

    @staticmethod
    def Pow(lhs: 'SymbolicReal', rhs: 'SymbolicReal') -> 'SymbolicReal':
        return SymbolicReal.from_expr(lhs.expr ** rhs.expr, lhs.constraints + rhs.constraints)

    @staticmethod
    def Eq(lhs: 'SymbolicReal', rhs: 'SymbolicReal') -> 'SymbolicReal':
        return SymbolicReal.from_expr(lhs.expr == rhs.expr, lhs.constraints + rhs.constraints)

    @staticmethod
    def Ne(lhs: 'SymbolicReal', rhs: 'SymbolicReal') -> 'SymbolicReal':
        return SymbolicReal.from_expr(lhs.expr != rhs.expr, lhs.constraints + rhs.constraints)

    @staticmethod
    def Lt(lhs: 'SymbolicReal', rhs: 'SymbolicReal') -> 'SymbolicReal':
        return SymbolicReal.from_expr(lhs.expr < rhs.expr, lhs.constraints + rhs.constraints)

    @staticmethod
    def LtE(lhs: 'SymbolicReal', rhs: 'SymbolicReal') -> 'SymbolicReal':
        return SymbolicReal.from_expr(lhs.expr <= rhs.expr, lhs.constraints + rhs.constraints)

    @staticmethod
    def Gt(lhs: 'SymbolicReal', rhs: 'SymbolicReal') -> 'SymbolicReal':
        return SymbolicReal.from_expr(lhs.expr > rhs.expr, lhs.constraints + rhs.constraints)

    @staticmethod
    def GtE(lhs: 'SymbolicReal', rhs: 'SymbolicReal') -> 'SymbolicReal':
        return SymbolicReal.from_expr(lhs.expr >= rhs.expr, lhs.constraints + rhs.constraints)

    @staticmethod
    def Select(cond: 'SymbolicInt', t: 'SymbolicReal', f: 'SymbolicReal') -> 'SymbolicReal':
        return SymbolicReal.from_expr(z3.If(cond.expr, t.expr, f.expr), cond.constraints + t.constraints + f.constraints)

    @staticmethod
    def Neg(x: 'SymbolicReal') -> 'SymbolicReal':
        return SymbolicReal.from_expr(-x.expr, x.constraints)

    @staticmethod
    def Abs(x: 'SymbolicReal') -> 'SymbolicReal':
        return SymbolicReal.from_expr(z3.If(x.expr < 0, -x.expr, x.expr), x.constraints)

    @staticmethod
    def Sign(x: 'SymbolicReal') -> 'SymbolicReal':
        return SymbolicReal.from_expr(z3.If(x.expr < 0.0, -1.0, z3.If(x.expr > 0, 1.0, 0.0)), x.constraints)

    @staticmethod
    def RealCast(x: 'SymbolicInt') -> 'SymbolicReal':
        return SymbolicReal.from_expr(z3.ToReal(x.expr), x.constraints)

    @staticmethod
    def Sqrt(x: 'SymbolicReal') -> 'SymbolicReal':
        return SymbolicReal.from_expr(z3.Sqrt(x.expr), x.constraints)

    @staticmethod
    def Sin(x: 'SymbolicReal') -> 'SymbolicReal':
        new_var = z3.Real('sin_' + str(x.expr))
        return SymbolicReal.from_expr(new_var, [new_var >= -1, new_var <= 1] + x.constraints)

    @staticmethod
    def Cos(x: 'SymbolicReal') -> 'SymbolicReal':
        new_var = z3.Real('cos_' + str(x.expr))
        return SymbolicReal.from_expr(new_var, [new_var >= -1, new_var <= 1] + x.constraints)

    @staticmethod
    def Tan(x: 'SymbolicReal') -> 'SymbolicReal':
        new_var = z3.Real('tan_' + str(x.expr))
        return SymbolicReal.from_expr(new_var, x.constraints)

    @staticmethod
    def Sinh(x: 'SymbolicReal') -> 'SymbolicReal':
        new_var = z3.Real('sinh_' + str(x.expr))
        return SymbolicReal.from_expr(new_var, [new_var >= -1, new_var <= 1] + x.constraints)

    @staticmethod
    def Cosh(x: 'SymbolicReal') -> 'SymbolicReal':
        new_var = z3.Real('cosh_' + str(x.expr))
        return SymbolicReal.from_expr(new_var, [new_var >= -1, new_var <= 1] + x.constraints)

    @staticmethod
    def Tanh(x: 'SymbolicReal') -> 'SymbolicReal':
        new_var = z3.Real('tanh_' + str(x.expr))
        return SymbolicReal.from_expr(new_var, x.constraints)

    @staticmethod
    def Exp(x: 'SymbolicReal') -> 'SymbolicReal':
        new_var = z3.Real('exp_' + str(x.expr))
        return SymbolicReal.from_expr(new_var, x.constraints)

    @staticmethod
    def Log(x: 'SymbolicReal') -> 'SymbolicReal':
        new_var = z3.Real('log_' + str(x.expr))
        return SymbolicReal.from_expr(new_var, x.constraints)
