from typing import Any, List

import z3


class SymbolicExpr:
    def __init__(self, expr: Any, constraints: List[Any]):
        self.expr = expr
        self.constraints = constraints
        solver = z3.Solver()
        for constraint in self.constraints:
            solver.add(constraint)
        assert solver.check() == z3.sat
