from typing import Any

from zinnia.compile.ast.ast_abs_op import ASTAbstractOperator
from zinnia.compile.ast.ast_expr import ASTExpression
from zinnia.debug.dbg_info import DebugInfo


class ASTBinaryOperator(ASTAbstractOperator):
    class Op:
        ADD = "add"
        SUB = "sub"
        MUL = "mul"
        DIV = "div"
        MOD = "mod"
        POW = "pow"
        FLOOR_DIV = "floor_div"
        MAT_MUL = "mat_mul"
        EQ = "eq"
        NE = "ne"
        LT = "lt"
        LTE = "lte"
        GT = "gt"
        GTE = "gte"
        AND = "and"
        OR = "or"

    def __init__(self, dbg: DebugInfo, op_type: str, lhs: ASTExpression, rhs: ASTExpression):
        super().__init__(dbg, [lhs, rhs], {})
        self.operator = op_type
        self.lhs = lhs
        self.rhs = rhs

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "operator": self.operator,
            "lhs": self.lhs.export(),
            "rhs": self.rhs.export(),
        }
