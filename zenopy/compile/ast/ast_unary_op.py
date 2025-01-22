from typing import Any

from zenopy.compile.ast.ast_abs_op import ASTAbstractOperator
from zenopy.compile.ast.ast_expr import ASTExpression
from zenopy.debug.dbg_info import DebugInfo


class ASTUnaryOperator(ASTAbstractOperator):
    class Op:
        USUB = "usub"
        UADD = "uadd"
        NOT = "not"

    def __init__(self, dbg: DebugInfo, op_type: str, operand: ASTExpression):
        super().__init__(dbg, [operand], {})
        self.operator = op_type
        self.operand = operand

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "operator": self.operator,
            "operand": self.operand.export(),
        }
