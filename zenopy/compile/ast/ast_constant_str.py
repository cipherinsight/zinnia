from typing import Any

from zenopy.compile.ast.ast_expr import ASTExpression
from zenopy.debug.dbg_info import DebugInfo


class ASTConstantString(ASTExpression):
    def __init__(self, dbg: DebugInfo, value: str):
        super().__init__(dbg)
        self.value = value

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "value": self.value,
        }
