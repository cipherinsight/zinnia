from typing import Any

from zinnia.compile.ast.ast_expr import ASTExpression
from zinnia.debug.dbg_info import DebugInfo


class ASTConstantBoolean(ASTExpression):
    def __init__(self, dbg: DebugInfo, value: bool):
        super().__init__(dbg)
        self.value = value

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "value": self.value,
        }
