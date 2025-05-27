from typing import Any

from zinnia.compile.ast.ast_expr import ASTExpression
from zinnia.debug.dbg_info import DebugInfo


class ASTConstantNone(ASTExpression):
    def __init__(self, dbg: DebugInfo):
        super().__init__(dbg)

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
        }
