from typing import Any

from zenopy.compile.ast.ast_expr import ASTExpression
from zenopy.debug.dbg_info import DebugInfo


class ASTLoad(ASTExpression):
    def __init__(self, dbg: DebugInfo, name: str):
        super().__init__(dbg)
        self.name = name

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "name": self.name,
        }
