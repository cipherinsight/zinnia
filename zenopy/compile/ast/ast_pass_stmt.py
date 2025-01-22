from typing import Any

from zenopy.compile.ast.ast_stmt import ASTStatement
from zenopy.debug.dbg_info import DebugInfo


class ASTPassStatement(ASTStatement):
    def __init__(self, dbg: DebugInfo):
        super().__init__(dbg)

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
        }
