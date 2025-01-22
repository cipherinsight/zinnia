from typing import List, Any

from zenopy.compile.ast.ast_expr import ASTExpression
from zenopy.compile.ast.ast_stmt import ASTStatement
from zenopy.debug.dbg_info import DebugInfo


class ASTCondStatement(ASTStatement):
    def __init__(self, dbg: DebugInfo, cond: ASTExpression, t_block: List[ASTStatement], f_block: List[ASTStatement]):
        super().__init__(dbg)
        self.cond = cond
        self.t_block = t_block
        self.f_block = f_block

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "cond": self.cond.export(),
            "t_block": [stmt.export() for stmt in self.t_block],
            "f_block": [stmt.export() for stmt in self.f_block],
        }
