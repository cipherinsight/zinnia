from typing import List, Any

from zinnia.compile.ast.ast_assign_target import ASTAssignTarget
from zinnia.compile.ast.ast_expr import ASTExpression
from zinnia.compile.ast.ast_stmt import ASTStatement
from zinnia.debug.dbg_info import DebugInfo


class ASTForInStatement(ASTStatement):
    def __init__(
            self,
            dbg: DebugInfo,
            target: ASTAssignTarget,
            iter_expr: ASTExpression,
            block: List[ASTStatement],
            orelse: List[ASTStatement]
    ):
        super().__init__(dbg)
        self.target = target
        self.iter_expr = iter_expr
        self.block = block
        self.orelse = orelse

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "target": self.target.export(),
            "iter_expr": self.iter_expr.export(),
            "block": [stmt.export() for stmt in self.block],
            "orelse": [stmt.export() for stmt in self.orelse],
        }
