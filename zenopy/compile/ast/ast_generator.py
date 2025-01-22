from typing import List, Any

from zenopy.compile.ast.ast_assign_target import ASTAssignTarget
from zenopy.compile.ast.ast_comp import ASTComponent
from zenopy.compile.ast.ast_expr import ASTExpression
from zenopy.debug.dbg_info import DebugInfo


class ASTGenerator(ASTComponent):
    def __init__(self, dbg: DebugInfo, target: ASTAssignTarget, _iter: ASTExpression, ifs: List[ASTExpression]):
        super().__init__(dbg)
        self.target = target
        self.iter = _iter
        self.ifs = ifs

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "target": self.target.export(),
            "iter": self.iter.export(),
            "ifs": [expr.export() for expr in self.ifs],
        }
