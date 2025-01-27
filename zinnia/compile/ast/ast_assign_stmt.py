from typing import List, Any

from zinnia.compile.ast.ast_assign_target import ASTAssignTarget
from zinnia.compile.ast.ast_expr import ASTExpression
from zinnia.compile.ast.ast_stmt import ASTStatement
from zinnia.debug.dbg_info import DebugInfo


class ASTAssignStatement(ASTStatement):
    def __init__(self, dbg: DebugInfo, targets: List[ASTAssignTarget], value: ASTExpression):
        super().__init__(dbg)
        self.targets = targets
        self.value = value

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "targets": [target.export() for target in self.targets],
            "value": self.value.export(),
        }
