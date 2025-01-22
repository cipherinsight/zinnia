from typing import Any

from zenopy.compile.ast import ASTAssignTarget, ASTExpression, ASTSlice
from zenopy.debug.dbg_info import DebugInfo


class ASTSubscriptAssignTarget(ASTAssignTarget):
    def __init__(self, dbg: DebugInfo, target: ASTExpression, slicing: ASTSlice, star: bool = False):
        super().__init__(dbg, star=star)
        self.target = target
        self.slicing = slicing

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "target": self.target.export(),
            "slicing": self.slicing.export(),
            "star": self.star,
        }
