from typing import Any

from zinnia.compile.ast.ast_expr import ASTExpression
from zinnia.compile.ast.ast_slice import ASTSlice
from zinnia.debug.dbg_info import DebugInfo


class ASTSubscriptExp(ASTExpression):
    def __init__(self, dbg: DebugInfo, val: ASTExpression, slicing: ASTSlice):
        super().__init__(dbg)
        self.val = val
        self.slicing = slicing

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "val": self.val.export(),
            "slicing": self.slicing.export(),
        }
