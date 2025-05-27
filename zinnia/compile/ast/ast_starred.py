from typing import Any

from zinnia.compile.ast import ASTExpression
from zinnia.debug.dbg_info import DebugInfo


class ASTStarredExpr(ASTExpression):
    def __init__(self, dbg: DebugInfo, inner_value: ASTExpression):
        super().__init__(dbg)
        self.value = inner_value

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "value": [self.value.export()],
        }
