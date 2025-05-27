from typing import List, Any

from zinnia.compile.ast.ast_expr import ASTExpression
from zinnia.debug.dbg_info import DebugInfo


class ASTJoinedStr(ASTExpression):
    def __init__(self, dbg: DebugInfo, values: List[ASTExpression]):
        super().__init__(dbg)
        self.values = values

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "values": [v.export() for v in self.values],
        }
