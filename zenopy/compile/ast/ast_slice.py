from typing import List, Tuple, Any

from zenopy.compile.ast.ast_comp import ASTComponent
from zenopy.compile.ast.ast_expr import ASTExpression
from zenopy.debug.dbg_info import DebugInfo


class ASTSlice(ASTComponent):
    def __init__(
            self,
            dbg: DebugInfo, data: List[ASTExpression | Tuple[ASTExpression, ASTExpression, ASTExpression]]
    ):
        super().__init__(dbg)
        self.data = data

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "data": [d.export() if isinstance(d, ASTExpression) else [d[0].export(), d[1].export(), d[2].export()] for d in self.data]
        }
