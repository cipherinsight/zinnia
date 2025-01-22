from typing import Dict, List, Any

from zenopy.compile.ast.ast_expr import ASTExpression
from zenopy.debug.dbg_info import DebugInfo


class ASTAbstractOperator(ASTExpression):
    def __init__(self, dbg: DebugInfo, args: List[ASTExpression], kwargs: Dict[str, ASTExpression]):
        super().__init__(dbg)
        self.args = args
        self.kwargs = kwargs

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "args": [arg.export() for arg in self.args],
            "kwargs": {key: value.export() for key, value in self.kwargs.items()},
        }
