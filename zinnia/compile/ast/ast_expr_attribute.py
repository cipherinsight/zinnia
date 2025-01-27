from typing import Dict, List, Any

from zinnia.compile.ast.ast_abs_op import ASTAbstractOperator
from zinnia.compile.ast.ast_expr import ASTExpression
from zinnia.debug.dbg_info import DebugInfo


class ASTExprAttribute(ASTAbstractOperator):
    def __init__(
            self,
            dbg: DebugInfo,
            target: ASTExpression,
            member: str,
            args: List[ASTExpression],
            kwargs: Dict[str, ASTExpression]
    ):
        super().__init__(dbg, args, kwargs)
        self.target = target
        self.member = member

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "target": self.target.export(),
            "member": self.member,
            "args": [arg.export() for arg in self.args],
            "kwargs": {k: v.export() for k, v in self.kwargs.items()},
        }
