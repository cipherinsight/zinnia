from typing import Optional, List, Dict, Any

from zinnia.compile.ast.ast_abs_op import ASTAbstractOperator
from zinnia.compile.ast.ast_expr import ASTExpression
from zinnia.debug.dbg_info import DebugInfo


class ASTNamedAttribute(ASTAbstractOperator):
    def __init__(self, dbg: DebugInfo, target: Optional[str], member: str, args: List[ASTExpression], kwargs: Dict[str, ASTExpression]):
        super().__init__(dbg, args, kwargs)
        self.target = target
        self.member = member

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "target": self.target,
            "member": self.member,
        }
