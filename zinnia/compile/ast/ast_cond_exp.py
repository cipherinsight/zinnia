from typing import Any

from zinnia.compile.ast.ast_expr import ASTExpression
from zinnia.debug.dbg_info import DebugInfo


class ASTCondExp(ASTExpression):
    def __init__(self, dbg: DebugInfo, cond: ASTExpression, t_expr: ASTExpression, f_expr: ASTExpression):
        super().__init__(dbg)
        self.cond = cond
        self.t_expr = t_expr
        self.f_expr = f_expr

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "cond": self.cond.export(),
            "t_expr": self.t_expr.export(),
            "f_expr": self.f_expr.export(),
        }
