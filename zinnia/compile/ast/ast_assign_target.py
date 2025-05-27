from typing import Any

from zinnia.compile.ast.ast_comp import ASTComponent
from zinnia.debug.dbg_info import DebugInfo


class ASTAssignTarget(ASTComponent):
    def __init__(self, dbg: DebugInfo, star: bool = False):
        super().__init__(dbg)
        self.star = star

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "star": self.star,
        }
