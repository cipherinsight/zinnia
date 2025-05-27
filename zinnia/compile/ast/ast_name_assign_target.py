from typing import Any

from zinnia.compile.ast import ASTAssignTarget
from zinnia.debug.dbg_info import DebugInfo


class ASTNameAssignTarget(ASTAssignTarget):
    def __init__(self, dbg: DebugInfo, name: str, star: bool = False):
        super().__init__(dbg, star=star)
        self.name = name

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "name": self.name,
            "star": self.star,
        }
