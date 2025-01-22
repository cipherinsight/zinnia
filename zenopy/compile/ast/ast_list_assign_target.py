from typing import Any, List

from zenopy.compile.ast import ASTAssignTarget
from zenopy.debug.dbg_info import DebugInfo


class ASTListAssignTarget(ASTAssignTarget):
    def __init__(self, dbg: DebugInfo, targets: List[ASTAssignTarget], star: bool = False):
        super().__init__(dbg, star=star)
        self.targets = targets

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "targets": [target.export() for target in self.targets],
            "star": self.star,
        }
