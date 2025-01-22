from typing import Tuple, Any

from zenopy.compile.ast import ASTAssignTarget
from zenopy.debug.dbg_info import DebugInfo


class ASTTupleAssignTarget(ASTAssignTarget):
    def __init__(self, dbg: DebugInfo, targets: Tuple[ASTAssignTarget, ...], star: bool = False):
        super().__init__(dbg, star=star)
        self.targets = targets

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "targets": [target.export() for target in self.targets],
            "star": self.star,
        }
