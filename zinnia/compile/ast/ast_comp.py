from typing import Any

from zinnia.debug.dbg_info import DebugInfo


class ASTComponent:
    def __init__(self, dbg: DebugInfo):
        self.dbg = dbg

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
        }
