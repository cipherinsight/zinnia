from typing import Any

from zinnia.compile.ast.ast_comp import ASTComponent
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.type_sys import DTDescriptor


class ASTAnnotation(ASTComponent):
    class Kind:
        PUBLIC = "Public"
        PRIVATE = "Private"
        HASHED = "Hashed"

    def __init__(self, dbg: DebugInfo, dt: DTDescriptor, kind: str | None):
        super().__init__(dbg)
        self.dt = dt
        self.kind = kind

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "kind": self.kind,
            "dt": self.dt.export(),
        }
