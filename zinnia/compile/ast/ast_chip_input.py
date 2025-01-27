from typing import Any

from zinnia.compile.ast.ast_anno import ASTAnnotation
from zinnia.compile.ast.ast_comp import ASTComponent
from zinnia.debug.dbg_info import DebugInfo


class ASTChipInput(ASTComponent):
    def __init__(
        self,
        dbg: DebugInfo,
        name: str,
        annotation: ASTAnnotation | None
    ):
        super().__init__(dbg)
        self.name = name
        self.annotation = annotation

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "name": self.name,
            "annotation": self.annotation.export() if self.annotation else None,
        }
