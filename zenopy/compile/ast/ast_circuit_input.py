from typing import Any

from zenopy.compile.ast.ast_anno import ASTAnnotation
from zenopy.compile.ast.ast_comp import ASTComponent
from zenopy.debug.dbg_info import DebugInfo


class ASTCircuitInput(ASTComponent):
    def __init__(
        self,
        dbg: DebugInfo,
        name: str,
        annotation: ASTAnnotation
    ):
        super().__init__(dbg)
        self.name = name
        self.annotation = annotation

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "name": self.name,
            "annotation": self.annotation.export(),
        }
