from typing import List, Any

from zinnia.compile.ast.ast_expr import ASTExpression
from zinnia.compile.ast.ast_generator import ASTGenerator
from zinnia.debug.dbg_info import DebugInfo


class ASTGeneratorExp(ASTExpression):
    class Kind:
        LIST = "list"
        TUPLE = "tuple"

    def __init__(self, dbg: DebugInfo, elt: ASTExpression, generators: List[ASTGenerator], kind: str):
        super().__init__(dbg)
        self.elt = elt
        self.generators = generators
        self.kind = kind

    def export(self) -> Any:
        return {
            "__class__": self.__class__.__name__,
            "elt": self.elt.export(),
            "generators": [gen.export() for gen in self.generators],
            "kind": self.kind,
        }
