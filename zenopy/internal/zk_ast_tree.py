from typing import Dict

from zenopy.compile.ast import ASTCircuit, ASTChip
from zenopy.internal.external_func_obj import ExternalFuncObj


class ZKAbstractSyntaxTree:
    def __init__(self, root: ASTCircuit, chips: Dict[str, ASTChip], externals: Dict[str, ExternalFuncObj]):
        self.root = root
        self.chips = chips
        self.externals = externals
