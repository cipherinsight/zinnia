from typing import Dict, Callable

from zenopy.ast.zk_ast import ASTProgram, ASTChip
from zenopy.internal.external_func_obj import ExternalFuncObj


class ZKAbstractSyntaxTree:
    def __init__(self, root: ASTProgram, chips: Dict[str, ASTChip], externals: Dict[str, ExternalFuncObj]):
        self.root = root
        self.chips = chips
        self.externals = externals
