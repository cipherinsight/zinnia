from typing import Dict

from zenopy.ast.zk_ast import ASTProgram, ASTChip


class ZKAbstractSyntaxTree:
    def __init__(self, root: ASTProgram, chips: Dict[str, ASTChip]):
        self.root = root
        self.chips = chips
