from pyzk.ast.zk_ast import ASTChip


class ChipObject:
    def __init__(self, chip_ast: ASTChip):
        self.chip_ast = chip_ast

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Invoking chip outside circuit is not allowed")
