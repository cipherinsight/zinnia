from zinnia.compile.ast import ASTChip
from zinnia.compile.type_sys.dt_descriptor import DTDescriptor


class InternalChipObject:
    def __init__(self, name: str, chip_ast: ASTChip, return_dt: DTDescriptor):
        self.name = name
        self.chip_ast = chip_ast
        self.return_dt = return_dt
