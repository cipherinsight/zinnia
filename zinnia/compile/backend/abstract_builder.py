from typing import List

from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.api.zk_compiled_program import ZKCompiledProgram


class AbstractProgramBuilder:
    def __init__(self, name: str, stmts: List[IRStatement]):
        self.name = name
        self.stmts = stmts

    def build(self) -> ZKCompiledProgram:
        raise NotImplementedError()
