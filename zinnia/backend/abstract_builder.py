from typing import List

from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.api.zk_program import ZKProgram


class AbstractProgramBuilder:
    def __init__(self, name: str, stmts: List[IRStatement]):
        self.name = name
        self.stmts = stmts

    def build(self) -> ZKProgram:
        raise NotImplementedError()
