from typing import List

from pyzk.backend.zk_program import ZKProgram
from pyzk.internal.prog_meta_data import ProgramMetadata
from pyzk.ir.ir_stmt import IRStatement


class AbstractProgramBuilder:
    def __init__(self, stmts: List[IRStatement], prog_metadata: ProgramMetadata):
        self.stmts = stmts
        self.prog_metadata = prog_metadata

    def build(self) -> ZKProgram:
        raise NotImplementedError()
