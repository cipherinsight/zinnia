from typing import List

from zenopy.compile.ir_stmt import IRStatement
from zenopy.config.mock_exec_config import MockExecConfig
from zenopy.opdef.ir_op.abstract_ir import AbstractIR
from zenopy.opdef.ir_op.ir_assert import AssertIR
from zenopy.opdef.ir_op.ir_read_float import ReadFloatIR
from zenopy.opdef.ir_op.ir_read_integer import ReadIntegerIR
from .executor import ZKProgramExecutor
from .exec_ctx import ExecutionContext
from ..backend.zk_program import ZKProgram
from .exec_result import ZKExecResult


class MockProgramExecutor(ZKProgramExecutor):
    def __init__(self, exec_ctx: ExecutionContext, zk_program: ZKProgram, stmts: List[IRStatement]):
        super().__init__(exec_ctx, zk_program)
        self.ir_stmts = stmts
        self.value_table = {}
        self.satisfied = True

    def exec(self) -> ZKExecResult:
        for stmt in self.ir_stmts:
            self.exec_stmt(stmt, MockExecConfig())
        return ZKExecResult(self.satisfied)

    def exec_stmt(self, stmt: IRStatement, config: MockExecConfig):
        typename = type(stmt.operator).__name__
        method_name = 'exec_' + typename
        method = getattr(self, method_name, None)
        if method is None:
            args = [self.value_table[x] for x in stmt.arguments]
            assert isinstance(stmt.operator, AbstractIR)
            self.value_table[stmt.stmt_id] = stmt.operator.mock_exec(stmt.operator.argparse(None, args, {}), config)
            return
        return method(stmt)

    def exec_ReadIntegerIR(self, stmt: IRStatement):
        assert isinstance(stmt.operator, ReadIntegerIR)
        idx_major = stmt.operator.major
        idx_minor = stmt.operator.minor
        val = self.exec_ctx.inputs[(idx_major, idx_minor)]
        self.value_table[stmt.stmt_id] = val

    def exec_ReadFloatIR(self, stmt: IRStatement):
        assert isinstance(stmt.operator, ReadFloatIR)
        idx_major = stmt.operator.major
        idx_minor = stmt.operator.minor
        val = self.exec_ctx.inputs[(idx_major, idx_minor)]
        self.value_table[stmt.stmt_id] = val

    def exec_ReadHashIR(self, stmt: IRStatement):
        raise NotImplementedError()

    def exec_AssertIR(self, stmt: IRStatement):
        assert isinstance(stmt.operator, AssertIR)
        args = [self.value_table[x] for x in stmt.arguments]
        assert len(args) == 1
        if args[0] == 0:
            self.satisfied = False
