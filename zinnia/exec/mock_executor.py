from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.config.mock_exec_config import MockExecConfig
from zinnia.config.zinnia_config import ZinniaConfig
from zinnia.opdef.ir_op.abstract_ir import AbstractIR
from zinnia.opdef.ir_op.ir_assert import AssertIR
from zinnia.opdef.ir_op.ir_read_float import ReadFloatIR
from zinnia.opdef.ir_op.ir_read_hash import ReadHashIR
from zinnia.opdef.ir_op.ir_read_integer import ReadIntegerIR
from .executor import ZKProgramExecutor
from .exec_ctx import ExecutionContext
from zinnia.api.zk_compiled_program import ZKCompiledProgram
from .exec_result import ZKExecResult


class MockProgramExecutor(ZKProgramExecutor):
    def __init__(self, exec_ctx: ExecutionContext, zk_program: ZKCompiledProgram, config: ZinniaConfig):
        super().__init__(exec_ctx, zk_program, config)
        self.ir_stmts = zk_program.zk_program_irs
        self.value_table = {}
        self.input_table = {}
        self.satisfied = True

    def exec(self, *args, **kwargs) -> ZKExecResult:
        zk_parsed_inputs = self.exec_ctx.argparse(*args, **kwargs)
        for entry in zk_parsed_inputs.get_entries():
            self.input_table[entry.get_indices()] = entry.get_value()
        for stmt in self.ir_stmts:
            self.exec_stmt(stmt, self.config.mock_config())
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
        val = self.input_table[stmt.operator.indices]
        self.value_table[stmt.stmt_id] = val

    def exec_ReadFloatIR(self, stmt: IRStatement):
        assert isinstance(stmt.operator, ReadFloatIR)
        val = self.input_table[stmt.operator.indices]
        self.value_table[stmt.stmt_id] = val

    def exec_ReadHashIR(self, stmt: IRStatement):
        assert isinstance(stmt.operator, ReadHashIR)
        val = self.input_table[stmt.operator.indices]
        self.value_table[stmt.stmt_id] = val

    def exec_AssertIR(self, stmt: IRStatement):
        assert isinstance(stmt.operator, AssertIR)
        args = [self.value_table[x] for x in stmt.arguments]
        assert len(args) == 1
        if args[0] == 0:
            self.satisfied = False
