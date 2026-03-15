from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.config.mock_exec_config import MockExecConfig
from zinnia.config.zinnia_config import ZinniaConfig
from zinnia.ir_def.abstract_ir import AbstractIR
from zinnia.ir_def.defs.ir_assert import AssertIR
from zinnia.ir_def.defs.ir_allocate_memory import AllocateMemoryIR
from zinnia.ir_def.defs.ir_read_memory import ReadMemoryIR
from zinnia.ir_def.defs.ir_print import PrintIR
from zinnia.ir_def.defs.ir_read_float import ReadFloatIR
from zinnia.ir_def.defs.ir_read_hash import ReadHashIR
from zinnia.ir_def.defs.ir_read_integer import ReadIntegerIR
from zinnia.ir_def.defs.ir_write_memory import WriteMemoryIR
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
        self.memories = {}
        self.memory_sizes = {}
        self.memory_init_values = {}
        self.satisfied = True

    def exec(self, *args, **kwargs) -> ZKExecResult:
        zk_parsed_inputs = self.exec_ctx.argparse(*args, **kwargs)
        for entry in zk_parsed_inputs.get_entries():
            self.input_table[entry.get_indices()] = entry.get_value()
        for stmt in self.ir_stmts:
            self.exec_stmt(stmt, self.config.mock_config())
        return ZKExecResult(self.satisfied)

    def exec_stmt(self, stmt: IRStatement, config: MockExecConfig):
        typename = type(stmt.ir_instance).__name__
        method_name = 'exec_' + typename
        method = getattr(self, method_name, None)
        if method is None:
            args = [self.value_table[x] for x in stmt.arguments]
            assert isinstance(stmt.ir_instance, AbstractIR)
            self.value_table[stmt.stmt_id] = stmt.ir_instance.mock_exec(args, config)
            return
        return method(stmt)

    def exec_PrintIR(self, stmt: IRStatement):
        assert isinstance(stmt.ir_instance, PrintIR)
        args = [self.value_table[x] for x in stmt.arguments]
        if args[0] != 0:
            print(args[1], end='', flush=True, sep='')

    def exec_ReadIntegerIR(self, stmt: IRStatement):
        assert isinstance(stmt.ir_instance, ReadIntegerIR)
        val = self.input_table[stmt.ir_instance.indices]
        self.value_table[stmt.stmt_id] = val

    def exec_ReadFloatIR(self, stmt: IRStatement):
        assert isinstance(stmt.ir_instance, ReadFloatIR)
        val = self.input_table[stmt.ir_instance.indices]
        self.value_table[stmt.stmt_id] = val

    def exec_ReadHashIR(self, stmt: IRStatement):
        assert isinstance(stmt.ir_instance, ReadHashIR)
        val = self.input_table[stmt.ir_instance.indices]
        self.value_table[stmt.stmt_id] = val

    def exec_AssertIR(self, stmt: IRStatement):
        assert isinstance(stmt.ir_instance, AssertIR)
        args = [self.value_table[x] for x in stmt.arguments]
        assert len(args) == 1
        if args[0] == 0:
            self.satisfied = False

    def exec_AllocateMemoryIR(self, stmt: IRStatement):
        assert isinstance(stmt.ir_instance, AllocateMemoryIR)
        segment_id = stmt.ir_instance.segment_id
        self.memory_sizes[segment_id] = stmt.ir_instance.size
        self.memory_init_values[segment_id] = stmt.ir_instance.init_value
        self.memories[segment_id] = {i: stmt.ir_instance.init_value for i in range(stmt.ir_instance.size)}

    def exec_WriteMemoryIR(self, stmt: IRStatement):
        assert isinstance(stmt.ir_instance, WriteMemoryIR)
        segment_id = stmt.ir_instance.segment_id
        address = self.value_table[stmt.arguments[0]]
        value = self.value_table[stmt.arguments[1]]
        self.memories[segment_id][address] = value

    def exec_ReadMemoryIR(self, stmt: IRStatement):
        assert isinstance(stmt.ir_instance, ReadMemoryIR)
        segment_id = stmt.ir_instance.segment_id
        address = self.value_table[stmt.arguments[0]]
        self.value_table[stmt.stmt_id] = self.memories[segment_id].get(address, self.memory_init_values[segment_id])
