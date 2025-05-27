from zinnia.config.zinnia_config import ZinniaConfig
from .exec_ctx import ExecutionContext
from zinnia.api.zk_compiled_program import ZKCompiledProgram
from .exec_result import ZKExecResult


class ZKProgramExecutor:
    def __init__(self, exec_ctx: ExecutionContext, zk_program: ZKCompiledProgram, config: ZinniaConfig):
        super().__init__()
        self.exec_ctx = exec_ctx
        self.zk_program = zk_program
        self.config = config

    def exec(self, *args, **kwargs) -> ZKExecResult:
        raise NotImplementedError()
