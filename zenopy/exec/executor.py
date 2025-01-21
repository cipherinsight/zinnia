from .exec_ctx import ExecutionContext
from ..backend.zk_program import ZKProgram
from .exec_result import ZKExecResult


class ZKProgramExecutor:
    def __init__(self, exec_ctx: ExecutionContext, zk_program: ZKProgram):
        super().__init__()
        self.exec_ctx = exec_ctx
        self.zk_program = zk_program

    def exec(self) -> ZKExecResult:
        raise NotImplementedError()
