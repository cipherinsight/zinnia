import json

from .exec_ctx import ExecutionContext
from ..backend.zk_program import Halo2ZKProgram, ZKProgram


class ZKProgramExecutor:
    def __init__(self, exec_ctx: ExecutionContext, zk_program: ZKProgram):
        super().__init__()
        self.exec_ctx = exec_ctx
        self.zk_program = zk_program

    def exec(self):
        raise NotImplementedError()


class Halo2ZKProgramExecutor(ZKProgramExecutor):
    def __init__(self, exec_ctx: ExecutionContext, zk_program: Halo2ZKProgram):
        super().__init__(exec_ctx, zk_program)

    def exec(self):
        assert isinstance(self.zk_program, Halo2ZKProgram)
        with open(f"/Users/zhantong/Projects/halo2-graph/examples/{self.zk_program.circuit_name}.rs", "w") as f:
            f.write(self.zk_program.source)
        with open(f"/Users/zhantong/Projects/halo2-graph/data/{self.zk_program.circuit_name}.in", "w") as f:
            parsed_data = self.exec_ctx.inputs
            json_dict = {}
            for key, val in parsed_data.items():
                json_dict[f"x_{key[0]}_{key[1]}"] = val
            f.write(json.dumps(json_dict))
