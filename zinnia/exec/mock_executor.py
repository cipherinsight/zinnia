import json

from zinnia.exec.exec_result import ZKExecResult
from zinnia.exec.input_parser import parse_inputs


class MockProgramExecutor:
    def __init__(self, exec_ctx, program, config):
        self.exec_ctx = exec_ctx
        self.program = program
        self.config = config

    def exec(self, *args) -> ZKExecResult:
        from zinnia.compile._bridge import mock_execute

        entries = parse_inputs(self.program.program_inputs, args)
        externals_dict = {}
        for name, ef in self.program.external_funcs.items():
            externals_dict[name] = ef.callable

        result_json = mock_execute(
            self.program.zk_program_irs_json,
            self.program.preprocess_irs_json,
            json.dumps(entries),
            externals_dict,
        )
        result = json.loads(result_json)
        return ZKExecResult(result["satisfied"], result.get("public_outputs"))
