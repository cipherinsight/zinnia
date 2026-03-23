from zinnia.exec.exec_result import ZKExecResult


class MockProgramExecutor:
    """Executes a compiled ZK program using the mock backend."""

    def __init__(self, exec_ctx, program, config):
        self.exec_ctx = exec_ctx
        self.program = program
        self.config = config

    def exec(self, *args) -> ZKExecResult:
        proof_result = self.program.prove(*args, backend="mock")
        satisfied = proof_result.proof_bytes_hex == "mock_satisfied"
        return ZKExecResult(satisfied, proof=proof_result)
