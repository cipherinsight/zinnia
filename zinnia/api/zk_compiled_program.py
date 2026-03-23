import json
from typing import List, Dict

from zinnia.api.zk_program_input import ZKProgramInput
from zinnia.internal.internal_external_func_object import InternalExternalFuncObject


class ZKCompiledProgram:
    """Result of compiling a ZK circuit. Stores IR as a JSON string (Rust-owned)."""

    def __init__(
        self,
        name: str,
        backend: str,
        ir_stmts_json: str,
        program_inputs: List[ZKProgramInput],
        external_funcs: Dict[str, InternalExternalFuncObject],
        eval_data: Dict = None,
    ):
        self.name = name
        self.backend = backend
        self.ir_stmts_json = ir_stmts_json
        self.program_inputs = program_inputs
        self.external_funcs = external_funcs
        self.eval_data = eval_data or {}

    def get_program_name(self) -> str:
        return self.name

    def get_target_backend_name(self) -> str:
        return self.backend

    def get_ir_stmts(self) -> list:
        """Returns the IR statements as a list of dicts."""
        return json.loads(self.ir_stmts_json)

    def get_program_inputs(self) -> List[ZKProgramInput]:
        return self.program_inputs

    def get_eval_data(self) -> Dict:
        return self.eval_data

    def argparse(self, *args):
        """Parse positional arguments into a ZKParsedInput."""
        from zinnia.exec.input_parser import parse_inputs_to_parsed_input
        return parse_inputs_to_parsed_input(self.program_inputs, args)

    def get_execution_context(self):
        """Validate externals and return an execution context dict."""
        from zinnia.debug.exception import ZinniaException

        serialized = json.loads(self.serialize())
        expected_names = set(serialized.get("external_funcs", []))
        provided_names = set(self.external_funcs.keys())

        for name in provided_names - expected_names:
            raise ZinniaException(f"External function {name} provided, but not expected")
        for name in expected_names - provided_names:
            raise ZinniaException(f"External function {name} expected, but not provided")

        return {
            "program": self,
            "external_funcs": self.external_funcs,
        }

    def mock_execute(self, *args, externals=None, config=None):
        """Convenience: parse inputs and mock-execute in one call."""
        from zinnia.exec.exec_result import ZKExecResult
        proof_result = self.prove(*args, backend="mock", externals=externals)
        satisfied = proof_result.proof_bytes_hex == "mock_satisfied"
        return ZKExecResult(satisfied, proof=proof_result)

    def prove(self, *args, backend="mock", params=None, externals=None):
        """Generate a proof for this compiled program.

        Args:
            *args: Positional arguments matching the circuit inputs.
            backend: "mock" (default, fast) or "halo2" (real ZK proof).
            params: Optional dict with proving parameters.
            externals: Optional dict of {name: callable} for external functions.

        Returns:
            ZKProofResult containing the proof artifact.
        """
        from zinnia.exec.proof_result import ZKProofResult
        from zinnia.exec.input_parser import parse_inputs
        from zinnia.compile._bridge import prove_circuit

        entries = parse_inputs(self.program_inputs, args)
        witness = {"entries": [
            [e["key"], {e["kind"]: e["value"]}] for e in entries
        ]}

        ext_dict = {}
        if externals:
            ext_dict = externals
        else:
            for name, ef in self.external_funcs.items():
                if hasattr(ef, 'callable'):
                    ext_dict[name] = ef.callable
                elif callable(ef):
                    ext_dict[name] = ef

        params_json = json.dumps(params) if params else None

        artifact_json = prove_circuit(
            self.ir_stmts_json,
            json.dumps(witness),
            ext_dict,
            backend,
            params_json,
        )
        return ZKProofResult.from_json(artifact_json)

    def verify(self, proof_result) -> bool:
        """Verify a proof artifact (backend auto-detected from the artifact)."""
        from zinnia.compile._bridge import verify_proof_artifact

        result_json = verify_proof_artifact(proof_result.to_json())
        result = json.loads(result_json)
        return result["valid"]

    def serialize(self) -> str:
        return json.dumps({
            "name": self.name,
            "backend": self.backend,
            "ir_stmts": json.loads(self.ir_stmts_json),
            "program_inputs": [pi.export() for pi in self.program_inputs],
            "external_funcs": [ef.name for ef in self.external_funcs.values()],
            "eval_data": self.eval_data,
        })

    @staticmethod
    def deserialize(data: str, external_funcs=None) -> 'ZKCompiledProgram':
        if external_funcs is None:
            external_funcs = []
        payload = json.loads(data)
        _program_inputs = [ZKProgramInput.import_from(pi) for pi in payload['program_inputs']]

        ef_map = {}
        for ef in external_funcs:
            if hasattr(ef, 'to_internal_object'):
                internal = ef.to_internal_object()
                ef_map[internal.name] = internal
            else:
                ef_map[ef.name] = ef

        return ZKCompiledProgram(
            name=payload['name'],
            backend=payload['backend'],
            ir_stmts_json=json.dumps(payload['ir_stmts']),
            program_inputs=_program_inputs,
            external_funcs=ef_map,
        )
