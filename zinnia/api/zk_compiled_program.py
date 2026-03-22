import json
from typing import List, Dict

from zinnia.api.zk_program_input import ZKProgramInput
from zinnia.internal.internal_external_func_object import InternalExternalFuncObject


class ZKCompiledProgram:
    """Result of compiling a ZK circuit. Stores IR as JSON strings (Rust-owned)."""

    def __init__(
        self,
        name: str,
        backend: str,
        zk_program_irs_json: str,
        preprocess_irs_json: str,
        program_inputs: List[ZKProgramInput],
        external_funcs: Dict[str, InternalExternalFuncObject],
        eval_data: Dict = None,
    ):
        self.name = name
        self.backend = backend
        self.zk_program_irs_json = zk_program_irs_json
        self.preprocess_irs_json = preprocess_irs_json
        self.program_inputs = program_inputs
        self.external_funcs = external_funcs
        self.eval_data = eval_data or {}

    def get_program_name(self) -> str:
        return self.name

    def get_target_backend_name(self) -> str:
        return self.backend

    def get_zk_program_irs(self) -> list:
        """Returns the ZK program IR statements as a list of dicts."""
        return json.loads(self.zk_program_irs_json)

    def get_preprocess_irs(self) -> list:
        """Returns the preprocessing IR statements as a list of dicts."""
        return json.loads(self.preprocess_irs_json)

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

        # Check serialized external_funcs names against what the IR expects
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
        from zinnia.exec.input_parser import parse_inputs
        from zinnia.compile._bridge import mock_execute

        entries = parse_inputs(self.program_inputs, args)
        ext_dict = {}
        if externals:
            for name, ext in externals.items():
                if hasattr(ext, 'callable'):
                    ext_dict[name] = ext.callable
                elif callable(ext):
                    ext_dict[name] = ext

        result_json = mock_execute(
            self.zk_program_irs_json,
            self.preprocess_irs_json,
            json.dumps(entries),
            ext_dict,
        )
        result = json.loads(result_json)
        return ZKExecResult(result["satisfied"], result.get("public_outputs"))

    def serialize(self) -> str:
        return json.dumps({
            "name": self.name,
            "backend": self.backend,
            "zk_program_irs": json.loads(self.zk_program_irs_json),
            "preprocess_irs": json.loads(self.preprocess_irs_json),
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

        # external_funcs can be ZKExternalFunc or InternalExternalFuncObject
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
            zk_program_irs_json=json.dumps(payload['zk_program_irs']),
            preprocess_irs_json=json.dumps(payload['preprocess_irs']),
            program_inputs=_program_inputs,
            external_funcs=ef_map,
        )
