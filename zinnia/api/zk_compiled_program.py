import json
from typing import List, Dict

from zinnia.api.zk_program_input import ZKProgramInput
from zinnia.internal.internal_external_func_object import InternalExternalFuncObject


class ZKCompiledProgram:
    """Result of compiling a ZK circuit. Stores an opaque CompiledIR handle (Rust-owned)."""

    def __init__(
        self,
        name: str,
        backend: str,
        compiled_ir,
        program_inputs: List[ZKProgramInput],
        external_funcs: Dict[str, InternalExternalFuncObject],
        eval_data: Dict = None,
    ):
        self.name = name
        self.backend = backend
        self._compiled_ir = compiled_ir
        self.program_inputs = program_inputs
        self.external_funcs = external_funcs
        self.eval_data = eval_data or {}

    def get_program_name(self) -> str:
        return self.name

    def get_target_backend_name(self) -> str:
        return self.backend

    def get_ir_stmts(self) -> list:
        """Returns the IR statements as a list of dicts (on-demand JSON export)."""
        from zinnia.compile._bridge import export_ir_json
        return json.loads(export_ir_json(self._compiled_ir))

    def get_program_inputs(self) -> List[ZKProgramInput]:
        return self.program_inputs

    def get_eval_data(self) -> Dict:
        return self.eval_data

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
        from zinnia.exec.input_parser import build_circuit_inputs
        from zinnia.compile._bridge import prove_circuit

        witness = build_circuit_inputs(self.program_inputs, args)

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
            self._compiled_ir,
            json.dumps(witness),
            ext_dict,
            backend,
            params_json,
        )
        return ZKProofResult.from_json(artifact_json)

    def get_elided_publics(self) -> list:
        """Constant output cells elided from the public instance.

        Returns a list of ``{"Int": v}`` / ``{"Float": v}`` dicts in return
        order — the compiler's claim of which output values are pinned by
        the kernel text itself rather than exposed. Output auditors check
        these against ground truth.
        """
        from zinnia.compile._bridge import get_elided_publics
        return json.loads(get_elided_publics(self._compiled_ir))

    def emit_structural(self, *args, backend="noir", params=None, externals=None) -> str:
        """Render the circuit STRUCTURE for the given witness without proving.

        Returns the emitted Noir `main.nr` text (backend="noir") or a
        halo2-base cell-statistics fingerprint (backend="halo2-base").
        Soundness contract: for one compiled program, every witness must
        yield byte-identical output — the witness-independence regression
        test (testing/lang/test_witness_independence.py) relies on this.
        """
        from zinnia.exec.input_parser import build_circuit_inputs
        from zinnia.compile._bridge import emit_structural_circuit

        witness = build_circuit_inputs(self.program_inputs, args)

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
        return emit_structural_circuit(
            self._compiled_ir,
            json.dumps(witness),
            ext_dict,
            backend,
            params_json,
        )

    def verify(self, proof_result) -> bool:
        """Verify a proof artifact (backend auto-detected from the artifact)."""
        from zinnia.compile._bridge import verify_proof_artifact

        result_json = verify_proof_artifact(proof_result.to_json())
        result = json.loads(result_json)
        return result["valid"]

    def serialize(self) -> str:
        from zinnia.compile._bridge import export_ir_json
        return json.dumps({
            "name": self.name,
            "backend": self.backend,
            "ir_stmts": json.loads(export_ir_json(self._compiled_ir)),
            "program_inputs": [pi.export() for pi in self.program_inputs],
            "external_funcs": [ef.name for ef in self.external_funcs.values()],
            "eval_data": self.eval_data,
        })

    @staticmethod
    def deserialize(data: str, external_funcs=None) -> 'ZKCompiledProgram':
        from zinnia.compile._bridge import import_ir_json
        from zinnia.debug.exception import ZinniaException
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

        # Validate externals against what the circuit expects
        expected_names = set(payload.get('external_funcs', []))
        provided_names = set(ef_map.keys())
        for name in provided_names - expected_names:
            raise ZinniaException(f"External function {name} provided, but not expected")
        for name in expected_names - provided_names:
            raise ZinniaException(f"External function {name} expected, but not provided")

        compiled_ir = import_ir_json(json.dumps(payload['ir_stmts']))

        return ZKCompiledProgram(
            name=payload['name'],
            backend=payload['backend'],
            compiled_ir=compiled_ir,
            program_inputs=_program_inputs,
            external_funcs=ef_map,
        )
