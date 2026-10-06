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
        # Set only for a program restored by `deserialize` (the Rust handle of
        # an imported IR carries no layout).
        self._restored_layout = None

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

        ext_dict = self._external_callables(externals)

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

        Returns ``{"slot": k, "Int": v}`` / ``{"slot": k, "Float": v}`` dicts
        in output order. Superseded by :meth:`get_output_layout`, which says
        where every output value is.
        """
        from zinnia.compile._bridge import get_elided_publics
        return json.loads(get_elided_publics(self._compiled_ir))

    def get_output_layout(self) -> list | None:
        """Where every output value of the circuit is.

        The outputs are the return value or, for a function that returns
        ``None``, the array arguments it writes (in parameter order). They
        are split into *parts* (the items of a returned tuple, and of lists
        or tuples holding arrays, are separate parts); one entry per part, in
        output order::

            slot    = {"public": i, "kind": "i" | "f"}      # public vector index i
                    | {"constant": v}                       # not in the public vector
            output  = {"return": [path...]} | {"argument": name}
            static  = {"output": output, "shape": [d...], "cells": [slot...]}
            dynamic = {"output": output, "dynamic": {"length": slot, "rank": slot,
                       "dims": [slot...], "cells": [slot...],
                       "strides": [s...], "offset": o}}     # strides/offset optional

        A part's cells are its elements in row-major order (complex parts,
        marked ``"complex": true``, have two cells per element: real, then
        imaginary). A runtime-shaped part's shape is the first ``rank`` of
        ``dims``, whose product is ``length``; element ``c`` (coordinates) is
        cell ``offset + sum(c[k] * strides[k])`` when strides are given, else
        cell ``j`` for row-major position ``j``. Kind ``f`` values are Q32
        fixed point. Public indices count every public value of the proof
        (public inputs included, see :meth:`get_public_inputs`); the mock
        backend reports the exposed outputs only. ``None`` for a program
        deserialized from an older payload.
        """
        if self._compiled_ir is not None and self._restored_layout is None:
            from zinnia.compile._bridge import get_output_layout
            return json.loads(get_output_layout(self._compiled_ir))
        return (self._restored_layout or {}).get("outputs")

    def get_public_inputs(self) -> list | None:
        """Inputs read as public values: ``[{"public": i, "input": path, "kind"}]``."""
        if self._compiled_ir is not None and self._restored_layout is None:
            from zinnia.compile._bridge import get_public_inputs
            return json.loads(get_public_inputs(self._compiled_ir))
        return (self._restored_layout or {}).get("inputs")

    def describe_publics(self) -> list:
        """What each value of the public vector is, by public index.

        Every entry is one of ``{"input": path}``,
        ``{"output": output, "element": j}`` (row-major element ``j`` of a
        static part; for a complex part also ``"component": "real" | "imag"``),
        ``{"output": output, "cell": k}`` (payload slot ``k`` of a
        runtime-shaped part) or
        ``{"output": output, "metadata": "length" | "rank" | "dim", "axis": a}``.
        Constant outputs are not in the public vector; see
        :meth:`get_output_layout`.
        """
        layout = self.get_output_layout()
        inputs = self.get_public_inputs()
        if layout is None or inputs is None:
            raise ValueError("this program carries no output layout")
        described: dict[int, dict] = {e["public"]: {"input": e["input"]} for e in inputs}

        def note(slot: dict, entry: dict) -> None:
            if "public" in slot:
                described[slot["public"]] = entry

        for part in layout:
            output = part["output"]
            per_element = 2 if part.get("complex") else 1
            if "dynamic" in part:
                dynamic = part["dynamic"]
                note(dynamic["length"], {"output": output, "metadata": "length"})
                note(dynamic["rank"], {"output": output, "metadata": "rank"})
                for axis, dim in enumerate(dynamic["dims"]):
                    note(dim, {"output": output, "metadata": "dim", "axis": axis})
                cells = dynamic["cells"]
            else:
                cells = part["cells"]
            # A runtime-shaped part's cells are buffer slots; which element a
            # slot holds depends on the runtime shape.
            unit = "cell" if "dynamic" in part else "element"
            for index, cell in enumerate(cells):
                entry = {"output": output, unit: index // per_element}
                if per_element == 2:
                    entry["component"] = "real" if index % 2 == 0 else "imag"
                note(cell, entry)
        return [described[i] for i in sorted(described)]

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

        ext_dict = self._external_callables(externals)

        params_json = json.dumps(params) if params else None
        return emit_structural_circuit(
            self._compiled_ir,
            json.dumps(witness),
            ext_dict,
            backend,
            params_json,
        )

    def setup(self, params=None):
        """Generate halo2-base proving and verifying keys, once, without a witness.

        ``params`` is ``{"k", "lookup_bits", "precision_bits"}``; it defaults
        to the backend's own ``estimate_params``. The returned handle exposes
        ``k``, ``lookup_bits``, ``precision_bits``, ``ir_hash``,
        ``keygen_seconds`` (IPA parameters plus keygen), ``vk_bytes()`` and
        ``save(directory)``, which writes the IPA parameters, both keys and
        their pinning so another process can :meth:`load_keys` them. Use it
        with :meth:`prove_with_keys` and :meth:`verify_with_keys`; unlike
        :meth:`prove`, neither repeats parameter generation or keygen.
        """
        from zinnia._zinnia_core import halo2_base_keygen

        return halo2_base_keygen(self._compiled_ir, json.dumps(params) if params else None)

    @staticmethod
    def load_keys(directory, role: str = "prover"):
        """Read keys written by ``keys.save(directory)``.

        ``role="prover"`` reads the IPA parameters and the proving key,
        ``role="verifier"`` the IPA parameters and the verifying key. The
        handle's ``load_seconds`` is ``{"params": s, "proving_key" |
        "verifying_key": s}``, the time spent reading each file.
        """
        from zinnia._zinnia_core import halo2_base_load_keys

        return halo2_base_load_keys(str(directory), role)

    def prove_with_keys(self, keys, *args, externals=None) -> Dict:
        """Prove with keys from :meth:`setup` or :meth:`load_keys` (role ``prover``).

        Returns ``{"proof": hex, "public_values": [hex field elements], "k"}``
        and the seconds of each phase, timed separately:
        ``parse_inputs_seconds`` (reading the inputs into the backend),
        ``preprocess_seconds`` (the IR evaluation pass that resolves external
        calls), ``witness_seconds`` (circuit synthesis),
        ``thread_pool_seconds`` (starting the proving thread pool) and
        ``prove_seconds`` (``create_proof`` plus transcript).
        """
        from zinnia.exec.input_parser import build_circuit_inputs
        from zinnia._zinnia_core import halo2_base_prove

        witness = build_circuit_inputs(self.program_inputs, args)
        out = halo2_base_prove(
            self._compiled_ir, keys, json.dumps(witness), self._external_callables(externals)
        )
        return json.loads(out)

    @staticmethod
    def verify_with_keys(keys, proof: Dict) -> bool:
        """Verify a :meth:`prove_with_keys` result against the keys' verifying key."""
        return ZKCompiledProgram.verify_with_keys_report(keys, proof)["valid"]

    @staticmethod
    def verify_with_keys_report(keys, proof: Dict) -> Dict:
        """:meth:`verify_with_keys` with the verdict's details.

        Returns ``{"valid": bool, "error": str | None, "verify_seconds"}``;
        ``verify_seconds`` covers the transcript and ``verify_proof`` only.
        Needs only the verifying key (keys from :meth:`setup`, or
        :meth:`load_keys` with either role), the proof and its public values.
        """
        from zinnia._zinnia_core import halo2_base_verify

        return json.loads(halo2_base_verify(keys, proof["proof"], proof["public_values"]))

    def _external_callables(self, externals=None) -> Dict:
        if externals:
            return externals
        ext_dict = {}
        for name, ef in self.external_funcs.items():
            if hasattr(ef, 'callable'):
                ext_dict[name] = ef.callable
            elif callable(ef):
                ext_dict[name] = ef
        return ext_dict

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
            "public_layout": {"outputs": self.get_output_layout(),
                              "inputs": self.get_public_inputs()},
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

        program = ZKCompiledProgram(
            name=payload['name'],
            backend=payload['backend'],
            compiled_ir=compiled_ir,
            program_inputs=_program_inputs,
            external_funcs=ef_map,
        )
        program._restored_layout = payload.get('public_layout') or {}
        return program
