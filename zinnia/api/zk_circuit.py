import inspect
import json
from typing import List, Any, Dict

from zinnia import ZKChip, ZKExternalFunc
from zinnia.api.zk_compiled_program import ZKCompiledProgram
from zinnia.compile.zinnia_compiler import ZinniaCompiler
from zinnia.config.zinnia_config import ZinniaConfig
from zinnia.debug.exception import InternalZinniaException
from zinnia.debug.prettifier import prettify_exception


class ZKCircuit:
    name: str
    source: str
    chips: Dict[str, ZKChip]
    externals: Dict[str, ZKExternalFunc]
    config: ZinniaConfig
    program: ZKCompiledProgram | None

    def __init__(
            self,
            name: str,
            source: str,
            chips: List[Any] = None,
            externals: List[Any] = None,
            config: ZinniaConfig = ZinniaConfig()
    ):
        if name == 'main':
            raise ValueError('Circuit name cannot be `main`, please use another name.')
        self.name = name
        self.source = source
        self.chips = ZKCircuit.__parse_chips(chips)
        self.externals = ZKCircuit.__parse_externals(externals)
        self.config = config
        self.program = None

    @staticmethod
    def __parse_chips(chips: List[Any] | None) -> Dict[str, ZKChip]:
        if chips is None:
            return {}
        result = {}
        for chip in chips:
            if isinstance(chip, ZKChip):
                result[chip.name] = chip
            else:
                chip_instance = ZKChip.from_method(chip)
                result[chip_instance.name] = chip_instance
        return result

    @staticmethod
    def __parse_externals(externals: List[Any] | None) -> Dict[str, ZKExternalFunc]:
        if externals is None:
            return {}
        result = {}
        for ext in externals:
            if isinstance(ext, ZKExternalFunc):
                result[ext.name] = ext
            else:
                ext_instance = ZKExternalFunc.from_method(ext)
                result[ext_instance.name] = ext_instance
        return result

    @staticmethod
    def from_source(
            name: str,
            source: str,
            chips: List[Any] = None,
            externals: List[Any] = None,
            config: ZinniaConfig = ZinniaConfig()
    ) -> 'ZKCircuit':
        return ZKCircuit(name, source, chips, externals, config)

    @staticmethod
    def from_method(
            method,
            chips: List[Any] = None,
            externals: List[Any] = None,
            config: ZinniaConfig = ZinniaConfig()
    ) -> 'ZKCircuit':
        if chips is None:
            chips = []
        func_name = method.__name__
        if func_name == '__zk_circuit_annotator_inner':
            # The @zk_circuit decorator captures source_code, method_name,
            # and any lifted nested-def chip sources in its closure.
            _source_code = None
            _method_name = None
            _lifted_chip_sources = None
            _decorated_method = None
            if method.__closure__:
                for cell in method.__closure__:
                    contents = cell.cell_contents
                    if isinstance(contents, str) and '\n' in contents and 'def ' in contents:
                        _source_code = contents
                    elif isinstance(contents, str) and '\n' not in contents:
                        _method_name = contents
                    elif isinstance(contents, list) and contents and \
                            all(isinstance(t, tuple) and len(t) == 2
                                and isinstance(t[0], str) and isinstance(t[1], str)
                                for t in contents):
                        _lifted_chip_sources = contents
                    elif callable(contents) and not isinstance(contents, type):
                        _decorated_method = contents
            if _source_code is not None and _method_name is not None:
                merged_chips = list(chips)
                merged_externals = list(externals) if externals else []
                seen_chip_ids = {id(c) for c in merged_chips if isinstance(c, ZKChip)}
                seen_external_ids = {id(e) for e in merged_externals if isinstance(e, ZKExternalFunc)}
                # Scan the decorated method's module globals — same as
                # `__zk_circuit_annotator_inner` does on direct invocation. This
                # is the path used by the sweep harness via from_method().
                module_globals = (getattr(_decorated_method, "__globals__", {}) or {}) if _decorated_method else {}
                for _, val in module_globals.items():
                    if isinstance(val, ZKChip) and id(val) not in seen_chip_ids:
                        merged_chips.append(val)
                        seen_chip_ids.add(id(val))
                    elif isinstance(val, ZKExternalFunc) and id(val) not in seen_external_ids:
                        merged_externals.append(val)
                        seen_external_ids.add(id(val))
                if _lifted_chip_sources:
                    for chip_name, chip_src in _lifted_chip_sources:
                        merged_chips.append(ZKChip.from_source(chip_name, chip_src))
                return ZKCircuit(_method_name, _source_code, merged_chips, merged_externals or None, config)
            # Fallback: try to find the original callable
            for cell in method.__closure__:
                contents = cell.cell_contents
                if callable(contents) and not isinstance(contents, type):
                    source_code = inspect.getsource(contents)
                    return ZKCircuit(contents.__name__, source_code, chips, externals, config)
            raise ValueError("Could not extract circuit source from decorated function")
        else:
            source_code = inspect.getsource(method)
        return ZKCircuit(func_name, source_code, chips, externals, config)

    def compile(self) -> ZKCompiledProgram:
        from zinnia.debug.exception.base import ZinniaException
        try:
            self.program = ZinniaCompiler(self.config).compile(
                self.source, self.name,
                {k: v.to_internal_object() for k, v in self.chips.items()},
                {k: v.to_internal_object() for k, v in self.externals.items()},
            )
        except InternalZinniaException as e:
            raise prettify_exception(e)
        except (ValueError, RuntimeError) as e:
            # Rust core may raise ValueError/RuntimeError for compilation errors
            raise ZinniaException(str(e))
        except BaseException as e:
            # PyO3 PanicException inherits from BaseException
            if type(e).__name__ == 'PanicException':
                raise ZinniaException(str(e))
            raise
        return self.program

    def __call__(self, *args):
        """Execute the circuit using the mock backend (default).

        Equivalent to `self.prove(*args, backend="mock")`, but returns
        a ZKExecResult for backward compatibility.
        """
        proof_result = self.prove(*args, backend="mock")
        from zinnia.exec.exec_result import ZKExecResult
        satisfied = proof_result.proof_bytes_hex == "mock_satisfied"
        return ZKExecResult(satisfied, proof=proof_result)

    def mock(self, *args):
        """Mock execution (alias for __call__)."""
        return self(*args)

    def prove(self, *args, backend="mock", params=None):
        """Compile (if needed), preprocess, and prove.

        Args:
            *args: Positional arguments matching the circuit inputs.
            backend: "mock" (default, fast) or "halo2" (real ZK proof).
            params: Optional dict with proving parameters.

        Returns:
            ZKProofResult containing the proof artifact.
        """
        if self.program is None:
            self.compile()
        externals_dict = {n: ext.callable for n, ext in self.externals.items()}
        return self.program.prove(*args, backend=backend, params=params,
                                  externals=externals_dict)

    def verify(self, proof_result) -> bool:
        """Verify a proof artifact (backend auto-detected from the artifact).

        Args:
            proof_result: A ZKProofResult from prove().

        Returns:
            True if the proof is valid.
        """
        if self.program is None:
            self.compile()
        return self.program.verify(proof_result)

    def get_name(self) -> str:
        return self.name


def zk_circuit(method, config: ZinniaConfig = ZinniaConfig()):
    from zinnia.compile.module_constants import (
        extract_module_constants, substitute_module_constants,
    )
    from zinnia.compile.nested_def_lift import autolift_nested_defs
    raw_source = inspect.getsource(method)
    module_consts = extract_module_constants(method)
    source_code = substitute_module_constants(raw_source, module_consts)
    module_globals = getattr(method, "__globals__", {}) or {}
    source_code, _lifted_chip_sources = autolift_nested_defs(source_code, module_globals)
    method_name = method.__name__
    # Capture the original method so the closure exposes both its source and
    # its __globals__ — `ZKCircuit.from_method` walks the closure to discover
    # module-level @zk_chip / @zk_external helpers without needing the caller
    # to pass them explicitly.
    _decorated_method = method
    _ = _decorated_method  # ensure the closure cell exists (closures capture by reference only when referenced inside the inner)

    def __zk_circuit_annotator_inner(*args, **kwargs):
        # Reference _decorated_method so it becomes part of the closure
        # `from_method` can find via __closure__.
        _capture_method_ref = _decorated_method
        defined_chips: List[ZKChip] = []
        defined_externals: List[ZKExternalFunc] = []
        seen_chip_ids: set = set()
        seen_external_ids: set = set()
        # Scan the decorated method's module globals first — that's where
        # benchmark sources put their @zk_chip helpers.
        for _, val in module_globals.items():
            if isinstance(val, ZKChip) and id(val) not in seen_chip_ids:
                defined_chips.append(val)
                seen_chip_ids.add(id(val))
            elif isinstance(val, ZKExternalFunc) and id(val) not in seen_external_ids:
                defined_externals.append(val)
                seen_external_ids.add(id(val))
        # Also scan the caller's locals, for tests/notebooks that define
        # helpers inline.
        for _, val in inspect.currentframe().f_back.f_locals.items():
            if isinstance(val, ZKChip) and id(val) not in seen_chip_ids:
                defined_chips.append(val)
                seen_chip_ids.add(id(val))
            elif isinstance(val, ZKExternalFunc) and id(val) not in seen_external_ids:
                defined_externals.append(val)
                seen_external_ids.add(id(val))
        for chip_name, chip_src in _lifted_chip_sources:
            defined_chips.append(ZKChip.from_source(chip_name, chip_src))
        circuit = ZKCircuit(method_name, source_code, defined_chips, defined_externals, config)
        return circuit(*args)
    return __zk_circuit_annotator_inner
