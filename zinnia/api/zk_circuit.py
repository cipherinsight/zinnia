import functools
import inspect
from dataclasses import dataclass
from typing import List, Any, Dict, Tuple

from zinnia import ZKChip, ZKExternalFunc
from zinnia.api.zk_compiled_program import ZKCompiledProgram
from zinnia.compile.lowering import read_decorated_circuit
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
        """A circuit from a ``@zk_circuit`` function or a plain function.

        For a decorated function the circuit uses the source the decorator
        prepared, the chips it lifted, and every chip and external function
        defined in the function's module, after the ``chips`` and
        ``externals`` given here. A plain function is compiled from its
        source as written, with only the given chips and externals.
        """
        decorated = getattr(method, _DECORATED_CIRCUIT, None)
        if decorated is None:
            return ZKCircuit(method.__name__, inspect.getsource(method), chips, externals, config)
        chips, externals = _with_registered_functions(
            list(chips or []), list(externals or []), [decorated.module_globals])
        chips += [ZKChip.from_source(name, source) for name, source in decorated.lifted_chips]
        return ZKCircuit(decorated.name, decorated.source, chips, externals or None, config)

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
            backend: "mock" (default, fast), "halo2-base" or "noir" (real ZK
                proof). The legacy "halo2" backend is retired (an error).
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


@dataclass(frozen=True)
class _DecoratedCircuit:
    """What ``@zk_circuit`` prepared for a function (see `zinnia.compile.lowering`)."""

    name: str
    source: str
    lifted_chips: List[Tuple[str, str]]
    module_globals: Dict[str, Any]


#: Attribute of a ``@zk_circuit`` wrapper that holds its `_DecoratedCircuit`.
_DECORATED_CIRCUIT = "__zinnia_circuit__"


def _with_registered_functions(
        chips: List[Any], externals: List[Any], namespaces: List[Dict[str, Any]]
) -> Tuple[List[Any], List[Any]]:
    """``chips`` and ``externals`` followed by every ``ZKChip`` and
    ``ZKExternalFunc`` value in ``namespaces`` (in order), each object once."""
    seen = {id(f) for f in chips + externals if isinstance(f, (ZKChip, ZKExternalFunc))}
    for namespace in namespaces:
        for value in namespace.values():
            if id(value) in seen:
                continue
            if isinstance(value, ZKChip):
                chips.append(value)
            elif isinstance(value, ZKExternalFunc):
                externals.append(value)
            else:
                continue
            seen.add(id(value))
    return chips, externals


def zk_circuit(method=None, *, config: ZinniaConfig = ZinniaConfig(), requires=None):
    """Declare a circuit. Calling the decorated function runs it on the mock backend.

    Usable bare (``@zk_circuit``) or with options (``@zk_circuit(config=...,
    requires=[...])``). ``requires`` is read from the source by the AST
    transformer (it is the source of truth for preconditions); the argument
    is accepted here so the decorator call itself succeeds.
    """
    if method is None:
        def _factory(m):
            return zk_circuit(m, config=config, requires=requires)
        return _factory
    source, lifted_chips = read_decorated_circuit(method)
    decorated = _DecoratedCircuit(
        name=method.__name__,
        source=source,
        lifted_chips=lifted_chips,
        module_globals=getattr(method, "__globals__", {}) or {},
    )

    def run_on_mock(*args, **kwargs):
        # Chips and externals defined in the module, then those defined
        # where the circuit is called (tests and notebooks define helpers
        # inline), then the chips lifted from the body.
        caller_locals = inspect.currentframe().f_back.f_locals
        chips, externals = _with_registered_functions([], [], [decorated.module_globals, caller_locals])
        chips += [ZKChip.from_source(name, chip_source) for name, chip_source in decorated.lifted_chips]
        return ZKCircuit(decorated.name, decorated.source, chips, externals, config)(*args)

    functools.update_wrapper(run_on_mock, method)
    setattr(run_on_mock, _DECORATED_CIRCUIT, decorated)
    return run_on_mock
