import inspect
from typing import List, Any, Dict

from zinnia import ZKChip, ZKExternalFunc
from zinnia.api.zk_parsed_input import ZKParsedInput
from zinnia.api.zk_compiled_program import ZKCompiledProgram
from zinnia.compile.zinnia_compiler import ZinniaCompiler
from zinnia.config.zinnia_config import ZinniaConfig
from zinnia.debug.exception import InternalZinniaException
from zinnia.debug.prettifier import prettify_exception
from zinnia.exec.exec_result import ZKExecResult
from zinnia.exec.mock_executor import MockProgramExecutor


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
        method_name = method.__name__
        if method_name == '__zk_circuit_annotator_inner':
            _method = None
            for cell in method.__closure__:
                if callable(cell.cell_contents):
                    _method = cell.cell_contents
                    break
            assert _method is not None
            source_code = inspect.getsource(_method)
            method_name = _method.__name__
        else:
            source_code = inspect.getsource(method)
        return ZKCircuit(method_name, source_code, chips, externals, config)

    def compile(self) -> ZKCompiledProgram:
        try:
            self.program = ZinniaCompiler(self.config).compile(
                self.source, self.name,
                {k: v.to_internal_object() for k, v in self.chips.items()},
                {k: v.to_internal_object() for k, v in self.externals.items()},
            )
        except InternalZinniaException as e:
            raise prettify_exception(e)
        return self.program

    def mock(self, *args, **kwargs) -> ZKExecResult:
        if self.program is None:
            self.compile()
        try:
            exec_ctx = self.program.get_execution_context()
            mock_executor = MockProgramExecutor(exec_ctx, self.program, self.config)
            return mock_executor.exec(*args, **kwargs)
        except InternalZinniaException as e:
            raise prettify_exception(e)

    def __call__(self, *args, **kwargs) -> ZKExecResult:
        return self.mock(*args, **kwargs)

    def get_name(self) -> str:
        return self.name

    def argparse(self, *args, **kwargs) -> ZKParsedInput:
        if self.program is None:
            self.compile()
        exec_ctx = self.program.get_execution_context()
        return exec_ctx.argparse(*args, **kwargs)


def zk_circuit(method, config: ZinniaConfig = ZinniaConfig()):
    def __zk_circuit_annotator_inner(*args, **kwargs):
        source_code = inspect.getsource(method)
        method_name = method.__name__
        defined_chips: List[ZKChip] = []
        defined_externals: List[ZKExternalFunc] = []
        for key, val in inspect.currentframe().f_back.f_locals.items():
            if isinstance(val, ZKChip):
                defined_chips.append(val)
            elif isinstance(val, ZKExternalFunc):
                defined_externals.append(val)
        circuit = ZKCircuit(method_name, source_code, defined_chips, defined_externals, config)
        return circuit(*args, **kwargs)
    return __zk_circuit_annotator_inner
