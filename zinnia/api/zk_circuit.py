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
            # The @zk_circuit decorator captures source_code and method_name
            # in its closure. Extract them.
            _source_code = None
            _method_name = None
            if method.__closure__:
                for cell in method.__closure__:
                    contents = cell.cell_contents
                    if isinstance(contents, str) and '\n' in contents and 'def ' in contents:
                        _source_code = contents
                    elif isinstance(contents, str) and '\n' not in contents:
                        _method_name = contents
            if _source_code is not None and _method_name is not None:
                return ZKCircuit(_method_name, _source_code, chips, externals, config)
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
        from zinnia.exec.exec_result import ZKExecResult
        from zinnia.exec.input_parser import parse_inputs
        from zinnia.compile._bridge import mock_execute

        if self.program is None:
            self.compile()

        entries = parse_inputs(self.program.program_inputs, args)
        externals_dict = {n: ext.callable for n, ext in self.externals.items()}

        result_json = mock_execute(
            self.program.zk_program_irs_json,
            self.program.preprocess_irs_json,
            json.dumps(entries),
            externals_dict,
        )
        result = json.loads(result_json)
        return ZKExecResult(result["satisfied"], result.get("public_outputs"))

    def mock(self, *args):
        return self(*args)

    def get_name(self) -> str:
        return self.name


def zk_circuit(method, config: ZinniaConfig = ZinniaConfig()):
    source_code = inspect.getsource(method)
    method_name = method.__name__

    def __zk_circuit_annotator_inner(*args, **kwargs):
        defined_chips: List[ZKChip] = []
        defined_externals: List[ZKExternalFunc] = []
        for key, val in inspect.currentframe().f_back.f_locals.items():
            if isinstance(val, ZKChip):
                defined_chips.append(val)
            elif isinstance(val, ZKExternalFunc):
                defined_externals.append(val)
        circuit = ZKCircuit(method_name, source_code, defined_chips, defined_externals, config)
        return circuit(*args)
    return __zk_circuit_annotator_inner
