import inspect
import ast
import sys
from typing import Dict, Tuple, Callable, Any, List

import astpretty

from zenopy.compile.transformer.ast_transformer import ZenoPyCircuitASTTransformer, ZenoPyChipASTTransformer, ZenoPyExternalFuncASTTransformer
from zenopy.internal.zk_ast_tree import ZKAbstractSyntaxTree
from zenopy.backend.halo2_builder import Halo2ProgramBuilder
from zenopy.backend.zk_program import ZKProgram
from zenopy.debug.exception import InternalZenoPyException
from zenopy.compile.ir_gen import IRGenerator
from zenopy.debug.prettifier import prettify_ir_stmts, prettify_exception
from zenopy.exec.exec_ctx import ExecutionContext
from zenopy.exec.mock_executor import MockProgramExecutor
from zenopy.exec.exec_result import ZKExecResult
from zenopy.internal.external_func_obj import ExternalFuncObj


def _fix_indentation(code: str) -> str:
    lines = code.split('\n')
    min_indent = float('inf')
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)
    return '\n'.join([line[min_indent:] for line in lines])


class ZKChip:
    def __init__(self, name: str, source: str):
        self.name = name
        self.source = source
        tree = ast.parse(_fix_indentation(self.source))
        try:
            transformer = ZenoPyChipASTTransformer(self.source, self.name)
            ir_comp_tree = transformer.visit(tree.body[0])
        except InternalZenoPyException as e:
            raise prettify_exception(e)
        self.ast_tree = ir_comp_tree

    @staticmethod
    def from_source(name: str, source: str) -> 'ZKChip':
        return ZKChip(name, source)

    @staticmethod
    def from_method(method) -> 'ZKChip':
        if isinstance(method, ZKChip):
            return method
        source_code = inspect.getsource(method)
        method_name = method.__name__
        return ZKChip(method_name, source_code)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError('ZK Chip is not callable outside of a circuit.')


class ZKExternalFunc:
    def __init__(self, name: str, source: str, the_callable: Callable):
        self.name = name
        self.source = source
        tree = ast.parse(_fix_indentation(self.source))
        try:
            transformer = ZenoPyExternalFuncASTTransformer(self.source, self.name)
            return_dt = transformer.visit(tree.body[0])
        except InternalZenoPyException as e:
            raise prettify_exception(e)
        self.callable = the_callable
        self.return_dt = return_dt

    def __call__(self, *args, **kwargs):
        return self.callable(*args, **kwargs)

    def get_external_function(self) -> ExternalFuncObj:
        return ExternalFuncObj(self.name, self.callable, self.return_dt)

    @staticmethod
    def from_method(method) -> 'ZKExternalFunc':
        if isinstance(method, ZKExternalFunc):
            return method
        source_code = inspect.getsource(method)
        method_name = method.__name__
        return ZKExternalFunc(method_name, source_code, method)


class ZKCircuit:
    def __init__(self, name: str, source: str, chips: List[Any] = None, externals: List[Any] = None, debug=False):
        if name == 'main':
            raise ValueError('Circuit name cannot be `main`, please use another name.')
        self.name = name
        self.source = source
        self.chips = ZKCircuit.__parse_chips(chips)
        self.externals = ZKCircuit.__parse_externals(externals)
        self.debug = debug
        self.prog_metadata = None
        self.zk_program = None
        self.ir_stmts = []
        self.preprocess_stmts = []

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
    def from_source(name: str, source: str, chips: List[Any] = None, externals: List[Any] = None, debug: bool = False) -> 'ZKCircuit':
        return ZKCircuit(name, source, chips, externals, debug)

    @staticmethod
    def from_method(method, chips: List[Any] = None, externals: List[Any] = None, debug: bool = False) -> 'ZKCircuit':
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
        return ZKCircuit(method_name, source_code, chips, externals, debug)

    def compile(self) -> ZKProgram:
        tree = ast.parse(_fix_indentation(self.source))
        try:
            if self.debug:
                print('*' * 20 + ' Original AST ' + '*' * 20, file=sys.stderr)
                print(astpretty.pformat(tree.body[0], show_offsets=True), file=sys.stderr)
            transformer = ZenoPyCircuitASTTransformer(self.source, self.name)
            ir_comp_tree = transformer.visit(tree.body[0])
            generator = IRGenerator()
            self.ir_stmts, self.preprocess_stmts, self.external_calls, self.prog_metadata = generator.generate(ZKAbstractSyntaxTree(
                ir_comp_tree,
                {key: chip.ast_tree for key, chip in self.chips.items()},
                {key: ext.get_external_function() for key, ext in self.externals.items()}
            ))
            if self.debug:
                print('*' * 20 + ' IR Statements ' + '*' * 20, file=sys.stderr)
                print(prettify_ir_stmts(self.ir_stmts), file=sys.stderr)
            self.prog_metadata.set_circuit_name(self.name)
            prog_builder = Halo2ProgramBuilder(self.ir_stmts, self.prog_metadata)
            self.zk_program = prog_builder.build()
            if self.debug:
                print('*' * 20 + ' Program Source ' + '*' * 20, file=sys.stderr)
                print(self.zk_program.source, file=sys.stderr)
        except InternalZenoPyException as e:
            raise prettify_exception(e)
        return self.zk_program

    def mock(self, *args, **kwargs) -> ZKExecResult:
        if self.zk_program is None or self.prog_metadata is None:
            self.compile()
        try:
            exec_ctx = ExecutionContext(self.prog_metadata, self.preprocess_stmts, {key: ext.get_external_function() for key, ext in self.externals.items()}, self.external_calls, args, kwargs)
            mock_executor = MockProgramExecutor(exec_ctx, self.zk_program, self.ir_stmts)
            return mock_executor.exec()
        except InternalZenoPyException as e:
            raise prettify_exception(e)

    def __call__(self, *args, **kwargs) -> ZKExecResult:
        return self.mock(*args, **kwargs)

    def argparse(self, *args, **kwargs) -> Dict[Tuple[int, ...], float | int]:
        exec_ctx = ExecutionContext(self.prog_metadata, self.preprocess_stmts, {key: ext.get_external_function() for key, ext in self.externals.items()}, self.external_calls, args, kwargs)
        return exec_ctx.inputs


def zk_circuit(method, debug=True):
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
        circuit = ZKCircuit(method_name, source_code, defined_chips, defined_externals, debug)
        return circuit(*args, **kwargs)
    return __zk_circuit_annotator_inner


def zk_chip(method):
    source_code = inspect.getsource(method)
    method_name = method.__name__
    return ZKChip(method_name, source_code)


def zk_external(method):
    source_code = inspect.getsource(method)
    method_name = method.__name__
    return ZKExternalFunc(method_name, source_code, method)
