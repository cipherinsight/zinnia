import inspect
import ast
import json
import sys
from typing import Dict, Tuple

import astpretty

from zenopy.ast.ast_transformer import PyZKCircuitASTTransformer, PyZKChipASTTransformer
from zenopy.backend.halo2_builder import Halo2ProgramBuilder
from zenopy.backend.zk_program import ZKProgram
from zenopy.debug.exception import InternalZenoPyException
from zenopy.ir.ir_gen import IRGenerator
from zenopy.internal.chip_object import ChipObject
from zenopy.debug.prettifier import prettify_zk_ast, prettify_ir_stmts, prettify_exception
from zenopy.exec.exec_ctx import ExecutionContext


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
        if name == 'main':
            raise ValueError('Chip name cannot be `main`, please use another name.')
        self.name = name
        self.source = source
        tree = ast.parse(_fix_indentation(self.source))
        try:
            transformer = PyZKChipASTTransformer(self.source, self.name)
            ir_comp_tree = transformer.visit(tree.body[0])
        except InternalZenoPyException as e:
            raise prettify_exception(e)
        self.chip = ChipObject(ir_comp_tree)

    def get_chip(self) -> ChipObject:
        return self.chip

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


class ZKCircuit:
    def __init__(self, name: str, source: str, chips: Dict[str, ZKChip | ChipObject], debug=False):
        if name == 'main':
            raise ValueError('Circuit name cannot be `main`, please use another name.')
        self.name = name
        self.source = source
        self.chips = chips
        self.debug = debug
        self.prog_metadata = None
        self.zk_program = None

    def __call__(self, *args, **kwargs):
        try:
            if self.zk_program is None or self.prog_metadata is None:
                self.compile()
            exec_ctx = ExecutionContext(self.prog_metadata, args, kwargs)
            raise NotImplementedError('Execution not supported yet. Please read the compiled program source instead.')
            # executor = Halo2ZKProgramExecutor(exec_ctx, self.zk_program)
            # executor.exec()
        except InternalZenoPyException as e:
            raise prettify_exception(e)

    @staticmethod
    def from_source(name: str, source: str, chips: Dict[str, ZKChip | ChipObject], debug=False) -> 'ZKCircuit':
        return ZKCircuit(name, source, chips, debug)

    @staticmethod
    def from_method(method, chips: Dict[str, ZKChip | ChipObject], debug=False) -> 'ZKCircuit':
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
        return ZKCircuit(method_name, source_code, chips, debug)

    def compile(self) -> ZKProgram:
        tree = ast.parse(_fix_indentation(self.source))
        try:
            if self.debug:
                print('*' * 20 + ' Original AST ' + '*' * 20, file=sys.stderr)
                print(astpretty.pformat(tree.body[0], show_offsets=True), file=sys.stderr)
            transformer = PyZKCircuitASTTransformer(
                self.source, self.name,
                {key: (chip.get_chip() if isinstance(chip, ZKChip) else chip) for key, chip in self.chips.items()}
            )
            ir_comp_tree = transformer.visit(tree.body[0])
            if self.debug:
                print('*' * 20 + ' Transformed AST ' + '*' * 20, file=sys.stderr)
                print(prettify_zk_ast(ir_comp_tree), file=sys.stderr)
            generator = IRGenerator()
            ir_stmts, self.prog_metadata = generator.generate(ir_comp_tree)
            if self.debug:
                print('*' * 20 + ' IR Statements ' + '*' * 20, file=sys.stderr)
                print(prettify_ir_stmts(ir_stmts), file=sys.stderr)
                print('*' * 20 + ' Program Metadata ' + '*' * 20, file=sys.stderr)
                print(json.dumps(self.prog_metadata.export()), file=sys.stderr)
            self.prog_metadata.set_circuit_name(self.name)
            prog_builder = Halo2ProgramBuilder(ir_stmts, self.prog_metadata)
            self.zk_program = prog_builder.build()
            if self.debug:
                print('*' * 20 + ' Program Source ' + '*' * 20, file=sys.stderr)
                print(self.zk_program.source, file=sys.stderr)
        except InternalZenoPyException as e:
            raise prettify_exception(e)
        return self.zk_program

    def argparse(self, *args, **kwargs) -> Dict[Tuple[int, int], float | int]:
        exec_ctx = ExecutionContext(self.prog_metadata, args, kwargs)
        return exec_ctx.parse_args_kwargs(args, kwargs)


def zk_circuit(method, debug=True):
    def __zk_circuit_annotator_inner(*args, **kwargs):
        source_code = inspect.getsource(method)
        method_name = method.__name__
        defined_chips: Dict[str, ChipObject] = {}
        for key, val in inspect.currentframe().f_back.f_locals.items():
            if isinstance(val, ZKChip):
                defined_chips[key] = val.get_chip()
        circuit = ZKCircuit(method_name, source_code, defined_chips, debug)
        return circuit(*args, **kwargs)
    return __zk_circuit_annotator_inner


def zk_chip(method):
    source_code = inspect.getsource(method)
    method_name = method.__name__
    return ZKChip(method_name, source_code)
