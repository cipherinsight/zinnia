import inspect
import ast
import json
import sys
from typing import Dict

import astpretty

from pyzk.ast.ast_transformer import PyZKCircuitASTTransformer, PyZKChipASTTransformer
from pyzk.backend.halo2_builder import Halo2ProgramBuilder
from pyzk.debug.exception import InternalPyzkException
from pyzk.ir.ir_gen import IRGenerator
from pyzk.internal.chip_object import ChipObject
from pyzk.debug.prettifier import prettify_zk_ast, prettify_ir_stmts, prettify_exception
from pyzk.exec.exec_ctx import ExecutionContext
from pyzk.exec.executor import Halo2ZKProgramExecutor


class PyZKChip:
    def __init__(self, name: str, source: str):
        self.name = name
        self.source = source
        tree = ast.parse(self.source)
        try:
            transformer = PyZKChipASTTransformer(self.source, self.name)
            ir_comp_tree = transformer.visit(tree.body[0])
        except InternalPyzkException as e:
            raise prettify_exception(e)
        self.chip = ChipObject(ir_comp_tree)

    def get_chip(self) -> ChipObject:
        return self.chip


class PyZKCircuit:
    def __init__(self, name: str, source: str, chips: Dict[str, PyZKChip | ChipObject], debug=False):
        self.name = name
        self.source = source
        self.chips = chips
        self.debug = debug
        tree = ast.parse(self.source)
        try:
            if debug:
                print('*' * 20 + ' Original AST ' + '*' * 20, file=sys.stderr)
                print(astpretty.pformat(tree.body[0], show_offsets=True), file=sys.stderr)
            transformer = PyZKCircuitASTTransformer(
                self.source, self.name,
                {key: (chip.get_chip() if isinstance(chip, PyZKChip) else chip) for key, chip in self.chips}
            )
            ir_comp_tree = transformer.visit(tree.body[0])
            if debug:
                print('*' * 20 + ' Transformed AST ' + '*' * 20, file=sys.stderr)
                print(prettify_zk_ast(ir_comp_tree), file=sys.stderr)
            generator = IRGenerator()
            ir_stmts, self.prog_metadata = generator.generate(ir_comp_tree)
            if debug:
                print('*' * 20 + ' IR Statements ' + '*' * 20, file=sys.stderr)
                print(prettify_ir_stmts(ir_stmts), file=sys.stderr)
                print('*' * 20 + ' Program Metadata ' + '*' * 20, file=sys.stderr)
                print(json.dumps(self.prog_metadata.export()), file=sys.stderr)
            self.prog_metadata.set_circuit_name(self.name)
            prog_builder = Halo2ProgramBuilder(ir_stmts, self.prog_metadata)
            self.zk_program = prog_builder.build()
            if debug:
                print('*' * 20 + ' Program Source ' + '*' * 20, file=sys.stderr)
                print(self.zk_program.source, file=sys.stderr)
        except InternalPyzkException as e:
            raise prettify_exception(e)

    def __call__(self, *args, **kwargs):
        try:
            exec_ctx = ExecutionContext(self.prog_metadata, args, kwargs)
            executor = Halo2ZKProgramExecutor(exec_ctx, self.zk_program)
            executor.exec()
        except InternalPyzkException as e:
            raise prettify_exception(e)


def pyzk_circuit(method, debug=True):
    def __inner(*args, **kwargs):
        source_code = inspect.getsource(method)
        method_name = method.__name__
        defined_chips: Dict[str, ChipObject] = {}
        for key, val in inspect.currentframe().f_back.f_locals.items():
            if isinstance(val, ChipObject):
                defined_chips[key] = val
        circuit = PyZKCircuit(method_name, source_code, defined_chips, debug)
        return circuit(*args, **kwargs)
    return __inner


def pyzk_chip(method):
    source_code = inspect.getsource(method)
    method_name = method.__name__
    return PyZKChip(method_name, source_code).get_chip()
