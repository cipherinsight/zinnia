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

def pyzk_circuit(method, debug=True):
    def __inner(*args, **kwargs):
        source_code = inspect.getsource(method)
        tree = ast.parse(source_code)
        defined_chips: Dict[str, ChipObject] = {}
        for key, val in inspect.currentframe().f_back.f_locals.items():
            if isinstance(val, ChipObject):
                defined_chips[key] = val
        try:
            if debug:
                print('*' * 20 + ' Original AST ' + '*' * 20, file=sys.stderr)
                print(astpretty.pformat(tree.body[0], show_offsets=True), file=sys.stderr)
            transformer = PyZKCircuitASTTransformer(source_code, method.__name__, defined_chips)
            ir_comp_tree = transformer.visit(tree.body[0])
            if debug:
                print('*' * 20 + ' Transformed AST ' + '*' * 20, file=sys.stderr)
                print(prettify_zk_ast(ir_comp_tree), file=sys.stderr)
            generator = IRGenerator()
            ir_stmts, prog_metadata = generator.generate(ir_comp_tree)
            if debug:
                print('*' * 20 + ' IR Statements ' + '*' * 20, file=sys.stderr)
                print(prettify_ir_stmts(ir_stmts), file=sys.stderr)
                print('*' * 20 + ' Program Metadata ' + '*' * 20, file=sys.stderr)
                print(json.dumps(prog_metadata.export()), file=sys.stderr)
            prog_metadata.set_circuit_name(method.__name__)
            prog_builder = Halo2ProgramBuilder(ir_stmts, prog_metadata)
            zk_program = prog_builder.build()
            if debug:
                print('*' * 20 + ' Program Source ' + '*' * 20, file=sys.stderr)
                print(zk_program.source, file=sys.stderr)
            exec_ctx = ExecutionContext(prog_metadata, args, kwargs)
            executor = Halo2ZKProgramExecutor(exec_ctx, zk_program)
            executor.exec()
        except InternalPyzkException as e:
            raise prettify_exception(e)
    return __inner


def pyzk_chip(method):
    source_code = inspect.getsource(method)
    tree = ast.parse(source_code)
    try:
        transformer = PyZKChipASTTransformer(source_code, method.__name__)
        ir_comp_tree = transformer.visit(tree.body[0])
    except InternalPyzkException as e:
        raise prettify_exception(e)
    return ChipObject(ir_comp_tree)
