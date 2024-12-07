import inspect
import ast
import json
from typing import Dict

import astpretty

from pyzk.ast.ast_transformer import PyZKCircuitASTTransformer, PyZKChipASTTransformer
from pyzk.debug.exception import InternalPyzkException
from pyzk.ir.ir_gen import IRGenerator
from pyzk.internal.chip_object import ChipObject
from pyzk.debug.prettifier import prettify_zk_ast, prettify_ir_stmts, prettify_exception


def pyzk_circuit(method):
    def __inner():
        source_code = inspect.getsource(method)
        tree = ast.parse(source_code)
        defined_chips: Dict[str, ChipObject] = {}
        for key, val in inspect.currentframe().f_back.f_locals.items():
            if isinstance(val, ChipObject):
                defined_chips[key] = val
        try:
            with open('./ast-log.txt', 'w') as f:
                f.write(astpretty.pformat(tree.body[0], show_offsets=True))
            transformer = PyZKCircuitASTTransformer(source_code, method.__name__, defined_chips)
            ir_comp_tree = transformer.visit(tree.body[0])
            with open('./ast-transformed.txt', 'w') as f:
                f.write(prettify_zk_ast(ir_comp_tree))
            generator = IRGenerator()
            ir_stmts, prog_ctx = generator.generate(ir_comp_tree)
            with open('./ir-stmts.txt', 'w') as f:
                f.write(prettify_ir_stmts(ir_stmts))
            with open('./prog-ctx.txt', 'w') as f:
                json.dump(prog_ctx.export(), f)
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
