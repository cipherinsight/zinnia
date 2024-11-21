import inspect
import ast
import json

import astpretty

from pyzk.ast.ast_transformer import PyZKASTTransformer
from pyzk.exception.base import InternalPyzkException
from pyzk.ir.ir_gen import IRGenerator
from pyzk.util.prettifier import prettify_zk_ast, prettify_ir_stmts, prettify_exception


def pyzk_circuit(method):
    def __inner():
        source_code = inspect.getsource(method)
        tree = ast.parse(source_code)
        try:
            with open('./ast-log.txt', 'w') as f:
                f.write(astpretty.pformat(tree.body[0], show_offsets=True))
            transformer = PyZKASTTransformer()
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
            raise prettify_exception(e, method.__name__, source_code)
    return __inner
