import ast

from zinnia.compile.ast import ASTCircuit, ASTCircuitInput, ASTAnnotation
from zinnia.compile.transformer.base import ZinniaBaseASTTransformer
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import InvalidProgramException, InvalidCircuitInputException


class ZinniaCircuitASTTransformer(ZinniaBaseASTTransformer):
    def __init__(self, source_code: str, method_name: str):
        super().__init__(source_code, method_name)

    def get_dbg(self, node) -> DebugInfo:
        return DebugInfo(self.method_name, self.source_code, True, node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        dbg_info = self.get_dbg(node)
        args = self.visit_arguments(node.args)
        if node.returns is not None:
            raise InvalidProgramException(dbg_info, "Circuit function must not have a return annotation. Note that circuits should not return anything.")
        return ASTCircuit(dbg_info, self.visit_block(node.body), args)

    def visit_arguments(self, node):
        results = []
        for arg in node.args:
            dbg = self.get_dbg(arg)
            name: str = arg.arg
            if arg.annotation is None:
                raise InvalidCircuitInputException(dbg, "Circuit input must be annotated, e.g. `x: Public[Integer]` or `x: Private[Float]` or `x: Integer`.")
            annotation = self.visit_annotation(arg.annotation, name)
            if annotation.kind is None:
                annotation.kind = ASTAnnotation.Kind.PRIVATE
            results.append(ASTCircuitInput(dbg, name, annotation))
        return results
