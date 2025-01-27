import ast

from zinnia.compile.ast import ASTChip, ASTChipInput
from zinnia.compile.transformer.base import ZinniaBaseASTTransformer
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import InvalidProgramException, InvalidAnnotationException


class ZinniaChipASTTransformer(ZinniaBaseASTTransformer):
    def __init__(self, source_code: str, method_name: str):
        super().__init__(source_code, method_name)

    def get_dbg(self, node) -> DebugInfo:
        return DebugInfo(self.method_name, self.source_code, False, node.lineno, node.col_offset, node.end_lineno,
                         node.end_col_offset)

    def visit(self, node):
        if isinstance(node, ast.FunctionDef):
            return self.visit_FunctionDef(node)
        raise InvalidProgramException(None, "Invalid code passed to the compiler! The chip must be a function.")

    def visit_FunctionDef(self, node: ast.FunctionDef):
        dbg = self.get_dbg(node)
        args = self.visit_arguments(node.args)
        if node.returns is not None:
            return_anno = self.visit_annotation(node.returns, None)
        else:
            raise InvalidAnnotationException(dbg, "Chip must have a return annotation. Please specify the return type as None if it does not return anything.")
        if return_anno.kind is not None:
            raise InvalidAnnotationException(self.get_dbg(node.returns),
                                             f"Invalid return annotation for chips. In chips, the return type should NOT be annotated by `Public` or `Private` because chip returns are not inputs. Please remove these specifiers and leave the corresponding datatype only.")
        return ASTChip(dbg, self.visit_block(node.body), args, return_anno.dt)

    def visit_arguments(self, node):
        results = []
        for arg in node.args:
            dbg_info = self.get_dbg(arg)
            name: str = arg.arg
            if arg.annotation is None:
                annotation = None
            else:
                annotation = self.visit_annotation(arg.annotation, name)
            results.append(ASTChipInput(dbg_info, name, annotation))
        return results
