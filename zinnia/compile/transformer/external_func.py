import ast

from zinnia.compile.transformer.base import ZinniaBaseASTTransformer
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import InvalidProgramException, InvalidAnnotationException


class ZinniaExternalFuncASTTransformer(ZinniaBaseASTTransformer):
    def __init__(self, source_code: str, method_name: str):
        super().__init__(source_code, method_name)

    def get_dbg(self, node) -> DebugInfo:
        return DebugInfo(self.method_name, self.source_code, False, node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)

    def visit(self, node):
        if isinstance(node, ast.FunctionDef):
            return self.visit_FunctionDef(node)
        raise InvalidProgramException(None, "Invalid code passed to the compiler! The external function must be a function.")

    def visit_FunctionDef(self, node: ast.FunctionDef):
        dbg = self.get_dbg(node)
        if node.returns is not None:
            return_anno = self.visit_annotation(node.returns, None)
        else:
            raise InvalidAnnotationException(dbg, "External Functions must have a return annotation.")
        if return_anno.kind is not None:
            raise InvalidAnnotationException(self.get_dbg(node.returns), f"Invalid return annotation for external functions. In external functions, the return type should NOT be annotated by `Public` or `Private` because chip returns are not inputs. Please remove these specifiers and leave the corresponding datatype only.")
        return return_anno.dt
