import ast

from zinnia.compile.transformer.base import ZinniaBaseASTTransformer
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import InvalidProgramException, InvalidAnnotationException


class ZinniaChipASTTransformer(ZinniaBaseASTTransformer):
    def __init__(self, source_code: str, method_name: str):
        super().__init__(source_code, method_name)
        self.return_dt_full = None

    def get_dbg(self, node) -> DebugInfo:
        # Operator-class AST nodes (ast.BitAnd, ast.NotIn, etc.) have
        # `_attributes = ()` per CPython grammar and do NOT carry source
        # location attributes. Fall back to 0 so that emitting a diagnostic
        # for an unsupported operator doesn't itself crash with AttributeError.
        return DebugInfo(self.method_name, self.source_code, False,
                         getattr(node, 'lineno', 0),
                         getattr(node, 'col_offset', 0),
                         getattr(node, 'end_lineno', 0),
                         getattr(node, 'end_col_offset', 0))

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
        if return_anno["kind"] is not None:
            raise InvalidAnnotationException(self.get_dbg(node.returns),
                                             f"Invalid return annotation for chips. In chips, the return type should NOT be annotated by `Public` or `Private` because chip returns are not inputs. Please remove these specifiers and leave the corresponding datatype only.")
        self.return_dt_full = return_anno["dt"]
        return {"__class__": "ASTChip",
                "block": self.visit_block(node.body), "inputs": args,
                "return_dt": return_anno["dt"]}

    def visit_arguments(self, node):
        results = []
        # ``ast.FunctionDef.args.defaults`` is the list of default values for
        # the *last* N positional args (where N == len(defaults)). Pair them
        # with their args by right-aligning.
        positional = list(node.args)
        defaults = list(node.defaults)
        n_no_default = len(positional) - len(defaults)
        for i, arg in enumerate(positional):
            dbg_info = self.get_dbg(arg)
            name: str = arg.arg
            if arg.annotation is None:
                annotation_dict = None
            else:
                annotation = self.visit_annotation(arg.annotation, name)
                annotation_dict = {
                    "__class__": "ASTAnnotation",
                    "kind": annotation["kind"],
                    "dt": annotation["dt"],
                }
            default_dict = None
            if i >= n_no_default:
                default_node = defaults[i - n_no_default]
                # Defaults are evaluated at decorator time in Python; for
                # zinnia chips we only support literal defaults and unary-negated
                # literals so they can become a constant Value at compile time.
                default_dict = self.visit_expr(default_node)
            results.append({
                "__class__": "ASTChipInput",
                "name": name,
                "annotation": annotation_dict,
                "default": default_dict,
            })
        return results
