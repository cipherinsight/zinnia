import ast

from zinnia.compile.transformer.base import ZinniaBaseASTTransformer
from zinnia.compile.transformer._precondition import extract_preconditions
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import InvalidProgramException, InvalidCircuitInputException


class ZinniaCircuitASTTransformer(ZinniaBaseASTTransformer):
    def __init__(self, source_code: str, method_name: str):
        super().__init__(source_code, method_name)
        self.program_inputs_data = []

    def get_dbg(self, node) -> DebugInfo:
        # Operator-class AST nodes (ast.BitAnd, ast.NotIn, etc.) have
        # `_attributes = ()` per CPython grammar and do NOT carry source
        # location attributes. Fall back to 0 so that emitting a diagnostic
        # for an unsupported operator doesn't itself crash with AttributeError.
        return DebugInfo(self.method_name, self.source_code, True,
                         getattr(node, 'lineno', 0),
                         getattr(node, 'col_offset', 0),
                         getattr(node, 'end_lineno', 0),
                         getattr(node, 'end_col_offset', 0))

    def visit_FunctionDef(self, node: ast.FunctionDef):
        dbg_info = self.get_dbg(node)
        args = self.visit_arguments(node.args)
        if node.returns is not None:
            raise InvalidProgramException(dbg_info, "Circuit function must not have a return annotation. Note that circuits should not return anything.")
        # Extract preconditions and split into structural vs. scalar
        # buckets. The Rust ASTCircuit has two parallel fields
        # (`requires` for structural-predicate calls, `scalar_requires`
        # for ContractTerm-shaped scalar/arithmetic/logical
        # preconditions); the split is by the spec dict's `__class__`.
        all_preconditions = extract_preconditions(node, self.source_code, self.method_name)
        structural = []
        scalar = []
        for spec in all_preconditions:
            cls = spec.get("__class__")
            if cls == "ASTRequires":
                structural.append(spec)
            elif cls == "ASTScalarRequires":
                scalar.append(spec)
            else:
                # Defensive: unexpected spec class. Fall back to
                # structural to preserve the existing behaviour.
                structural.append(spec)
        return {
            "__class__": "ASTCircuit",
            "block": self.visit_block(node.body),
            "inputs": args,
            "requires": structural,
            "scalar_requires": scalar,
        }

    def visit_arguments(self, node):
        results = []
        for arg in node.args:
            dbg = self.get_dbg(arg)
            name: str = arg.arg
            if arg.annotation is None:
                raise InvalidCircuitInputException(dbg, "Circuit input must be annotated, e.g. `x: Public[Integer]` or `x: Private[Float]` or `x: Integer`.")
            annotation = self.visit_annotation(arg.annotation, name)
            if annotation["kind"] is None:
                annotation["kind"] = "Private"
            # Store full type dict for program inputs extraction
            self.program_inputs_data.append({
                "name": name,
                "dt": annotation["dt"],
                "kind": annotation["kind"],
            })
            results.append({
                "__class__": "ASTCircuitInput",
                "name": name,
                "annotation": {
                    "__class__": "ASTAnnotation",
                    "kind": annotation["kind"],
                    "dt": annotation["dt"],
                }
            })
        return results
