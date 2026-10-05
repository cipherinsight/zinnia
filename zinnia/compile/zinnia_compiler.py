import ast
import json
import time
from typing import Dict

from zinnia.compile.transformer import ZinniaExternalFuncASTTransformer, ZinniaCircuitASTTransformer
from zinnia.compile.transformer.chip import ZinniaChipASTTransformer
from zinnia.compile.transformer.base import _eval_shape_expr, _parse_shape_expr_token
from zinnia.compile._bridge import analyze_shapes, build_candidates, compile_circuit
from zinnia.compile.function_lowering import lower_function_source
from zinnia.config.zinnia_config import ZinniaConfig
from zinnia.api.zk_compiled_program import ZKCompiledProgram
from zinnia.api.zk_program_input import ZKProgramInput
from zinnia.internal.internal_chip_object import InternalChipObject
from zinnia.internal.internal_external_func_object import InternalExternalFuncObject


class ZinniaCompiler:
    def __init__(self, config: ZinniaConfig):
        self.config = config

    def _core_payloads(
            self,
            source: str, name: str,
            chips: Dict[str, InternalChipObject],
            externals: Dict[str, InternalExternalFuncObject]
    ) -> tuple[ZinniaCircuitASTTransformer, dict, dict, dict, dict]:
        """Build the four JSON payloads the Rust core reads.

        Returns ``(transformer, ast_dict, config_dict, chips_dict,
        externals_dict)``; the transformer carries the input side-data that
        :meth:`compile` needs afterwards. Shared by :meth:`compile` and
        :meth:`analyze_shapes` so the two always hand the core the same
        program.
        """
        fixed_source = ZinniaCompiler.fix_source_indentation(source)
        fixed_source, generated_helpers = lower_function_source(fixed_source, name)
        from .counted_loops import normalize_counted_loops
        python_ast = normalize_counted_loops(ast.parse(fixed_source))
        transformer = ZinniaCircuitASTTransformer(fixed_source, name)
        ast_dict = transformer.visit(python_ast.body[0])

        config_dict = {
            "backend": self.config.get_backend(),
            "loop_limit": self.config.loop_limit(),
            "recursion_limit": self.config.recursion_limit(),
            "enable_memory_consistency": (
                self.config.memory_consistency_enabled()
            ),
            "enable_dynamic_ndarray_meta_assertions": (
                self.config.memory_consistency_enabled()
                or _ast_has_bounded_dynamic_ndarray(ast_dict)
            ),
            "optimization": {
                "shortcut_optimization": self.config.optimization_config().shortcut_optimization(),
                "constant_fold": self.config.optimization_config().constant_fold(),
                "dead_code_elimination": self.config.optimization_config().dead_code_elimination(),
                "always_satisfied_elimination": self.config.optimization_config().always_satisfied_elimination(),
                "duplicate_code_elimination": self.config.optimization_config().duplicate_code_elimination(),
            }
        }
        # Phase 0's work budget (`revision/phase0/implementation.md` §8).
        # `ZinniaConfig` has no accessor of its own for it, so it is read
        # through the generic one: a caller raises it with
        # `config.set("phase0_work_budget", n)`. Omitted when unset, so the
        # Rust default applies; `compile_circuit` ignores the key.
        work_budget = self.config.get("phase0_work_budget")
        if work_budget is not None:
            config_dict["phase0_work_budget"] = work_budget
        # Build chips dict: name -> {chip_ast, return_dt}
        chips_dict = {}
        for chip_name, chip in chips.items():
            chips_dict[chip_name] = {
                "chip_ast": chip.chip_ast,
                "return_dt": chip.return_dt,
            }
        for helper_name, helper_source in generated_helpers:
            if helper_name in chips_dict or helper_name in externals:
                raise ValueError(f"Generated helper name collides with registered function: {helper_name}")
            helper_ast, return_dt = self.chip_ast_parse(helper_source, helper_name)
            chips_dict[helper_name] = {"chip_ast": helper_ast, "return_dt": return_dt}
        # Build externals dict: name -> {return_dt}
        externals_dict = {}
        for ext_name, ext in externals.items():
            externals_dict[ext_name] = {
                "return_dt": ext.return_dt,
            }
        return transformer, ast_dict, config_dict, chips_dict, externals_dict

    def analyze_shapes(
            self,
            source: str, name: str,
            chips: Dict[str, InternalChipObject],
            externals: Dict[str, InternalExternalFuncObject],
            pretty: bool = True
    ) -> dict:
        """Run phase 0 (array-level shape analysis) without compiling.

        Returns the parsed JSON of `revision/phase0/export.rs`'s schema, or
        ``{"rejected": {"reason": ..., "detail": ...}}`` when phase 0 refuses
        the program.

        ``pretty`` writes the ``*_pretty`` fields, which are a convenience:
        the term DAG and the SMT-LIB text carry the same information. Turning
        them off makes the record of a large unrolled circuit much smaller.
        """
        _, ast_dict, config_dict, chips_dict, externals_dict = self._core_payloads(
            source, name, chips, externals,
        )
        return json.loads(analyze_shapes(
            json.dumps(ast_dict), json.dumps(config_dict),
            json.dumps(chips_dict), json.dumps(externals_dict),
            pretty,
        ))

    def build_candidates(
            self,
            source: str, name: str,
            chips: Dict[str, InternalChipObject],
            externals: Dict[str, InternalExternalFuncObject],
            query_timeout_ms: int = 3000,
            program_timeout_ms: int = 60000,
    ) -> dict:
        """Run phase 0 and then phase 1 (`revision/phase1/README.md`).

        Returns the parsed JSON of `src/phase1/export.rs`'s schema, or
        ``{"phase": n, "rejected": {"reason": ..., "detail": ...}}`` when one
        of the two phases refuses the program.
        """
        _, ast_dict, config_dict, chips_dict, externals_dict = self._core_payloads(
            source, name, chips, externals,
        )
        return json.loads(build_candidates(
            json.dumps(ast_dict), json.dumps(config_dict),
            json.dumps(chips_dict), json.dumps(externals_dict),
            query_timeout_ms, program_timeout_ms,
        ))

    def compile(
            self,
            source: str, name: str,
            chips: Dict[str, InternalChipObject],
            externals: Dict[str, InternalExternalFuncObject]
    ) -> ZKCompiledProgram:
        circuit_name = name
        time_checkpoint_s = time.time()

        # 1. Parse Python source -> AST dict (no intermediate AST objects)
        transformer, ast_dict, config_dict, chips_dict, externals_dict = self._core_payloads(
            source, name, chips, externals,
        )
        time_transform = time.time() - time_checkpoint_s

        # 2. Compile in Rust: IR generation + all optimization passes in one call.
        #    Returns a CompiledIR handle (opaque, no JSON round-trip).
        time_checkpoint_compile_s = time.time()
        compiled_ir = compile_circuit(
            json.dumps(ast_dict), json.dumps(config_dict),
            json.dumps(chips_dict), json.dumps(externals_dict),
        )
        time_compile = time.time() - time_checkpoint_compile_s
        resolved_inputs = _resolved_program_input_dtypes(ast_dict)

        # 3. Extract program inputs from transformer side-data
        program_inputs = [
            ZKProgramInput(pi["name"], resolved_inputs.get(pi["name"], pi["dt"]), pi["kind"])
            for pi in transformer.program_inputs_data
        ]

        total_time = time.time() - time_checkpoint_s

        return ZKCompiledProgram(
            name=circuit_name,
            backend=self.config.get_backend(),
            compiled_ir=compiled_ir,
            program_inputs=program_inputs,
            external_funcs=externals,
            eval_data={
                'time_transform': time_transform,
                'time_compile': time_compile,
                'total_time': total_time,
            },
        )

    @staticmethod
    def chip_ast_parse(source: str, name: str):
        """Parse chip source code. Returns (ast_dict, return_dt_full)."""
        fixed_source = ZinniaCompiler.fix_source_indentation(source)
        from .counted_loops import normalize_counted_loops
        python_ast = normalize_counted_loops(ast.parse(fixed_source))
        transformer = ZinniaChipASTTransformer(fixed_source, name)
        ast_dict = transformer.visit(python_ast.body[0])
        return ast_dict, transformer.return_dt_full

    @staticmethod
    def external_func_ast_parse(source: str, name: str):
        """Parse external function source code. Returns full type dict."""
        fixed_source = ZinniaCompiler.fix_source_indentation(source)
        from .counted_loops import normalize_counted_loops
        python_ast = normalize_counted_loops(ast.parse(fixed_source))
        return ZinniaExternalFuncASTTransformer(fixed_source, name).visit(python_ast.body[0])

    @staticmethod
    def circuit_ast_parse(source: str, name: str) -> dict:
        """Parse circuit source code. Returns AST dict."""
        fixed_source = ZinniaCompiler.fix_source_indentation(source)
        fixed_source, _ = lower_function_source(fixed_source, name)
        from .counted_loops import normalize_counted_loops
        python_ast = normalize_counted_loops(ast.parse(fixed_source))
        return ZinniaCircuitASTTransformer(fixed_source, name).visit(python_ast.body[0])

    @staticmethod
    def fix_source_indentation(code: str) -> str:
        lines = code.split('\n')
        min_indent = float('inf')
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, indent)
        return '\n'.join([line[min_indent:] for line in lines])


def _resolved_program_input_dtypes(ast_dict: dict) -> Dict[str, object]:
    exact, upper = _shape_facts(ast_dict.get("scalar_requires", []))
    resolved = {}
    for inp in ast_dict.get("inputs", []):
        name = inp.get("name")
        dt = inp.get("annotation", {}).get("dt")
        new_dt = _resolve_symbolic_program_dt(dt, exact, upper)
        if new_dt is not None:
            resolved[name] = new_dt
    return resolved


def _ast_has_bounded_dynamic_ndarray(node) -> bool:
    if isinstance(node, dict):
        if "DynamicNDArray" in node:
            return True
        if "SymbolicNDArray" in node:
            shape = node["SymbolicNDArray"].get("shape", [])
            if any(isinstance(dim, dict) and "upper" in dim for dim in shape):
                return True
        return any(_ast_has_bounded_dynamic_ndarray(value) for value in node.values())
    if isinstance(node, list):
        return any(_ast_has_bounded_dynamic_ndarray(value) for value in node)
    return False


def _shape_facts(scalar_requires: list) -> tuple[Dict[str, int], Dict[str, int]]:
    exact: Dict[str, int] = {}
    upper: Dict[str, int] = {}

    def shape_name(t):
        if isinstance(t, dict) and "Var" in t and isinstance(t["Var"], dict):
            return t["Var"].get("ShapeParam")
        return None

    def lit_int(t):
        if isinstance(t, dict) and "LitInt" in t:
            return int(t["LitInt"])
        return None

    def walk(term):
        if not isinstance(term, dict):
            return
        if "BoolComb" in term:
            for operand in term["BoolComb"].get("operands", []):
                walk(operand)
            return
        if "Cmp" not in term:
            return
        cmp_term = term["Cmp"]
        op = cmp_term.get("op")
        lhs = cmp_term.get("lhs")
        rhs = cmp_term.get("rhs")
        lhs_name = shape_name(lhs)
        rhs_name = shape_name(rhs)
        lhs_int = lit_int(lhs)
        rhs_int = lit_int(rhs)
        if op == "Eq" and lhs_name and rhs_int is not None and rhs_int > 0:
            exact[lhs_name] = rhs_int
        elif op == "Eq" and rhs_name and lhs_int is not None and lhs_int > 0:
            exact[rhs_name] = lhs_int
        elif op == "Le" and lhs_name and rhs_int is not None and rhs_int > 0:
            upper[lhs_name] = min(upper.get(lhs_name, rhs_int), rhs_int)
        elif op == "Ge" and rhs_name and lhs_int is not None and lhs_int > 0:
            upper[rhs_name] = min(upper.get(rhs_name, lhs_int), lhs_int)

    for req in scalar_requires:
        walk(req.get("term"))
    return exact, upper


def _resolve_symbolic_program_dt(dt, exact: Dict[str, int], upper: Dict[str, int]):
    if not isinstance(dt, dict) or "SymbolicNDArray" not in dt:
        return None
    data = dt["SymbolicNDArray"]
    bindings = {**exact, **(data.get("bindings") or {})}
    shape = []
    max_shape = []
    runtime_rank = 0
    total_bound = 1
    all_exact = True
    for dim in data["shape"]:
        if isinstance(dim, int):
            value = dim
            bound = dim
            runtime_rank += 1
        elif isinstance(dim, str):
            try:
                value = _eval_shape_expr(_parse_shape_expr_token(dim), bindings)
            except ValueError:
                value = None
            if value is not None:
                bound = value
            elif dim in upper:
                all_exact = False
                bound = upper[dim]
            else:
                return None
            runtime_rank += 1
        elif isinstance(dim, dict) and "symbol" in dim and "upper" in dim:
            symbol = dim["symbol"]
            bound = int(dim["upper"])
            value = bindings.get(symbol)
            if value is None:
                all_exact = False
            runtime_rank += 1
        else:
            return None
        if value is not None:
            shape.append(int(value))
        max_shape.append(int(bound))
        total_bound *= int(bound)
    if all_exact:
        return {"NDArray": {"shape": shape, "dtype": data["dtype"]}}
    return {
        "DynamicNDArray": {
            "dtype": data["dtype"],
            "max_length": total_bound,
            "max_rank": runtime_rank,
            "max_shape": max_shape,
        }
    }
