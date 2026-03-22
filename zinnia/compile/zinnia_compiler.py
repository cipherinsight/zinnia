import ast
import json
import time
from typing import Dict

from zinnia.compile.transformer import ZinniaExternalFuncASTTransformer, ZinniaCircuitASTTransformer
from zinnia.compile.transformer.chip import ZinniaChipASTTransformer
from zinnia.compile._bridge import compile_circuit
from zinnia.config.zinnia_config import ZinniaConfig
from zinnia.api.zk_compiled_program import ZKCompiledProgram
from zinnia.api.zk_program_input import ZKProgramInput
from zinnia.internal.internal_chip_object import InternalChipObject
from zinnia.internal.internal_external_func_object import InternalExternalFuncObject


class ZinniaCompiler:
    def __init__(self, config: ZinniaConfig):
        self.config = config

    def compile(
            self,
            source: str, name: str,
            chips: Dict[str, InternalChipObject],
            externals: Dict[str, InternalExternalFuncObject]
    ) -> ZKCompiledProgram:
        time_checkpoint_s = time.time()

        # 1. Parse Python source -> AST dict (no intermediate AST objects)
        fixed_source = ZinniaCompiler.fix_source_indentation(source)
        python_ast = ast.parse(fixed_source)
        transformer = ZinniaCircuitASTTransformer(fixed_source, name)
        ast_dict = transformer.visit(python_ast.body[0])
        time_transform = time.time() - time_checkpoint_s

        # 2. Compile in Rust: IR generation + all optimization passes in one call
        time_checkpoint_compile_s = time.time()
        config_dict = {
            "backend": self.config.get_backend(),
            "loop_limit": self.config.loop_limit(),
            "recursion_limit": self.config.recursion_limit(),
            "enable_memory_consistency": self.config.memory_consistency_enabled(),
            "optimization": {
                "shortcut_optimization": self.config.optimization_config().shortcut_optimization(),
                "constant_fold": self.config.optimization_config().constant_fold(),
                "dead_code_elimination": self.config.optimization_config().dead_code_elimination(),
                "always_satisfied_elimination": self.config.optimization_config().always_satisfied_elimination(),
                "duplicate_code_elimination": self.config.optimization_config().duplicate_code_elimination(),
            }
        }
        # Build chips dict: name -> {chip_ast, return_dt}
        chips_dict = {}
        for name, chip in chips.items():
            chips_dict[name] = {
                "chip_ast": chip.chip_ast,
                "return_dt": chip.return_dt,
            }
        # Build externals dict: name -> {return_dt}
        externals_dict = {}
        for name, ext in externals.items():
            externals_dict[name] = {
                "return_dt": ext.return_dt,
            }
        result_json = compile_circuit(
            json.dumps(ast_dict), json.dumps(config_dict),
            json.dumps(chips_dict), json.dumps(externals_dict),
        )
        result = json.loads(result_json)
        time_compile = time.time() - time_checkpoint_compile_s

        # 3. Extract program inputs from transformer side-data
        program_inputs = [
            ZKProgramInput(pi["name"], pi["dt"], pi["kind"])
            for pi in transformer.program_inputs_data
        ]

        total_time = time.time() - time_checkpoint_s

        return ZKCompiledProgram(
            name=name,
            backend=self.config.get_backend(),
            zk_program_irs_json=json.dumps(result["zk_program_irs"]),
            preprocess_irs_json=json.dumps(result["preprocess_irs"]),
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
        python_ast = ast.parse(fixed_source)
        transformer = ZinniaChipASTTransformer(fixed_source, name)
        ast_dict = transformer.visit(python_ast.body[0])
        return ast_dict, transformer.return_dt_full

    @staticmethod
    def external_func_ast_parse(source: str, name: str):
        """Parse external function source code. Returns full type dict."""
        fixed_source = ZinniaCompiler.fix_source_indentation(source)
        python_ast = ast.parse(fixed_source)
        return ZinniaExternalFuncASTTransformer(fixed_source, name).visit(python_ast.body[0])

    @staticmethod
    def circuit_ast_parse(source: str, name: str) -> dict:
        """Parse circuit source code. Returns AST dict."""
        fixed_source = ZinniaCompiler.fix_source_indentation(source)
        python_ast = ast.parse(fixed_source)
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
