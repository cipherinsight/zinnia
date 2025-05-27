import ast
from typing import Dict, List

from zinnia.compile.backend.circom_builder import CircomProgramBuilder
from zinnia.compile.backend.halo2_builder import Halo2ProgramBuilder
from zinnia.compile.ast import ASTChip, ASTCircuit
from zinnia.compile.ir.ir_gen import IRGenerator
from zinnia.compile.ir.ir_graph import IRGraph
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.compile.optim_pass.always_satisfied_elimination import AlwaysSatisfiedEliminationIRPass
from zinnia.compile.optim_pass.constant_fold import ConstantFoldIRPass
from zinnia.compile.optim_pass.dead_code_elimination import DeadCodeEliminationIRPass
from zinnia.compile.optim_pass.duplicate_code_elimination import DuplicateCodeEliminationIRPass
from zinnia.compile.optim_pass.external_call_remover import ExternalCallRemoverIRPass
from zinnia.compile.optim_pass.shortcut_optimization_pass import ShortcutOptimIRPass
from zinnia.compile.transformer import ZinniaExternalFuncASTTransformer, ZinniaCircuitASTTransformer
from zinnia.compile.transformer.chip import ZinniaChipASTTransformer
from zinnia.compile.type_sys.dt_descriptor import DTDescriptor
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
        fixed_source = ZinniaCompiler.fix_source_indentation(source)
        python_ast = ast.parse(ZinniaCompiler.fix_source_indentation(fixed_source))
        transformer = ZinniaCircuitASTTransformer(ZinniaCompiler.fix_source_indentation(fixed_source), name)
        ast_tree: ASTCircuit = transformer.visit(python_ast.body[0])
        generator = IRGenerator(self.config)
        ir_graph = generator.generate(ast_tree, chips, externals)
        zk_program_ir = self.run_passes_for_zk_program(ir_graph)
        preprocess_ir = self.run_passes_for_input_preprocess(ir_graph)
        if self.config.get_backend() == ZinniaConfig.BACKEND_HALO2:
            prog_builder = Halo2ProgramBuilder(name, zk_program_ir)
        elif self.config.get_backend() == ZinniaConfig.BACKEND_CIRCOM:
            prog_builder = CircomProgramBuilder(name, zk_program_ir)
        else:
            raise NotImplementedError(f"Backend {self.config.get_backend()} is not supported.")
        compiled_source = prog_builder.build()
        program_inputs = []
        for inp in ast_tree.inputs:
            program_inputs.append(ZKProgramInput(inp.name, inp.annotation.dt, inp.annotation.kind))
        return ZKCompiledProgram(name, compiled_source, self.config.get_backend(), preprocess_ir, zk_program_ir, program_inputs, externals)

    def run_passes_for_zk_program(self, ir_graph: IRGraph) -> List[IRStatement]:
        ir_graph = ExternalCallRemoverIRPass().exec(ir_graph)
        if self.config.optimization_config().shortcut_optimization():
            ir_graph = ShortcutOptimIRPass().exec(ir_graph)
        if self.config.optimization_config().constant_fold():
            ir_graph = ConstantFoldIRPass().exec(ir_graph)
        if self.config.optimization_config().dead_code_elimination():
            ir_graph = DeadCodeEliminationIRPass().exec(ir_graph)
        if self.config.optimization_config().always_satisfied_elimination():
            ir_graph = AlwaysSatisfiedEliminationIRPass().exec(ir_graph)
        if self.config.optimization_config().duplicate_code_elimination():
            ir_graph = DuplicateCodeEliminationIRPass().exec(ir_graph)
        return ir_graph.export_stmts()

    def run_passes_for_input_preprocess(self, ir_graph: IRGraph) -> List[IRStatement]:
        if self.config.optimization_config().shortcut_optimization():
            ir_graph = ShortcutOptimIRPass().exec(ir_graph)
        if self.config.optimization_config().constant_fold():
            ir_graph = ConstantFoldIRPass().exec(ir_graph)
        if self.config.optimization_config().dead_code_elimination():
            ir_graph = DeadCodeEliminationIRPass().exec(ir_graph)
        if self.config.optimization_config().always_satisfied_elimination():
            ir_graph = AlwaysSatisfiedEliminationIRPass().exec(ir_graph)
        if self.config.optimization_config().duplicate_code_elimination():
            ir_graph = DuplicateCodeEliminationIRPass().exec(ir_graph)
        return ir_graph.export_stmts()

    @staticmethod
    def chip_ast_parse(source: str, name: str) -> ASTChip:
        fixed_source = ZinniaCompiler.fix_source_indentation(source)
        python_ast = ast.parse(fixed_source)
        return ZinniaChipASTTransformer(fixed_source, name).visit(python_ast.body[0])

    @staticmethod
    def external_func_ast_parse(source: str, name: str) -> DTDescriptor:
        fixed_source = ZinniaCompiler.fix_source_indentation(source)
        python_ast = ast.parse(fixed_source)
        return ZinniaExternalFuncASTTransformer(fixed_source, name).visit(python_ast.body[0])

    @staticmethod
    def circuit_ast_parse(source: str, name: str) -> ASTCircuit:
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
