import json
from typing import List, Dict

from zinnia.debug.exception import ZinniaException
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.exec.exec_ctx import ExecutionContext
from zinnia.api.zk_external_func import ZKExternalFunc
from zinnia.api.zk_program_input import ZKProgramInput
from zinnia.internal.internal_external_func_object import InternalExternalFuncObject


class ZKCompiledProgram:
    def __init__(
        self,
        name: str,
        source: str,
        backend: str,
        preprocess_irs: List[IRStatement],
        zk_program_irs: List[IRStatement],
        program_inputs: List[ZKProgramInput],
        external_funcs: Dict[str, InternalExternalFuncObject],
        eval_data: Dict = None,
    ):
        self.name = name
        self.source = source
        self.backend = backend
        self.preprocess_irs = preprocess_irs
        self.zk_program_irs = zk_program_irs
        self.program_inputs = program_inputs
        self.external_funcs = external_funcs
        self.eval_data = eval_data
        for key, ef in external_funcs.items():
            assert ef.name == key

    def get_program_name(self) -> str:
        return self.name

    def get_compiled_source(self) -> str:
        return self.source

    def get_target_backend_name(self) -> str:
        return self.backend

    def get_execution_context(self) -> ExecutionContext:
        return ExecutionContext(self.program_inputs, self.preprocess_irs, self.external_funcs)

    def argparse(self, *args, **kwargs):
        return self.get_execution_context().argparse(*args, **kwargs)

    @staticmethod
    def deserialize(data: str, external_funcs: List[ZKExternalFunc] = None) -> 'ZKCompiledProgram':
        if external_funcs is None:
            external_funcs = []
        payload = json.loads(data)
        _name = payload['name']
        _source = payload['source']
        _backend = payload['backend']
        _preprocess_irs = [IRStatement.import_from(ir) for ir in payload['preprocess_irs']]
        _zk_program_irs = [IRStatement.import_from(ir) for ir in payload['zk_program_irs']]
        _program_inputs = [ZKProgramInput.import_from(pi) for pi in payload['program_inputs']]
        _external_funcs = payload['external_funcs']
        _provided_external_funcs = []
        for ef in external_funcs:
            if not ef.name in _external_funcs:
                raise ZinniaException(f'External function {ef.name} provided, but not expected.')
            _provided_external_funcs.append(ef.name)
        for ef in _external_funcs:
            if ef not in _provided_external_funcs:
                raise ZinniaException(f'External function {ef} expected, but not provided.')
        return ZKCompiledProgram(
            name=_name,
            source=_source,
            backend=_backend,
            preprocess_irs=_preprocess_irs,
            zk_program_irs=_zk_program_irs,
            program_inputs=_program_inputs,
            external_funcs={ef.name: ef for ef in external_funcs},
        )

    def serialize(self) -> str:
        return json.dumps({
            "name": self.name,
            "source": self.source,
            "backend": self.backend,
            "preprocess_irs": [ir.export() for ir in self.preprocess_irs],
            "zk_program_irs": [ir.export() for ir in self.zk_program_irs],
            "program_inputs": [pi.export() for pi in self.program_inputs],
            "external_funcs": [ef.name for ef in self.external_funcs.values()],
            "eval_data": self.eval_data
        })
