from typing import Dict, Tuple, List, Any

from zenopy.compile.ir_stmt import IRStatement
from zenopy.config.mock_exec_config import MockExecConfig
from zenopy.internal.external_call import ExternalCall
from zenopy.internal.external_func_obj import ExternalFuncObj
from zenopy.internal.internal_ndarray import InternalNDArray
from zenopy.debug.exception.execution import ZKCircuitParameterException
from zenopy.internal.dt_descriptor import IntegerDTDescriptor, NDArrayDTDescriptor, FloatDTDescriptor, DTDescriptor, \
    FloatType, IntegerType, TupleDTDescriptor, ListDTDescriptor
from zenopy.internal.prog_meta_data import ProgramMetadata
from zenopy.opdef.ir_op.ir_assert import AssertIR
from zenopy.opdef.ir_op.ir_export_external_f import ExportExternalFIR
from zenopy.opdef.ir_op.ir_export_external_i import ExportExternalIIR
from zenopy.opdef.ir_op.ir_invoke_external import InvokeExternalIR
from zenopy.opdef.ir_op.ir_read_float import ReadFloatIR
from zenopy.opdef.ir_op.ir_read_integer import ReadIntegerIR


class ExecutionContext:
    def __init__(self, prog_metadata: ProgramMetadata, preprocess_stmts: List[IRStatement], external_funcs: Dict[str, ExternalFuncObj], external_calls: List[ExternalCall], circuit_args, circuit_kwargs):
        self.prog_metadata = prog_metadata
        self.external_funcs = external_funcs
        self.external_calls = external_calls
        self.inputs = self.argparse(circuit_args, circuit_kwargs)
        self.inputs = self.preprocess_externals(self.inputs, preprocess_stmts)

    def _inner_parse(self, dt: DTDescriptor, value: Any, indices: Tuple[int, ...], name: str) -> List[Tuple[Tuple[int, ...], Any]]:
        if isinstance(dt, IntegerDTDescriptor):
            if not isinstance(value, int):
                raise ZKCircuitParameterException(None, f'Input datatype mismatch for {name}')
            return [(indices, value)]
        elif isinstance(dt, FloatDTDescriptor):
            if isinstance(value, int):
                return [(indices, float(value))]
            elif isinstance(value, float):
                return [(indices, value)]
            else:
                raise ZKCircuitParameterException(None, f'Input datatype mismatch for {name}')
        elif isinstance(dt, TupleDTDescriptor):
            if not isinstance(value, tuple):
                raise ZKCircuitParameterException(None, f'Input datatype mismatch for {name}')
            parsed_result = []
            for i, v in enumerate(value):
                parsed_result.extend(self._inner_parse(dt.elements_dtype[i], v, indices + (i, ), name))
            return parsed_result
        elif isinstance(dt, ListDTDescriptor):
            if not isinstance(value, list):
                raise ZKCircuitParameterException(None, f'Input datatype mismatch for {name}')
            parsed_result = []
            for i, v in enumerate(value):
                parsed_result.extend(self._inner_parse(dt.elements_dtype[i], v, indices + (i, ), name))
            return parsed_result
        elif isinstance(dt, NDArrayDTDescriptor):
            if not isinstance(value, List):
                raise ZKCircuitParameterException(None, f'Input datatype mismatch for {name}')
            if not InternalNDArray.check_list_shape_matches(dt.shape, value):
                raise ZKCircuitParameterException(None, f'Input datatype mismatch for {name}')
            ndarray = InternalNDArray(dt.shape, value)
            dtype = dt.dtype
            def _dt_verifier(_, v):
                if dtype == IntegerType:
                    if not isinstance(v, int):
                        raise ZKCircuitParameterException(None, f'Input datatype mismatch for {name}')
                    return v
                elif dtype == FloatType:
                    if isinstance(v, int):
                        return float(v)
                    if isinstance(v, float):
                        return v
                    raise ZKCircuitParameterException(None, f'Input datatype mismatch for {name}')
                return v

            ndarray.for_each(_dt_verifier)
            _current_idx = 0
            parsed_result = []

            def _for_each_iterator(_, v):
                nonlocal _current_idx, parsed_result
                parsed_result.extend(self._inner_parse(dtype, v, indices + (_current_idx,), name))
                _current_idx += 1
                return v

            ndarray.for_each(_for_each_iterator)
            return parsed_result
        else:
            raise ZKCircuitParameterException(None, "Unsupported datatype for circuit")

    def argparse(self, circuit_args, circuit_kwargs) -> Dict[Tuple[int, ...], float | int]:
        inputs = self.prog_metadata.inputs
        arg_dict = {}
        for i, arg in enumerate(circuit_args):
            if i > len(inputs):
                raise ZKCircuitParameterException(None, f'Too many positional argument provided for the circuit')
            arg_dict[inputs[i].name] = arg
        for key, val in circuit_kwargs.items():
            if key not in [x.name for x in inputs]:
                raise ZKCircuitParameterException(None, f'Unknown keyword argument {key} in circuit')
            if key in arg_dict:
                raise ZKCircuitParameterException(None, f'Duplicate keyword argument {key} in circuit')
            arg_dict[key] = val
        for inp in inputs:
            if inp.name not in arg_dict:
                raise ZKCircuitParameterException(None, f'Circuit missing argument {inp.name}')
        parsed_result_items = []
        for i, inp in enumerate(inputs):
            parsed_result_items += self._inner_parse(inp.dt, arg_dict[inp.name], (0, i, ), inp.name)
        return {k: v for k, v in parsed_result_items}

    def _find_external_call(self, store_idx: int) -> ExternalCall:
        for call in self.external_calls:
            if call.call_id == store_idx:
                return call
        raise ValueError(f'Internal Error: External call with store index {store_idx} not found')

    def _generate_arg(self, exported_args: Dict, for_which: int, key: int | str, indices: Tuple[int, ...], dt: DTDescriptor):
        if dt == IntegerType:
            return exported_args[(for_which, key, indices)]
        elif dt == FloatType:
            return exported_args[(for_which, key, indices)]
        elif isinstance(dt, TupleDTDescriptor):
            elements = []
            for i in range(len(dt.elements_dtype)):
                elements.append(self._generate_arg(exported_args, for_which, key, indices + (i, ), dt.elements_dtype[i]))
            return tuple(elements)
        elif isinstance(dt, ListDTDescriptor):
            elements = []
            for i in range(len(dt.elements_dtype)):
                elements.append(self._generate_arg(exported_args, for_which, key, indices + (i, ), dt.elements_dtype[i]))
            return list(elements)
        elif isinstance(dt, NDArrayDTDescriptor):
            shape = dt.shape
            elements = []
            elements_count = 1
            for s in shape:
                elements_count *= s
            for i in range(elements_count):
                elements.append(self._generate_arg(exported_args, for_which, key, indices + (i, ), dt.dtype))
            ndarray = InternalNDArray.from_1d_values_and_shape(elements, shape)
            return ndarray.values
        raise NotImplementedError()

    def preprocess_externals(self, provided_inputs: Dict[Tuple[int, ...], float | int], stmts: List[IRStatement]) -> Dict[Tuple[int, ...], float | int]:
        value_table = {}
        exported_args = {}
        for stmt in stmts:
            if isinstance(stmt.operator, ReadIntegerIR):
                value_table[stmt.stmt_id] = provided_inputs[stmt.operator.indices]
            elif isinstance(stmt.operator, ReadFloatIR):
                value_table[stmt.stmt_id] = provided_inputs[stmt.operator.indices]
            elif isinstance(stmt.operator, InvokeExternalIR):
                external_call = self._find_external_call(stmt.operator.store_idx)
                _callable = self.external_funcs[external_call.method_name].callable
                return_dt = self.external_funcs[external_call.method_name].return_dt
                args, kwargs = [], {}
                for i, dt in enumerate(external_call.args):
                    args.append(self._generate_arg(exported_args, external_call.call_id, i, (), dt))
                for key, dt in external_call.kwargs.items():
                    kwargs[key] = self._generate_arg(exported_args, external_call.call_id, key, (), dt)
                invoke_result = _callable(*args, **kwargs)
                parsed_result = self._inner_parse(return_dt, invoke_result, (stmt.operator.store_idx, ), f'external_{stmt.operator.store_idx}')
                for k, v in parsed_result:
                    provided_inputs[k] = v
            elif isinstance(stmt.operator, ExportExternalIIR):
                exported_args[(stmt.operator.for_which, stmt.operator.key, stmt.operator.indices)] = value_table[stmt.arguments[0]]
            elif isinstance(stmt.operator, ExportExternalFIR):
                exported_args[(stmt.operator.for_which, stmt.operator.key, stmt.operator.indices)] = value_table[stmt.arguments[0]]
            elif isinstance(stmt.operator, AssertIR):
                pass
            else:
                args = [value_table[x] for x in stmt.arguments]
                value_table[stmt.stmt_id] = stmt.operator.mock_exec(stmt.operator.argparse(None, args, {}), MockExecConfig())
        return provided_inputs
