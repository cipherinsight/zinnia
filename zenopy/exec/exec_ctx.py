from typing import Dict, Tuple, List

from zenopy.internal.internal_ndarray import InternalNDArray
from zenopy.debug.exception.execution import ZKCircuitParameterException
from zenopy.internal.dt_descriptor import IntegerDTDescriptor, NDArrayDTDescriptor, FloatDTDescriptor
from zenopy.internal.prog_meta_data import ProgramMetadata


class ExecutionContext:
    def __init__(self, prog_metadata: ProgramMetadata, circuit_args, circuit_kwargs):
        self.prog_metadata = prog_metadata
        self.inputs = self.parse_args_kwargs(circuit_args, circuit_kwargs)

    def parse_args_kwargs(self, circuit_args, circuit_kwargs) -> Dict[Tuple[int, int], float | int]:
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
        parsed_result_dict = {}
        for i, inp in enumerate(inputs):
            if isinstance(inp.dt, IntegerDTDescriptor):
                if not isinstance(arg_dict[inp.name], int):
                    raise ZKCircuitParameterException(None, f'Expected integer for argument {inp.name}')
                parsed_result_dict[(i, 0)] = arg_dict[inp.name]
            elif isinstance(inp.dt, FloatDTDescriptor):
                if isinstance(arg_dict[inp.name], int):
                    parsed_result_dict[(i, 0)] = float(arg_dict[inp.name])
                elif not isinstance(arg_dict[inp.name], float):
                    raise ZKCircuitParameterException(None, f'Expected float for argument {inp.name}')
                else:
                    parsed_result_dict[(i, 0)] = arg_dict[inp.name]
            elif isinstance(inp.dt, NDArrayDTDescriptor):
                if not isinstance(arg_dict[inp.name], List):
                    raise ZKCircuitParameterException(None, f'Expected pure Python list for argument {inp.name}')
                try:
                    ndarray = InternalNDArray(inp.dt.shape, arg_dict[inp.name])
                except AssertionError:
                    raise ZKCircuitParameterException(None, f'NDArray shape mismatch for {inp.name}')
                def _dt_verifier(indices, v):
                    if isinstance(inp.dt.dtype, IntegerDTDescriptor):
                        if not isinstance(v, int):
                            raise ZKCircuitParameterException(None, f'Expected integer for argument {inp.name}, position {indices}')
                    elif isinstance(inp.dt.dtype, FloatDTDescriptor):
                        if isinstance(v, int):
                            return float(v)
                        if not isinstance(v, float):
                            raise ZKCircuitParameterException(None, f'Expected float for argument {inp.name}, position {indices}')
                    else:
                        raise ZKCircuitParameterException(None, f'Unsupported NDArray element for argument {inp.name}, position {indices}')
                    return v

                ndarray.for_each(_dt_verifier)
                _current_idx = 0
                def _for_each_iterator(indices, v):
                    nonlocal _current_idx
                    parsed_result_dict[(i, _current_idx)] = v
                    _current_idx += 1
                    return v

                ndarray.for_each(_for_each_iterator)
            else:
                raise ZKCircuitParameterException(None, "Unsupported datatype for circuit")
        return parsed_result_dict
