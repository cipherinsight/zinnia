from typing import Dict, Tuple, List, Any

from zinnia.lang.type import PoseidonHashed, NDArray, Integer, Float
from zinnia.api.zk_parsed_input import ZKParsedInput
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.config.mock_exec_config import MockExecConfig
from zinnia.api.zk_program_input import ZKProgramInput
from zinnia.internal.internal_external_func_object import InternalExternalFuncObject
from zinnia.internal.internal_ndarray import InternalNDArray
from zinnia.debug.exception.execution import ZKCircuitParameterException
from zinnia.compile.type_sys import NDArrayDTDescriptor, FloatDTDescriptor, DTDescriptor, \
    FloatType, IntegerType, TupleDTDescriptor, ListDTDescriptor, PoseidonHashedDTDescriptor, BooleanType, \
    BooleanDTDescriptor
from zinnia.ir_def.defs.ir_assert import AssertIR
from zinnia.ir_def.defs.ir_export_external_f import ExportExternalFIR
from zinnia.ir_def.defs.ir_export_external_i import ExportExternalIIR
from zinnia.ir_def.defs.ir_invoke_external import InvokeExternalIR
from zinnia.ir_def.defs.ir_read_float import ReadFloatIR
from zinnia.ir_def.defs.ir_read_hash import ReadHashIR
from zinnia.ir_def.defs.ir_read_integer import ReadIntegerIR


class ExecutionContext:
    def __init__(
            self,
            circuit_inputs: List[ZKProgramInput],
            preprocess_stmts: List[IRStatement],
            external_funcs: Dict[str, InternalExternalFuncObject],
    ):
        self.circuit_inputs = circuit_inputs
        self.preprocess_stmts = preprocess_stmts
        self.external_funcs = external_funcs

    def _recursive_infer_datatype(self, value: Any) -> DTDescriptor:
        if isinstance(value, int):
            return IntegerType
        if isinstance(value, bool):
            return IntegerType
        if isinstance(value, float):
            return FloatType
        if isinstance(value, list):
            items_dt = [self._recursive_infer_datatype(x) for x in value]
            return ListDTDescriptor(items_dt)
        if isinstance(value, tuple):
            items_dt = [self._recursive_infer_datatype(x) for x in value]
            return TupleDTDescriptor(tuple(items_dt))
        if isinstance(value, PoseidonHashed):
            inner_dtype = self._recursive_infer_datatype(value.actual_value)
            return PoseidonHashedDTDescriptor(inner_dtype)
        try:
            import numpy as np

            if isinstance(value, np.bool):
                return IntegerType
            if ExecutionContext._is_numpy_integer(value):
                return IntegerType
            if ExecutionContext._is_numpy_float(value):
                return FloatType
            if isinstance(value, np.ndarray):
                shape = value.shape
                dtype = self._recursive_infer_datatype(value.flatten().tolist()[0])
                return NDArrayDTDescriptor(shape, dtype)
        except ImportError:
            pass
        if isinstance(value, NDArray):
            shape = value.shape
            dtype = None
            if value.dtype == Integer:
                dtype = IntegerType
            elif value.dtype == Float:
                dtype = FloatType
            assert dtype is not None
            return NDArrayDTDescriptor(shape, dtype)

        raise ValueError(f'Unrecognizable value as circuit input: {value}')

    def _recursive_verify_datatype_matches(
            self,
            expected: DTDescriptor,
            got: DTDescriptor
    ) -> bool:
        if expected == got:
            return True
        elif expected == FloatType and got == IntegerType:
            return True
        elif expected == IntegerType and got == FloatType:
            return False
        elif expected == BooleanType and got == IntegerType:
            return True
        elif expected == IntegerType and got == BooleanType:
            return True
        elif isinstance(expected, ListDTDescriptor) and isinstance(got, ListDTDescriptor):
            if len(expected.elements_type) != len(got.elements_type):
                return False
            for i in range(len(expected.elements_type)):
                if not self._recursive_verify_datatype_matches(expected.elements_type[i], got.elements_type[i]):
                    return False
            return True
        elif isinstance(expected, TupleDTDescriptor) and isinstance(got, TupleDTDescriptor):
            if len(expected.elements_type) != len(got.elements_type):
                return False
            for i in range(len(expected.elements_type)):
                if not self._recursive_verify_datatype_matches(expected.elements_type[i], got.elements_type[i]):
                    return False
            return True
        elif isinstance(expected, NDArrayDTDescriptor) and isinstance(got, NDArrayDTDescriptor):
            if expected.shape != got.shape:
                return False
            return self._recursive_verify_datatype_matches(expected.dtype, got.dtype)
        elif isinstance(expected, NDArrayDTDescriptor) and isinstance(got, ListDTDescriptor):
            if expected.shape[0] != len(got.elements_type):
                return False
            if len(expected.shape) > 1:
                for element_dtype in got.elements_type:
                    if not self._recursive_verify_datatype_matches(NDArrayDTDescriptor(expected.shape[1:], expected.dtype), element_dtype):
                        return False
            if len(expected.shape) == 1:
                for element_dtype in got.elements_type:
                    if not self._recursive_verify_datatype_matches(expected.dtype, element_dtype):
                        return False
            return True
        elif isinstance(expected, PoseidonHashedDTDescriptor) and isinstance(got, PoseidonHashedDTDescriptor):
            return self._recursive_verify_datatype_matches(expected.dtype, got.dtype)
        return False

    def _recursive_parse_value_to_entries(
            self,
            indices: Tuple[int, ...],
            dt: DTDescriptor,
            value: Any
    ) -> List[ZKParsedInput.Entry]:
        if dt == IntegerType:
            if isinstance(value, int):
                return [ZKParsedInput.Entry(indices, ZKParsedInput.Kind.INTEGER, value)]
            if isinstance(value, bool):
                return [ZKParsedInput.Entry(indices, ZKParsedInput.Kind.INTEGER, 1 if value else 0)]
            try:
                import numpy as np

                if isinstance(value, np.bool):
                    return [ZKParsedInput.Entry(indices, ZKParsedInput.Kind.INTEGER, 1 if value else 0)]
                if ExecutionContext._is_numpy_integer(value):
                    return [ZKParsedInput.Entry(indices, ZKParsedInput.Kind.INTEGER, int(value))]
            except ImportError:
                pass
            raise NotImplementedError()
        elif isinstance(dt, FloatDTDescriptor):
            if isinstance(value, int):
                return [ZKParsedInput.Entry(indices, ZKParsedInput.Kind.FLOAT, float(value))]
            if isinstance(value, bool):
                return [ZKParsedInput.Entry(indices, ZKParsedInput.Kind.FLOAT, 1.0 if value else 0.0)]
            elif isinstance(value, float):
                return [ZKParsedInput.Entry(indices, ZKParsedInput.Kind.FLOAT, value)]
            try:
                import numpy as np

                if isinstance(value, np.bool):
                    return [ZKParsedInput.Entry(indices, ZKParsedInput.Kind.FLOAT, 1.0 if value else 0.0)]
                if ExecutionContext._is_numpy_float(value):
                    return [ZKParsedInput.Entry(indices, ZKParsedInput.Kind.FLOAT, float(value))]
            except ImportError:
                pass
            raise NotImplementedError()
        elif isinstance(dt, BooleanDTDescriptor):
            if isinstance(value, int):
                return [ZKParsedInput.Entry(indices, ZKParsedInput.Kind.INTEGER, 1 if value != 0 else 0)]
            if isinstance(value, bool):
                return [ZKParsedInput.Entry(indices, ZKParsedInput.Kind.INTEGER, 1 if value else 0)]
            try:
                import numpy as np

                if isinstance(value, np.bool):
                    return [ZKParsedInput.Entry(indices, ZKParsedInput.Kind.INTEGER, 1 if value else 0)]
                if ExecutionContext._is_numpy_integer(value):
                    return [ZKParsedInput.Entry(indices, ZKParsedInput.Kind.INTEGER, 1 if value != 0 else 0)]
            except ImportError:
                pass
            raise NotImplementedError()
        elif isinstance(dt, TupleDTDescriptor):
            assert isinstance(value, tuple)
            parsed_result = []
            for i, v in enumerate(value):
                parsed_result.extend(self._recursive_parse_value_to_entries(indices + (i,), dt.elements_type[i], v))
            return parsed_result
        elif isinstance(dt, ListDTDescriptor):
            assert isinstance(value, list)
            parsed_result = []
            for i, v in enumerate(value):
                parsed_result.extend(self._recursive_parse_value_to_entries(indices + (i,), dt.elements_type[i], v))
            return parsed_result
        elif isinstance(dt, NDArrayDTDescriptor):
            ndarray = None
            if isinstance(value, list):
                ndarray = InternalNDArray(dt.shape, value)
            try:
                import numpy as np

                if isinstance(value, np.ndarray):
                    ndarray = InternalNDArray(dt.shape, value.tolist())
            except ImportError:
                pass
            if isinstance(value, NDArray):
                ndarray = value._NDArray__ndarray
            assert ndarray is not None
            flattened_values = ndarray.flatten()
            parsed_result = []
            for i, val in enumerate(flattened_values):
                parsed_result += self._recursive_parse_value_to_entries(indices + (i, ), dt.dtype, val)
            return parsed_result
        elif isinstance(dt, PoseidonHashedDTDescriptor):
            assert isinstance(value, PoseidonHashed)
            parsed_result = [ZKParsedInput.Entry(indices + (1, ), ZKParsedInput.Kind.HASH, value.get_hash())]
            parsed_result += self._recursive_parse_value_to_entries(indices + (0, ), dt.dtype, value.get_value())
            return parsed_result
        raise NotImplementedError(f"Unsupported datatype {dt} for circuit")

    def argparse(self, *args, **kwargs) -> ZKParsedInput:
        inputs: List[ZKProgramInput] = self.circuit_inputs
        arg_dict = {}
        for i, arg in enumerate(args):
            if i >= len(inputs):
                raise ZKCircuitParameterException(None, f'Too many positional argument provided for the circuit')
            arg_dict[inputs[i].name] = arg
        for key, val in kwargs.items():
            if key not in [x.name for x in inputs]:
                raise ZKCircuitParameterException(None, f'Unknown keyword argument {key} in circuit')
            if key in arg_dict:
                raise ZKCircuitParameterException(None, f'Duplicate keyword argument {key} in circuit')
            arg_dict[key] = val
        for inp in inputs:
            if inp.name not in arg_dict:
                raise ZKCircuitParameterException(None, f'Circuit missing required argument {inp.name}')
        parsed_result_input_entries = []
        for i, inp in enumerate(inputs):
            inferred_dtype = self._recursive_infer_datatype(arg_dict[inp.name])
            if not self._recursive_verify_datatype_matches(inp.get_dt(), inferred_dtype):
                raise ZKCircuitParameterException(None, f'Input datatype mismatch for `{inp.name}`. Expected {inp.get_dt()}, got {inferred_dtype}')
            parsed_result_input_entries += self._recursive_parse_value_to_entries((0, i,), inp.get_dt(), arg_dict[inp.name])
        new_inputs = self._execute_external_calls(parsed_result_input_entries)
        return ZKParsedInput(parsed_result_input_entries + new_inputs)

    def _recursive_construct_python_object(
            self,
            exported_values: Dict,
            for_which: int,
            key: int | str,
            indices: Tuple[int, ...],
            dt: DTDescriptor
    ):
        if dt == IntegerType:
            return int(exported_values[(for_which, key, indices)])
        elif dt == FloatType:
            return float(exported_values[(for_which, key, indices)])
        elif isinstance(dt, TupleDTDescriptor):
            elements = []
            for i in range(len(dt.elements_type)):
                elements.append(self._recursive_construct_python_object(exported_values, for_which, key, indices + (i,), dt.elements_type[i]))
            return tuple(elements)
        elif isinstance(dt, ListDTDescriptor):
            elements = []
            for i in range(len(dt.elements_type)):
                elements.append(self._recursive_construct_python_object(exported_values, for_which, key, indices + (i,), dt.elements_type[i]))
            return list(elements)
        elif isinstance(dt, NDArrayDTDescriptor):
            shape = dt.shape
            elements = []
            elements_count = 1
            for s in shape:
                elements_count *= s
            for i in range(elements_count):
                elements.append(self._recursive_construct_python_object(exported_values, for_which, key, indices + (i,), dt.dtype))
            ndarray = InternalNDArray.from_1d_values_and_shape(elements, shape)
            try:
                import numpy as np

                if dt.dtype == IntegerType:
                    return np.asarray(ndarray.values, dtype=int)
                elif dt.dtype == FloatType:
                    return np.asarray(ndarray.values, dtype=float)
                else:
                    raise NotImplementedError()
            except ImportError:
                pass
            return ndarray.values
        raise NotImplementedError()

    def _execute_external_calls(
            self,
            provided_inputs: List[ZKParsedInput.Entry]
    ) -> List[ZKParsedInput.Entry]:
        input_table = {}
        value_table = {}
        new_inputs = []
        exported_values = {}
        for entry in provided_inputs:
            input_table[entry.get_indices()] = entry.get_value()
        for stmt in self.preprocess_stmts:
            if isinstance(stmt.ir_instance, ReadIntegerIR):
                value_table[stmt.stmt_id] = input_table[stmt.ir_instance.indices]
            elif isinstance(stmt.ir_instance, ReadFloatIR):
                value_table[stmt.stmt_id] = input_table[stmt.ir_instance.indices]
            elif isinstance(stmt.ir_instance, ReadHashIR):
                value_table[stmt.stmt_id] = input_table[stmt.ir_instance.indices]
            elif isinstance(stmt.ir_instance, InvokeExternalIR):
                external_func = None
                for name, ef in self.external_funcs.items():
                    if name == stmt.ir_instance.func_name:
                        external_func = ef
                if external_func is None:
                    raise ValueError(f'External call with name {stmt.ir_instance.func_name} not found')
                _callable = external_func.callable
                return_dt = external_func.return_dt
                args, kwargs = [], {}
                for i, dt in enumerate(stmt.ir_instance.args):
                    args.append(self._recursive_construct_python_object(exported_values, stmt.ir_instance.store_idx, i, (), dt))
                for key, dt in stmt.ir_instance.kwargs.items():
                    kwargs[key] = self._recursive_construct_python_object(exported_values, stmt.ir_instance.store_idx, key, (), dt)
                invoke_result = _callable(*args, **kwargs)
                parsed_entries = self._recursive_parse_value_to_entries((stmt.ir_instance.store_idx,), return_dt, invoke_result)
                for entry in parsed_entries:
                    input_table[entry.get_indices()] = entry.get_value()
                new_inputs += parsed_entries
            elif isinstance(stmt.ir_instance, ExportExternalIIR):
                exported_values[(stmt.ir_instance.for_which, stmt.ir_instance.key, stmt.ir_instance.indices)] = value_table[stmt.arguments[0]]
            elif isinstance(stmt.ir_instance, ExportExternalFIR):
                exported_values[(stmt.ir_instance.for_which, stmt.ir_instance.key, stmt.ir_instance.indices)] = value_table[stmt.arguments[0]]
            elif isinstance(stmt.ir_instance, AssertIR):
                pass
            else:
                args = [value_table[x] for x in stmt.arguments]
                value_table[stmt.stmt_id] = stmt.ir_instance.mock_exec(args, MockExecConfig())
        return new_inputs

    @staticmethod
    def _is_numpy_integer(value: Any) -> bool:
        try:
            import numpy as np

            try:
                if isinstance(value, np.int_):
                    return True
            except AttributeError:
                pass
            try:
                if isinstance(value, np.intc):
                    return True
            except AttributeError:
                pass
            try:
                if isinstance(value, np.intp):
                    return True
            except AttributeError:
                pass
            try:
                if isinstance(value, np.int8):
                    return True
            except AttributeError:
                pass
            try:
                if isinstance(value, np.int16):
                    return True
            except AttributeError:
                pass
            try:
                if isinstance(value, np.int32):
                    return True
            except AttributeError:
                pass
            try:
                if isinstance(value, np.int64):
                    return True
            except AttributeError:
                pass
            try:
                if isinstance(value, np.int128):
                    return True
            except AttributeError:
                pass
            try:
                if isinstance(value, np.int256):
                    return True
            except AttributeError:
                pass
            try:
                if isinstance(value, np.uint):
                    return True
            except AttributeError:
                pass
            try:
                if isinstance(value, np.ulong):
                    return True
            except AttributeError:
                pass
            try:
                if isinstance(value, np.uint8):
                    return True
            except AttributeError:
                pass
            try:
                if isinstance(value, np.uintc):
                    return True
            except AttributeError:
                pass
            try:
                if isinstance(value, np.uintp):
                    return True
            except AttributeError:
                pass
            try:
                if isinstance(value, np.uint16):
                    return True
            except AttributeError:
                pass
            try:
                if isinstance(value, np.uint32):
                    return True
            except AttributeError:
                pass
            try:
                if isinstance(value, np.uint64):
                    return True
            except AttributeError:
                pass
            try:
                if isinstance(value, np.uint128):
                    return True
            except AttributeError:
                pass
            try:
                if isinstance(value, np.uint256):
                    return True
            except AttributeError:
                pass
            try:
                if isinstance(value, np.ulonglong):
                    return True
            except AttributeError:
                pass
        except ImportError:
            pass
        return False

    @staticmethod
    def _is_numpy_float(value: Any) -> bool:
        try:
            import numpy as np

            try:
                if isinstance(value, np.float16):
                    return True
            except AttributeError:
                pass
            try:
                if isinstance(value, np.float32):
                    return True
            except AttributeError:
                pass
            try:
                if isinstance(value, np.float64):
                    return True
            except AttributeError:
                pass
            try:
                if isinstance(value, np.float80):
                    return True
            except AttributeError:
                pass
            try:
                if isinstance(value, np.float96):
                    return True
            except AttributeError:
                pass
            try:
                if isinstance(value, np.float128):
                    return True
            except AttributeError:
                pass
            try:
                if isinstance(value, np.float256):
                    return True
            except AttributeError:
                pass
        except ImportError:
            pass
        return False
