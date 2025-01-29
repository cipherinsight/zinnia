from typing import Tuple, Any, Dict, Optional, List

from zinnia.compile.type_sys.dt_descriptor import DTDescriptor
from zinnia.compile.type_sys.float import FloatDTDescriptor
from zinnia.compile.type_sys.integer import IntegerDTDescriptor
from zinnia.compile.type_sys.number import NumberDTDescriptor
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import InvalidAnnotationException


class NDArrayDTDescriptor(DTDescriptor):
    def __init__(self, shape: Tuple[int, ...], dtype: NumberDTDescriptor):
        super().__init__()
        self.shape = shape
        self.dtype = dtype

    def __str__(self) -> str:
        return f'{self.get_typename()}[{self.dtype}, {",".join([str(x) for x in self.shape])}]'

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and self.shape == other.shape and self.dtype == other.dtype

    @classmethod
    def get_typename(cls) -> str:
        return "NDArray"

    @classmethod
    def get_alise_typenames(cls) -> List[str]:
        return ["NDArray"]

    @classmethod
    def from_annotation(cls, dbg_i: Optional[DebugInfo], args: Tuple[DTDescriptor | int, ...]) -> 'NDArrayDTDescriptor':
        if len(args) <= 1:
            raise InvalidAnnotationException(dbg_i, "Annotation `NDArray` requires 2 or more arguments")
        if isinstance(args[0], DTDescriptor):
            if isinstance(args[0], FloatDTDescriptor) or isinstance(args[0], IntegerDTDescriptor):
                dtype = args[0]
            else:
                raise InvalidAnnotationException(dbg_i, f"Unsupported `NDArray` dtype `{args[0]}`")
            args = args[1:]
        else:
            raise InvalidAnnotationException(dbg_i, f"Annotation `NDArray` missing a required argument dtype")
        if any([not isinstance(arg, int) for arg in args]):
            raise InvalidAnnotationException(dbg_i, "Annotation `NDArray` only accepts integers as dimension sizes")
        return NDArrayDTDescriptor(tuple(arg for arg in args), dtype)

    def get_number_of_elements(self) -> int:
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    def export(self) -> Dict[str, Any]:
        from zinnia.compile.type_sys.dt_factory import DTDescriptorFactory

        return {
            "dtype": DTDescriptorFactory.export(self.dtype),
            "shape": list(self.shape)
        }

    @staticmethod
    def import_from(data: Dict) -> 'NDArrayDTDescriptor':
        from zinnia.compile.type_sys.dt_factory import DTDescriptorFactory

        dtype = DTDescriptorFactory.import_from(data['dtype'])
        assert isinstance(dtype, NumberDTDescriptor)
        return NDArrayDTDescriptor(tuple(data['shape']), dtype)
