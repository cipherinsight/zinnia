from typing import Any, Dict, List, Optional, Tuple

from zinnia.compile.type_sys.dt_descriptor import DTDescriptor
from zinnia.compile.type_sys.float import FloatDTDescriptor
from zinnia.compile.type_sys.integer import IntegerDTDescriptor
from zinnia.compile.type_sys.ndarray import NDArrayDTDescriptor
from zinnia.compile.type_sys.number import NumberDTDescriptor
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import InvalidAnnotationException


class DynamicNDArrayDTDescriptor(NDArrayDTDescriptor):
    def __init__(self, dtype: NumberDTDescriptor, max_length: int, max_rank: int):
        # Dynamic arrays are modeled as bounded flat storage with runtime metadata.
        super().__init__((max_length,), dtype)
        self.max_length = max_length
        self.max_rank = max_rank

    def __str__(self) -> str:
        return f"{self.get_typename()}[{self.dtype}, {self.max_length}, {self.max_rank}]"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, DynamicNDArrayDTDescriptor)
            and self.dtype == other.dtype
            and self.max_length == other.max_length
            and self.max_rank == other.max_rank
        )

    @classmethod
    def get_typename(cls) -> str:
        return "DynamicNDArray"

    @classmethod
    def get_alise_typenames(cls) -> List[str]:
        return ["DynamicNDArray"]

    @classmethod
    def from_annotation(
        cls, dbg_i: Optional[DebugInfo], args: Tuple[DTDescriptor | int, ...]
    ) -> "DynamicNDArrayDTDescriptor":
        if len(args) != 3:
            raise InvalidAnnotationException(
                dbg_i,
                "Annotation `DynamicNDArray` requires exactly 3 arguments: dtype, max_length, max_rank",
            )
        dtype_arg, max_length_arg, max_rank_arg = args
        if not isinstance(dtype_arg, DTDescriptor):
            raise InvalidAnnotationException(
                dbg_i,
                "Annotation `DynamicNDArray` missing required dtype argument",
            )
        if not isinstance(dtype_arg, (FloatDTDescriptor, IntegerDTDescriptor)):
            raise InvalidAnnotationException(dbg_i, f"Unsupported `DynamicNDArray` dtype `{dtype_arg}`")
        if not isinstance(max_length_arg, int) or max_length_arg <= 0:
            raise InvalidAnnotationException(
                dbg_i,
                "Annotation `DynamicNDArray` requires `max_length` to be a positive integer",
            )
        if not isinstance(max_rank_arg, int) or max_rank_arg <= 0:
            raise InvalidAnnotationException(
                dbg_i,
                "Annotation `DynamicNDArray` requires `max_rank` to be a positive integer",
            )
        return DynamicNDArrayDTDescriptor(dtype_arg, max_length_arg, max_rank_arg)

    def export(self) -> Dict[str, Any]:
        from zinnia.compile.type_sys.dt_factory import DTDescriptorFactory

        return {
            "dtype": DTDescriptorFactory.export(self.dtype),
            "max_length": self.max_length,
            "max_rank": self.max_rank,
        }

    @staticmethod
    def import_from(data: Dict) -> "DynamicNDArrayDTDescriptor":
        from zinnia.compile.type_sys.dt_factory import DTDescriptorFactory

        dtype = DTDescriptorFactory.import_from(data["dtype"])
        assert isinstance(dtype, NumberDTDescriptor)
        return DynamicNDArrayDTDescriptor(
            dtype=dtype,
            max_length=data["max_length"],
            max_rank=data["max_rank"],
        )