from typing import Tuple, Optional, Any, Dict

from pyzk.debug.exception.transforming import InvalidAnnotationException
from pyzk.debug.dbg_info import DebugInfo


class DTDescriptor(object):
    def __init__(self):
        pass

    def __new__(cls, *args, **kwargs):
        if cls is DTDescriptor:
            raise TypeError(f"<DTDescriptor> must be subclassed.")
        return object.__new__(cls)

    def __str__(self) -> str:
        return self.get_typename()

    def __eq__(self, other) -> bool:
        return self.get_typename() == other.get_typename()

    def export(self) -> Dict[str, Any]:
        return {
            'typename': self.get_typename(),
        }

    @classmethod
    def get_typename(cls):
        raise NotImplementedError()

    @classmethod
    def from_annotation(cls, dbg_i: Optional[DebugInfo], args: Tuple[Any, ...]) -> 'DTDescriptor':
        raise NotImplementedError()


class NDArrayDTDescriptor(DTDescriptor):
    def __init__(self, shape: Tuple[int, ...]):
        super().__init__()
        self.shape = shape

    def __str__(self) -> str:
        return f'{self.get_typename()}[{",".join([str(x) for x in self.shape])}]'

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and self.shape == other.shape

    def export(self) -> Dict[str, Any]:
        result = super().export()
        result['shape'] = list(self.shape)
        return result

    @classmethod
    def get_typename(cls):
        return "NDArray"

    @classmethod
    def from_annotation(cls, dbg_i: Optional[DebugInfo], args: Tuple[DTDescriptor | int, ...]) -> 'NDArrayDTDescriptor':
        if len(args) == 0:
            raise InvalidAnnotationException(dbg_i, "Annotation `NDArray` requires 1 or more arguments")
        if any([not isinstance(arg, int) for arg in args]):
            raise InvalidAnnotationException(dbg_i, "Annotation `NDArray` only accepts integers as dimension sizes")
        return NDArrayDTDescriptor(tuple(arg for arg in args))


class TupleDTDescriptor(DTDescriptor):
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def __str__(self) -> str:
        return f'{self.get_typename()}[{self.length}]'

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and self.length == other.length

    def export(self) -> Dict[str, Any]:
        result = super().export()
        result['length'] = self.length
        return result

    @classmethod
    def get_typename(cls):
        return "Tuple"

    @classmethod
    def from_annotation(cls, dbg_i: Optional[DebugInfo], args: Tuple[DTDescriptor | int, ...]) -> 'TupleDTDescriptor':
        if len(args) != 1:
            raise InvalidAnnotationException(dbg_i, "Annotation `Tuple` requires exactly 1 argument")
        if not isinstance(args[0], int):
            raise InvalidAnnotationException(dbg_i, "Annotation `Tuple` accepts only one integer as length")
        return TupleDTDescriptor(args[0])


class NumberDTDescriptor(DTDescriptor):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_typename(cls):
        return "Number"

    @classmethod
    def from_annotation(cls, dbg_i: Optional[DebugInfo], args: Tuple[DTDescriptor | int, ...]) -> 'NumberDTDescriptor':
        return NumberDTDescriptor()


class NoneDTDescriptor(DTDescriptor):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_typename(cls):
        return "None"

    @classmethod
    def from_annotation(cls, dbg_i: Optional[DebugInfo], args: Tuple[DTDescriptor | int, ...]) -> 'DTDescriptor':
        return NoneDTDescriptor()


class DTDescriptorFactory:
    DATATYPE_REGISTRY = [NDArrayDTDescriptor, TupleDTDescriptor, NumberDTDescriptor, NoneDTDescriptor]

    @staticmethod
    def create(dbg_i: Optional[DebugInfo], typename: str, args: Tuple[DTDescriptor | int, ...] = None) -> DTDescriptor:
        if args is None:
            args = tuple()
        for datatype in DTDescriptorFactory.DATATYPE_REGISTRY:
            if datatype.get_typename() == typename:
                return datatype.from_annotation(dbg_i, args)
        raise InvalidAnnotationException(dbg_i, f'`{typename}` is not a valid type name')

    @staticmethod
    def is_typename(typename: str) -> bool:
        for datatype in DTDescriptorFactory.DATATYPE_REGISTRY:
            if datatype.get_typename() == typename:
                return True
        return False
