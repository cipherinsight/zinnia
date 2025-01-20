from typing import Tuple, Optional, Any, Dict, Type, List

from zenopy.debug.exception.transforming import InvalidAnnotationException
from zenopy.debug.dbg_info import DebugInfo


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
        return self.__class__ == other.__class__ and self.get_typename() == other.get_typename()

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

    def is_t(self, other: Type) -> bool:
        return isinstance(self, other)


class NumberDTDescriptor(DTDescriptor):
    def __init__(self):
        super().__init__()

    def __new__(cls, *args, **kwargs):
        if cls is DTDescriptor:
            raise TypeError(f"<NumberDTDescriptor> must be subclassed.")
        return object.__new__(cls)

    @classmethod
    def get_typename(cls):
        return "Number"

    @classmethod
    def from_annotation(cls, dbg_i: Optional[DebugInfo], args: Tuple[DTDescriptor | int, ...]) -> 'NumberDTDescriptor':
        raise NotImplementedError()


class NDArrayDTDescriptor(DTDescriptor):
    def __init__(self, shape: Tuple[int, ...], dtype: NumberDTDescriptor):
        super().__init__()
        self.shape = shape
        self.dtype = dtype

    def __str__(self) -> str:
        return f'{self.get_typename()}[{self.dtype}, {",".join([str(x) for x in self.shape])}]'

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and self.shape == other.shape and self.dtype == other.dtype

    def export(self) -> Dict[str, Any]:
        result = super().export()
        result['shape'] = list(self.shape)
        result['dtype'] = self.dtype.export()
        return result

    @classmethod
    def get_typename(cls):
        return "NDArray"

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


class TupleDTDescriptor(DTDescriptor):
    def __init__(self, elements_dtype: Tuple[DTDescriptor, ...]):
        super().__init__()
        self.elements_dtype = elements_dtype

    def __str__(self) -> str:
        return f'{self.get_typename()}[{", ".join([str(x) for x in self.elements_dtype])}]'

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and self.elements_dtype == other.elements_dtype

    def export(self) -> Dict[str, Any]:
        result = super().export()
        result['elements'] = [
            x.export() for x in self.elements_dtype
        ]
        return result

    @classmethod
    def get_typename(cls):
        return "Tuple"

    @classmethod
    def from_annotation(cls, dbg_i: Optional[DebugInfo], args: Tuple[DTDescriptor | int, ...]) -> 'TupleDTDescriptor':
        if len(args) == 0:
            raise InvalidAnnotationException(dbg_i, "Annotation `Tuple` requires 1 or more arguments")
        return TupleDTDescriptor(tuple(arg for arg in args))


class ListDTDescriptor(DTDescriptor):
    def __init__(self, elements_dtype: List[DTDescriptor]):
        super().__init__()
        self.elements_dtype = elements_dtype

    def __str__(self) -> str:
        return f'{self.get_typename()}[{", ".join([str(x) for x in self.elements_dtype])}]'

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and self.elements_dtype == other.elements_dtype

    def export(self) -> Dict[str, Any]:
        result = super().export()
        result['elements'] = [
            x.export() for x in self.elements_dtype
        ]
        return result

    @classmethod
    def get_typename(cls):
        return "List"

    @classmethod
    def from_annotation(cls, dbg_i: Optional[DebugInfo], args: List[DTDescriptor | int]) -> 'ListDTDescriptor':
        if len(args) == 0:
            raise InvalidAnnotationException(dbg_i, "Annotation `List` requires 1 or more arguments")
        return ListDTDescriptor([arg for arg in args])


class IntegerDTDescriptor(NumberDTDescriptor):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_typename(cls):
        return "Integer"

    @classmethod
    def from_annotation(cls, dbg_i: Optional[DebugInfo], args: Tuple[DTDescriptor | int, ...]) -> 'IntegerDTDescriptor':
        return IntegerDTDescriptor()


class FloatDTDescriptor(NumberDTDescriptor):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_typename(cls):
        return "Float"

    @classmethod
    def from_annotation(cls, dbg_i: Optional[DebugInfo], args: Tuple[DTDescriptor | int, ...]) -> 'FloatDTDescriptor':
        return FloatDTDescriptor()


class ClassDTDescriptor(DTDescriptor):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_typename(cls):
        return "Class"

    @classmethod
    def from_annotation(cls, dbg_i: Optional[DebugInfo], args: Tuple[DTDescriptor | int, ...]) -> 'ClassDTDescriptor':
        return ClassDTDescriptor()


class NoneDTDescriptor(DTDescriptor):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_typename(cls):
        return "None"

    @classmethod
    def from_annotation(cls, dbg_i: Optional[DebugInfo], args: Tuple[DTDescriptor | int, ...]) -> 'DTDescriptor':
        return NoneDTDescriptor()


class StringDTDescriptor(DTDescriptor):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_typename(cls):
        return "String"

    @classmethod
    def from_annotation(cls, dbg_i: Optional[DebugInfo], args: Tuple[DTDescriptor | int, ...]) -> 'DTDescriptor':
        return StringDTDescriptor()


class HashedDTDescriptor(DTDescriptor):
    def __init__(self, dtype: DTDescriptor):
        super().__init__()
        self.dtype = dtype

    @classmethod
    def get_typename(cls):
        return f"Hashed"

    def __str__(self) -> str:
        return f'{self.get_typename()}[{self.dtype}]'

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and self.dtype == other.dtype

    def export(self) -> Dict[str, Any]:
        result = super().export()
        result['dtype'] = self.dtype.export()
        return result


class DTDescriptorFactory:
    DATATYPE_REGISTRY = [NDArrayDTDescriptor, TupleDTDescriptor, IntegerDTDescriptor, FloatDTDescriptor, NoneDTDescriptor, ClassDTDescriptor]

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


FloatType = FloatDTDescriptor()
IntegerType = IntegerDTDescriptor()
NumberType = NumberDTDescriptor()
NoneType = NoneDTDescriptor()
StringType = StringDTDescriptor()
