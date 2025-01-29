from typing import Optional, Tuple, Dict

from zinnia.compile.type_sys import PoseidonHashedDTDescriptor
from zinnia.compile.type_sys.clazz import ClassDTDescriptor
from zinnia.compile.type_sys.dt_descriptor import DTDescriptor
from zinnia.compile.type_sys.float import FloatDTDescriptor
from zinnia.compile.type_sys.integer import IntegerDTDescriptor
from zinnia.compile.type_sys.list import ListDTDescriptor
from zinnia.compile.type_sys.ndarray import NDArrayDTDescriptor
from zinnia.compile.type_sys.none import NoneDTDescriptor
from zinnia.compile.type_sys.tuple import TupleDTDescriptor
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import InvalidAnnotationException


class DTDescriptorFactory:
    DATATYPE_REGISTRY = [
        NDArrayDTDescriptor, TupleDTDescriptor, ListDTDescriptor, IntegerDTDescriptor, FloatDTDescriptor,
        NoneDTDescriptor, ClassDTDescriptor, PoseidonHashedDTDescriptor
    ]

    @staticmethod
    def create(dbg_i: Optional[DebugInfo], typename: str, args: Tuple[DTDescriptor | int, ...] = None) -> DTDescriptor:
        if args is None:
            args = tuple()
        for datatype in DTDescriptorFactory.DATATYPE_REGISTRY:
            if typename in datatype.get_alise_typenames():
                return datatype.from_annotation(dbg_i, args)
        raise InvalidAnnotationException(dbg_i, f'`{typename}` is not a valid type name')

    @staticmethod
    def is_typename(typename: str) -> bool:
        for datatype in DTDescriptorFactory.DATATYPE_REGISTRY:
            if datatype.get_typename() == typename:
                return True
        return False

    @staticmethod
    def export(dt: DTDescriptor) -> Dict:
        return {
            "__class__": dt.__class__.__name__,
            "dt_data": dt.export()
        }

    @staticmethod
    def import_from(data: Dict) -> DTDescriptor:
        class_name = data['__class__']
        for datatype in DTDescriptorFactory.DATATYPE_REGISTRY:
            if datatype.__name__ == class_name:
                return datatype.import_from(data['dt_data'])
        raise NotImplementedError(f"Internal Error: DTDescriptorFactory: {class_name} is not implemented")
