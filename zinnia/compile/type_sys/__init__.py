from zinnia.compile.type_sys.clazz import ClassDTDescriptor
from zinnia.compile.type_sys.dt_descriptor import DTDescriptor
from zinnia.compile.type_sys.float import FloatDTDescriptor
from zinnia.compile.type_sys.integer import IntegerDTDescriptor
from zinnia.compile.type_sys.list import ListDTDescriptor
from zinnia.compile.type_sys.ndarray import NDArrayDTDescriptor
from zinnia.compile.type_sys.none import NoneDTDescriptor
from zinnia.compile.type_sys.tuple import TupleDTDescriptor
from zinnia.compile.type_sys.hashed import PoseidonHashedDTDescriptor
from zinnia.compile.type_sys.number import NumberDTDescriptor
from zinnia.compile.type_sys.string import StringDTDescriptor
from zinnia.compile.type_sys.dt_factory import DTDescriptorFactory

FloatType = FloatDTDescriptor()
IntegerType = IntegerDTDescriptor()
NumberType = NumberDTDescriptor()
NoneType = NoneDTDescriptor()
StringType = StringDTDescriptor()
