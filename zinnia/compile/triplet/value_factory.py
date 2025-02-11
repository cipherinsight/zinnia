from zinnia.compile.triplet.value import Value, ClassValue, FloatValue, IntegerValue, ListValue, NDArrayValue, \
    NoneValue, StringValue, TupleValue
from zinnia.compile.triplet.store import ValueStore


class ValueFactory:
    VALUE_CLASSES = [
        ClassValue, FloatValue, IntegerValue, ListValue, NDArrayValue, NoneValue, StringValue, TupleValue
    ]

    @classmethod
    def from_value_store(cls, value_store: ValueStore, type_locked: bool = False) -> Value:
        for c in cls.VALUE_CLASSES:
            r = c.from_value_store(value_store, type_locked)
            if r is not None:
                return r
        raise NotImplementedError()
