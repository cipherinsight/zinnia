from zinnia.compile.type_sys import DTDescriptor, IntegerType, BooleanType
from zinnia.op_def.abstract.abstract_arithemetic import AbstractArithemetic


class AbstractCompare(AbstractArithemetic):
    def __init__(self):
        super().__init__()

    def get_expected_result_dt(self, lhs_dt: DTDescriptor, rhs_dt: DTDescriptor):
        return BooleanType
