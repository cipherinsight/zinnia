from zinnia.compile.type_sys import DTDescriptor, IntegerType
from zinnia.opdef.abstract.abstract_arithemetic import AbstractArithemetic


class AbstractCompare(AbstractArithemetic):
    def __init__(self):
        super().__init__()

    def get_expected_result_dt(self, lhs_dt: DTDescriptor, rhs_dt: DTDescriptor):
        return IntegerType
