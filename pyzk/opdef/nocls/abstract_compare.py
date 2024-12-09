from pyzk.internal.dt_descriptor import IntegerDTDescriptor, DTDescriptor
from pyzk.opdef.nocls.abstract_arithemetic import AbstractArithemetic


class AbstractCompare(AbstractArithemetic):
    def __init__(self):
        super().__init__()

    def get_expected_result_dt(self, lhs_dt: DTDescriptor, rhs_dt: DTDescriptor):
        return IntegerDTDescriptor()
