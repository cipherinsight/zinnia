from pyzk.lang.metatype import NDArrayMeta

Number = int

class NDArray(metaclass=NDArrayMeta):
    def __init__(self, *args):
        self.size = list(args)

    @staticmethod
    def all_zeros(*args) -> 'NDArray':
        return NDArray(*args)

    @staticmethod
    def all_ones(*args) -> 'NDArray':
        return NDArray(*args)

    @staticmethod
    def identity(size: int) -> 'NDArray':
        return NDArray(size, size)

    def sum(self, axis: int = None) -> Number:
        raise NotImplementedError("Cannot perform action outside circuit method.")
