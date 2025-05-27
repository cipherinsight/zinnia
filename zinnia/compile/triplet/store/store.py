class ValueStore:
    def __copy__(self):
        raise NotImplementedError()

    def __deepcopy__(self, memo):
        return self.__copy__()

    def assign(self, value: 'ValueStore') -> 'ValueStore':
        raise NotImplementedError()
