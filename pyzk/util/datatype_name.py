class DataTypeName:
    NUMBER = 'Number'
    NDARRAY = 'NDArray'
    TUPLE = 'Tuple'

    @staticmethod
    def is_datatype_name(name: str) -> bool:
        return name == DataTypeName.NUMBER or name == DataTypeName.NDARRAY or name == DataTypeName.TUPLE
