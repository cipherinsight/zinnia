from zinnia.lang.type import NDArray


class NamespaceNP:
    @staticmethod
    def asarray(*args, **kwargs) -> NDArray:
        pass

    @staticmethod
    def transpose(*args, **kwargs) -> NDArray:
        pass

    @staticmethod
    def moveaxis(*args, **kwargs) -> NDArray:
        pass
