import inspect

from zinnia.compile.zinnia_compiler import ZinniaCompiler
from zinnia.debug.exception import InternalZinniaException, ZinniaException
from zinnia.debug.prettifier import prettify_exception
from zinnia.internal.internal_chip_object import InternalChipObject


class ZKChip:
    def __init__(self, name: str, source: str):
        self.name = name
        self.source = source
        try:
            ast_tree = ZinniaCompiler.chip_ast_parse(source, name)
        except InternalZinniaException as e:
            raise prettify_exception(e)
        self.ast_tree = ast_tree

    @staticmethod
    def from_source(name: str, source: str) -> 'ZKChip':
        return ZKChip(name, source)

    @staticmethod
    def from_method(method) -> 'ZKChip':
        from zinnia.api.zk_external_func import ZKExternalFunc

        if isinstance(method, ZKChip):
            return method
        if isinstance(method, ZKExternalFunc):
            raise ZinniaException('Cannot convert a ZKExternalFunc into ZKChip.')
        source_code = inspect.getsource(method)
        method_name = method.__name__
        return ZKChip(method_name, source_code)

    def __call__(self, *args, **kwargs):
        raise ZinniaException('ZK Chip is not callable outside of a circuit.')

    def to_internal_object(self) -> InternalChipObject:
        return InternalChipObject(self.name, self.ast_tree, self.ast_tree.return_dt)

    def get_name(self) -> str:
        return self.name


def zk_chip(method):
    source_code = inspect.getsource(method)
    method_name = method.__name__
    return ZKChip(method_name, source_code)
