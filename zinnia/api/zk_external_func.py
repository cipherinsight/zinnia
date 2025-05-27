import inspect
from typing import Callable

from zinnia.debug.exception import InternalZinniaException, ZinniaException
from zinnia.debug.prettifier import prettify_exception
from zinnia.internal.internal_external_func_object import InternalExternalFuncObject


class ZKExternalFunc:
    def __init__(self, name: str, source: str, the_callable: Callable):
        self.name = name
        self.source = source
        try:
            from zinnia.compile.zinnia_compiler import ZinniaCompiler

            return_dt = ZinniaCompiler.external_func_ast_parse(source, name)
        except InternalZinniaException as e:
            raise prettify_exception(e)
        self.callable = the_callable
        self.return_dt = return_dt

    def __call__(self, *args, **kwargs):
        return self.callable(*args, **kwargs)

    @staticmethod
    def from_method(method) -> 'ZKExternalFunc':
        from zinnia.api.zk_chip import ZKChip

        if isinstance(method, ZKExternalFunc):
            return method
        if isinstance(method, ZKChip):
            raise ZinniaException('Cannot convert a ZKChip into ZKExternalFunc.')
        source_code = inspect.getsource(method)
        method_name = method.__name__
        return ZKExternalFunc(method_name, source_code, method)

    def to_internal_object(self) -> InternalExternalFuncObject:
        return InternalExternalFuncObject(self.name, self.callable, self.return_dt)

    def get_name(self) -> str:
        return self.name


def zk_external(method):
    source_code = inspect.getsource(method)
    method_name = method.__name__
    return ZKExternalFunc(method_name, source_code, method)
