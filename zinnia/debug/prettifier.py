from typing import List

from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import InternalZinniaException, ZinniaException
from zinnia.compile.ir.ir_stmt import IRStatement


def prettify_ir_stmts(stmts: List[IRStatement]):
    results = []
    for i, stmt in enumerate(stmts):
        s = f'#{stmt.stmt_id}\t{stmt.ir_instance.get_signature()}\t'
        s += f'({", ".join([f"%{arg}" for arg in stmt.arguments])})'
        results.append(s)
    return "\n".join(results)


def prettify_debug_info(dbg: DebugInfo):
    source_lines = dbg.source_code.splitlines()
    error_report_msg = f'  In method "{dbg.method_name}", line {dbg.lineno}'
    if dbg.end_lineno != dbg.lineno:
        error_report_msg += f' to {dbg.end_lineno}'
    error_report_msg += '\n'
    if dbg.end_lineno != dbg.lineno:
        for i, line in enumerate(source_lines):
            line_no = i + 1
            if line_no == dbg.lineno:
                error_report_msg += f'    {line}'
                error_report_msg += '\n'
                error_report_msg += ' ' * (4 + dbg.col_offset)
                error_report_msg += '^' * (len(line) - dbg.col_offset)
                error_report_msg += '\n'
            elif line_no == dbg.end_lineno:
                error_report_msg += f'    {line}'
                error_report_msg += '\n'
                error_report_msg += '    ' + '^' * (len(line) - dbg.end_col_offset)
                error_report_msg += '\n'
            elif dbg.lineno < line_no < dbg.end_lineno:
                error_report_msg += f'    {line}'
                error_report_msg += '\n'
                error_report_msg += '    ' + '^' * len(line)
                error_report_msg += '\n'
    else:
        error_report_msg += f'    {source_lines[dbg.lineno - 1]}'
        error_report_msg += '\n'
        error_report_msg += ' ' * (4 + dbg.col_offset)
        error_report_msg += '^' * (dbg.end_col_offset - dbg.col_offset)
        error_report_msg += '\n'
    return error_report_msg


def prettify_exception(exception: InternalZinniaException) -> ZinniaException:
    if isinstance(exception, InternalZinniaException):
        if exception.dbg_i is None:
            return ZinniaException(f'{type(exception).__name__}: {exception.msg}')
        error_report_msg = prettify_debug_info(exception.dbg_i)
        error_report_msg += type(exception).__name__ + ': ' + exception.msg
        return ZinniaException(error_report_msg)
    raise NotImplementedError
