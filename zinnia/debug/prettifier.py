from typing import List

from zinnia.debug.exception import InternalZinniaException, ZinniaException
from zinnia.compile.ir.ir_stmt import IRStatement


def prettify_ir_stmts(stmts: List[IRStatement]):
    results = []
    for i, stmt in enumerate(stmts):
        s = f'#{stmt.stmt_id}\t{stmt.ir_instance.get_signature()}\t'
        s += f'({", ".join([f"%{arg}" for arg in stmt.arguments])})'
        results.append(s)
    return "\n".join(results)


def prettify_exception(exception: InternalZinniaException) -> ZinniaException:
    if isinstance(exception, InternalZinniaException):
        if exception.dbg_i is None:
            return ZinniaException(f'{type(exception).__name__}: {exception.msg}')
        source_lines = exception.dbg_i.source_code.splitlines()
        error_report_msg = f'  In method "{exception.dbg_i.method_name}", line {exception.dbg_i.lineno}'
        if exception.dbg_i.end_lineno != exception.dbg_i.lineno:
            error_report_msg += f' to {exception.dbg_i.end_lineno}'
        error_report_msg += '\n'
        if exception.dbg_i.end_lineno != exception.dbg_i.lineno:
            for i, line in enumerate(source_lines):
                line_no = i + 1
                if line_no == exception.dbg_i.lineno:
                    error_report_msg += f'    {line}'
                    error_report_msg += '\n'
                    error_report_msg += ' ' * (4 + exception.dbg_i.col_offset)
                    error_report_msg += '^' * (len(line) - exception.dbg_i.col_offset)
                    error_report_msg += '\n'
                elif line_no == exception.dbg_i.end_lineno:
                    error_report_msg += f'    {line}'
                    error_report_msg += '\n'
                    error_report_msg += '    ' + '^' * (len(line) - exception.dbg_i.end_col_offset)
                    error_report_msg += '\n'
                elif exception.dbg_i.lineno < line_no < exception.dbg_i.end_lineno:
                    error_report_msg += f'    {line}'
                    error_report_msg += '\n'
                    error_report_msg += '    ' + '^' * len(line)
                    error_report_msg += '\n'
        else:
            error_report_msg += f'    {source_lines[exception.dbg_i.lineno - 1]}'
            error_report_msg += '\n'
            error_report_msg += ' ' * (4 + exception.dbg_i.col_offset)
            error_report_msg += '^' * (exception.dbg_i.end_col_offset - exception.dbg_i.col_offset)
            error_report_msg += '\n'
        error_report_msg += type(exception).__name__ + ': ' + exception.msg
        return ZinniaException(error_report_msg)
    raise NotImplementedError
