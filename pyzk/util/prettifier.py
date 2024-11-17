from typing import List

from pyzk.ast.zk_ast import ASTComponent, ASTAssignStatement, ASTSlicingAssignStatement, \
    ASTAssertStatement, ASTOperator, ASTConstant, ASTSlicing, ASTLoad, ASTSlicingData, ASTPassStatement, \
    ASTProgram, ASTProgramInput, ASTAnnotation, ASTForInStatement, ASTCondStatement, \
    ASTExpression, ASTCreateNDArray, ASTSlicingAssignData, ASTContinueStatement, ASTBreakStatement
from pyzk.exception.base import InternalPyzkException, PyZKException
from pyzk.ir.ir_builder import IRStatement
from pyzk.util.op_name import OpName


def prettify_zk_ast(node: ASTComponent) -> str:
    def _inner(depth: int, n: ASTComponent | None, name: str = '') -> List[str]:
        prefix = ''
        if depth == 1:
            prefix = '* '
        elif depth > 1:
            prefix = ' ' * 2 * (depth - 1) + '* '
        prefix += f'[{name}:({type(n).__name__})]'
        prefix = str(prefix)
        if n is None:
            res = [prefix]
            return res
        elif isinstance(n, ASTSlicingAssignStatement):
            res = [prefix + f' -> {n.assignee}']
            res += _inner(depth + 1, n.slicing, 'slicing')
            if n.annotation is not None:
                res += _inner(depth + 1, n.annotation, 'annotation')
            res += _inner(depth + 1, n.value, 'value')
            return res
        elif isinstance(n, ASTAssignStatement):
            res = [prefix + f' -> {n.assignee}']
            if n.annotation is not None:
                res += _inner(depth + 1, n.annotation, 'annotation')
            res += _inner(depth + 1, n.value, 'value')
            return res
        elif isinstance(n, ASTAssertStatement):
            res = [prefix]
            res += _inner(depth + 1, n.expr, 'expr')
            return res
        elif isinstance(n, ASTOperator):
            res = [prefix + f' &{n.op}']
            for i, arg in enumerate(n.args):
                res += _inner(depth + 1, arg, f'arg_{i + 1}')
            return res
        elif isinstance(n, ASTConstant):
            res = [prefix + f' = {n.value}']
            return res
        elif isinstance(n, ASTSlicing):
            res = [prefix]
            res += _inner(depth + 1, n.val, 'value')
            res += _inner(depth + 1, n.slicing, 'slicing')
            return res
        elif isinstance(n, ASTLoad):
            res = [prefix + f' <- {n.name}']
            return res
        elif isinstance(n, ASTSlicingData):
            res = [prefix]
            for i, slicing_data in enumerate(n.data):
                if isinstance(slicing_data, ASTExpression):
                    res += _inner(depth + 1, slicing_data, f'slicing_integer')
                else:
                    res += _inner(depth + 1, slicing_data[0], f'slicing_{i + 1}_l')
                    res += _inner(depth + 1, slicing_data[1], f'slicing_{i + 1}_r')
            return res
        elif isinstance(n, ASTSlicingAssignData):
            res = [prefix]
            assert n.data is not None
            for i, sli in enumerate(n.data):
                res += _inner(depth + 1, sli, f'slicing_{i + 1}')
            return res
        elif isinstance(n, ASTPassStatement):
            res = [prefix]
            return res
        elif isinstance(n, ASTContinueStatement):
            res = [prefix]
            return res
        elif isinstance(n, ASTBreakStatement):
            res = [prefix]
            return res
        elif isinstance(n, ASTProgram):
            res = [prefix]
            for i, inp in enumerate(n.inputs):
                res += _inner(depth + 1, inp, f'input_{i + 1}')
            for stmt in n.block:
                res += _inner(depth + 1, stmt)
            return res
        elif isinstance(n, ASTProgramInput):
            res = [prefix + f' -> {n.name} ({"Public" if n.public else "Private"})']
            if n.annotation is not None:
                res += _inner(depth + 1, n.annotation, 'annotation')
            return res
        elif isinstance(n, ASTAnnotation):
            res = [prefix + f' {n.typename}[{", ".join([str(x) for x in n.shape])}] ({"public" if n.public else "private"})']
            return res
        elif isinstance(n, ASTForInStatement):
            res = [prefix]
            res += _inner(depth + 1, n.iter_expr, f'iter_elts')
            for stmt in n.block:
                res += _inner(depth + 1, stmt)
            return res
        elif isinstance(n, ASTCondStatement):
            res = [prefix]
            res += _inner(depth + 1, None, 'true')
            for stmt in n.t_block:
                res += _inner(depth + 1, stmt)
            res += _inner(depth + 1, None, 'false')
            for stmt in n.f_block:
                res += _inner(depth + 1, stmt)
            return res
        elif isinstance(n, ASTCreateNDArray):
            res = [prefix]
            for i, expr in enumerate(n.values):
                res += _inner(depth + 1, expr, f'val_{i + 1}')
            return res
        else:
            raise NotImplementedError(type(n))
    return '\n'.join(_inner(0, node))


def prettify_ir_stmts(stmts: List[IRStatement]):
    results = []
    for i, stmt in enumerate(stmts):
        s = f'#{stmt.stmt_id}\t{stmt.op}\t'
        if stmt.op == OpName.Special.CONSTANT:
            s += f'<= {stmt.constant_value}'
        elif stmt.op == OpName.Special.INPUT:
            s += f'<= {stmt.constant_args[0]}'
        elif stmt.op == OpName.Special.READ_INT:
            s += f'<= {stmt.constant_args[0]}, {stmt.constant_args[1]}'
        else:
            s += f'({", ".join(["%" + str(arg) for arg in stmt.args])})'
            if (OpName.is_constant_arg_operator(stmt.op) or OpName.is_constant_operator(stmt.op)) and stmt.constant_args is not None:
                s += f'<{", ".join([str(arg) for arg in stmt.constant_args])}>'
        if stmt.slicing_args:
            s += f'[{", ".join([str(x) if isinstance(x, int) else (str(x[0]) + ":" + str(x[1])) for x in stmt.slicing_args])}]'
        if stmt.slicing_assign_args:
            s += '['
            items = []
            for sli in stmt.slicing_assign_args:
                items.append(f'[{", ".join([str(x) if isinstance(x, int) else (str(x[0]) + ":" + str(x[1])) for x in sli])}]')
            s += ', '.join(items)
            s += ']'
        if stmt.annotation is not None:
            s += ' {'
            items = []
            if stmt.annotation.typename is not None:
                items.append(f'{stmt.annotation.typename}')
            if stmt.annotation.shape is not None:
                items.append(f'{stmt.annotation.shape}')
            if stmt.annotation.public is not None:
                items.append("Public" if stmt.annotation.public else "Private")
            s += '-'.join(items)
            s += '}'
        results.append(s)
    return "\n".join(results)


def prettify_exception(exception: InternalPyzkException, method_name: str, source_code: str) -> PyZKException:
    if isinstance(exception, InternalPyzkException):
        if exception.source_pos is None:
            return PyZKException(f'{type(exception).__name__}: {exception.msg}')
        source_lines = source_code.splitlines()
        error_report_msg = f'  In method "{method_name}", line {exception.source_pos.lineno}'
        if exception.source_pos.end_lineno != exception.source_pos.lineno:
            error_report_msg += f' to {exception.source_pos.end_lineno}'
        error_report_msg += '\n'
        if exception.source_pos.end_lineno != exception.source_pos.lineno:
            for i, line in enumerate(source_lines):
                line_no = i + 1
                if line_no == exception.source_pos.lineno:
                    error_report_msg += f'    {line}'
                    error_report_msg += '\n'
                    error_report_msg += ' ' * (4 + exception.source_pos.col_offset)
                    error_report_msg += '^' * (len(line) - exception.source_pos.col_offset)
                    error_report_msg += '\n'
                elif line_no == exception.source_pos.end_lineno:
                    error_report_msg += f'    {line}'
                    error_report_msg += '\n'
                    error_report_msg += '    ' + '^' * (len(line) - exception.source_pos.end_col_offset)
                    error_report_msg += '\n'
                elif exception.source_pos.lineno < line_no < exception.source_pos.end_lineno:
                    error_report_msg += f'    {line}'
                    error_report_msg += '\n'
                    error_report_msg += '    ' + '^' * len(line)
                    error_report_msg += '\n'
        else:
            error_report_msg += f'    {source_lines[exception.source_pos.lineno - 1]}'
            error_report_msg += '\n'
            error_report_msg += ' ' * (4 + exception.source_pos.col_offset)
            error_report_msg += '^' * (exception.source_pos.end_col_offset - exception.source_pos.col_offset)
            error_report_msg += '\n'
        error_report_msg += type(exception).__name__ + ': ' + exception.msg
        return PyZKException(error_report_msg)
    raise NotImplementedError
