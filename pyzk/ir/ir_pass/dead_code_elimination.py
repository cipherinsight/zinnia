from pyzk.ir.ir_builder import IRGraph, IRStatement
from pyzk.ir.ir_pass.abstract_pass import AbstractIRPass
from pyzk.util.op_name import OpName


class DeadCodeEliminationIRPass(AbstractIRPass):
    def __init__(self):
        super().__init__()

    def exec(self, ir_graph: IRGraph) -> IRGraph:
        stmts = ir_graph.export_stmts()
        ensuring_keep_stmts = [(True if self._is_must_keep_node(stmt) else False) for stmt in stmts]
        in_d, out_d = ir_graph.get_io_degrees()
        killing_queue = []
        to_be_eliminated = []
        for i, stmt in enumerate(stmts):
            if out_d[i] == 0 and not ensuring_keep_stmts[i]:
                killing_queue.append(i)
        while len(killing_queue) > 0:
            to_be_killed = killing_queue.pop()
            to_be_eliminated.append(to_be_killed)
            referring_tos = stmts[to_be_killed].args
            for t in referring_tos:
                in_d[to_be_killed] -= 1
                out_d[t] -= 1
                if out_d[t] == 0 and not ensuring_keep_stmts[t]:
                    killing_queue.append(t)
        ir_graph.remove_stmt_bunch(to_be_eliminated)
        return ir_graph

    def _is_must_keep_node(self, n: IRStatement):
        return n.op == OpName.Special.ASSERT or n.op == OpName.Special.INPUT or n.op == OpName.Special.EXPOSE_PUBLIC
