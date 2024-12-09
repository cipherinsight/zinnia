from pyzk.ir.ir_builder import IRGraph
from pyzk.ir.ir_pass.abstract_pass import AbstractIRPass


class DeadCodeEliminationIRPass(AbstractIRPass):
    def __init__(self):
        super().__init__()

    def exec(self, ir_graph: IRGraph) -> IRGraph:
        stmts = ir_graph.export_stmts()
        ensuring_keep_stmts = [stmt.operator.dce_keep() for stmt in stmts]
        in_d, out_d = ir_graph.get_io_degrees()
        killing_queue = []
        to_be_eliminated = []
        for i, stmt in enumerate(stmts):
            if out_d[i] == 0 and not ensuring_keep_stmts[i]:
                killing_queue.append(i)
        while len(killing_queue) > 0:
            to_be_killed = killing_queue.pop()
            to_be_eliminated.append(to_be_killed)
            referring_tos = stmts[to_be_killed].arguments
            for k, t in referring_tos.items():
                if t is None:
                    continue
                in_d[to_be_killed] -= 1
                out_d[t] -= 1
                if out_d[t] == 0 and not ensuring_keep_stmts[t]:
                    killing_queue.append(t)
        ir_graph.remove_stmt_bunch(to_be_eliminated)
        return ir_graph

