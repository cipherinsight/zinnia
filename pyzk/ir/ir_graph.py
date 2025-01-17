from typing import List

from pyzk.ir.ir_stmt import IRStatement


class IRGraphMetadata:
    def __init__(
        self,
        ndarray_flattened: bool = False,
        annotated: bool = False,
    ):
        self.ndarray_flattened = ndarray_flattened
        self.annotated = annotated


class IRGraph:
    def __init__(self, stmts: List[IRStatement], metadata: IRGraphMetadata):
        self.stmts = []
        self.in_d = self.out_d = []
        self.in_links = self.out_links = []
        self.update_graph(stmts)
        self.metadata = metadata

    def update_graph(self, stmts: List[IRStatement]):
        self.stmts = stmts
        for i, stmt in enumerate(self.stmts):
            assert i == stmt.stmt_id
        in_d = [0 for _ in range(len(self.stmts))]
        out_d = [0 for _ in range(len(self.stmts))]
        being_referred_bys: List[List[int]] = [[] for _ in range(len(self.stmts))]
        for i, stmt in enumerate(self.stmts):
            referring_tos = stmt.arguments
            for t in referring_tos:
                if t is None:
                    continue
                out_d[t] += 1
                in_d[i] += 1
                being_referred_bys[t].append(i)
        self.in_d = in_d
        self.out_d = out_d
        self.in_links = [stmt.arguments for stmt in self.stmts]
        self.out_links = being_referred_bys

    def get_io_degrees(self):
        return self.in_d.copy(), self.out_d.copy()

    def get_io_links(self):
        return self.in_links.copy(), self.out_links.copy()

    def get_topological_order_ids(self, reverse: bool = False) -> List[int]:
        in_d = self.in_d.copy()
        out_d = self.out_d.copy()
        visiting_order = []
        stmt_queue = []
        if reverse:
            for i, stmt in enumerate(self.stmts):
                if out_d[i] == 0:
                    stmt_queue.append(i)
            stmt_queue = sorted(stmt_queue, reverse=False)
            while len(stmt_queue) > 0:
                curr = stmt_queue.pop()
                visiting_order.append(curr)
                referring_to = self.in_links[curr]
                for t in referring_to:
                    out_d[t] -= 1
                    in_d[curr] -= 1
                    if out_d[t] == 0:
                        stmt_queue.append(t)
                stmt_queue = sorted(stmt_queue, reverse=False)
        else:
            for i, stmt in enumerate(self.stmts):
                if in_d[i] == 0:
                    stmt_queue.append(i)
            stmt_queue = sorted(stmt_queue, reverse=True)
            while len(stmt_queue) > 0:
                curr = stmt_queue.pop()
                visiting_order.append(curr)
                being_referred_by = self.out_links[curr]
                for t in being_referred_by:
                    in_d[t] -= 1
                    out_d[curr] -= 1
                    if in_d[t] == 0:
                        stmt_queue.append(t)
                stmt_queue = sorted(stmt_queue, reverse=True)
        return visiting_order

    def get_topological_order(self, reverse: bool = False) -> List[IRStatement]:
        return [self.retrieve_stmt_with_id(x) for x in self.get_topological_order_ids(reverse)]

    def retrieve_stmt_with_id(self, idx: int) -> IRStatement:
        return self.stmts[idx]

    def topological_refresh(self) -> "IRGraph":
        topological_order = self.get_topological_order_ids()
        id_mapping = {}
        for new_id, old_id in enumerate(topological_order):
            id_mapping[old_id] = new_id
        for stmt in self.stmts:
            stmt.stmt_id = id_mapping[stmt.stmt_id]
            for i, arg in enumerate(stmt.arguments):
                stmt.arguments[i] = id_mapping[arg]
        self.update_graph(self.stmts)
        return self

    def remove_stmt(self, idx: int):
        self.stmts.pop(idx)
        id_mapping = {}
        for new_id, stmt in enumerate(self.stmts):
            id_mapping[stmt.stmt_id] = new_id
        for stmt in self.stmts:
            stmt.stmt_id = id_mapping[stmt.stmt_id]
            for i, arg in enumerate(stmt.arguments):
                stmt.arguments[i] = id_mapping[arg]
        self.update_graph(self.stmts)

    def remove_stmt_bunch(self, indices: List[int]):
        self.stmts = [stmt for i, stmt in enumerate(self.stmts) if i not in indices]
        id_mapping = {}
        for new_id, stmt in enumerate(self.stmts):
            id_mapping[stmt.stmt_id] = new_id
        for stmt in self.stmts:
            stmt.stmt_id = id_mapping[stmt.stmt_id]
            for i, arg in enumerate(stmt.arguments):
                stmt.arguments[i] = id_mapping[arg]
        self.update_graph(self.stmts)

    def export_stmts(self) -> List[IRStatement]:
        return self.stmts
