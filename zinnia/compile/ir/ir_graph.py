import copy
from typing import List

from zinnia.compile.ir.ir_stmt import IRStatement



class IRGraph:
    def __init__(self, stmts: List[IRStatement]):
        self.stmts = []
        self.in_d = self.out_d = []
        self.in_links = self.out_links = []
        self.update_graph(stmts)

    def __copy__(self):
        return IRGraph(self.stmts)

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

    def get_topological_order(self, reverse: bool = False) -> List[IRStatement]:
        if reverse:
            return list(reversed(self.stmts))
        return self.stmts

    def retrieve_stmt_with_id(self, idx: int) -> IRStatement:
        return self.stmts[idx]

    def remove_stmt(self, idx: int):
        after_removal_stmts = self.stmts[:idx] + self.stmts[idx + 1:]
        after_removal_stmts = [copy.copy(stmt) for stmt in after_removal_stmts]
        id_mapping = {}
        for new_id, stmt in enumerate(after_removal_stmts):
            id_mapping[stmt.stmt_id] = new_id
        for stmt in after_removal_stmts:
            stmt.stmt_id = id_mapping[stmt.stmt_id]
            for i, arg in enumerate(stmt.arguments):
                stmt.arguments[i] = id_mapping[arg]
        self.update_graph(after_removal_stmts)

    def remove_stmt_bunch(self, indices: List[int]):
        after_removal_stmts = [copy.copy(stmt) for i, stmt in enumerate(self.stmts) if i not in indices]
        id_mapping = {}
        for new_id, stmt in enumerate(after_removal_stmts):
            id_mapping[stmt.stmt_id] = new_id
        for stmt in after_removal_stmts:
            stmt.stmt_id = id_mapping[stmt.stmt_id]
            for i, arg in enumerate(stmt.arguments):
                stmt.arguments[i] = id_mapping[arg]
        self.update_graph(after_removal_stmts)

    def export_stmts(self) -> List[IRStatement]:
        return self.stmts
