Task A (folder A/): characterize the accepted programs and the performance set.

1. Of the programs the acceptance account accepted (results.json, outcome accepted_correct; 214), how many
   create an array whose shape depends on the input (e.g. a boolean-mask result, np.flatnonzero,
   np.nonzero, np.unique, np.argwhere, a slice with runtime bounds, np.delete/insert at runtime position)?
   Determine this from the compiler, not by regex on source: compile each accepted program with the final
   compiler and detect bounded (runtime-shaped) arrays in the compiler's IR or types (find the right
   introspection in the code: bounded-dynamic array types / runtime length values / padded buffers).
   Give the count per corpus (DS-1000, NPBench, Pythran) and the list.
2. How many accepted programs read or write an array at a position computed from the input (runtime-index
   gather/scatter/memory access, as opposed to static indices)? Again from the compiler (e.g. operator
   placeholders the lowering layer receives: gather, scatter, memory read/write with non-constant index,
   sort/argsort permutation reads). Count per corpus, with the list.
3. The performance set datasets/case_sets/npbench_pythran_perf.json has 79 programs (status perf), while the
   acceptance account accepts 82 NPBench+Pythran programs. List exactly which accepted programs are not in
   the performance set and why (from the case set's status/reason fields).
