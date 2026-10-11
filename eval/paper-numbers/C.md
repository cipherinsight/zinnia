Task C (folder C/): SMT statistics for the appendix.

The paper's appendix wants "compile times and the number of queries that exhaust their step budget,
per corpus". Compile times (remote, compiler 4115cecb) are in
/home/zhantong/Projects/zinnia-project/eval-redo/compile-timing/remote/ct_fin.jsonl (config default;
set perf/acceptance; derive the corpus from the name). Compute per corpus (DS-1000, NPBench, Pythran) over
the accepted programs: median, p90, max compile time, and the SMT share (resolver_smt_s / compile_s).
Then, with the final compiler 3ebe0558 locally, count per corpus: the number of SMT queries issued and the
number that exhaust their step budget (find the telemetry the compiler exposes, e.g. env vars or debug
counters in src/ for the resolver / SMT tier; if none exists say so precisely). Local counts are fine for
query counts (they are deterministic); do not report local times.
