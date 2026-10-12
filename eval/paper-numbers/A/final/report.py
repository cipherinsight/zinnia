import json,hashlib,collections,subprocess
from analyze import OUT,ROOT,ACCOUNT,rows,key
records=json.loads((OUT/'classification.json').read_text())
perf_path=ROOT/'benchmarking/datasets/case_sets/npbench_pythran_perf.json'
perf=json.loads(perf_path.read_text()); cases=perf['cases']; pm={(c['program'],c['function']):c for c in cases}
corpora=['DS-1000','NPBench','Pythran']
accepted={(r['program'],r['function']) for r in rows() if r['dataset'] in ['npbench','pythran']}
performance={(r['program'],r['function']) for r in cases if r['status']=='perf'}
assert len(records)==214
# The performance case set changed after 3ebe0558; its difference with the accepted set is reported, not asserted.
COMMIT=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
missing=[pm[k] for k in sorted(accepted-performance) if k in pm]
(OUT/'performance_difference.json').write_text(json.dumps(missing,indent=2))
for r in records:
 r['without_sort_only']=any(k not in ['Sort','SortRuntimeRead'] for k in r['index_evidence']) or r['explicit_runtime_address']
def count(field,c=None):return sum(bool(r.get(field)) for r in records if c is None or r['corpus']==c)
def ids(rs):return ', '.join('`'+r['program']+'::'+r['function']+'`' for r in sorted(rs,key=lambda r:(r['program'],r['function']))) or 'None.'
lines=['# Task A — accepted programs and performance set','',
'Generated from compiler evidence by `report.py`; the unit is a **(corpus, program, entry function)**, not a distinct source file. The supplied acceptance account selects only `outcome == "accepted_correct"`. Its 214 entries comprise 132 DS-1000, 32 NPBench, and 50 Pythran entries. All 214 compile successfully with final compiler `'+COMMIT+'` (remote host shuaicpu6, container zhantong-dev).', '',
'## Results','',
'| Definition | DS-1000 | NPBench | Pythran | Total |','|---|---:|---:|---:|---:|']
for label,field in [('Runtime-shaped array (Q1, primary)','runtime_shape'),('Any DynamicNDArray type (Q1 alternative)','dynamic_type'),('Runtime shape OR runtime-position insert (Q1 example-inclusive alternative)','shape_or_insert'),('Input-dependent array selection/movement (Q2, primary)','runtime_index'),('Q2 limited to lowering input, omitting pre-expanded compaction','lowering_runtime_index'),('Q2 explicit runtime addresses only (strict alternative)','explicit_runtime_address'),('Q2 excluding programs counted only through Sort placeholders','without_sort_only')]:
 lines.append('| '+label+' | '+' | '.join(str(count(field,c)) for c in corpora)+' | '+str(count(field))+' |')
lines+=['','## Definitions and compiler evidence','',
'**Q1.** A program qualifies if compilation encounters `Value::DynamicNDArray` whose `runtime_length`, `runtime_rank`, or at least one `runtime_shape` scalar has `static_val == None`. The bound is `envelope.total_bound`. This includes intermediate, discarded, and returned arrays, and views such as input-dependent reshape. It does not require the final output to have runtime shape. A constant length alone does not imply constant shape (e.g. reshape `(nrow, -1)`). This is the compiler\'s static-versus-runtime classification, not a claim that we proved two distinct feasible shapes with an SMT solver.', '',
'`instrument.py` copies the compiler under `A/compiler/` and adds read-only logging at IR-generator expression results (recursively through composites) and `DynamicNDArrayData::max_length`, which also observes internal temporary buffers. It records metadata without minting compiler IDs or emitting instructions. See `instrumentation.diff`, `compiler/src/types/composite.rs`, and `records/*/dynamic.jsonl`. All programs with this type also have nonconstant shape metadata, so the broader type-only count is identical.', '',
'**Fixed-size insert is not runtime-shaped.** Inserting one element at a runtime position into a fixed-size array changes element placement, but always increases length by one. The user\'s example therefore spans two definitions. The example-inclusive alternative additionally counts `case0363::kernel`, `case0364::kernel`, and `case0365::kernel`, all DS-1000; their lowering has an input-dependent `Insert` position. They are included in Q2, not Q1\'s primary count.', '',
'**Q2.** The primary definition counts input-dependent selection or placement of array elements as recognized by the compiler: nonconstant input-dependent addresses in `ArrayReadIR`/`ArrayWriteIR` (or ROM/RAM counterparts), a runtime read of a sorted output, input-dependent `Window` bounds, `Insert` positions, `PermutationScatter` destinations, `MaskWrite` predicates, sort/argsort permutations over input-dependent values, and boolean-mask compaction. These are program-level unions, not operator counts.', '',
'For the structured IR entering `ArrayLowering`, `summarize.py` propagates input dependence along statement operands, starting at `ReadInteger`, `ReadFloat`, `ReadHash`, external-result and witness reads. Static indices occupy the `ArrayAccess.offset` field; only input-dependent runtime axis operands count. Static unrolled loop indices and static shape arithmetic do not qualify. Sort selection is detected from input-dependent shared values; `Window`, `Insert`, and scatter are tested only on their position/mask operands. Reductions returning an index (such as argmax) alone are not counted unless that index is subsequently used for selection.', '',
'Boolean-mask gathers can be expanded into a scalar one-hot shift network **before** the lowering input. `compaction.py` logs actual calls to `compact_by_mask` with a nonconstant mask, and checks their final IR against the stock compiler. This adds nine entries to the lowering-input-only count: '+ids([r for r in records if r['runtime_index'] and not r['lowering_runtime_index']])+'. All such compactions produce dynamic arrays, so the revisit covers every dynamic-type program.', '',
'**Q2 ambiguity.** The strict alternative counts only explicit runtime addresses and runtime Sort-output reads; it excludes mask predicates and compiler operators whose placement is implicit. The primary definition is an operational compiler count: the frontend also implements some `min`/`max` reductions with `Sort` placeholders, so programs such as NPBench MLP and softmax qualify even though the source need not contain a sorting call or explicit indexed load. The final alternative removes programs whose only evidence is such a Sort placeholder. These alternatives, with their exact membership, prevent conflating compiler lowering behavior with source-level indexing syntax. A direct array access simplified away before lowering is outside the lowering-input definitions; scalar conditional selections alone are not treated as indexed loads.', '',
'**Validation.** For all 214 entries, stock and instrumented compilations have identical final exported-IR SHA-256 hashes, output layouts, and byte-for-byte identical pre-lowering text dumps. The instrumented extension disables optional proving backends when built; compilation still targets the default `halo2` lowering, and equality checks establish that this does not change the measured IR. No proof, key, timing benchmark, or mock execution was requested or run. Internal compile-time bookkeeping in the existing compiler is not reported.', '',
'## Q1 per-program lists','']
for c in corpora:lines += ['### '+c+' ('+str(count('runtime_shape',c))+')','',ids([r for r in records if r['corpus']==c and r['runtime_shape']]),'']
lines+=['## Q2 per-program lists (primary definition)','']
for c in corpora:lines += ['### '+c+' ('+str(count('runtime_index',c))+')','',ids([r for r in records if r['corpus']==c and r['runtime_index']]),'']
lines+=['## Performance set','',f'{len(performance)} `status == "perf"` entries; accepted NPBench/Pythran entries not in it: {len(accepted-performance)}; perf entries not accepted: {len(performance-accepted)}.','']
lines+=[
'## Reproduce and audit','',
'```sh','# remote (shuaicpu6, container zhantong-dev)','cd /home/zhantong/zinnia-eval-run/paper-numbers-final/A','./reproduce.sh','```','',
'The script uses the requested worktree `.venv/bin/python`, reuses `benchmarking.harness.kernels.compile_kernel` (including decorated-source extraction and lifted helpers), and compiles the acceptance-account kernel path and entry function. Inputs are not fixture-specialized; the same annotations/source used by the acceptance harness determine the compilation. No fixture values are needed for this compiler-only characterization. All new files, copied compiler source, build outputs, and logs stay in `A/`; Python bytecode writes are disabled. The container has no systemd; each compilation worker runs under `ulimit -v` (32 GiB virtual memory), with 16 workers per stage.', '',
'For a fast report-only rebuild from saved evidence:','',
'```sh',str(ROOT/'.venv/bin/python')+' summarize.py',str(ROOT/'.venv/bin/python')+' report.py','```','',
'- `classification.json`: all 214 rows, flags, dynamic metadata, representative IR statements and per-family counts.',
'- `records/<dataset>__<program>__<function>/`: original account row, kernel hash, stock/instrumented final IR hash and output layout, raw lowering text, structured lowering IR (`layer.json.gz`), dynamic metadata, and compaction evidence/checks.',
'- `performance_difference.json`: full case-set records for all three exclusions.',
'- `provenance.json`: input hashes and compiler revision.',
'- `phase0.json`: auxiliary shape analysis only. This separate analysis rejects 12 otherwise compilable programs, so it is **not** used in the reported classifications. The actual IR-generation instrumentation covers all 214.', '',
'## Complete membership table','',
'`S`: Q1 primary; `S+I`: shape-or-insert alternative; `I`: Q2 primary; `L`: lowering-input-only alternative; `A`: strict explicit-address alternative; `NS`: exclude Sort-only evidence. “—” means false. This table lists all accepted entries, including negatives; the dynamic-type alternative has exactly the `S` membership.', '',
'| Corpus | Program::function | S | S+I | I | L | A | NS | Q2 compiler evidence |','|---|---|---:|---:|---:|---:|---:|---:|---|']
for r in sorted(records,key=lambda r:(corpora.index(r['corpus']),r['program'],r['function'])):
 lines.append('| '+r['corpus']+' | `'+r['program']+'::'+r['function']+'` | '+' | '.join('yes' if r[f] else '—' for f in ['runtime_shape','shape_or_insert','runtime_index','lowering_runtime_index','explicit_runtime_address','without_sort_only'])+' | '+(', '.join(r['index_evidence']) or '—')+' |')
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
provenance={'compiler_commit':COMMIT,'host':'shuaicpu6 (docker zhantong-dev)','files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ACCOUNT,perf_path,ROOT/'zinnia/_zinnia_core.cpython-314-x86_64-linux-gnu.so'] if p.exists()},'accepted':len(records),'stock_instrumented_equal':len(records)}
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
print('REPORT.md written')
