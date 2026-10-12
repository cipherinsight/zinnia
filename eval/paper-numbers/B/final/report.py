"""Generate REPORT.md, summary.json, and CSV tables from the saved measurements."""
import measure,json,math,statistics,collections,csv,hashlib,subprocess
from pathlib import Path
HERE=measure.HERE
rows=json.loads((HERE/'arrays.json').read_text());audit=json.loads((HERE/'audit.json').read_text())
assert len(audit)==214 and not json.loads((HERE/'array_issues.json').read_text())
by=collections.defaultdict(list)
for r in rows:by[r['program']].append(r)
def gm(xs):
 return math.exp(sum(math.log(x) for x in xs)/len(xs)) if xs else None
def ratio(r,f):return r['capacity']/r['lengths'][f] if r['lengths'][f] else math.inf
def fmt(x):return '∞' if x==math.inf else ('—' if x is None else f'{x:.6g}')
def stats(xs):return dict(n=len(xs),geomean=gm(xs),median=statistics.median(xs),min=min(xs),max=max(xs))
prog={p:{f:gm([ratio(r,f) for r in rr])for f in ['A','B']} for p,rr in by.items()}
for p in prog:prog[p]['AB']=gm([prog[p]['A'],prog[p]['B']])
finite_ps=[p for p in by if math.isfinite(prog[p]['AB'])]
summary={'accepted':len(audit),'bounded_programs':len(by),'bounded_array_values':len(rows),'zero_length_arrays':sum(r['lengths']['A']==0 or r['lengths']['B']==0 for r in rows),'program_weighted':{},'array_weighted':{}}
for f in ['A','B','AB']:
 vals=[prog[p][f]for p in by];summary['program_weighted'][f]=stats(vals)
 summary.setdefault('program_weighted_finite_complete',{})[f]=stats([prog[p][f]for p in finite_ps])
 vals=[ratio(r,g)for r in rows for g in (['A','B']if f=='AB'else[f])]
 summary['array_weighted'][f]=stats(vals);summary.setdefault('array_weighted_finite',{})[f]=stats([x for x in vals if math.isfinite(x)])
# Literal first-axis length alternative (element capacity / Python len).
first_axis={p:gm([r['capacity']/r['shapes'][f][0] if r['shapes'][f][0] else math.inf for r in rr for f in ['A','B']]) for p,rr in by.items()}
summary['literal_len_alternative']={'all_programs':stats(list(first_axis.values())),'finite_complete_programs':stats([first_axis[p]for p in finite_ps]),'per_program':first_axis}
# Original-output-only alternative, without observational rewrites.
from benchmarking.harness.reference import run_kernel_as_numpy
output_rows=[]
for p in measure.entries():
 c=json.loads((HERE/'census'/(measure.ident(p)+'.json')).read_text());ds=[x for x in c['layout']if 'dynamic'in x]
 if not ds:continue
 outs={f:run_kernel_as_numpy(p.kernel_path,p.function,measure.load_fixture(p.fixture_path).arguments(f))for f in ['A','B']}
 for x in ds:
  lengths={}
  for f in ['A','B']:
   v=outs[f]
   for i in x['output']['return']:v=v[i]
   lengths[f]=int(v.size)
  output_rows.append(dict(program=measure.ident(p),array=str(x['output']),capacity=len(x['dynamic']['cells'])//(2 if x.get('complex')else 1),lengths=lengths))
op=collections.defaultdict(list)
for r in output_rows:op[r['program']].append(r)
summary['output_only']={'programs':len(op),'arrays':len(output_rows),'program_weighted':stats([gm([ratio(r,f)for r in rr for f in ['A','B']])for rr in op.values()]),'array_weighted':stats([ratio(r,f)for r in output_rows for f in ['A','B']])}
# Same-source tuple extraction is an alias in case0481 (nonzero tuple element).
dedup=[r for r in rows if not (r['program']=='ds1000_numpy__case0481__kernel' and r['expression']=='idx_list.nonzero()[0]')]
summary['without_tuple_alias']={'arrays':len(dedup),'finite_array_weighted':stats([ratio(r,f)for r in dedup for f in ['A','B']if r['lengths'][f]>0])}
# Circuit sizes.
from bounds_cost import CONTROLS
costrows=[]
for case in CONTROLS:
 name='ds1000_numpy__case'+case+'__kernel';r=json.loads((HERE/'costs'/name/'result.json').read_text())
 for f in ['A','B']:
  pair=r['common'][f];d=pair['default'];v=pair[f];ng=r['variants']['default']['noir']['gates'];nv=r['variants'][f]['noir']['gates']
  assert r['variants'][f]['check_'+f]['status']=='match'
  costrows.append(dict(program=name,fixture=f,lookup_bits=d['lookup_bits'],default_cells=d['advice']+d['lookup'],tight_cells=v['advice']+v['lookup'],halo2_growth=(d['advice']+d['lookup'])/(v['advice']+v['lookup']),default_gates=ng,tight_gates=nv,noir_growth=ng/nv,controls=','.join(r['controls'])))
summary['costs']={k:stats([gm([r[k]for r in costrows if r['program']==p])for p in sorted({r['program']for r in costrows})])for k in ['halo2_growth','noir_growth']}
slack={p for p in by if any(r['capacity']>r['lengths'][f] for r in by[p]for f in ['A','B'])}
summary['costs_positive_slack']={k:stats([gm([r[k]for r in costrows if r['program']==p])for p in sorted({r['program']for r in costrows}&slack)])for k in ['halo2_growth','noir_growth']}
# Fail rather than publish incomplete validation.
validations=[json.loads(p.read_text())for p in (HERE/'validation').glob('*.json')]
original_checks=[json.loads(p.read_text())for p in (HERE/'original_validation').glob('*.json')]
cost_checks=json.loads((HERE/'cost_validation.json').read_text())
assert {v['id']for v in validations}==set(by) and all('error'not in v and all(x['match']for x in v['checks'])for v in validations)
assert {v['id']for v in original_checks}==set(by) and all('error'not in v and all(x['satisfied']and x['check']=='match'for x in v['fixtures'].values())for v in original_checks)
assert len(cost_checks)==len(costrows) and all(v['rejected']and v['exact_output_capacity']for v in cost_checks)
# JSON uses strings for infinity, rather than nonstandard Infinity tokens.
def safe(x):
 if isinstance(x,float)and not math.isfinite(x):return 'infinity'
 if isinstance(x,dict):return {k:safe(v)for k,v in x.items()}
 if isinstance(x,list):return [safe(v)for v in x]
 return x
(HERE/'summary.json').write_text(json.dumps(safe(summary),indent=2))
(HERE/'output_arrays.json').write_text(json.dumps(output_rows,indent=2))
with (HERE/'arrays.csv').open('w')as f:
 w=csv.writer(f);w.writerow(['program','array','expression','capacity','length_A','length_B','ratio_A','ratio_B'])
 for r in rows:w.writerow([r['program'],r['array'],r['expression'],r['capacity'],r['lengths']['A'],r['lengths']['B'],fmt(ratio(r,'A')),fmt(ratio(r,'B'))])
with (HERE/'costs.csv').open('w')as f:
 w=csv.DictWriter(f,fieldnames=list(costrows[0]));w.writeheader();w.writerows(costrows)
# Provenance and content hashes.
import zinnia._zinnia_core as core
manifest={'compiler_commit':subprocess.check_output(['git','-C',str(measure.ROOT),'rev-parse','HEAD'],text=True).strip(),'extension':core.__file__,'extension_sha256':hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),'acceptance_sha256':hashlib.sha256(measure.ACCEPT.read_bytes()).hexdigest(),'files':[]}
for p in measure.entries():manifest['files'].append(dict(id=measure.ident(p),kernel=str(p.kernel_path),kernel_sha256=hashlib.sha256(p.kernel_path.read_bytes()).hexdigest(),fixture=str(p.fixture_path),fixture_sha256=hashlib.sha256(p.fixture_path.read_bytes()).hexdigest()))
for tool,path in [('nargo',Path.home()/'.nargo/bin/nargo'),('bb',Path.home()/'.bb/bb')]:manifest[tool]=dict(path=str(path),version=subprocess.check_output([str(path),'--version'],text=True).strip(),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(HERE/'manifest.json').write_text(json.dumps(manifest,indent=2))
S=summary;finite=S['program_weighted_finite_complete']['AB'];ac=S['array_weighted_finite']['AB'];oc=S['output_only'];hc=S['costs']['halo2_growth'];nc=S['costs']['noir_growth']
lines=['# Task B: cost of bounds','',f'Compiler `{manifest["compiler_commit"]}`; acceptance membership is the `accepted_correct` account of the final compiler (epoch fz, `accept-merged.json`), the same 214 entries as the older account. Remote host shuaicpu6. All {len(audit)} accepted entry functions compile on the final compiler.','',
f'**Result:** {len(by)} programs contain bounded runtime-shaped source array values ({len(rows)} expression-result array occurrences). The equal-program capacity/actual geometric mean is **∞**, because two programs contain empty arrays on both fixtures. Over the {len(finite_ps)} programs with no empty measured array it is **{fmt(finite["geomean"])}×**, median **{fmt(finite["median"])}×**. Sound checked shape-control specialization was measured for **{hc["n"]} programs**: default/specialized Halo2 cells have geometric mean **{fmt(hc["geomean"])}×** (range {fmt(hc["min"])}–{fmt(hc["max"])}×); Noir gates **{fmt(nc["geomean"])}×** (range {fmt(nc["min"])}–{fmt(nc["max"])}×). These circuit ratios include savings from knowing shape-control values; they are not an isolated padding-only causal effect.','',
'## Definitions and scope','',
'An array is a NumPy ndarray-valued source expression result (including intermediate arithmetic results, views, tuple/list elements, and each unrolled occurrence). Plain references to variables are not counted again. Boolean-mask subscript augmented assignments are desugared to a read, arithmetic result, and write, matching the compiler transformer; their implicit array temporaries count too. The shape is runtime-dependent when the compiler exports a witness-dependent length, rank, or dimension. Fixed-total reshapes count, even when capacity/length is one. Scalar arrays use total element count (`size`), not Python `len` (first-axis size). Complex payload pairs count as one element. Fixed-shape Boolean masks and scalar values do not count.','',
'Capacity is the number of payload element slots in the compiler’s exported dynamic layout, including constant padding. For internal values, an observational source rewrite lifts each expression to a temporary and returns the collected values; the same unmodified compiler then exports their layouts. The resulting sources are in `probes/`. A/B actual lengths and shapes come from the harness fixture loader and independent NumPy execution. Mock evaluation checks the exported shape metadata. This measures the compiler’s **logical bounded buffer envelope**, not distinct allocated storages or live resident memory: views can share storage, and lowering can fuse or eliminate buffers. Exposing intermediates can prevent such elimination. Consequently 36 is a source-value census, not a claim that all 36 optimized circuits physically allocate padding. The compiler has no public original-program census of internal buffer identity/shape; an exact unique-storage cost cannot be inferred from these results.','',
'Probe handling: branch-local crop results in 0507/0509/0510 use the original compiled output layout; their row/column index arrays use a prefix probe. In periodic_dist the three periodic branches are exposed unconditionally for inspection; both fixtures enable all three. Window helper results are enumerated individually. Static-only probe failures/oversized probes are source-audited, with reasons recorded in `audit.json`. The trace container itself is excluded. `nonzero()[0]` is a same-array alias of the ndarray inside `nonzero()`; the alias-excluded alternative below removes this one extra occurrence. Other view expressions remain logical values.','',
'For program p and fixture f, r[p,f] is the geometric mean of C/L over its arrays. The program’s combined ratio is sqrt(r[p,A] r[p,B]); the headline geometric mean and median weight programs equally. C>0,L=0 is +∞, never silently omitted or replaced by one. “Finite complete programs” drops the entire program if any measured array is empty. The finite-observation alternative drops only empty array/fixture pairs. A/B are the supplied fixtures, not a representative input distribution.','',
'## (a) Capacity divided by actual elements','',
'| Population / weighting | A GM | B GM | A+B GM | A+B median |', '|---|---:|---:|---:|---:|']
for title,key in [('All programs, equal program weight','program_weighted'),('Finite complete programs, equal program weight','program_weighted_finite_complete'),('All array/fixture pairs, equal array weight','array_weighted'),('Finite array/fixture pairs, equal array weight','array_weighted_finite')]:
 x=S[key];lines.append(f'| {title} | {fmt(x["A"]["geomean"])} | {fmt(x["B"]["geomean"])} | {fmt(x["AB"]["geomean"])} | {fmt(x["AB"]["median"])} |')
lines += ['',f'There are {S["zero_length_arrays"]} empty array values, on both fixtures ({2*S["zero_length_arrays"]} array/fixture pairs), all in `pythran_periodic_dist/dist` and `pythran_tax/tax`. The finite-array summary uses {ac["n"]} array/fixture pairs. The all-program median (including infinite programs) is {fmt(S["program_weighted"]["AB"]["median"])}×.','',
f'**Alternative: original outputs only.** {oc["programs"]} programs, {oc["arrays"]} arrays: program-weighted GM {fmt(oc["program_weighted"]["geomean"])}×, median {fmt(oc["program_weighted"]["median"])}×; array-weighted GM {fmt(oc["array_weighted"]["geomean"])}×, median {fmt(oc["array_weighted"]["median"])}×. These use unmodified original circuits, and all lengths are positive. Per-array data: `output_arrays.json`. This excludes internal-only case0490, periodic_dist, projection_simplex, and tax.','',
f'**Alternative: omit the explicit tuple extraction alias in case0481.** {S["without_tuple_alias"]["arrays"]} logical array values; finite array-weighted GM {fmt(S["without_tuple_alias"]["finite_array_weighted"]["geomean"])}×, median {fmt(S["without_tuple_alias"]["finite_array_weighted"]["median"])}×. Equal-program figures are unchanged.','',
f'**Alternative: literal Python len.** Dividing element capacity by first-axis length instead of total elements gives finite-complete-program GM {fmt(S["literal_len_alternative"]["finite_complete_programs"]["geomean"])}×, median {fmt(S["literal_len_alternative"]["finite_complete_programs"]["median"])}× (the full GM is still ∞). This mixes element and row units for higher-rank arrays and is not recommended. All per-program values are in `summary.json`; dimensions are in `arrays.json`.', '', '### Per-program list','', '| Program | Array values | GM A | GM B | GM A+B | Median array ratio A+B |', '|---|---:|---:|---:|---:|---:|']
for p,rr in by.items():lines.append(f'| {p} | {len(rr)} | {fmt(prog[p]["A"])} | {fmt(prog[p]["B"])} | {fmt(prog[p]["AB"])} | {fmt(statistics.median([ratio(r,f)for r in rr for f in ["A","B"]]))} |')
lines += ['', '## (b) Circuit sizes','',
'General mechanism: retain the private input and assert equality of each shape-control value to its fixture value, then bind that local variable to the asserted constant before executing the unchanged kernel body. An array control (`size`) is checked element by element. Payload inputs remain private and unconstrained beyond the original domain. Every variant is exactly the original function on this explicitly restricted input domain; substituting an asserted equal constant is sound. Source for each default/A/B variant and emitted Noir are in `costs/<program>/`. All 17 programs have identical shape-control values on A and B, so their A/B sizes coincide.','',
'This makes the relevant source arrays statically shaped with element counts equal to the fixture’s actual lengths. Static windows can be exported as lists of rows; validation groups them by original window. The exact output capacities are checked in `cost_validation.json`; source-level internal slices/reshapes/reductions have constant fixture dimensions as well. The comparison is useful as the cost of retaining general shape controls, but it also removes shape arithmetic, dynamic indexing/routing, and metadata. It must not be called a pure cost of over-reservation.','',
'Halo2 uses Zinnia’s witness-free `halo2_base_cell_stats`: advice + lookup cells, at the **same lookup width per pair**, the maximum of the two auto-selected widths (k may differ; counts depend on width). Fixed cells are excluded, as in the harness cost definition. Raw own-width stats, k, fixed cells and common-width stats remain in each result JSON. Noir is compiled for the Noir target, emitted with `emit_structural`, then `nargo compile` and `bb gates --scheme ultra_honk --verifier_target noir-recursive`; the metric is `functions[0].circuit_size`, not padded powers of two. Gate counting reuses `benchmarking.noir_gates.gate_count`, with temporary packages under B/tmp. No keys, real proofs, or timing runs were made.','',
'| Program | Controls fixed | Common lookup bits | Default cells | Exact-shape cells | Growth | Default gates | Exact-shape gates | Growth |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in costrows[::2]:lines.append(f'| {r["program"]} | {r["controls"]} | {r["lookup_bits"]} | {r["default_cells"]} | {r["tight_cells"]} | {fmt(r["halo2_growth"])} | {r["default_gates"]} | {r["tight_gates"]} | {fmt(r["noir_growth"])} |')
lines += ['',f'Across {hc["n"]} programs: Halo2 GM {fmt(hc["geomean"])}×, median {fmt(hc["median"])}×, range {fmt(hc["min"])}–{fmt(hc["max"])}×; Noir GM {fmt(nc["geomean"])}×, median {fmt(nc["median"])}×, range {fmt(nc["min"])}–{fmt(nc["max"])}×. A/B rows are retained separately in `costs.csv`.','',
f'Excluding the zero-slack fixed-total reshape programs 0301, 0302, 0389 leaves {S["costs_positive_slack"]["halo2_growth"]["n"]} programs: Halo2 GM {fmt(S["costs_positive_slack"]["halo2_growth"]["geomean"])}×, range {fmt(S["costs_positive_slack"]["halo2_growth"]["min"])}–{fmt(S["costs_positive_slack"]["halo2_growth"]["max"])}×; Noir GM {fmt(S["costs_positive_slack"]["noir_growth"]["geomean"])}×, range {fmt(S["costs_positive_slack"]["noir_growth"]["min"])}–{fmt(S["costs_positive_slack"]["noir_growth"]["max"])}×. This remains specialization, not isolated padding cost.','',
'### Programs without a sound exact-capacity comparison','',
'No general capacity override exists in the Python API. `requirements_probe.json` records representative attempts: assertions and scalar `@requires` leave the exported capacities unchanged (0303: 23; 0383: 100; 0391: 200); an asserted mask count in 0412 leaves capacity 10, and asserted unique count in 0298 leaves output capacity 25. `@requires` rejects the tested aggregate/unique-count predicates. Filtering/nonzero compaction is sized from the input envelope in `src/ops/dyn_ndarray/memory_ops.rs`, before ordinary scalar optimization. Tightening the result afterwards does not shrink that construction. A truncated output or fixed fixture payload would not be a sound capacity-only comparison. Implementing a different compactor/unique algorithm could be sound, but would measure another algorithm, so none is presented here.','',
'| Program | Reason exact-capacity comparison is unavailable in this method |','|---|---|']
measured={r['program']for r in costrows}
for p in by:
 if p in measured:continue
 if 'case0298' in p:reason='Unique cardinality is data-dependent; both unique values and the zero matrix depend on it.'
 elif any(x in p for x in ['case0507','case0508','case0509','case0510']):reason='Nonzero/argwhere intermediate cardinality and crop extrema depend on payload; fixing crop endpoints alone would leave the index buffer oversized.'
 elif 'periodic_dist' in p:reason='Masked intermediate arrays are empty on both fixtures; scalar L specialization does not constrain payload-dependent masks or remove their envelope.'
 elif 'tax' in p:reason='Several mask results and dependent arithmetic arrays are empty; a count assertion does not change mask-compaction capacity.'
 elif 'projection_simplex' in p:reason='Two filtered intermediates depend on a predicate over sorted payload values; fixing z does not fix their lengths.'
 else:reason='Boolean filtering or nonzero/argwhere cardinality depends on payload; count preconditions do not resize the current compaction construction.'
 lines.append(f'| {p} | {reason} |')
lines += ['', '## Validation and reproduction','',
f'`validation/*.json` contains {sum(len(v["checks"])for v in validations)} successful A/B length-and-shape checks for {sum(len(v["checks"])for v in validations)//2} exposed internal/source array occurrences. The other three occurrences are the original branch-merged crop outputs. In addition, `original_validation/` checks all 36 original programs on A and B: mock constraints are satisfied, every logical output matches NumPy, and all 62 dynamic output shapes/lengths match. `cost_validation.json` contains 34 successful valid-domain mock checks, exact output-capacity checks, and rejection checks after changing a constrained shape control. Cost result files also contain NumPy logical-output comparisons for original A/B and each specialized fixture. Mock satisfaction is checked using `proof_bytes_hex == "mock_satisfied"`; no cryptographic proof is generated.','',
'`manifest.json` records the compiler/extension hash, tool versions and hashes, acceptance hash, and every kernel/fixture hash. All artifacts and generated scripts are under this directory. Measured on the remote host shuaicpu6 (container zhantong-dev, no systemd): every process runs under `ulimit -v` (32 GiB virtual memory), with 16 workers per stage. Stages must run sequentially. Resource deadlines are failure guards, not performance measurements.','',
'From this folder, run:', '', '```sh','bash reproduce.sh','```','',
'The scripts reuse existing census/probe files. For an independent fresh run, move the generated `census/`, `probes/`, `validation/`, and `costs/` directories aside within B before running. To regenerate just this report from the saved measurements:', '', '```sh','/home/zhantong/zinnia-eval-run/zinnia-sizefix/.venv/bin/python -B report.py','```','',
'## Per-array list','',
'Array identifiers are source span `line:column:end_line:end_column:AST-kind`, invocation ordinal, and tuple/list element path. For windows, the element path distinguishes each returned window. Capacity and length are element counts; ratios are also in `arrays.csv`. The full compiler layout and NumPy shapes are retained in `arrays.json`.','',
'| Program | Array ID | Expression | Capacity | Actual A | Actual B | C/A | C/B |','|---|---|---|---:|---:|---:|---:|---:|']
for r in rows:
 expr=r['expression'].replace('|','&#124;');lines.append(f'| {r["program"]} | {r["array"]} | `{expr}` | {r["capacity"]} | {r["lengths"]["A"]} | {r["lengths"]["B"]} | {fmt(ratio(r,"A"))} | {fmt(ratio(r,"B"))} |')
lines += ['', '## Full accepted-program audit','', '| Program | Bounded array values | Note |','|---|---:|---|']
for a in audit:lines.append(f'| {a["id"]} | {a["arrays"]} | {(a.get("repair") or "compiled; expression probe").replace("|","/")} |')
(HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(safe(summary),indent=2))
