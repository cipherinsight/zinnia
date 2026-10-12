"""Source-reviewed static cases where exposing every temporary causes excessive output."""
import measure,json,ast
cases={'npbench_scattering_self_energies_scattering_self_energies':'All array extents are input shape dimensions or literals. Loop indices range over fixed input shape extents; indexed/subarray temporary shapes are fixed.', 'pythran_matmul':'zero(len(m0), len(m1[0])) constructs a static nested list; all loop bounds are fixed input lengths and all arithmetic is scalar indexing. No value-dependent array shape.'}
for p in measure.entries():
 if p.program not in cases:continue
 c=json.loads((measure.HERE/'census'/(measure.ident(p)+'.json')).read_text())
 assert not any('dynamic'in x for x in c['layout'])
 (measure.HERE/'probes'/(measure.ident(p)+'.json')).write_text(json.dumps(dict(id=measure.ident(p),dynamic=[],layout=[],repair=cases[p.program],source=p.kernel_path.read_text()),indent=2))
