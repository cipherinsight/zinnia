from pathlib import Path
import shutil, subprocess, os
ROOT=Path(os.environ.get('ZINNIA_CHECKOUT','/home/zhantong/zinnia-eval-run/zinnia-sizefix'))
OUT=Path(__file__).resolve().parent
COPY=OUT/'compiler'
COPY.mkdir(exist_ok=True)
for f in ['Cargo.toml','Cargo.lock','build.rs']: shutil.copy2(ROOT/f,COPY/f)
for f in ['src','zinnia']: shutil.copytree(ROOT/f,COPY/f,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.so'))
p=COPY/'src/ir_gen/mod.rs'; t=p.read_text(); needle='    fn visit(&mut self, node: &ASTNode) -> Value {'
t=t.replace(needle,'''    fn visit(&mut self, node: &ASTNode) -> Value {
        let result = self.visit_audit_original(node);
        crate::types::composite::audit_value(&result);
        result
    }
    fn visit_audit_original(&mut self, node: &ASTNode) -> Value {''')
p.write_text(t)
p=COPY/'src/types/composite.rs'; t=p.read_text(); t=t.replace('        self.envelope.total_bound','        audit_dynamic(self);\n        self.envelope.total_bound')
t+='''
// Measurement only: observe compiler values, without allocating compiler IDs or IR.
pub(crate) fn audit_value(value: &Value) {
    match value {
        Value::DynamicNDArray(d) => audit_dynamic(d),
        Value::List(d) | Value::Tuple(d) => { for v in &d.values { audit_value(v); } },
        _ => {},
    }
}
fn audit_dynamic(d: &DynamicNDArrayData) {
    let Some(path) = std::env::var_os("ZINNIA_AUDIT_DYNAMIC") else { return };
    use std::io::Write;
    let scalar = |v: &ScalarValue<i64>| serde_json::json!({"constant": v.static_val, "stmt": v.stmt_id});
    let row = serde_json::json!({
        "value_id": format!("{:?}", d.value_id), "segment": d.segment_id,
        "bound": d.envelope.total_bound,
        "length": scalar(&d.meta.runtime_length), "rank": scalar(&d.meta.runtime_rank),
        "shape": d.meta.runtime_shape.iter().map(scalar).collect::<Vec<_>>(),
        "offset": scalar(&d.meta.runtime_offset),
    });
    // Log each distinct metadata record once per worker process.
    static SEEN: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> = std::sync::OnceLock::new();
    let line = row.to_string();
    if SEEN.get_or_init(Default::default).lock().unwrap().insert(line.clone()) {
        let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();
        writeln!(file, "{}", line).unwrap();
    }
}
'''
p.write_text(t)
# Add structured pre-lowering IR export (same graph as existing text export).
p=COPY/'src/lib.rs';t=p.read_text();needle='fn dump_layer_input(graph: &IRGraph) {'
t=t.replace(needle,needle+'''
    if let Some(path) = std::env::var_os("ZINNIA_AUDIT_IR") {
        std::fs::write(path, serde_json::to_string(&graph.export_stmts()).unwrap()).unwrap();
    }
''');p.write_text(t)
# Save exact instrumentation patch for review.
with (OUT/'instrumentation.diff').open('w') as f:
 for name in ['src/ir_gen/mod.rs','src/types/composite.rs','src/lib.rs']:
  subprocess.run(['diff','-u',str(ROOT/name),str(COPY/name)],stdout=f)
# Capture compaction before it becomes scalar one-hot selects (no ArrayOp survives).
p=COPY/'src/ops/dyn_ndarray/memory_ops.rs';t=p.read_text();needle=') -> Compaction {\n'
assert t.count(needle)==1
t=t.replace(needle,needle+'''    if mask_cells.iter().any(|v| v.bool_val().is_none() && v.int_val().is_none()) {
        if let Some(path) = std::env::var_os("ZINNIA_AUDIT_COMPACTION") {
            use std::io::Write;
            let row = serde_json::json!({"rows": mask_cells.len(), "trailing": trailing_total,
                "runtime_masks": mask_cells.iter().filter_map(|v| v.stmt_id()).collect::<Vec<_>>()});
            let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();
            writeln!(file, "{}", row).unwrap();
        }
    }
''');p.write_text(t)
with (OUT/'instrumentation.diff').open('a') as f:
 subprocess.run(['diff','-u',str(ROOT/'src/ops/dyn_ndarray/memory_ops.rs'),str(p)],stdout=f)
