//! Fuse private constant min/max chains after the input window is checked.
use super::*;

#[derive(Clone)]
pub(super) struct Plan {
    root: crate::types::ValueId,
    low: i128,
    high: i128,
    finish: u32,
}

pub(super) fn plan(ir: &IRGraph) -> HashMap<u32, Plan> {
    use crate::ir_defs::IR;
    let mut uses = HashMap::<u32, usize>::new();
    for s in &ir.stmts { for a in &s.arguments { *uses.entry(*a).or_default() += 1; } }
    // Recognize select(x < c, x, c) = min and select(x > c,x,c) = max,
    // including swapped branches/operands and equality at the endpoint.
    let extreme = |id: u32| -> Option<(u32, i128, bool, u32)> {
        let s = ir.stmts.get(id as usize)?;
        if !matches!(s.ir, IR::SelectI | IR::SelectF) { return None }
        let [cmp, t, f] = s.arguments.as_slice() else { return None };
        let c = ir.stmts.get(*cmp as usize)?;
        let [a, b] = c.arguments.as_slice() else { return None };
        let less = match c.ir {
            IR::LtI | IR::LteI | IR::LtF | IR::LteF => true,
            IR::GtI | IR::GteI | IR::GtF | IR::GteF => false,
            _ => return None,
        };
        let min = if t == a && f == b { less }
            else if t == b && f == a { !less } else { return None };
        let constant = |i: u32| match ir.stmts.get(i as usize).map(|s| &s.ir) {
            Some(IR::ConstantInt { value }) => Some(*value as i128),
            Some(IR::ConstantFloat { value }) => Some(crate::prove::field::quantize(*value, Q_BITS)),
            _ => None,
        };
        let (x, k) = constant(*a).map(|k| (*b, k)).or_else(|| constant(*b).map(|k| (*a, k)))?;
        Some((x, k, min, *cmp))
    };
    let mut result = HashMap::new();
    for s in &ir.stmts {
        let Some((inner, outer_k, outer_min, outer_cmp)) = extreme(s.stmt_id) else { continue };
        let Some((root, inner_k, inner_min, inner_cmp)) = extreme(inner) else { continue };
        if inner_min == outer_min || uses.get(&inner) != Some(&2)
            || uses.get(&inner_cmp) != Some(&1) || uses.get(&outer_cmp) != Some(&1) { continue }
        let (low, high) = if outer_min { (inner_k, outer_k) } else { (outer_k, inner_k) };
        if low < -(1i128 << 63) || high >= 1i128 << 63 || high < low || high - low >= 1i128 << 32 { continue }
        let p = Plan { root: ir.stmts[root as usize].value_id, low, high, finish: s.stmt_id };
        for id in [inner_cmp, inner, outer_cmp, s.stmt_id] { result.insert(id, p.clone()); }
    }
    result
}

impl NoirEmitter<'_> {
    pub(super) fn clip_override(&mut self, comparison: bool) -> Option<String> {
        let id = self.current_stmt?;
        let p = self.clip_plan.get(&id)?.clone();
        let root = self.var_of_vid.get(&p.root)?.clone();
        let (lo, hi) = self.known_range(&root)?;
        if lo < -(1i128 << 63) || hi >= 1i128 << 63 { return None }
        if comparison { return Some(self.literal_int(0)); }
        if id != p.finish { return Some(self.bind(&root)); }
        self.used_clip_helper = true;
        let k = (1u128 << 64) / (p.high - p.low + 1) as u128 + 1;
        let out = self.bind(&format!("zinnia_clip_const({root}, {}, {}, {k})",
            signed_to_field_decimal(p.low), signed_to_field_decimal(p.high)));
        self.proven_range.insert(out.clone(), (p.low, p.high));
        Some(out)
    }
}

pub(super) const HELPERS: &str = r#"
unconstrained fn zinnia_clip_hint(x: Field, lo: Field, hi: Field) -> (Field, Field) {
    let w = (x + 9223372036854775808) as u64;
    let l = (lo + 9223372036854775808) as u64;
    let h = (hi + 9223372036854775808) as u64;
    ((w < l) as Field, (w > h) as Field)
}
fn zinnia_clip_const(x: Field, lo: Field, hi: Field, k: Field) -> Field {
    let (l, h) = unsafe { zinnia_clip_hint(x, lo, hi) };
    assert(l * (l - 1) == 0);
    assert(h * (h - 1) == 0);
    assert(l * h == 0);
    let m = 1 - l - h;
    let d = hi - lo;
    let u = l * (lo - 1 - x) + h * (x - hi - 1) + m * (d * k - (x - lo) * (k - 1));
    u.assert_max_bit_size::<64>();
    x + l * (lo - x) + h * (hi - x)
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir::IRStatement, ir_defs::IR, types::ValueId};

    #[test]
    fn parity_emitter_clip_requires_private_chain_and_checked_input() {
        let specs = vec![
            (IR::ReadInteger { path: InputPath::new("x", vec![]), is_public: false }, vec![]),
            (IR::ConstantInt { value: 2 }, vec![]),
            (IR::GtI, vec![0, 1]),
            (IR::SelectI, vec![2, 0, 1]),
            (IR::ConstantInt { value: 10 }, vec![]),
            (IR::LtI, vec![3, 4]),
            (IR::SelectI, vec![5, 3, 4]),
        ];
        let mut ir = IRGraph::new(specs.into_iter().enumerate().map(|(i, (ir, args))| {
            let values = args.iter().map(|a| ValueId(*a as u64)).collect();
            IRStatement::new(i as u32, ValueId(i as u64), ir, args, values, None)
        }).collect());
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = NoirEmitter::new(Vec::new(), HashMap::new(), &prov);
        em.clip_plan = plan(&ir);
        assert_eq!(em.clip_plan.len(), 4);
        em.var_of_vid.insert(ValueId(0), "x".into());
        em.current_stmt = Some(2);
        assert!(em.clip_override(true).is_none());
        em.record_signed64("x");
        assert!(em.clip_override(true).is_some());
        em.current_stmt = Some(3);
        assert!(em.clip_override(false).is_some());
        em.current_stmt = Some(6);
        let clipped = em.clip_override(false).unwrap();
        assert_eq!(em.known_range(&clipped), Some((2, 10)));
        assert!(em.body.contains("zinnia_clip_const(x, 2, 10,"));
        ir.stmts.push(IRStatement::new(7, ValueId(7), IR::AddI, vec![3, 0], vec![ValueId(3), ValueId(0)], None));
        assert!(plan(&ir).is_empty());
    }
}
