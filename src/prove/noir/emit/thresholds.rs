//! Reuse one checked bit decomposition across many constant thresholds.
use super::*;

pub(super) fn plan(ir: &IRGraph) -> HashSet<u32> {
    use crate::ir_defs::IR;
    let mut groups = HashMap::<u32, Vec<u32>>::new();
    for stmt in &ir.stmts {
        if !matches!(stmt.ir, IR::LtI | IR::LteI | IR::GtI | IR::GteI) { continue }
        let [a, b] = stmt.arguments.as_slice() else { continue };
        let is_constant = |i: u32| matches!(ir.stmts[i as usize].ir, IR::ConstantInt { .. });
        let x = match (is_constant(*a), is_constant(*b)) {
            (true, false) => *b, (false, true) => *a, _ => continue,
        };
        groups.entry(x).or_default().push(stmt.stmt_id);
    }
    // A shared decision DAG pays off only for a large family of thresholds.
    groups.into_values().filter(|v| v.len() >= 32).flatten().collect()
}

impl NoirEmitter<'_> {
    pub(super) fn shared_threshold_compare(&mut self, a: &str, b: &str, cmp: Cmp) -> Option<String> {
        if !self.current_stmt.is_some_and(|id| self.threshold_plan.contains(&id)) { return None }
        let (x, k, cmp) = match (self.raw_constant(a), self.raw_constant(b)) {
            (None, Some(k)) => (a, k, cmp),
            (Some(k), None) => (b, k, cmp.mirrored()),
            _ => return None,
        };
        let (lo, hi) = self.known_range(x)?;
        // Bounds must be established by constraints, not speculative analysis.
        // All original comparison gaps then fit their signed/unsigned windows.
        if lo < 0 || hi >= 1 << 12 || k < 0 || k > hi { return None }
        let width = (128 - (hi as u128).leading_zeros()).max(1);
        let (limit, negate) = match cmp {
            Cmp::Gt => (k, false), Cmp::Le => (k, true),
            Cmp::Ge => (k - 1, false), Cmp::Lt => (k - 1, true),
        };
        if limit < 0 { return Some(self.literal_int((!negate) as i64)); }
        let (bits, width) = if let Some(pair) = self.threshold_bits.get(x) {
            pair.clone()
        } else {
            let bits = self.fresh();
            let _ = writeln!(self.body, "    let {bits}: [bool; {width}] = {x}.to_le_bits();");
            self.threshold_bits.insert(x.to_string(), (bits.clone(), width));
            (bits, width)
        };
        // Starting at the least significant bit, extend [x_low > k_low]:
        // k_i=0 gives x_i OR previous; k_i=1 gives x_i AND previous.
        // Text CSE shares identical lower-bit decisions across thresholds.
        let mut greater = "false".to_string();
        for i in 0..width {
            let bit = format!("{bits}[{i}]");
            if limit & (1i128 << i) != 0 {
                if greater != "false" { greater = self.bind_typed(&format!("{bit} & {greater}"), "bool"); }
            } else if greater == "false" {
                greater = bit;
            } else {
                greater = self.bind_typed(&format!("{bit} | {greater}"), "bool");
            }
        }
        let flag = if negate { format!("(!{greater}) as Field") } else { format!("({greater}) as Field") };
        Some(self.bind_flag(&flag))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parity_emitter_thresholds_require_checked_ranges_and_share_bits() {
        let prov = crate::provenance::ProvenanceTable::default();
        let mut em = NoirEmitter::new(Vec::new(), HashMap::new(), &prov);
        em.threshold_plan.insert(5);
        em.current_stmt = Some(5);
        em.interval_of.insert("x".into(), (0, 1024));
        let k = em.constant_int(10).unwrap();
        assert!(em.shared_threshold_compare("x", &k, Cmp::Gt).is_none());
        em.derived_range.insert("x".into(), (0, 1024));
        let gt = em.shared_threshold_compare("x", &k, Cmp::Gt).unwrap();
        assert_eq!(Some(gt), em.shared_threshold_compare(&k, "x", Cmp::Lt));
        let le = em.shared_threshold_compare("x", &k, Cmp::Le).unwrap();
        assert_eq!(em.known_range(&le), Some((0, 1)));
        assert_eq!(em.body.matches("x.to_le_bits()").count(), 1);
        assert!(em.body.contains("[bool; 11]"));
        em.current_stmt = None;
        assert!(em.shared_threshold_compare("x", &k, Cmp::Gt).is_none());
    }
}
