//! End-to-end checks of the emitted Noir: each test builds a small IR graph,
//! runs the circuit with `nargo execute` (and `bb` where noted) and checks
//! that the honest input gives the mock's answer while a malicious input is
//! rejected by a failing constraint. Tests skip (pass with a notice) when
//! the toolchain is missing.
//!
//! The execution helper stages the package itself from `emit_circuit`'s
//! output, so the same tests also run against the emitter before these
//! fixes (that is how their "fails before" status was checked).

use crate::circuit_input::{CircuitInputs, InputNode, InputParam, InputPath, ResolvedWitness};
use crate::ir::{IRGraph, IRStatement};
use crate::ir_defs::IR;
use crate::prove::traits::ProverBackend;
use crate::prove::types::ProvingParams;
use crate::provenance::{AffineView, Interval};
use crate::types::ValueId;

use super::driver::DriverPaths;
use super::emit::emit_circuit;
use super::NoirProverBackend;

/// Small IR graph builder: statement ids are indices, value ids are fresh.
#[derive(Default)]
struct Graph {
    stmts: Vec<IRStatement>,
    vids: Vec<ValueId>,
}

impl Graph {
    fn push(&mut self, ir: IR, args: &[u32]) -> u32 {
        let id = self.stmts.len() as u32;
        let vid = ValueId::next();
        let arg_values = args.iter().map(|a| self.vids[*a as usize]).collect();
        self.stmts
            .push(IRStatement::new(id, vid, ir, args.to_vec(), arg_values, None));
        self.vids.push(vid);
        id
    }

    fn read_int(&mut self, name: &str) -> u32 {
        self.push(
            IR::ReadInteger {
                path: InputPath::new(name, vec![]),
                is_public: false,
            },
            &[],
        )
    }

    fn int(&mut self, value: i64) -> u32 {
        self.push(IR::ConstantInt { value }, &[])
    }

    fn expose(&mut self, cell: u32) {
        self.push(IR::ExposePublicI, &[cell]);
    }

    fn vid(&self, id: u32) -> ValueId {
        self.vids[id as usize]
    }

    fn finish(self) -> IRGraph {
        IRGraph::new(self.stmts)
    }
}

fn witness(ints: &[(&str, i64)]) -> ResolvedWitness {
    let mut inputs = CircuitInputs::new();
    for (name, value) in ints {
        inputs.params.push(InputParam {
            name: name.to_string(),
            is_public: false,
            dtype: serde_json::Value::Null,
            value: InputNode::Int(*value),
        });
    }
    ResolvedWitness::new(inputs, 32)
}

fn toolchain_present(need_bb: bool) -> bool {
    let paths = DriverPaths::default();
    let ok = paths.nargo.exists() && (!need_bb || paths.bb.exists());
    if !ok {
        eprintln!("skipping: Noir toolchain not installed");
    }
    ok
}

/// Signed reading of a `0x`-prefixed big-endian field element (the mock's
/// `fp_to_i64`).
fn signed(hex: &str) -> i128 {
    use crate::prove::field::{PrimeField, F};
    let digits = format!("{:0>64}", hex.trim_start_matches("0x"));
    let mut le = [0u8; 32];
    for i in 0..32 {
        le[31 - i] = u8::from_str_radix(&digits[2 * i..2 * i + 2], 16).unwrap();
    }
    let mut repr = <F as PrimeField>::Repr::default();
    repr.as_mut().copy_from_slice(&le);
    let fp = Option::<F>::from(F::from_repr(repr)).expect("canonical field element");
    crate::prove::kernel::fp_to_i64(fp) as i128
}

/// Emit, stage and `nargo execute` the circuit. `Ok` holds the returned
/// values; `Err` holds nargo's output when execution failed.
fn execute(graph: &IRGraph, w: &ResolvedWitness) -> Result<Vec<i128>, String> {
    let out = emit_circuit(graph, w, &ProvingParams::default()).map_err(|e| e.to_string())?;
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("zinnia-noir-test-{}-{}", std::process::id(), nanos));
    std::fs::create_dir_all(dir.join("src")).unwrap();
    std::fs::create_dir_all(dir.join("noir_fpc").join("src")).unwrap();
    std::fs::write(
        dir.join("Nargo.toml"),
        "[package]\nname = \"zinnia_circuit\"\ntype = \"bin\"\nauthors = [\"\"]\n\n\
         [dependencies]\nnoir_fpc = { path = \"./noir_fpc\" }\n",
    )
    .unwrap();
    std::fs::write(dir.join("src").join("main.nr"), &out.main_nr).unwrap();
    std::fs::write(dir.join("Prover.toml"), &out.prover_toml).unwrap();
    std::fs::write(
        dir.join("noir_fpc").join("Nargo.toml"),
        include_str!("../../../benchmarking/noir_fpc/Nargo.toml"),
    )
    .unwrap();
    std::fs::write(
        dir.join("noir_fpc").join("src").join("lib.nr"),
        include_str!("../../../benchmarking/noir_fpc/src/lib.nr"),
    )
    .unwrap();
    let run = std::process::Command::new(&DriverPaths::default().nargo)
        .args(["execute", "--overwrite-return"])
        .current_dir(&dir)
        .output()
        .unwrap();
    let result = if run.status.success() {
        let toml = std::fs::read_to_string(dir.join("Prover.toml")).unwrap();
        let line = toml.lines().find(|l| l.starts_with("return =")).unwrap_or("");
        Ok(line
            .split(|c| c == '[' || c == ']' || c == ',' || c == '=')
            .map(|t| t.trim().trim_matches('"'))
            .filter(|t| t.starts_with("0x"))
            .map(signed)
            .collect())
    } else {
        Err(format!(
            "{}{}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        ))
    };
    let _ = std::fs::remove_dir_all(&dir);
    result
}

/// The input was rejected by a failing constraint (not, say, by a parse
/// error in Prover.toml).
fn assert_rejected(result: Result<Vec<i128>, String>) {
    match result {
        Ok(vals) => panic!("malicious input accepted, returned {:?}", vals),
        Err(msg) => assert!(
            ["Failed constraint", "Failed assertion", "Cannot satisfy constraint", "Assertion failed"]
                .iter()
                .any(|m| msg.contains(m)),
            "rejected for another reason: {}",
            msg
        ),
    }
}

/// B4(a): `AssertNonNegI` used to be lowered through the interval the
/// builder derives from that very assert (a `u64 >= 0` compare of a
/// truncated cast). It must reject a negative value.
#[test]
fn assert_non_neg_rejects_negative_input() {
    if !toolchain_present(false) {
        return;
    }
    let mut g = Graph::default();
    let x = g.read_int("x");
    g.push(IR::AssertNonNegI, &[x]);
    g.expose(x);
    let vx = g.vid(x);
    let mut graph = g.finish();
    // What IRBuilder::deposit_provenance records for this assert.
    graph.provenance.set_interval(vx, Interval { lo: 0, hi: i64::MAX });
    assert_eq!(execute(&graph, &witness(&[("x", 3)])).unwrap(), vec![3]);
    assert_rejected(execute(&graph, &witness(&[("x", -3)])));
}

/// B4(b): with u32 promotion on (>= 20 intrinsic ops, an emitter policy
/// since removed), `AssertU128Lt` used to check only the low 32 bits of the
/// index (`idx as u32 < hi`).
#[test]
fn assert_u128_lt_rejects_index_above_u32() {
    if !toolchain_present(false) {
        return;
    }
    let mut g = Graph::default();
    let idx = g.read_int("idx");
    let hi = g.int(10);
    for _ in 0..20 {
        g.push(IR::AssertU128Lt, &[idx, hi]);
    }
    g.expose(idx);
    let vidx = g.vid(idx);
    let mut graph = g.finish();
    graph.provenance.set_interval(vidx, Interval { lo: 0, hi: 9 });
    assert_eq!(execute(&graph, &witness(&[("idx", 3)])).unwrap(), vec![3]);
    assert_rejected(execute(&graph, &witness(&[("idx", (1i64 << 32) + 3)])));
    assert_rejected(execute(&graph, &witness(&[("idx", -1)])));
}

/// `0 <= idx < 8` is one 3-bit range check (a power-of-two bound needs no
/// check of `7 - idx`): 7 is accepted, 8 and -1 are rejected.
#[test]
fn power_of_two_bound_is_one_range_check() {
    if !toolchain_present(false) {
        return;
    }
    let mut g = Graph::default();
    let idx = g.read_int("idx");
    let hi = g.int(8);
    g.push(IR::AssertU128Lt, &[idx, hi]);
    g.expose(idx);
    let graph = g.finish();
    let honest = witness(&[("idx", 7)]);
    let emitted = emit_circuit(&graph, &honest, &ProvingParams::default()).unwrap();
    assert_eq!(emitted.main_nr.matches("assert_max_bit_size").count(), 1, "{}", emitted.main_nr);
    assert_eq!(execute(&graph, &honest).unwrap(), vec![7]);
    assert_rejected(execute(&graph, &witness(&[("idx", 8)])));
    assert_rejected(execute(&graph, &witness(&[("idx", -1)])));
}

/// NumPy's index wrap-around `y = x + 8 if x < 0 else x` with `y < 8`
/// asserted: the comparison `x < 0` uses an 8-bit gap check (every accepted
/// x lies in [-8, 7]); in-range indices of both signs are accepted, and
/// out-of-range ones (including one far outside 8 bits) are rejected.
#[test]
fn wrapped_index_comparison_is_narrow_and_sound() {
    if !toolchain_present(false) {
        return;
    }
    let mut g = Graph::default();
    let x = g.read_int("x");
    let zero = g.int(0);
    let n = g.int(8);
    let negative = g.push(IR::LtI, &[x, zero]);
    let shifted = g.push(IR::AddI, &[x, n]);
    let wrapped = g.push(IR::SelectI, &[negative, shifted, x]);
    g.push(IR::AssertU128Lt, &[wrapped, n]);
    g.expose(wrapped);
    let graph = g.finish();
    let honest = witness(&[("x", -3)]);
    let emitted = emit_circuit(&graph, &honest, &ProvingParams::default()).unwrap();
    assert!(emitted.main_nr.contains("qlt_bits::<8>("), "{}", emitted.main_nr);
    assert_eq!(execute(&graph, &honest).unwrap(), vec![5]);
    assert_eq!(execute(&graph, &witness(&[("x", 7)])).unwrap(), vec![7]);
    assert_eq!(execute(&graph, &witness(&[("x", -8)])).unwrap(), vec![0]);
    assert_rejected(execute(&graph, &witness(&[("x", 8)])));
    assert_rejected(execute(&graph, &witness(&[("x", -9)])));
    assert_rejected(execute(&graph, &witness(&[("x", 1 << 40)])));
}

/// Equality tests of a wrapped index y = x + 5 if x < 0 else x against
/// each slot j are one zero test of (x - j)(x - j + 5); they equal the
/// mock's for indices of both signs, and in-range slot tests stay exact for
/// an out-of-range x (no slot fires for x = 5 or -6).
#[test]
fn wrapped_index_equalities_match_the_mock() {
    if !toolchain_present(false) {
        return;
    }
    let mut g = Graph::default();
    let x = g.read_int("x");
    let zero = g.int(0);
    let n = g.int(5);
    let negative = g.push(IR::LtI, &[x, zero]);
    let shifted = g.push(IR::AddI, &[x, n]);
    let wrapped = g.push(IR::SelectI, &[negative, shifted, x]);
    for j in 0..5 {
        let slot = g.int(j);
        let hit = g.push(IR::EqI, &[wrapped, slot]);
        g.expose(hit);
    }
    let graph = g.finish();
    let emitted = emit_circuit(&graph, &witness(&[("x", 0)]), &ProvingParams::default()).unwrap();
    assert!(emitted.main_nr.contains(") * ("), "{}", emitted.main_nr);
    for value in [-5, -2, 0, 3, 4, 5, -6] {
        let w = witness(&[("x", value)]);
        let (satisfied, mock) = mock_outputs(&graph, &w);
        assert!(satisfied);
        let noir: Vec<i128> = execute(&graph, &w).unwrap();
        assert_eq!(noir, mock, "x = {value}");
    }
}

/// A witness whose inputs are raw field elements (hex), e.g. values outside
/// the signed 64-bit window.
fn raw_witness(inputs: &[(&str, &str)]) -> ResolvedWitness {
    let mut w = CircuitInputs::new();
    for (name, hex) in inputs {
        w.params.push(InputParam {
            name: name.to_string(),
            is_public: false,
            dtype: serde_json::Value::Null,
            value: InputNode::Hex(hex.to_string()),
        });
    }
    ResolvedWitness::new(w, 32)
}

/// The input-domain checks `x >= i64::MIN` and `x <= i64::MAX` (each used
/// only by an assert) cost one 64-bit check of x - i64::MIN: the second is
/// decided by the range the first proves. An asserted `y < 10` is one check
/// of 9 - y. Values in the window pass, values outside it are rejected.
#[test]
fn asserted_comparisons_check_their_outcome_directly() {
    if !toolchain_present(false) {
        return;
    }
    let mut g = Graph::default();
    let x = g.read_int("x");
    let y = g.read_int("y");
    let min = g.int(i64::MIN);
    let max = g.int(i64::MAX);
    let above = g.push(IR::GteI, &[x, min]);
    g.push(IR::Assert, &[above]);
    let below = g.push(IR::LteI, &[x, max]);
    g.push(IR::Assert, &[below]);
    let ten = g.int(10);
    let small = g.push(IR::LtI, &[y, ten]);
    g.push(IR::Assert, &[small]);
    g.expose(x);
    g.expose(y);
    let graph = g.finish();
    let honest = witness(&[("x", -5), ("y", 9)]);
    let emitted = emit_circuit(&graph, &honest, &ProvingParams::default()).unwrap();
    assert_eq!(emitted.main_nr.matches("assert_max_bit_size::<64>").count(), 2, "{}", emitted.main_nr);
    assert!(!emitted.main_nr.contains("qlt_bits"), "{}", emitted.main_nr);
    assert_eq!(execute(&graph, &honest).unwrap(), vec![-5, 9]);
    for x in [i64::MIN, i64::MAX] {
        assert_eq!(execute(&graph, &witness(&[("x", x), ("y", -3)])).unwrap(), vec![x as i128, -3]);
    }
    assert_rejected(execute(&graph, &witness(&[("x", 0), ("y", 10)])));
    // 2^63 and -2^63 - 1 (p - 2^63 - 1) leave the window.
    let p_minus = "0x30644e72e131a029b85045b68181585d2833e84879b97090c3e1f593f0000000";
    assert_rejected(execute(&graph, &raw_witness(&[("x", "0x8000000000000000"), ("y", "0x0")])));
    assert_rejected(execute(&graph, &raw_witness(&[("x", p_minus), ("y", "0x0")])));
}

/// An input's domain checks are implied by a bounds check of the same input
/// (`AssertU128Lt(x, 8)`): the circuit has the bounds check only, which
/// accepts 0..7 and rejects 8, -1 and a value outside the 64-bit window.
#[test]
fn domain_checks_implied_by_a_bounds_check_are_dropped() {
    if !toolchain_present(false) {
        return;
    }
    let mut g = Graph::default();
    let x = g.read_int("x");
    let min = g.int(i64::MIN);
    let max = g.int(i64::MAX);
    let above = g.push(IR::GteI, &[x, min]);
    g.push(IR::Assert, &[above]);
    let below = g.push(IR::LteI, &[x, max]);
    g.push(IR::Assert, &[below]);
    let eight = g.int(8);
    g.push(IR::AssertU128Lt, &[x, eight]);
    g.expose(x);
    let graph = g.finish();
    let honest = witness(&[("x", 7)]);
    let emitted = emit_circuit(&graph, &honest, &ProvingParams::default()).unwrap();
    assert_eq!(emitted.main_nr.matches("assert_max_bit_size").count(), 1, "{}", emitted.main_nr);
    assert!(emitted.main_nr.contains("assert_max_bit_size::<3>"), "{}", emitted.main_nr);
    assert_eq!(execute(&graph, &honest).unwrap(), vec![7]);
    assert_eq!(execute(&graph, &witness(&[("x", 0)])).unwrap(), vec![0]);
    assert_rejected(execute(&graph, &witness(&[("x", 8)])));
    assert_rejected(execute(&graph, &witness(&[("x", -1)])));
    assert_rejected(execute(&graph, &raw_witness(&[("x", "0x8000000000000000")])));
}

/// M5: a dynamic-address write outside the segment used to update no slot
/// and succeed; the mock rejects it.
#[test]
fn dynamic_write_out_of_range_is_rejected() {
    if !toolchain_present(false) {
        return;
    }
    let mut g = Graph::default();
    g.push(
        IR::AllocateMemory {
            segment_id: 0,
            size: 4,
            init_value: 0,
        },
        &[],
    );
    let i = g.read_int("i");
    let seven = g.int(7);
    g.push(IR::WriteMemory { segment_id: 0 }, &[i, seven]);
    let zero = g.int(0);
    let r = g.push(IR::ReadMemory { segment_id: 0 }, &[zero]);
    g.expose(r);
    let graph = g.finish();
    assert_eq!(execute(&graph, &witness(&[("i", 0)])).unwrap(), vec![7]);
    assert_eq!(execute(&graph, &witness(&[("i", 2)])).unwrap(), vec![0]);
    assert_rejected(execute(&graph, &witness(&[("i", 4)])));
    assert_rejected(execute(&graph, &witness(&[("i", -1)])));
}

/// M5: a constant-address write past the segment end is a compile error
/// (it used to be dropped silently).
#[test]
fn constant_write_out_of_range_is_an_error() {
    let mut g = Graph::default();
    g.push(
        IR::AllocateMemory {
            segment_id: 0,
            size: 2,
            init_value: 0,
        },
        &[],
    );
    let addr = g.int(5);
    let v = g.int(1);
    g.push(IR::WriteMemory { segment_id: 0 }, &[addr, v]);
    let graph = g.finish();
    let err = super::emit::emit_circuit(&graph, &witness(&[]), &ProvingParams::default());
    assert!(err.is_err());
}

/// M5: a two-root gather checked each root against its own value set but not
/// the combined address, so an out-of-range combination gathered 0.
#[test]
fn two_root_read_out_of_range_is_rejected() {
    if !toolchain_present(false) {
        return;
    }
    let mut g = Graph::default();
    g.push(
        IR::AllocateMemory {
            segment_id: 0,
            size: 6,
            init_value: 0,
        },
        &[],
    );
    for slot in 0..6 {
        let a = g.int(slot);
        let v = g.int(10 + slot);
        g.push(IR::WriteMemory { segment_id: 0 }, &[a, v]);
    }
    let r = g.read_int("r");
    let c = g.read_int("c");
    let four = g.int(4);
    let row = g.push(IR::MulI, &[r, four]);
    let addr = g.push(IR::AddI, &[row, c]);
    let read = g.push(IR::ReadMemory { segment_id: 0 }, &[addr]);
    g.expose(read);
    let (vr, vc, vaddr) = (g.vid(r), g.vid(c), g.vid(addr));
    let mut graph = g.finish();
    graph.provenance.set_affine(
        vaddr,
        AffineView {
            terms: vec![(vr, 4), (vc, 1)],
            offset: 0,
        },
    );
    assert_eq!(execute(&graph, &witness(&[("r", 1), ("c", 1)])).unwrap(), vec![15]);
    // r = 1 and c = 3 are each feasible for some slot, but 4*1 + 3 = 7 >= 6.
    assert_rejected(execute(&graph, &witness(&[("r", 1), ("c", 3)])));
}

/// Dynamic reads through a native Noir array (chosen when several dynamic
/// reads share one array): honest addresses read the written values, and
/// the addresses -1 and len are rejected by `array_index`'s own checks (the
/// pinned bb does not bound a memory index at p - 1 by itself).
#[test]
fn native_array_read_rejects_out_of_range_addresses() {
    if !toolchain_present(false) {
        return;
    }
    let mut g = Graph::default();
    g.push(
        IR::AllocateMemory {
            segment_id: 0,
            size: 64,
            init_value: 0,
        },
        &[],
    );
    for slot in 0..64 {
        let a = g.int(slot);
        let v = g.int(10 + slot);
        g.push(IR::WriteMemory { segment_id: 0 }, &[a, v]);
    }
    let addrs: Vec<u32> = ["x", "y"].iter().map(|n| g.read_int(n)).collect();
    for addr in addrs {
        let read = g.push(IR::ReadMemory { segment_id: 0 }, &[addr]);
        g.expose(read);
    }
    let graph = g.finish();
    let honest = witness(&[("x", 0), ("y", 63)]);
    let emitted = emit_circuit(&graph, &honest, &ProvingParams::default()).unwrap();
    assert!(emitted.main_nr.contains("array_index::<8>("), "{}", emitted.main_nr);
    assert_eq!(execute(&graph, &honest).unwrap(), vec![10, 73]);
    assert_rejected(execute(&graph, &witness(&[("x", -1), ("y", 5)])));
    assert_rejected(execute(&graph, &witness(&[("x", 0), ("y", 64)])));
}

/// M6: logic ops and selects use Python truthiness, as the mock: `not 5` is/// M6: logic ops and selects use Python truthiness, as the mock: `not 5` is
/// 0 (it was 1 - 5 = -4) and a select on 5 takes the true branch.
#[test]
fn logic_ops_use_truthiness() {
    if !toolchain_present(false) {
        return;
    }
    let mut g = Graph::default();
    let x = g.read_int("x");
    let not_x = g.push(IR::LogicalNot, &[x]);
    g.expose(not_x);
    let one = g.int(1);
    let two = g.int(2);
    let sel = g.push(IR::SelectI, &[x, one, two]);
    g.expose(sel);
    let and = g.push(IR::LogicalAnd, &[x, x]);
    g.expose(and);
    let graph = g.finish();
    assert_eq!(execute(&graph, &witness(&[("x", 5)])).unwrap(), vec![0, 1, 1]);
    assert_eq!(execute(&graph, &witness(&[("x", 0)])).unwrap(), vec![1, 2, 0]);
}

/// M6: the bound of `a // b` was copied from the dividend, so with a
/// negative divisor `(7 // y) < 2` compared a truncated u8 copy of the
/// quotient and gave 0. Python and the mock: 7 // -2 = -4, (-4 < 2) = 1,
/// 7 % -2 = -1; a zero divisor gives 0 (the mock's convention).
#[test]
fn floor_div_by_negative_divisor_compares_correctly() {
    if !toolchain_present(false) {
        return;
    }
    let mut g = Graph::default();
    let seven = g.int(7);
    let y = g.read_int("y");
    let q = g.push(IR::FloorDivI, &[seven, y]);
    let two = g.int(2);
    let lt = g.push(IR::LtI, &[q, two]);
    g.expose(q);
    g.expose(lt);
    let r = g.push(IR::ModI, &[seven, y]);
    g.expose(r);
    let graph = g.finish();
    assert_eq!(execute(&graph, &witness(&[("y", -2)])).unwrap(), vec![-4, 1, -1]);
    assert_eq!(execute(&graph, &witness(&[("y", 2)])).unwrap(), vec![3, 0, 1]);
    assert_eq!(execute(&graph, &witness(&[("y", 0)])).unwrap(), vec![0, 1, 0]);
}

/// Floor division and remainder by a positive constant (the
/// `divmod_by_positive` path), including a negative dividend.
#[test]
fn constant_divisor_division_matches_python() {
    if !toolchain_present(false) {
        return;
    }
    let mut g = Graph::default();
    let x = g.read_int("x");
    let three = g.int(3);
    let q = g.push(IR::FloorDivI, &[x, three]);
    let r = g.push(IR::ModI, &[x, three]);
    g.expose(q);
    g.expose(r);
    let graph = g.finish();
    assert_eq!(execute(&graph, &witness(&[("x", -7)])).unwrap(), vec![-3, 2]);
    assert_eq!(execute(&graph, &witness(&[("x", 7)])).unwrap(), vec![2, 1]);
}

/// H-10: the artifact's public values are the values bb proved, and
/// `verify` checks the proof against them: a tampered output is rejected.
#[test]
fn verify_rejects_tampered_public_value() {
    if !toolchain_present(true) {
        return;
    }
    let mut g = Graph::default();
    let x = g.read_int("x");
    let one = g.int(1);
    let y = g.push(IR::AddI, &[x, one]);
    g.expose(y);
    let graph = g.finish();
    let backend = NoirProverBackend;
    let artifact = backend
        .prove(&graph, &witness(&[("x", 41)]), &ProvingParams::default())
        .expect("prove");
    assert_eq!(artifact.public_values.len(), 1);
    assert_eq!(signed(&artifact.public_values[0]), 42);
    assert!(backend.verify(&artifact).unwrap().valid);
    let mut tampered = artifact.clone();
    tampered.public_values[0] = format!("0x{:064x}", 43);
    assert!(!backend.verify(&tampered).unwrap().valid);
}

// ── Every lowered op against the mock backend ────────────────────────────

/// The mock backend's public outputs of `graph`, in statement order, as
/// signed integers (Q32 for floats), plus whether its asserts held.
fn mock_outputs(graph: &IRGraph, w: &ResolvedWitness) -> (bool, Vec<i128>) {
    use crate::prove::interpreter::interpret_ir;
    use crate::prove::mock::synthesizer::MockSynthesizer;
    let mut synth = MockSynthesizer::new(w.clone(), ProvingParams::default());
    interpret_ir(graph, &mut synth).expect("mock run");
    let mut outs: Vec<(u32, i128)> = synth
        .public_outputs
        .iter()
        .map(|(label, value)| {
            let id = label.rsplit('_').next().unwrap().parse::<u32>().unwrap();
            (id, value.parse::<i128>().unwrap())
        })
        .collect();
    outs.sort();
    (synth.satisfied, outs.into_iter().map(|(_, v)| v).collect())
}

fn float_witness(ints: &[(&str, i64)], floats: &[(&str, f64)]) -> ResolvedWitness {
    let mut w = witness(ints);
    for (name, value) in floats {
        w.input.params.push(InputParam {
            name: name.to_string(),
            is_public: false,
            dtype: serde_json::Value::Null,
            value: InputNode::Float(*value),
        });
    }
    w
}

/// Every IR op the emitter lowers, on signed, fractional and zero operands,
/// compared position by position with the mock backend: exact for every op
/// the mock computes in fixed point (including sqrt, sin, cos, tan and
/// arcsin, which run the kernel's algorithms), within 1e-6 (relative +
/// absolute) for the ops the mock evaluates in f64 (pow, sinh, cosh, tanh,
/// exp, log, arccos, arctan2). Ops whose mock semantics departs from NumPy
/// are listed (and excluded) at the end.
#[test]
fn every_lowered_op_matches_the_mock() {
    if !toolchain_present(false) {
        return;
    }
    let mut g = Graph::default();
    let mut approx: Vec<bool> = Vec::new();
    let expose_i = |g: &mut Graph, approx: &mut Vec<bool>, c: u32| {
        g.push(IR::ExposePublicI, &[c]);
        approx.push(false);
    };
    let expose_f = |g: &mut Graph, approx: &mut Vec<bool>, c: u32, is_approx: bool| {
        g.push(IR::ExposePublicF, &[c]);
        approx.push(is_approx);
    };
    let read_f = |g: &mut Graph, name: &str| {
        g.push(
            IR::ReadFloat {
                path: InputPath::new(name, vec![]),
                is_public: false,
            },
            &[],
        )
    };

    // Integers: x = -7, y = 3, z = 5 (a non-0/1 truth value).
    let x = g.read_int("x");
    let y = g.read_int("y");
    let z = g.read_int("z");
    let zero_i = g.read_int("zi");
    let zero = g.int(0);
    let neg3 = g.int(-3);
    let c21 = g.int(21);
    let c13 = g.int(13);
    let c10 = g.int(10);
    let neg3_dyn = g.read_int("w");
    for (ir, args) in [
        (IR::AddI, vec![x, y]),
        (IR::SubI, vec![x, y]),
        (IR::MulI, vec![x, y]),
        (IR::DivI, vec![c21, y]),
        (IR::FloorDivI, vec![x, y]),
        (IR::ModI, vec![x, y]),
        (IR::FloorDivI, vec![x, neg3]),
        (IR::ModI, vec![x, neg3]),
        (IR::FloorDivI, vec![c13, y]),
        (IR::FloorDivI, vec![x, zero_i]),
        (IR::ModI, vec![x, zero_i]),
        (IR::DivI, vec![c21, zero_i]),
        (IR::FloorDivIPinned { divisor_bits: 4 }, vec![c13, y]),
        (IR::AbsI, vec![x]),
        (IR::SignI, vec![x]),
        (IR::EqI, vec![x, x]),
        (IR::NeI, vec![x, y]),
        (IR::LtI, vec![x, y]),
        (IR::LteI, vec![y, x]),
        (IR::GtI, vec![x, y]),
        (IR::GteI, vec![y, y]),
        (IR::LtIBounded, vec![y, z]),
        (IR::GtIBounded, vec![y, z]),
        (IR::IndexInBox { rank: 1 }, vec![y, zero, z]),
        (IR::IndexInBox { rank: 1 }, vec![z, zero, y]),
        (IR::LogicalAnd, vec![z, zero]),
        (IR::LogicalOr, vec![z, zero]),
        (IR::LogicalNot, vec![z]),
        (IR::SelectI, vec![z, x, y]),
        (IR::SelectManyI { len: 1, lane: 0 }, vec![zero, x, y]),
        (IR::BoolCast, vec![x]),
        (IR::AssertU128LtBit { hi: 8, bit: 1, bits: 3 }, vec![y]),
        // Bitwise (two's complement, x = -7 = ...11111001): constant masks
        // of both signs, dynamic operands, and `~`.
        (IR::BitAndI, vec![x, c13]),
        (IR::BitAndI, vec![c13, x]),
        (IR::BitAndI, vec![x, neg3]),
        (IR::BitAndI, vec![x, y]),
        (IR::BitAndI, vec![x, neg3_dyn]),
        (IR::BitOrI, vec![x, c10]),
        (IR::BitOrI, vec![x, y]),
        (IR::BitXorI, vec![x, c21]),
        (IR::BitXorI, vec![y, z]),
        (IR::BitXorI, vec![x, neg3_dyn]),
        (IR::BitNotI, vec![x]),
    ] {
        let r = g.push(ir, &args);
        expose_i(&mut g, &mut approx, r);
    }
    // Bit tests: (x & 2^k) compared with 0 (a scaled bit), and a multi-bit
    // mask compared with 0 on both sides.
    for k in 0..4 {
        let mask = g.int(1 << k);
        let bit = g.push(IR::BitAndI, &[x, mask]);
        for (ir, args) in [
            (IR::GtI, vec![bit, zero]),
            (IR::NeI, vec![bit, zero]),
            (IR::EqI, vec![zero, bit]),
            (IR::LteI, vec![bit, zero]),
        ] {
            let r = g.push(ir, &args);
            expose_i(&mut g, &mut approx, r);
        }
    }
    let low3 = g.push(IR::BitAndI, &[y, c13]);
    for (ir, args) in [
        (IR::GtI, vec![low3, zero]),
        (IR::LtI, vec![zero, low3]),
        (IR::GteI, vec![low3, zero]),
        (IR::LtI, vec![low3, zero]),
    ] {
        let r = g.push(ir, &args);
        expose_i(&mut g, &mut approx, r);
    }
    let eq = g.push(IR::EqI, &[x, x]);
    g.push(IR::Assert, &[eq]);
    let lt = g.push(IR::LtI, &[x, y]);
    g.push(IR::AssertBit, &[lt]);
    g.push(IR::AssertNonNegI, &[y]);
    g.push(IR::AssertU128Lt, &[y, c10]);

    // Floats: f = -1.75, h = 2.5, s = 0.6.
    let f = read_f(&mut g, "f");
    let h = read_f(&mut g, "h");
    let s = read_f(&mut g, "s");
    let neg_s = g.push(IR::ConstantFloat { value: -0.6 }, &[]);
    let zero_f = read_f(&mut g, "zf");
    let big = g.push(IR::ConstantFloat { value: 277.0 }, &[]);
    let three = g.int(3);
    let one = g.int(1);
    // Integral floats (exact products): float(x) = -7.0 and the constant -3.0.
    let float_x = g.push(IR::FloatCast, &[x]);
    let neg_three_f = g.push(IR::ConstantFloat { value: -3.0 }, &[]);
    // Constant divisors: integral (power of two or not), fractional, negative, zero.
    let divisors: Vec<u32> = [2.0, 3.0, -4.0, 0.5, 0.6, -0.6, 30.0, 0.0]
        .iter()
        .map(|&value| g.push(IR::ConstantFloat { value }, &[]))
        .collect();
    for &d in &divisors {
        for numerator in [f, h, s] {
            let q = g.push(IR::DivF, &[numerator, d]);
            expose_f(&mut g, &mut approx, q, false);
            let fq = g.push(IR::FloorDivF, &[numerator, d]);
            expose_f(&mut g, &mut approx, fq, false);
            let m = g.push(IR::ModF, &[numerator, d]);
            expose_f(&mut g, &mut approx, m, false);
        }
    }
    for divisor in [4, 7, 1] {
        let d = g.int(divisor);
        let q = g.push(IR::FloorDivI, &[x, d]);
        expose_i(&mut g, &mut approx, q);
        let m = g.push(IR::ModI, &[x, d]);
        expose_i(&mut g, &mut approx, m);
    }
    for (ir, args, is_approx) in [
        (IR::MulF, vec![float_x, f], false),
        (IR::MulF, vec![s, float_x], false),
        (IR::MulF, vec![neg_three_f, s], false),
        (IR::ProductSumF { len: 3 }, vec![float_x, f, s, h, f, neg_three_f], false),
        (IR::ProductSumF { len: 2 }, vec![float_x, f, neg_three_f, s], false),
        (IR::AddF, vec![f, s], false),
        (IR::SubF, vec![f, h], false),
        (IR::MulF, vec![f, s], false),
        (IR::DivF, vec![f, s], false),
        (IR::DivF, vec![f, neg_s], false),
        (IR::SumF { len: 3 }, vec![f, h, s], false),
        (IR::ProductSumF { len: 2 }, vec![f, s, h, f], false),
        (IR::AbsF, vec![f], false),
        (IR::FloatCast, vec![x], false),
        (IR::SelectF, vec![lt, f, h], false),
        (IR::SelectBitF, vec![one, f, h], false),
        (IR::SelectManyF { len: 1, lane: 0 }, vec![z, f, h], false),
        (IR::FloorDivF, vec![f, s], false),
        (IR::FloorDivF, vec![h, neg_s], false),
        (IR::ModF, vec![f, s], false),
        (IR::ModF, vec![h, neg_s], false),
        (IR::DivF, vec![f, zero_f], false),
        (IR::FloorDivF, vec![f, zero_f], false),
        (IR::ModF, vec![f, zero_f], false),
        (IR::SinF, vec![h], false),
        (IR::SinF, vec![big], false),
        (IR::CosF, vec![f], false),
        (IR::TanF, vec![s], false),
        (IR::SqrtF, vec![h], false),
        (IR::SqrtF, vec![f], false),
        (IR::ArcSinF, vec![s], false),
        (IR::ArcSinF, vec![f], false),
        (IR::PowF, vec![h, s], true),
        (IR::PowFI, vec![h, three], true),
        (IR::SinHF, vec![f], true),
        (IR::CosHF, vec![s], true),
        (IR::TanHF, vec![f], true),
        (IR::ExpF, vec![f], true),
        (IR::LogF, vec![h], true),
        (IR::ArcCosF, vec![s], true),
        (IR::ArcTan2F, vec![f, h], true),
    ] {
        let r = g.push(ir, &args);
        expose_f(&mut g, &mut approx, r, is_approx);
    }
    for ir in [IR::EqF, IR::NeF, IR::LtF, IR::LteF, IR::GtF, IR::GteF] {
        let r = g.push(ir, &[f, s]);
        expose_i(&mut g, &mut approx, r);
    }
    // Comparisons with zero share the sign of `sign_abs` with abs, sqrt and
    // int casts of the same value (f < 0, h > 0, zf == 0).
    let zero_float = g.push(IR::ConstantFloat { value: 0.0 }, &[]);
    for v in [f, h, zero_f] {
        for (ir, args) in [
            (IR::LtF, vec![v, zero_float]),
            (IR::GteF, vec![v, zero_float]),
            (IR::GtF, vec![v, zero_float]),
            (IR::LteF, vec![v, zero_float]),
            (IR::LtF, vec![zero_float, v]),
            (IR::GteF, vec![zero_float, v]),
        ] {
            let r = g.push(ir, &args);
            expose_i(&mut g, &mut approx, r);
        }
    }
    for (ir, args) in [
        (IR::LtI, vec![x, zero]),
        (IR::GtI, vec![x, zero]),
        (IR::LteI, vec![zero, x]),
        (IR::GteI, vec![y, zero]),
    ] {
        let r = g.push(ir, &args);
        expose_i(&mut g, &mut approx, r);
    }
    let i = g.push(IR::IntCast, &[f]);
    expose_i(&mut g, &mut approx, i);

    let graph = g.finish();
    let w = float_witness(
        &[("x", -7), ("y", 3), ("z", 5), ("zi", 0), ("w", -3)],
        &[("f", -1.75), ("h", 2.5), ("s", 0.6), ("zf", 0.0)],
    );
    let (satisfied, mock) = mock_outputs(&graph, &w);
    assert!(satisfied);
    let noir = execute(&graph, &w).expect("noir execute");
    assert_eq!(noir.len(), mock.len());
    let q = (1u64 << 32) as f64;
    for (pos, ((m, n), is_approx)) in mock.iter().zip(&noir).zip(&approx).enumerate() {
        if *is_approx {
            let tol = 1e-6 * q + 1e-6 * (*m as f64).abs();
            assert!(((m - n) as f64).abs() <= tol, "output {}: mock {} noir {}", pos, m, n);
        } else {
            assert_eq!(m, n, "output {}", pos);
        }
    }

    // Not compared: SignF (the mock returns the integer +-1 rather than
    // +-1.0; Noir returns +-1.0 as NumPy) and PowFI with a negative base
    // (the mock evaluates exp(n * log(base)), NaN for base < 0).
}

// ── Prover hints ──────────────────────────────────────────────────────────

/// `IR::Hint` lanes from `noir_fpc`'s unconstrained sort equal the mock's
/// (`kernel::hint_lane`): constant ranks index one hint array, a runtime
/// rank uses the `_at` form, and ties keep their original order.
#[test]
fn hint_lanes_match_the_mock() {
    use crate::ir_defs::HintKind;
    if !toolchain_present(false) {
        return;
    }
    let values = [("a0", 30), ("a1", -10), ("a2", 20), ("a3", -10), ("a4", 0)];
    let mut g = Graph::default();
    let cells: Vec<u32> = values.iter().map(|(name, _)| g.read_int(name)).collect();
    let rank = g.read_int("r");
    for kind in [HintKind::StableArgsort, HintKind::SortedValues] {
        for constant_rank in [0, 1, 4, 5] {
            let mut operands = cells.clone();
            operands.push(g.int(constant_rank));
            let lane = g.push(IR::Hint { kind, len: 5 }, &operands);
            g.expose(lane);
        }
        let mut operands = cells.clone();
        operands.push(rank);
        let lane = g.push(IR::Hint { kind, len: 5 }, &operands);
        g.expose(lane);
    }
    let graph = g.finish();
    for runtime_rank in [2, 7, -1] {
        let mut inputs = values.to_vec();
        inputs.push(("r", runtime_rank));
        let w = witness(&inputs);
        let (ok, mock) = mock_outputs(&graph, &w);
        assert!(ok);
        assert_eq!(execute(&graph, &w).expect("nargo execute"), mock, "rank {runtime_rank}");
    }
}

// ── Sort lowerings (optim::sort_lowering) ─────────────────────────────────

/// Each sort lowering with hints, emitted to Noir with its hints turned
/// into free private inputs (a malicious prover's choice): the honest
/// values give NumPy's stable sort, and every other index for an argsort
/// lane (a duplicate, a non-permutation, a tie partner: an unstable order)
/// or every other candidate for an order statistic fails a constraint.
#[test]
fn sort_hints_are_pinned_in_noir() {
    use crate::optim::sort_lowering::test_support::{
        hints_as_inputs, sort_graph, tampered_inputs, Forced, Output,
    };
    if !toolchain_present(false) {
        return;
    }
    let values = [3, 1, 3, 0, 1];
    let n = values.len() as i64;
    let every: Vec<Output> = (0..n as u32).flat_map(|r| [Output::Value(r), Output::Index(r)]).collect();
    let cases: Vec<(Forced, Vec<Output>)> = vec![
        (Forced::Permutation, every),
        (Forced::Runs, vec![Output::Index(3), Output::Index(4)]),
        (Forced::Counting, vec![Output::Value(2)]),
        (Forced::Runs, vec![Output::ValueAt(1), Output::IndexAt(3)]),
    ];
    for (forced, outputs) in cases {
        let sort = sort_graph(&values, &outputs, forced);
        let (graph, sites) = hints_as_inputs(&sort);
        let as_witness = |inputs: &[(String, i64)]| {
            let named: Vec<(&str, i64)> = inputs.iter().map(|(k, v)| (k.as_str(), *v)).collect();
            witness(&named)
        };
        let honest = tampered_inputs(&sort, &sites, usize::MAX, 0);
        let got = execute(&graph, &as_witness(&honest)).expect("honest hints");
        let expected: Vec<i128> = sort.expected.as_ref().unwrap().iter().map(|&v| v as i128).collect();
        assert_eq!(got, expected, "{forced:?} {outputs:?}");
        for (k, site) in sites.iter().enumerate() {
            let value_pin = matches!(forced, Forced::Counting);
            let candidates: Vec<i64> = if site.honest >= 0 && site.honest < n && !value_pin {
                (-1..=n).collect() // an argsort lane
            } else {
                values.iter().flat_map(|&v| [v - 1, v, v + 1]).collect()
            };
            for value in candidates.into_iter().filter(|&v| v != site.honest) {
                assert_rejected(execute(&graph, &as_witness(&tampered_inputs(&sort, &sites, k, value))));
            }
        }
    }
}

/// `IR::KeyLess` by one gap check of packed keys: equal to the mock on ties,
/// index order and window extremes; asserted, it rejects a false order.
#[test]
fn key_comparison_matches_the_mock_and_rejects_false_assertions() {
    if !toolchain_present(false) {
        return;
    }
    let (max, min) = (i64::MAX, i64::MIN);
    let pairs: [(i64, i64, i64, i64); 7] =
        [(1, 5, 2, 0), (2, 0, 1, 5), (3, 4, 3, 5), (3, 5, 3, 4), (3, 5, 3, 5), (min, 127, max, 0), (max, 0, min, 127)];
    let mut g = Graph::default();
    let cells: Vec<u32> = ["v1", "i1", "v2", "i2"].iter().map(|n| g.read_int(n)).collect();
    let less = g.push(IR::KeyLess { index_bits: 7 }, &cells);
    g.expose(less);
    let graph = g.finish();
    for (v1, i1, v2, i2) in pairs {
        let w = witness(&[("v1", v1), ("i1", i1), ("v2", v2), ("i2", i2)]);
        let (ok, mock) = mock_outputs(&graph, &w);
        assert!(ok);
        assert_eq!(mock, vec![((v1, i1) < (v2, i2)) as i128]);
        assert_eq!(execute(&graph, &w).expect("nargo execute"), mock, "({v1}, {i1}) < ({v2}, {i2})");
    }
    // Asserted: the direct gap check holds for a true order, fails otherwise.
    let mut g = Graph::default();
    let cells: Vec<u32> = ["v1", "i1", "v2", "i2"].iter().map(|n| g.read_int(n)).collect();
    let less = g.push(IR::KeyLess { index_bits: 7 }, &cells);
    g.push(IR::Assert, &[less]);
    let one = g.int(1);
    g.expose(one);
    let asserted = g.finish();
    for (v1, i1, v2, i2) in pairs {
        let w = witness(&[("v1", v1), ("i1", i1), ("v2", v2), ("i2", i2)]);
        if (v1, i1) < (v2, i2) {
            assert!(execute(&asserted, &w).is_ok(), "({v1}, {i1}) < ({v2}, {i2})");
        } else {
            assert_rejected(execute(&asserted, &w));
        }
    }
}
