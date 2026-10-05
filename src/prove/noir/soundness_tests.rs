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

/// B4(b): with u32 promotion on (>= 20 intrinsic ops), `AssertU128Lt` used
/// to check only the low 32 bits of the index (`idx as u32 < hi`).
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

/// M6: logic ops and selects use Python truthiness, as the mock: `not 5` is
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
/// negative divisor `(7 // y) < 2` compared a truncated u8 copy of -3 and
/// returned 0. The mock (Euclidean `//`) gives 1.
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
    assert_eq!(execute(&graph, &witness(&[("y", -2)])).unwrap(), vec![-3, 1, 1]);
    assert_eq!(execute(&graph, &witness(&[("y", 2)])).unwrap(), vec![3, 0, 1]);
    assert_rejected(execute(&graph, &witness(&[("y", 0)])));
}

/// Euclidean division and remainder by a positive constant (the
/// `euclid_divmod_small` path), including a negative dividend.
#[test]
fn constant_divisor_division_matches_euclid() {
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
