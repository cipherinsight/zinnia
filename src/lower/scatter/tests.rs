//! Mock-backend tests of the scatter family: NumPy equality on edge shapes,
//! no witness where NumPy raises, and every hint pinned (exhaustively over
//! the prover's choices for small axes).

use super::insert::Form;
use super::position::Positions;
use super::test_support::*;

/// The honest witness gives NumPy's one-hot and thermometer for every index
/// that names a position, and there is no witness for any other index.
#[test]
fn position_one_hot_matches_numpy_and_rejects_bad_indices() {
    for positions in [Positions::Element, Positions::InsertionPoint] {
        for n in 0..=4usize {
            for x in -(n as i64) - 2..=n as i64 + 2 {
                let case = position_graph(x, n, positions);
                let (ok, outputs) = mock_run(&case.graph, &case.inputs);
                match &case.expected {
                    Some(expected) => {
                        assert!(ok, "{positions:?} n={n} x={x}: rejected");
                        assert_eq!(&outputs, expected, "{positions:?} n={n} x={x}");
                    }
                    None => assert!(!ok, "{positions:?} n={n} x={x}: accepted, NumPy raises"),
                }
            }
        }
    }
}

/// Every assignment of the hint lanes (all 0/1 vectors, and each lane set
/// to 2 or -1) other than the honest one is rejected, for every index.
#[test]
fn position_one_hot_hints_are_pinned() {
    for positions in [Positions::Element, Positions::InsertionPoint] {
        for n in 1..=3usize {
            for x in -(n as i64) - 1..=n as i64 + 1 {
                let case = position_graph(x, n, positions);
                let (graph, sites) = hints_as_inputs(&case);
                let honest: Vec<i64> = sites.iter().map(|s| s.honest).collect();
                let mut choices: Vec<Vec<i64>> = (0..1u32 << sites.len())
                    .map(|bits| (0..sites.len()).map(|k| (bits >> k & 1) as i64).collect())
                    .collect();
                for k in 0..sites.len() {
                    for v in [2, -1] {
                        let mut c = honest.clone();
                        c[k] = v;
                        choices.push(c);
                    }
                }
                for hints in choices {
                    let (ok, _) = mock_run(&graph, &with_hints(&case, &sites, &hints));
                    let should = case.expected.is_some() && hints == honest;
                    assert_eq!(ok, should, "{positions:?} n={n} x={x} hints={hints:?}");
                }
            }
        }
    }
}

/// `np.insert` along axis 0 equals NumPy for every position, including
/// negative ones, on empty, single-row and multi-lane arrays, in both forms;
/// there is no witness for a position NumPy rejects.
#[test]
fn insert_matches_numpy() {
    let shapes: [(usize, usize); 5] = [(0, 1), (1, 1), (3, 1), (2, 3), (4, 2)];
    for form in [Form::Complement, Form::SharedProduct] {
        for (n, lanes) in shapes {
            let rows: Vec<Vec<i64>> =
                (0..n).map(|k| (0..lanes).map(|j| 10 * k as i64 + j as i64 + 1).collect()).collect();
            let inserted: Vec<i64> = (0..lanes).map(|j| -7 - j as i64).collect();
            for pos in -(n as i64) - 2..=n as i64 + 2 {
                let case = insert_graph(&rows, &inserted, pos, form);
                let (ok, outputs) = mock_run(&case.graph, &case.inputs);
                match &case.expected {
                    Some(expected) => {
                        assert!(ok, "{form:?} {n}x{lanes} pos={pos}: rejected");
                        assert_eq!(&outputs, expected, "{form:?} {n}x{lanes} pos={pos}");
                    }
                    None => assert!(!ok, "{form:?} {n}x{lanes} pos={pos}: accepted, NumPy raises"),
                }
            }
        }
    }
}

/// The insertion's only hints are its position one-hot: each one changed
/// (to the other bit, or to a non-bit) is rejected.
#[test]
fn insert_hints_are_pinned() {
    let rows = vec![vec![5, 6], vec![7, 8], vec![9, 10]];
    for form in [Form::Complement, Form::SharedProduct] {
        for pos in [-3, -1, 0, 2, 3] {
            let case = insert_graph(&rows, &[1, 2], pos, form);
            let (graph, sites) = hints_as_inputs(&case);
            assert_eq!(sites.len(), rows.len() + 1);
            let honest: Vec<i64> = sites.iter().map(|s| s.honest).collect();
            assert!(mock_run(&graph, &with_hints(&case, &sites, &honest)).0);
            for k in 0..sites.len() {
                for v in [1 - honest[k], 2, -1] {
                    let mut hints = honest.clone();
                    hints[k] = v;
                    assert!(!mock_run(&graph, &with_hints(&case, &sites, &hints)).0, "{form:?} pos={pos} hint {k} = {v}");
                }
            }
        }
    }
}

/// A chain of masked writes (0490's three writes) lowered as a decision
/// list equals the replayed writes for cells below, at and above both
/// thresholds, in either threshold order; it uses fewer selects than the
/// select per cell, which gives the same values.
#[test]
fn mask_write_chain_matches_replayed_writes() {
    use super::{MaskWriteByDecisionLists, MaskWriteBySelects};
    for (t1, t2) in [(3, 7), (7, 3), (5, 5)] {
        for x in -1..=9 {
            let list = mask_write_chain_graph(x, t1, t2, &MaskWriteByDecisionLists);
            let plain = mask_write_chain_graph(x, t1, t2, &MaskWriteBySelects);
            for case in [&list, &plain] {
                let (ok, outputs) = mock_run(&case.graph, &case.inputs);
                assert!(ok, "x={x} t1={t1} t2={t2}");
                assert_eq!(Some(outputs), case.expected, "x={x} t1={t1} t2={t2}");
            }
            assert!(selects(&list.graph) < selects(&plain.graph), "the decision list was not used");
        }
    }
}
