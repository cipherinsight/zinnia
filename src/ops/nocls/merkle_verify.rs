use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct MerkleVerifyOp;

impl MerkleVerifyOp {
    const PARAMS: [ParamEntry; 4] = [
        ParamEntry::required("leaf"),
        ParamEntry::required("root"),
        ParamEntry::required("siblings"),
        ParamEntry::required("directions"),
    ];
}

impl Op for MerkleVerifyOp {
    fn name(&self) -> &'static str { "merkle_verify" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let leaf = args.require("leaf");
        let root = args.require("root");
        let siblings = args.require("siblings");
        let directions = args.require("directions");

        let (sib_vals, dir_vals) = match (siblings, directions) {
            (Value::List(s), Value::List(d)) => (&s.values, &d.values),
            (Value::Tuple(s), Value::Tuple(d)) => (&s.values, &d.values),
            (Value::Tuple(s), Value::List(d)) => (&s.values, &d.values),
            (Value::List(s), Value::Tuple(d)) => (&s.values, &d.values),
            _ => panic!("merkle_verify: siblings and directions must be List or Tuple"),
        };
        assert_eq!(
            sib_vals.len(),
            dir_vals.len(),
            "merkle_verify: siblings and directions must have the same length"
        );

        let mut acc = leaf.clone();
        for (sib, dir) in sib_vals.iter().zip(dir_vals.iter()) {
            // dir == 0 means acc is on the left, sib on the right
            // dir == 1 means sib is on the left, acc on the right
            let left = builder.ir_select_i(dir, sib, &acc);
            let right = builder.ir_select_i(dir, &acc, sib);
            acc = builder.ir_poseidon_hash(&[left, right]);
        }

        builder.ir_equal_i(&acc, root)
    }
}
